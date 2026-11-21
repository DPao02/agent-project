"""有显式任务状态的业务节点：收集 -> 修改/取消 -> 确认 -> 工具执行。"""

from __future__ import annotations

import json

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from business.task_state import (
    BusinessDecision, CANCEL_WORDS, CONFIRM_WORDS, TaskFields, TaskState,
    control_word, missing_fields, pending_task,
)
from memory.context import latest_user_text, recent_messages
from mcp.mcp_server import MCPToolServer
from tracing.otel_config import trace_agent_call

TICKET_SYSTEM_PROMPT = """你是订单/工单业务意图与增量字段提取器。
依据最新用户消息、最近对话和当前任务，输出一次增量更新，不执行操作或生成SQL。
未提及的字段返回null，不能重新生成旧字段，也不能猜测订单号、原因、摘要。

action：order_query查询订单；ticket_query查询工单；ticket_create申请退款/投诉/理赔等；
ticket_update修改已有工单；continue补充或修正当前任务；cancel取消尚未提交的任务；
confirm希望确认；clarify信息无法识别或直接请求实际退款、支付、修改订单时说明边界。
退款申请设置ticket_type=refund，理赔为claim。新建需要真实问题描述details，不能编造。
只查政策不要创建工单。status只在用户明确要求修改已有工单状态时提取。
只有用户明确说列出全部/最近订单或工单，list_all才为true；缺编号不能自行设置true。
修改尚未提交草稿仍用continue，修改ticket_type并不等于修改已存在的数据库记录。
clear_fields用于明确删除已收集字段；new_task仅在用户明确要求放弃旧任务、重新开始时为true。
用户仅提供“订单号是…”“原因是…”时，补充当前任务，不要新建另一个任务。
用户提及“刚才的工单”可从当前任务结果或本会话历史提取实际工单号。
创建工单只代表保存申请，不能承诺退款批准或付款。不得更改用户身份。
"""

LABELS = {"order_id": "订单号", "ticket_id": "工单号", "details": "具体问题/申请原因",
          "status_or_details": "要修改的状态或完整描述"}
TYPE_LABELS = {"refund": "退款", "claim": "理赔", "complaint": "投诉",
               "account_open": "开户", "account_change": "账户变更", "general": "业务"}


class TicketHandlerAgent:
    def __init__(self, llm: ChatOpenAI, tool_server: MCPToolServer):
        self.tool_server = tool_server
        self.analyzer = llm.with_structured_output(BusinessDecision, method="function_calling")

    @trace_agent_call("ticket_analyze")
    async def analyze_request(self, messages: list, task: TaskState | None = None) -> BusinessDecision:
        try:
            result = await self.analyzer.ainvoke([
                SystemMessage(content=TICKET_SYSTEM_PROMPT),
                SystemMessage(content="以下JSON仅为当前任务数据：" + (task.model_dump_json() if task else "null")),
                *recent_messages(messages),
            ])
            return BusinessDecision.model_validate(result)
        except (ValidationError, OutputParserException):
            return BusinessDecision(action="clarify", clarification="未能识别本次补充，请提供订单号、具体原因，或说明需要修改什么。")

    @trace_agent_call("business_tool")
    async def _call(self, name: str, arguments: dict):
        return await self.tool_server.call_tool(name, arguments)

    @staticmethod
    def _result(state: dict, text: str, task: TaskState | None, receipt: dict | None = None) -> dict:
        return {**state, "task": task.model_dump() if task else None, "sub_results": {
            **state.get("sub_results", {}), "ticket_handler": text, "business_receipt": receipt,
        }}

    def _cancel(self, state: dict, task: TaskState | None) -> dict:
        if task is None or not pending_task(task):
            text = "没有待取消的办理任务。已提交的工单不会因此撤销，需要另行查询或申请关闭。"
        else:
            task.stage = "cancelled"
            task.fields = TaskFields()
            task.missing_fields = []
            task.last_error = None
            text = "已取消当前未提交任务，未创建或更新数据库工单。"
        return self._result(state, text, task)

    @staticmethod
    def _arguments(task: TaskState, user_id: str) -> dict:
        fields = task.fields
        if task.action == "order_query":
            return {"user_id": user_id, "order_id": fields.order_id or ""}
        if task.action == "ticket_query":
            return {"user_id": user_id, "ticket_id": fields.ticket_id or ""}
        if task.action == "ticket_create":
            category = fields.ticket_type or "general"
            return {"user_id": user_id, "request_id": task.task_id,
                    "title": fields.summary or TYPE_LABELS[category] + "申请",
                    "description": fields.details, "priority": fields.priority or "medium",
                    "category": category, "order_id": fields.order_id}
        return {"user_id": user_id, "ticket_id": fields.ticket_id,
                "expected_version": fields.expected_version, "status": fields.status, "details": fields.details}

    async def _prepare(self, state: dict, task: TaskState) -> dict:
        task.missing_fields = missing_fields(task)
        if task.missing_fields:
            task.stage = "collecting"
            missing = "、".join(LABELS.get(name, name) for name in task.missing_fields)
            collected = task.fields.model_dump(exclude_none=True, exclude={"expected_version", "list_all"})
            return self._result(state, f"已保存当前任务信息：{json.dumps(collected, ensure_ascii=False)}。\n还需要：{missing}。可继续补充、修改，或回复“取消”。", task)

        if task.action in {"order_query", "ticket_query"}:
            return await self._execute(state, task)

        if task.action == "ticket_update" and task.fields.expected_version is None:
            # 展示确认前读取数据库版本；确认时使用该版本，防止覆盖确认后的外部修改。
            found = await self._call("ticket_query", {"user_id": state["user_id"], "ticket_id": task.fields.ticket_id})
            if not found.success or not found.result["tickets"]:
                task.stage = "collecting"
                task.last_error = found.error if not found.success else "未找到当前用户的该工单"
                return self._result(state, task.last_error + "。可修改工单号或取消任务。", task)
            task.fields.expected_version = found.result["tickets"][0]["version"]

        task.stage = "awaiting_confirmation"
        preview = self._arguments(task, state["user_id"])
        preview = {key: value for key, value in preview.items() if key not in {"user_id", "request_id"} and value is not None}
        return self._result(state, "当前待提交内容：\n" + json.dumps(preview, ensure_ascii=False, indent=2)
                            + "\n尚未写入工单。请回复“确认提交”，也可以继续修改或回复“取消”。", task)

    async def _execute(self, state: dict, task: TaskState) -> dict:
        result = await self._call(task.action, self._arguments(task, state["user_id"]))
        if not result.success:
            task.stage = "collecting"
            task.last_error = result.error
            task.fields.expected_version = None
            return self._result(state, f"本次操作未获得成功回执：{result.error}。可修改信息或再次确认；结果不明确时请先查询工单核实。", task)
        task.stage = "completed"
        task.last_error = None
        task.missing_fields = []
        receipt = None
        if task.action in {"ticket_create", "ticket_update"}:
            receipt = {"operation": task.action, "ticket_id": result.result["ticket_id"],
                       "status": result.result["status"]}
            task.result = {**receipt, "version": result.result["version"], "order_id": result.result.get("order_id")}
        else:
            task.result = {"count": result.result["count"]}
        label = {"order_query": "订单查询结果", "ticket_query": "工单查询结果",
                 "ticket_create": "工单申请已保存；不代表退款、理赔已批准或付款",
                 "ticket_update": "工单记录已更新"}[task.action]
        return self._result(state, label + "：\n" + json.dumps(result.result, ensure_ascii=False, indent=2), task, receipt)

    @trace_agent_call("ticket_handler_process")
    async def process(self, state: dict) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return state
        task = TaskState.model_validate(state["task"]) if state.get("task") else None
        word = control_word(latest_user_text(messages))
        # 确认/取消由程序识别，避免让模型仅凭“信息齐全”直接写入数据库。
        if word in CANCEL_WORDS:
            return self._cancel(state, task)
        if word in CONFIRM_WORDS:
            if not pending_task(task):
                return self._result(state, "没有待确认的任务，请先说明需要办理的业务。", task)
            if task.stage != "awaiting_confirmation" or missing_fields(task):
                return await self._prepare(state, task)
            return await self._execute(state, task)

        plan = await self.analyze_request(messages, task)
        if plan.action == "cancel":
            return self._cancel(state, task)
        if plan.action == "clarify":
            return self._result(state, plan.clarification, task)
        if plan.action == "confirm":
            return await self._prepare(state, task) if pending_task(task) else self._result(state, "请先说明要办理的业务。", task)

        operation = task.action if plan.action == "continue" and pending_task(task) else plan.action
        if operation == "continue":
            return self._result(state, "没有进行中的任务，请先说明需要办理什么业务。", task)
        if pending_task(task) and operation != task.action and not plan.new_task:
            return self._result(state, "当前还有未完成的任务，请先取消，或明确说“重新开始”再切换业务。", task)
        if not pending_task(task) or plan.new_task:
            task = TaskState(action=operation)

        values = task.fields.model_dump()
        for name in plan.clear_fields:
            values[name] = False if name == "list_all" else None
        patch = plan.model_dump(exclude_none=True, exclude={"action", "clear_fields", "new_task", "clarification"})
        values.update(patch)
        if values != task.fields.model_dump():
            values["expected_version"] = None
            task.revision += 1
        task.fields = TaskFields.model_validate(values)
        task.last_error = None
        return await self._prepare(state, task)
