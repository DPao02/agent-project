"""
Action Agent（待办Agent）
- 从转写文本中提取行动项（谁/做什么/截止时间）
- 自动同步到 Jira Cloud 和飞书任务
- 支持幂等性保证（防止重复创建）
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from ..integrations.feishu_client import FeishuClient
from ..integrations.jira_client import JiraClient
from ..integrations.deepseek_client import LLMClient
from ..models.schemas import ActionItem, ActionResult, Priority


ACTION_SYSTEM_PROMPT = """你是一位专业的任务提取助手。你的任务是从会议转写文本中提取所有行动项/待办事项。

提取规则：
1. 识别明确分配给某人的任务
2. 提取任务的截止时间（如果提到的话）
3. 判断任务优先级（根据语气和上下文）
4. 记录任务的上下文（为什么要做这件事）

注意：
- 只提取明确的行动项，不要凭空创造
- 截止时间格式为 YYYY-MM-DD
- 如果没有明确截止时间，留空
- 原话明确点名负责人时保留原文姓名，例如“李明负责预算”应提取assignee为“李明”，task为预算任务；不需要将姓名映射为说话人编号。
- 本人说“我来负责”时，负责人使用这句话的说话人编号；没有明确负责人时才填“未明确”。
- 不猜测姓名、职务，也不能把发出指令者自动当成任务执行者。

你必须严格按照JSON格式输出："""

ACTION_USER_PROMPT = """请从以下会议转写文本中提取所有行动项/待办事项。

今天的日期是: {today}

## 会议转写文本
{transcript}

## 输出格式（严格JSON）
{{
  "action_items": [
    {{
      "assignee": "原文明示的负责人姓名，或本人承诺时的SPEAKER_00等编号；未指派时填未明确",
      "task": "具体任务描述",
      "deadline": "YYYY-MM-DD 或空字符串",
      "priority": "low/medium/high/urgent",
      "context": "这个任务的背景说明"
    }}
  ]
}}"""


class ActionAgent:
    """
    待办Agent - 并行阶段的节点之一

    架构说明:
    1. 从 state 读取 transcript_text
    2. LLM 提取行动项三元组（谁/做什么/截止时间）
    3. 并行同步到 Jira 和飞书
    4. 记录同步状态

    面试考点:
    - 如何保证 Jira 同步的幂等性？（基于 meeting_id + task hash 去重）
    - 如何处理人名和 Jira 用户的映射？（企业通讯录 + 模糊匹配）
    - 如果 Jira/飞书不可用怎么办？（降级到本地存储 + 异步重试）
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        jira_client: JiraClient | None = None,
        feishu_client: FeishuClient | None = None,
    ):
        self.llm = llm_client or LLMClient()
        self.jira = jira_client or JiraClient()
        self.feishu = feishu_client or FeishuClient()

    async def process(self, state: dict) -> dict:
        """
        LangGraph 节点函数 —— 提取待办并同步

        与 Summary Agent、Unresolved Agent 并行执行。
        """
        meeting_id = state.get("meeting_id", "unknown")
        logger.info(f"[ActionAgent] Processing meeting: {meeting_id}")

        transcript_text = state.get("transcript_text", "")
        if not transcript_text:
            logger.warning("[ActionAgent] No transcript text available")
            return {"actions": ActionResult(
                meeting_id=meeting_id, action_items=[]
            )}

        errors = []
        try:
            speaker_ids = list(getattr(state.get("transcript"), "speaker_ids", []))
            action_items = await self._extract_actions(transcript_text, speaker_ids)
            synced_items = await self._sync_to_external(action_items, meeting_id)

            actions = ActionResult(
                meeting_id=meeting_id,
                action_items=synced_items,
                sync_status={
                    "jira": "enabled" if self.jira.is_enabled else "disabled",
                    "feishu": "enabled" if self.feishu.is_enabled else "disabled",
                },
            )
            logger.info(
                f"[ActionAgent] Extracted {len(synced_items)} action items"
            )
        except Exception as e:
            logger.error(f"[ActionAgent] Error: {e}")
            errors = [
                f"ActionAgent: {str(e)}"
            ]
            actions = ActionResult(
                meeting_id=meeting_id, action_items=[]
            )

        # errors只包含本步新错误，由MeetingState的reducer追加历史。
        return {"actions": actions, "errors": errors}

    async def _extract_actions(self, transcript: str, speaker_ids: list[str]) -> list[ActionItem]:
        """调用 LLM 提取行动项"""
        today = datetime.now().strftime("%Y-%m-%d")
        messages = [
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ACTION_USER_PROMPT.format(
                    today=today, transcript=transcript
                ),
            },
        ]

        result = await self.llm.chat_json(
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
        )

        items = []
        allowed = set(speaker_ids)
        for raw in result.get("action_items", []):
            priority_str = raw.get("priority", "medium").lower()
            try:
                priority = Priority(priority_str)
            except ValueError:
                priority = Priority.MEDIUM

            assignee = str(raw.get("assignee") or "未明确").strip()
            context = str(raw.get("context") or "")
            # 发言分组用编号，任务分配保留原文明示的姓名，两者不需要身份映射。
            # 姓名应来自转写；模型输出的编号则必须属于本次会议。
            named_in_transcript = (
                bool(assignee)
                and not assignee.startswith("SPEAKER_")
                and assignee not in {"UNKNOWN", "未明确", "未指定", "我", "你", "他", "她", "大家"}
                and assignee in transcript
            )
            if assignee not in allowed and not named_in_transcript:
                assignee = "未明确"
            items.append(
                ActionItem(
                    assignee=assignee,
                    task=raw.get("task", ""),
                    deadline=raw.get("deadline", ""),
                    priority=priority,
                    context=context,
                )
            )

        return items

    async def _sync_to_external(
        self, items: list[ActionItem], meeting_id: str
    ) -> list[ActionItem]:
        """将行动项同步到 Jira 和飞书"""
        synced = []
        for item in items:
            # Jira 同步
            if self.jira.is_enabled:
                try:
                    jira_result = self.jira.create_issue(
                        summary=f"[会议待办] {item.task}",
                        description=(
                            f"来源：会议 {meeting_id}\n"
                            f"负责人：{item.assignee}\n"
                            f"上下文：{item.context}"
                        ),
                        assignee=self.jira.resolve_user(item.assignee),
                        due_date=item.deadline or None,
                        priority=JiraClient.map_priority(item.priority.value),
                        labels=["meeting-auto", f"meeting-{meeting_id}"],
                    )
                    item.jira_issue_key = jira_result["key"]
                except Exception as e:
                    logger.error(
                        f"Failed to sync to Jira: {item.task} - {e}"
                    )

            # 飞书同步
            if self.feishu.is_enabled:
                try:
                    due_ts = None
                    if item.deadline:
                        due_dt = datetime.strptime(item.deadline, "%Y-%m-%d")
                        due_ts = int(due_dt.timestamp())

                    feishu_result = await self.feishu.create_task(
                        summary=f"[会议待办] {item.task}",
                        description=(
                            f"负责人：{item.assignee}\n"
                            f"来源会议：{meeting_id}\n"
                            f"上下文：{item.context}"
                        ),
                        due_timestamp=due_ts,
                    )
                    item.feishu_task_id = feishu_result.get("task_id")
                except Exception as e:
                    logger.error(
                        f"Failed to sync to Feishu: {item.task} - {e}"
                    )

            synced.append(item)

        return synced
