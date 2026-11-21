"""跨多轮办理的显式状态。任务字段来自用户，缺失字段由程序计算。"""

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from business.schemas import InputModel, Priority, TicketStatus, TicketType

Operation = Literal["order_query", "ticket_create", "ticket_query", "ticket_update"]
Slot = Literal["order_id", "ticket_id", "ticket_type", "priority", "summary", "details", "status", "list_all"]


class TaskFields(InputModel):
    order_id: str | None = Field(default=None, min_length=1, max_length=64)
    ticket_id: str | None = Field(default=None, min_length=1, max_length=64)
    ticket_type: TicketType | None = None
    priority: Priority | None = None
    summary: str | None = Field(default=None, min_length=1, max_length=200)
    details: str | None = Field(default=None, min_length=1, max_length=4000)
    status: TicketStatus | None = None
    list_all: bool = False
    # 来自数据库，不允许模型填写。
    expected_version: int | None = Field(default=None, ge=1)


class TaskState(BaseModel):
    task_id: str = Field(default_factory=lambda: "task-" + uuid.uuid4().hex)
    action: Operation
    stage: Literal["collecting", "awaiting_confirmation", "completed", "cancelled"] = "collecting"
    fields: TaskFields = Field(default_factory=TaskFields)
    missing_fields: list[str] = Field(default_factory=list)
    revision: int = 1
    last_error: str | None = None
    result: dict | None = None


class BusinessDecision(InputModel):
    """单轮增量：未提及的字段为None，不覆盖已经收集的信息。"""
    action: Literal["order_query", "ticket_create", "ticket_query", "ticket_update", "continue", "cancel", "confirm", "clarify"]
    order_id: str | None = Field(default=None, min_length=1, max_length=64)
    ticket_id: str | None = Field(default=None, min_length=1, max_length=64)
    ticket_type: TicketType | None = None
    priority: Priority | None = None
    summary: str | None = Field(default=None, min_length=1, max_length=200)
    details: str | None = Field(default=None, min_length=1, max_length=4000)
    status: TicketStatus | None = None
    list_all: bool | None = None
    clear_fields: list[Slot] = Field(default_factory=list)
    new_task: bool = False
    clarification: str = "请补充需要办理的业务信息。"


CONFIRM_WORDS = {"确认", "确认提交", "确认修改", "确定提交", "提交申请", "确认办理"}
CANCEL_WORDS = {"取消", "取消办理", "取消申请", "取消当前任务", "不办了", "先不办了", "算了"}


def control_word(text: str) -> str:
    return text.strip().rstrip("。！! ")


def pending_task(value: dict | TaskState | None) -> bool:
    if value is None:
        return False
    task = value if isinstance(value, TaskState) else TaskState.model_validate(value)
    return task.stage in {"collecting", "awaiting_confirmation"}


def missing_fields(task: TaskState) -> list[str]:
    fields = task.fields
    if task.action == "order_query":
        return [] if fields.order_id or fields.list_all else ["order_id"]
    if task.action == "ticket_query":
        return [] if fields.ticket_id or fields.list_all else ["ticket_id"]
    if task.action == "ticket_create":
        missing = []
        if fields.ticket_type in {"refund", "claim"} and not fields.order_id:
            missing.append("order_id")
        if not fields.details:
            missing.append("details")
        return missing
    missing = [] if fields.ticket_id else ["ticket_id"]
    if fields.status is None and fields.details is None:
        missing.append("status_or_details")
    return missing
