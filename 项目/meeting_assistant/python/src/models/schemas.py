"""统一数据模型：Agent、LangGraph 和 HTTP 输出共用这些结构。

Pydantic 模型约束结果字段；MeetingState 定义图通道。节点返回的字典是
本步的增量更新，errors 使用列表相加合并多个并行节点的新增错误。
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from operator import add
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


class MeetingStatus(str, Enum):
    CREATED = "created"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class MeetingMode(str, Enum):
    REAL = "real"
    DEMO = "demo"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class IssueCategory(str, Enum):
    DECISION = "decision"
    DISAGREEMENT = "disagreement"
    MISSING_INFORMATION = "missing_information"
    DEPENDENCY = "dependency"


class TranscriptSegment(BaseModel):
    # 编号模式：speaker兼容旧输出字段，与speaker_id保持一致。
    speaker_id: str = "UNKNOWN"
    speaker: str = "Unknown"
    text: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    confidence: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def check_time_range(self):
        if self.end < self.start:
            raise ValueError("转写片段的结束时间不能早于开始时间")
        return self


class TranscriptResult(BaseModel):
    meeting_id: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    language: str = "zh"
    duration_seconds: float = Field(default=0.0, ge=0)
    full_text: str = ""
    diarization_available: bool = False
    warnings: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def speaker_ids(self) -> list[str]:
        return list(dict.fromkeys(segment.speaker_id for segment in self.segments))


class TopicSummary(BaseModel):
    title: str
    discussion_points: list[str] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    conclusion: str = ""


class SpeakerSummary(BaseModel):
    speaker_id: str
    points: list[str] = Field(default_factory=list)


class MeetingSummary(BaseModel):
    title: str = "会议纪要"
    date: str = ""
    participants: list[str] = Field(default_factory=list)
    topics: list[TopicSummary] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    speaker_summaries: list[SpeakerSummary] = Field(default_factory=list)


class ActionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ActionItem(BaseModel):
    action_id: str | None = None
    assignee: str = "未明确"
    task: str = Field(min_length=1)
    deadline: str = ""
    priority: Priority = Priority.MEDIUM
    context: str = ""
    jira_issue_key: str | None = None
    feishu_task_id: str | None = None
    status: ActionStatus = ActionStatus.PENDING
    completed_at: datetime | None = None

    @field_validator("deadline")
    @classmethod
    def valid_deadline(cls, value):
        return date.fromisoformat(value).isoformat() if value else ""


class ActionResult(BaseModel):
    meeting_id: str
    action_items: list[ActionItem] = Field(default_factory=list)
    sync_status: dict[str, str] = Field(default_factory=dict)


class IssueEvidence(BaseModel):
    """原文引用；说话人和时间由程序从对应转写片段读取。"""
    segment_index: int = Field(ge=0)
    speaker_id: str
    quote: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def check_time_range(self):
        if self.end < self.start:
            raise ValueError("引用的结束时间不能早于开始时间")
        return self


class UnresolvedIssue(BaseModel):
    issue_id: str
    topic: str = Field(min_length=1)
    description: str = Field(min_length=1)
    category: IssueCategory
    priority: Priority = Priority.MEDIUM
    impact: str = ""
    suggested_next_step: str = ""
    evidence: list[IssueEvidence] = Field(min_length=1)


class UnresolvedResult(BaseModel):
    meeting_id: str
    issues: list[UnresolvedIssue] = Field(default_factory=list)


class FollowUpResult(BaseModel):
    meeting_id: str
    summary_sent: bool = False
    recipients: list[str] = Field(default_factory=list)
    jira_issues_created: list[str] = Field(default_factory=list)
    feishu_tasks_created: list[str] = Field(default_factory=list)
    reminders_scheduled: int = Field(default=0, ge=0, description="数据库中尚待执行的提醒数量")
    report_url: str | None = None
    report_markdown: str = ""


class MeetingState(TypedDict, total=False):
    meeting_id: str
    mode: MeetingMode
    status: MeetingStatus
    audio_data: bytes
    transcript: TranscriptResult | None
    transcript_text: str
    summary: MeetingSummary | None
    actions: ActionResult | None
    unresolved: UnresolvedResult | None
    followup: FollowUpResult | None
    # reducer 接收旧列表和本步新增列表；节点不能再次返回历史错误。
    errors: Annotated[list[str], add]


def create_initial_state(
    meeting_id: str, audio_data: bytes = b"", *, mode: MeetingMode | str = MeetingMode.REAL,
    transcript: TranscriptResult | None = None,
) -> MeetingState:
    """每场会议创建独立状态，尤其不能共用可变列表。"""
    if not meeting_id.strip():
        raise ValueError("meeting_id 不能为空")
    return MeetingState(
        meeting_id=meeting_id,
        mode=MeetingMode(mode),
        status=MeetingStatus.CREATED,
        audio_data=audio_data,
        transcript=transcript,
        transcript_text="",
        summary=None,
        actions=None,
        unresolved=None,
        followup=None,
        errors=[],
    )
