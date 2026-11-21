"""MySQL保存会议事实；SQLAlchemy负责连接、表映射和事务。

会议快照是查询依据，转写分段/行动项另存结构化表。每次处理使用新版本，
旧分段不会覆盖新会议；运行令牌避免同一会议被两个请求同时写入。
"""
from __future__ import annotations

import hashlib
from copy import deepcopy
import os
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, URL, create_engine, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import load_environment
from .models.schemas import (
    ActionResult, ActionStatus, FollowUpResult, UnresolvedResult, MeetingMode, MeetingStatus,
    MeetingSummary, TranscriptResult, create_initial_state,
)
from .reminders import reminder_plan


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class StorageError(Exception):
    def __init__(self, message, status_code=409):
        super().__init__(message)
        self.status_code = status_code


class Base(DeclarativeBase):
    pass


class MeetingRecord(Base):
    __tablename__ = "ma_meetings"
    meeting_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(255), default="")
    mode: Mapped[str] = mapped_column(String(16), default="real")
    status: Mapped[str] = mapped_column(String(32), default="created")
    revision: Mapped[int] = mapped_column(Integer, default=0)
    run_token: Mapped[str | None] = mapped_column(String(32), nullable=True)
    participants: Mapped[list] = mapped_column(JSON, default=list)
    # 兼容已经创建的旧表列，不再用于输入、输出或姓名映射。
    _legacy_speaker_map: Mapped[dict] = mapped_column("speaker_map", JSON, default=dict)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    audio_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    index_status: Mapped[str] = mapped_column(String(32), default="not_indexed")
    index_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SegmentRecord(Base):
    __tablename__ = "ma_transcript_segments"
    meeting_id: Mapped[str] = mapped_column(ForeignKey("ma_meetings.meeting_id"), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, primary_key=True)
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    speaker_id: Mapped[str] = mapped_column(String(128))
    speaker_name: Mapped[str] = mapped_column(String(128))
    data: Mapped[dict] = mapped_column(JSON)


class ActionRecord(Base):
    __tablename__ = "ma_action_items"
    action_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("ma_meetings.meeting_id"), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    assignee: Mapped[str] = mapped_column(String(128))
    data: Mapped[dict] = mapped_column(JSON)


class ReminderRecord(Base):
    __tablename__ = "ma_action_reminders"
    # ID由待办ID、截止日和提前天数共同确定，重复保存不会生成第二份计划。
    reminder_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    action_id: Mapped[str] = mapped_column(ForeignKey("ma_action_items.action_id"), index=True)
    deadline: Mapped[str] = mapped_column(String(10))
    days_before: Mapped[int] = mapped_column(Integer)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    channel: Mapped[str | None] = mapped_column(String(20), nullable=True)
    claim_token: Mapped[str | None] = mapped_column(String(32), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    send_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


def as_utc(value=None):
    value = value or utcnow()
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


def iso_utc(value):
    return value.replace(tzinfo=timezone.utc).isoformat() if value else None


def serialize_state(state):
    """只保存可重建的业务结果；音频字节由独立文件保存。"""
    result = {
        "meeting_id": state["meeting_id"], "mode": MeetingMode(state["mode"]).value,
        "status": MeetingStatus(state["status"]).value,
        "transcript_text": state.get("transcript_text", ""), "errors": list(state.get("errors", [])),
    }
    for key in ("transcript", "summary", "actions", "unresolved", "followup"):
        value = state.get(key)
        result[key] = value.model_dump(mode="json") if value is not None else None
    return result


def restore_state(result):
    state = create_initial_state(result["meeting_id"], mode=result["mode"])
    state.update(status=MeetingStatus(result["status"]), errors=list(result.get("errors", [])),
                 transcript_text=result.get("transcript_text", ""))
    for key, model in {"transcript": TranscriptResult, "summary": MeetingSummary, "actions": ActionResult,
                       "unresolved": UnresolvedResult, "followup": FollowUpResult}.items():
        state[key] = model.model_validate(result[key]) if result.get(key) is not None else None
    return state


class MeetingRepository:
    def __init__(self, engine):
        self.engine = engine
        self.sessions = sessionmaker(bind=engine, expire_on_commit=False)

    @classmethod
    def from_env(cls):
        load_environment()
        required = ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD")
        missing = [key for key in required if not os.getenv(key)]
        if missing:
            raise ValueError("请配置会议项目MySQL连接：" + ", ".join(missing))
        url = URL.create("mysql+pymysql", host=os.environ["MYSQL_HOST"], port=int(os.environ["MYSQL_PORT"]),
                         database=os.environ["MYSQL_DATABASE"], username=os.environ["MYSQL_USER"],
                         password=os.environ["MYSQL_PASSWORD"], query={"charset": "utf8mb4"})
        return cls(create_engine(url, pool_pre_ping=True, pool_recycle=1800, pool_size=5, max_overflow=5,
                                 hide_parameters=True, connect_args={"connect_timeout": 5, "read_timeout": 30}))

    def initialize(self):
        # 仅创建缺失的ma_表，不修改客服项目的表。
        Base.metadata.create_all(self.engine)
        # 旧待办表继续使用已有JSON字段保存状态，无需删表或改动原有列。
        # 同时为已有会议的当前待办补齐提醒计划，重启可重复执行。
        with self.sessions.begin() as session:
            actions = session.scalars(select(ActionRecord).join(MeetingRecord).where(
                ActionRecord.revision == MeetingRecord.revision)).all()
            for action in actions:
                self._plan_reminders(session, action)

    @staticmethod
    def _dict(row):
        return {"meeting_id": row.meeting_id, "user_id": row.user_id, "title": row.title,
                "mode": row.mode, "status": row.status, "revision": row.revision,
                "participants": row.participants,
                "result": row.result, "audio_path": row.audio_path, "index_status": row.index_status,
                "index_error": row.index_error, "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat()}

    @staticmethod
    def _owned(session, meeting_id, user_id, lock=False):
        stmt = select(MeetingRecord).where(MeetingRecord.meeting_id == meeting_id)
        row = session.scalar(stmt.with_for_update() if lock else stmt)
        if row is None or row.user_id != user_id:
            raise StorageError("会议不存在或不属于当前用户", 404)
        return row

    def create(self, meeting_id, user_id="anonymous", title="", allow_existing=False):
        with self.sessions.begin() as session:
            existing = session.get(MeetingRecord, meeting_id)
            if existing is not None:
                if existing.user_id != user_id:
                    raise StorageError("会议不存在或不属于当前用户", 404)
                if not allow_existing:
                    raise StorageError("会议编号已存在")
                return self._dict(existing)
            row = MeetingRecord(meeting_id=meeting_id, user_id=user_id, title=title, participants=[])
            session.add(row)
            session.flush()
            result = self._dict(row)
        return result

    def get(self, meeting_id, user_id="anonymous"):
        with self.sessions() as session:
            row = self._owned(session, meeting_id, user_id)
            result = self._dict(row)
            if result["result"]:
                result["result"] = deepcopy(result["result"])
                # 旧会议可读取原转写后重新分析；旧洞察不能当作新未决事项返回。
                result["result"].pop("insights", None)
                result["result"].setdefault("unresolved", None)
            # 会议正文保留分析快照；待办状态始终从结构化表读取，完成操作即时可见。
            if result["result"] and result["result"].get("actions") is not None:
                actions = self._current_actions(session, row)
                result["result"]["actions"]["action_items"] = [self._action_dict(a) for a in actions]
                if result["result"].get("followup"):
                    result["result"]["followup"]["reminders_scheduled"] = self._pending_count(session, actions)
            return result

    def list(self, user_id="anonymous", limit=20):
        with self.sessions() as session:
            rows = session.scalars(select(MeetingRecord).where(MeetingRecord.user_id == user_id)
                                   .order_by(MeetingRecord.updated_at.desc()).limit(limit)).all()
            return [{k: v for k, v in self._dict(row).items() if k not in {"result", "audio_path"}} for row in rows]

    def claim(self, meeting_id, user_id, mode, expected_revision=None):
        with self.sessions.begin() as session:
            row = self._owned(session, meeting_id, user_id, lock=True)
            if row.run_token is not None:
                raise StorageError("本会议正在处理，请等待当前请求完成")
            if expected_revision is not None and row.revision != expected_revision:
                raise StorageError("会议内容已被更新，请重新读取后操作")
            row.run_token = uuid.uuid4().hex
            row.revision += 1
            row.mode, row.status = MeetingMode(mode).value, "processing"
            row.result = None
            row.index_status, row.index_error = "pending", None
            row.updated_at = utcnow()
            result = {**self._dict(row), "run_token": row.run_token}
        return result

    def _save_children(self, session, row, result):
        for position, segment in enumerate((result.get("transcript") or {}).get("segments", [])):
            session.add(SegmentRecord(meeting_id=row.meeting_id, revision=row.revision, position=position,
                                      speaker_id=segment["speaker_id"], speaker_name=segment["speaker"], data=segment))
        old = session.scalars(select(ActionRecord).where(ActionRecord.meeting_id == row.meeting_id)
                              .order_by(ActionRecord.revision)).all()
        identity = lambda data: (" ".join(data["task"].split()), data["assignee"].strip())
        known = {identity(action.data): action for action in old}
        seen = set()
        saved = []
        for item in (result.get("actions") or {}).get("action_items", []):
            key = identity(item)
            if key in seen:
                continue
            seen.add(key)
            action = known.get(key)
            data = dict(item)
            if action is None:
                action_id = hashlib.sha256(f"{row.meeting_id}:{key}".encode()).hexdigest()
                action = ActionRecord(action_id=action_id, meeting_id=row.meeting_id,
                                      revision=row.revision, assignee=item["assignee"], data={})
                session.add(action)
            else:
                # 重跑同一会议时保留完成状态，不把已经完成的任务重新变成待办。
                data["status"] = action.data.get("status", "pending")
                data["completed_at"] = action.data.get("completed_at")
                for field in ("jira_issue_key", "feishu_task_id"):
                    data[field] = action.data.get(field) or data.get(field)
            data["action_id"] = action.action_id
            data.setdefault("status", "pending")
            action.revision, action.data = row.revision, data
            session.flush()
            self._plan_reminders(session, action)
            saved.append(data)
        # 新一版不再包含的待办，不再触发旧提醒。
        for action in old:
            if identity(action.data) not in seen:
                self._cancel_reminders(session, action.action_id)
        if result.get("actions") is not None:
            result["actions"]["action_items"] = saved
        if result.get("followup") is not None:
            result["followup"]["reminders_scheduled"] = self._pending_count(
                session, self._current_actions(session, row))

    def finish(self, meeting_id, user_id, token, result, audio_path=None):
        with self.sessions.begin() as session:
            row = self._owned(session, meeting_id, user_id, lock=True)
            if row.run_token != token:
                raise StorageError("会议处理版本已变化")
            row.result, row.status = result, result["status"]
            row.participants = list(dict.fromkeys(segment["speaker_id"]
                for segment in (result.get("transcript") or {}).get("segments", [])))
            row.audio_path = audio_path
            if result.get("summary"):
                row.title = result["summary"]["title"]
            row.run_token, row.updated_at = None, utcnow()
            self._save_children(session, row, result)
            # JSON列赋新对象，使结构化待办ID和真正的提醒数量一并进入会议快照。
            row.result = deepcopy(result)
        return self.get(meeting_id, user_id)

    @staticmethod
    def _current_actions(session, meeting):
        return session.scalars(select(ActionRecord).where(
            ActionRecord.meeting_id == meeting.meeting_id, ActionRecord.revision == meeting.revision)
            .order_by(ActionRecord.action_id)).all()

    @staticmethod
    def _action_dict(action):
        return {**action.data, "action_id": action.action_id,
                "status": action.data.get("status", "pending"),
                "completed_at": action.data.get("completed_at")}

    @staticmethod
    def _pending_count(session, actions):
        ids = [a.action_id for a in actions]
        return len(session.scalars(select(ReminderRecord.reminder_id).where(
            ReminderRecord.action_id.in_(ids),
            ReminderRecord.status.in_(["pending", "sending"]))).all()) if ids else 0

    @staticmethod
    def _cancel_reminders(session, action_id):
        session.execute(update(ReminderRecord).where(ReminderRecord.action_id == action_id,
            ReminderRecord.status.in_(["pending", "sending"])).values(status="cancelled"))

    def _plan_reminders(self, session, action, *, now=None):
        now = as_utc(now)
        data = self._action_dict(action)
        if data["status"] in {"completed", "cancelled"}:
            self._cancel_reminders(session, action.action_id)
            return
        deadline = data.get("deadline", "")
        session.execute(update(ReminderRecord).where(ReminderRecord.action_id == action.action_id,
            ReminderRecord.deadline != deadline, ReminderRecord.status.in_(["pending", "sending"]))
            .values(status="cancelled"))
        try:
            plans = reminder_plan(deadline, hour=int(os.getenv("REMINDER_HOUR", "9")))
        except ValueError as exc:
            raise StorageError("截止日期或REMINDER_HOUR配置不正确", 422) from exc
        for days_before, scheduled_at, expires_at in plans:
            key = hashlib.sha256(f"{action.action_id}:{deadline}:{days_before}".encode()).hexdigest()
            reminder = session.get(ReminderRecord, key)
            status = "skipped" if expires_at <= now else "pending"
            if reminder is None:
                session.add(ReminderRecord(reminder_id=key, action_id=action.action_id, deadline=deadline,
                    days_before=days_before, scheduled_at=scheduled_at, expires_at=expires_at, status=status))
            elif reminder.status == "cancelled" and reminder.claim_token is None:
                # 未发送过的计划可在重新打开任务时恢复；sent/failed/uncertain不自动重发。
                reminder.status = status
        session.flush()

    def update_action(self, meeting_id, user_id, action_id, *, status=None, deadline=None, now=None):
        now = as_utc(now)
        with self.sessions.begin() as session:
            meeting = self._owned(session, meeting_id, user_id, lock=True)
            if meeting.run_token:
                raise StorageError("会议正在重新分析，请处理完成后再修改待办")
            action = session.get(ActionRecord, action_id, with_for_update=True)
            if action is None or action.meeting_id != meeting_id or action.revision != meeting.revision:
                raise StorageError("待办不存在于本次会议的当前版本", 404)
            data = self._action_dict(action)
            if status is not None:
                try:
                    data["status"] = ActionStatus(status).value
                except ValueError as exc:
                    raise StorageError("待办状态不正确", 422) from exc
                data["completed_at"] = (data.get("completed_at") or iso_utc(now)) if data["status"] == "completed" else None
            if deadline is not None:
                data["deadline"] = deadline
            action.data = data
            self._plan_reminders(session, action, now=now)
            meeting.updated_at = now
        return data

    def list_reminders(self, meeting_id, user_id="anonymous"):
        with self.sessions() as session:
            meeting = self._owned(session, meeting_id, user_id)
            rows = session.scalars(select(ReminderRecord).join(ActionRecord).where(
                ActionRecord.meeting_id == meeting_id, ActionRecord.revision == meeting.revision)
                .order_by(ReminderRecord.scheduled_at, ReminderRecord.reminder_id)).all()
            return [{"reminder_id": r.reminder_id, "action_id": r.action_id, "deadline": r.deadline,
                     "days_before": r.days_before, "scheduled_at": iso_utc(r.scheduled_at),
                     "status": r.status, "channel": r.channel, "attempts": r.attempts,
                     "sent_at": iso_utc(r.sent_at), "send_result": r.send_result,
                     "last_error": r.last_error} for r in rows]

    def claim_due_reminders(self, *, now=None, limit=100, meeting_id=None):
        """先筛候选，再按会议→待办→提醒的顺序加锁，多个扫描器只会领取一份。"""
        now = as_utc(now)
        with self.sessions.begin() as session:
            # 验收可限定一场会议，避免测试时扫描其他会议的真实提醒。
            scope = [] if meeting_id is None else [ReminderRecord.action_id.in_(
                select(ActionRecord.action_id).where(ActionRecord.meeting_id == meeting_id))]
            session.execute(update(ReminderRecord).where(*scope, ReminderRecord.status == "pending",
                ReminderRecord.expires_at <= now).values(status="skipped"))
            # 进程可能在外部发送后退出；超时的发送标为待核对，不当作未发送重试。
            session.execute(update(ReminderRecord).where(*scope, ReminderRecord.status == "sending",
                ReminderRecord.claimed_at < now - timedelta(minutes=10)).values(
                    status="uncertain", claim_token=None, last_error="发送进程中断，结果待核对"))
            candidates = session.execute(select(ReminderRecord.reminder_id, ActionRecord.meeting_id)
                .join(ActionRecord).join(MeetingRecord, MeetingRecord.meeting_id == ActionRecord.meeting_id)
                .where(*scope, MeetingRecord.run_token.is_(None),
                    ReminderRecord.status == "pending", ReminderRecord.scheduled_at <= now,
                    ReminderRecord.expires_at > now).order_by(ReminderRecord.scheduled_at).limit(limit)).all()
        jobs = []
        for reminder_id, meeting_id in candidates:
            with self.sessions.begin() as session:
                meeting = session.get(MeetingRecord, meeting_id, with_for_update=True)
                if meeting.run_token:
                    continue
                reminder = session.get(ReminderRecord, reminder_id)
                action = session.get(ActionRecord, reminder.action_id, with_for_update=True)
                reminder = session.scalar(select(ReminderRecord).where(
                    ReminderRecord.reminder_id == reminder_id).with_for_update().execution_options(populate_existing=True))
                if reminder.status != "pending":
                    continue
                if action.revision != meeting.revision or action.data.get("status", "pending") in {"completed", "cancelled"}:
                    reminder.status = "cancelled"
                    continue
                reminder.status, reminder.claim_token, reminder.claimed_at = "sending", uuid.uuid4().hex, now
                reminder.attempts += 1
                jobs.append({"reminder_id": reminder_id, "claim_token": reminder.claim_token,
                    "meeting_id": meeting_id, "title": meeting.title, "mode": meeting.mode,
                    "assignee": action.assignee, "task": action.data["task"], "deadline": reminder.deadline,
                    "days_before": reminder.days_before})
        return jobs

    def reminder_is_active(self, reminder_id, token, *, now=None):
        with self.sessions.begin() as session:
            reminder = session.get(ReminderRecord, reminder_id)
            if reminder is None or reminder.status != "sending" or reminder.claim_token != token:
                return False
            action = session.get(ActionRecord, reminder.action_id)
            meeting = session.get(MeetingRecord, action.meeting_id)
            if (meeting.run_token or action.revision != meeting.revision or reminder.expires_at <= as_utc(now)
                    or action.data.get("status", "pending") in {"completed", "cancelled"}):
                reminder.status, reminder.claim_token = "cancelled", None
                return False
            return True

    def finish_reminder(self, reminder_id, token, *, status, channel, result, error=None, now=None):
        with self.sessions.begin() as session:
            changed = session.execute(update(ReminderRecord).where(ReminderRecord.reminder_id == reminder_id,
                ReminderRecord.claim_token == token, ReminderRecord.status.in_(["sending", "cancelled"])).values(
                    status=status, channel=channel, send_result=result, last_error=error,
                    sent_at=as_utc(now) if status == "sent" else None, claim_token=None)).rowcount
        return changed == 1

    def mark_index(self, meeting_id, user_id, revision, status, error=None):
        with self.sessions.begin() as session:
            changed = session.execute(update(MeetingRecord).where(
                MeetingRecord.meeting_id == meeting_id, MeetingRecord.user_id == user_id,
                MeetingRecord.revision == revision,
            ).values(index_status=status, index_error=error)).rowcount
        return changed == 1

    def close(self):
        self.engine.dispose()
