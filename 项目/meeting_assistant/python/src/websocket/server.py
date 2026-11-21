"""录完再处理的会议API：共享模型/图，按会议保存MySQL结果和Milvus索引。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import date
import json
import os
from pathlib import Path
import re
from typing import Annotated
from urllib.parse import quote
import uuid

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy.exc import SQLAlchemyError

from ..config import PROJECT_DIR, load_environment
from ..graph.meeting_graph import MeetingPipeline
from ..integrations.milvus_client import MeetingVectorStore
from ..models.schemas import ActionStatus, MeetingMode, MeetingStatus, create_initial_state
from ..reminders import ReminderService
from ..storage import MeetingRepository, StorageError, restore_state, serialize_state

UserId = Annotated[str, Query(min_length=1, max_length=64)]


def checked_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise HTTPException(422, "会议编号只允许1到64位字母、数字、下划线或连字符")
    return value


class MeetingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meeting_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    user_id: str = Field(default="anonymous", min_length=1, max_length=64)
    title: str = Field(default="", max_length=255)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(default="anonymous", min_length=1, max_length=64)
    meeting_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    mode: MeetingMode | None = MeetingMode.REAL


class ActionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: ActionStatus | None = None
    deadline: str | None = Field(default=None, pattern=r"^(?:\d{4}-\d{2}-\d{2})?$")

    @field_validator("deadline")
    @classmethod
    def check_deadline(cls, value):
        return date.fromisoformat(value).isoformat() if value else value

    @model_validator(mode="after")
    def check_update(self):
        if self.status is None and self.deadline is None:
            raise ValueError("请提供status或deadline；deadline为空字符串表示清除截止日期")
        return self


def public_result(record):
    result = {key: record[key] for key in ("meeting_id", "user_id", "mode", "status", "revision",
              "title", "participants", "index_status", "index_error")}
    result.update(record.get("result") or {})
    return result


async def index_result(app, record):
    """先持久化MySQL再更新向量；索引失败不丢失会议，允许单独重建。"""
    repo, vectors = app.state.repository, app.state.vectors
    args = record["meeting_id"], record["user_id"], record["revision"]
    if not vectors.enabled:
        await asyncio.to_thread(repo.mark_index, *args, "disabled")
    elif (record.get("result") or {}).get("transcript"):
        try:
            await asyncio.to_thread(vectors.index, record)
            await asyncio.to_thread(repo.mark_index, *args, "indexed")
        except Exception as exc:
            logger.warning("Meeting index failed: {} ({})", record["meeting_id"], type(exc).__name__)
            await asyncio.to_thread(repo.mark_index, *args, "failed", str(exc)[:1000])
    else:
        await asyncio.to_thread(repo.mark_index, *args, "not_indexed")
    return await asyncio.to_thread(repo.get, record["meeting_id"], record["user_id"])


async def process_meeting(app, meeting_id, user_id, *, mode, audio_data=b"", transcribe_only=False, analyze_existing=False):
    meeting_id = checked_id(meeting_id)
    repo = app.state.repository
    original = await asyncio.to_thread(repo.create, meeting_id, user_id, allow_existing=True)
    transcript = None
    if analyze_existing:
        if not original["result"] or not original["result"].get("transcript"):
            raise HTTPException(409, "还没有成功转写的内容，请先上传音频或运行演示")
        transcript = restore_state(original["result"])["transcript"]
        mode = original["mode"]
    reservation = await asyncio.to_thread(repo.claim, meeting_id, user_id, mode, original["revision"])
    audio_path = original["audio_path"] if analyze_existing else None
    try:
        if audio_data:
            path = app.state.data_dir / "audio" / meeting_id / (reservation["run_token"] + ".audio")
            def save_audio():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(audio_data)
            await asyncio.to_thread(save_audio)
            audio_path = str(path)
        state = await app.state.pipeline.run(meeting_id, audio_data, mode=mode,
            transcript=transcript, transcribe_only=transcribe_only)
    except asyncio.CancelledError:
        state = create_initial_state(meeting_id, mode=mode)
        state.update(status=MeetingStatus.FAILED, errors=["本次处理被中断，请重新提交"])
        await asyncio.shield(asyncio.to_thread(repo.finish, meeting_id, user_id, reservation["run_token"],
                                              serialize_state(state), audio_path))
        raise
    except Exception as exc:
        logger.exception("Meeting processing failed: {}", meeting_id)
        state = create_initial_state(meeting_id, mode=mode)
        state.update(status=MeetingStatus.FAILED, errors=[f"会议处理失败：{type(exc).__name__}: {exc}"])
    if state.get("followup"):
        state["followup"].report_url = f"/api/v1/meeting/{meeting_id}/report/download?user_id={quote(user_id, safe='')}"
    record = await asyncio.to_thread(repo.finish, meeting_id, user_id, reservation["run_token"],
                                     serialize_state(state), audio_path)
    return await index_result(app, record)


def create_app(*, repository=None, pipeline=None, vectors=None, data_dir=None, reminders_enabled=None):
    """测试可注入依赖；正常运行由lifespan只创建一次。"""
    @asynccontextmanager
    async def lifespan(app):
        load_environment()
        repo = worker = store = reminders = None
        try:
            repo = repository or await asyncio.to_thread(MeetingRepository.from_env)
            await asyncio.to_thread(repo.initialize)
            worker = pipeline or MeetingPipeline()
            store = vectors or MeetingVectorStore()
            app.state.repository, app.state.pipeline, app.state.vectors = repo, worker, store
            directory = Path(data_dir or os.getenv("MEETING_DATA_DIR", str(PROJECT_DIR / "data")))
            app.state.data_dir = directory if directory.is_absolute() else PROJECT_DIR / directory
            app.state.max_audio_bytes = int(os.getenv("MAX_AUDIO_MB", "256")) * 1024 * 1024
            reminders = ReminderService(repo, worker.feishu)
            app.state.reminders = reminders
            # 注入测试仓库时可显式开启扫描；正常服务默认随应用启动，随应用停止。
            enabled = reminders_enabled if reminders_enabled is not None else (
                repository is None and os.getenv("REMINDERS_ENABLED", "true").lower() == "true")
            if enabled:
                reminders.start()
            yield
        finally:
            try:
                try:
                    if reminders is not None:
                        await reminders.close()
                finally:
                    if worker is not None:
                        await worker.close()
            finally:
                try:
                    if store is not None:
                        await asyncio.to_thread(store.close)
                finally:
                    if repo is not None:
                        await asyncio.to_thread(repo.close)

    app = FastAPI(title="多Agent智能会议助手", version="1.2.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(StorageError)
    async def storage_error(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request, exc):
        logger.error("Meeting database operation failed: {}", type(exc).__name__)
        return JSONResponse(status_code=503, content={"detail": "会议数据库暂不可用"})

    @app.get("/")
    async def root():
        return {"name": "多Agent智能会议助手", "version": "1.2.0", "docs": "/docs"}

    @app.post("/api/v1/meeting/start")
    async def start_meeting(request: Request, data: MeetingCreate | None = Body(default=None)):
        data = data or MeetingCreate()
        record = await asyncio.to_thread(request.app.state.repository.create,
            data.meeting_id or uuid.uuid4().hex, data.user_id, data.title)
        return public_result(record)

    @app.get("/api/v1/meetings")
    async def list_meetings(request: Request, user_id: UserId = "anonymous", limit: int = Query(default=20, ge=1, le=100)):
        return {"meetings": await asyncio.to_thread(request.app.state.repository.list, user_id, limit)}

    @app.post("/api/v1/meeting/{meeting_id}/upload")
    async def upload_audio(request: Request, meeting_id: str, file: UploadFile = File(...),
                           user_id: UserId = "anonymous", transcribe_only: bool = False):
        audio = await file.read(request.app.state.max_audio_bytes + 1)
        if len(audio) > request.app.state.max_audio_bytes:
            raise HTTPException(413, "音频超过配置的大小限制")
        return public_result(await process_meeting(request.app, meeting_id, user_id, mode=MeetingMode.REAL,
                                                  audio_data=audio, transcribe_only=transcribe_only))

    @app.post("/api/v1/meeting/{meeting_id}/demo")
    async def run_demo(request: Request, meeting_id: str, user_id: UserId = "anonymous"):
        return public_result(await process_meeting(request.app, meeting_id, user_id, mode=MeetingMode.DEMO))

    @app.post("/api/v1/meeting/{meeting_id}/analyze")
    async def analyze(request: Request, meeting_id: str, user_id: UserId = "anonymous"):
        return public_result(await process_meeting(request.app, meeting_id, user_id, mode=MeetingMode.REAL, analyze_existing=True))

    @app.patch("/api/v1/meeting/{meeting_id}/actions/{action_id}")
    async def update_action(request: Request, meeting_id: str, action_id: str, data: ActionUpdate,
                            user_id: UserId = "anonymous"):
        return await asyncio.to_thread(request.app.state.repository.update_action,
            checked_id(meeting_id), user_id, action_id, status=data.status, deadline=data.deadline)

    @app.get("/api/v1/meeting/{meeting_id}/reminders")
    async def list_reminders(request: Request, meeting_id: str, user_id: UserId = "anonymous"):
        return {"reminders": await asyncio.to_thread(request.app.state.repository.list_reminders,
                                                     checked_id(meeting_id), user_id)}

    @app.post("/api/v1/meeting/{meeting_id}/reindex")
    async def reindex(request: Request, meeting_id: str, user_id: UserId = "anonymous"):
        record = await asyncio.to_thread(request.app.state.repository.get, checked_id(meeting_id), user_id)
        if not record.get("result"):
            raise HTTPException(409, "会议尚无结果")
        return public_result(await index_result(request.app, record))

    @app.post("/api/v1/meetings/search")
    async def search(request: Request, data: SearchRequest):
        repo, store = request.app.state.repository, request.app.state.vectors
        if not store.enabled:
            raise HTTPException(503, "未启用Milvus语义检索")
        if data.meeting_id:
            await asyncio.to_thread(repo.get, checked_id(data.meeting_id), data.user_id)
        try:
            hits = await asyncio.to_thread(store.search, data.query, data.user_id, data.top_k, data.meeting_id)
        except Exception as exc:
            logger.warning("Meeting semantic search failed: {}", type(exc).__name__)
            raise HTTPException(503, "语义检索暂不可用，请检查Milvus和Embedding配置") from exc
        current, matched = {}, []
        for hit in hits:
            try:
                if hit["meeting_id"] not in current:
                    current[hit["meeting_id"]] = await asyncio.to_thread(repo.get, hit["meeting_id"], data.user_id)
                row = current[hit["meeting_id"]]
                if (row["revision"] == hit["revision"] and row["index_status"] == "indexed"
                        and (data.mode is None or row["mode"] == data.mode)):
                    matched.append({**hit, "mode": row["mode"], "title": row["title"]})
            except StorageError:
                continue
            if len(matched) == data.top_k:
                break
        return {"matches": matched}

    @app.get("/api/v1/meeting/{meeting_id}/report/download")
    async def download_report(request: Request, meeting_id: str, user_id: UserId = "anonymous"):
        row = await asyncio.to_thread(request.app.state.repository.get, checked_id(meeting_id), user_id)
        text = ((row.get("result") or {}).get("followup") or {}).get("report_markdown")
        if not text:
            raise HTTPException(409, "报告尚未生成，请先完成分析")
        return Response(text, media_type="text/markdown", headers={"Content-Disposition": f'attachment; filename="{meeting_id}.md"'})

    def result_route(field):
        async def get_result(request: Request, meeting_id: str, user_id: UserId = "anonymous"):
            row = await asyncio.to_thread(request.app.state.repository.get, checked_id(meeting_id), user_id)
            value = (row.get("result") or {}).get(field)
            if value is None:
                raise HTTPException(409, f"{field}尚不可用，会议状态：{row['status']}")
            return value
        return get_result

    for field in ("transcript", "summary", "actions", "unresolved"):
        app.add_api_route(f"/api/v1/meeting/{{meeting_id}}/{field}", result_route(field), methods=["GET"], name=f"get_{field}")

    @app.get("/api/v1/meeting/{meeting_id}/report")
    @app.get("/api/v1/meeting/{meeting_id}")
    async def get_meeting(request: Request, meeting_id: str, user_id: UserId = "anonymous"):
        return public_result(await asyncio.to_thread(request.app.state.repository.get, checked_id(meeting_id), user_id))

    @app.websocket("/ws/meeting/{meeting_id}")
    async def websocket_meeting(websocket: WebSocket, meeting_id: str):
        await websocket.accept()
        user_id = websocket.query_params.get("user_id", "anonymous")
        audio_buffer = bytearray()  # 每个连接自己的音频，不保存在共享Pipeline中。
        try:
            checked_id(meeting_id)
            if not 1 <= len(user_id) <= 64:
                raise ValueError("user_id长度应为1到64")
            await asyncio.to_thread(websocket.app.state.repository.create, meeting_id, user_id, allow_existing=True)
            await websocket.send_json({"type": "connected", "meeting_id": meeting_id})
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes"):
                    audio_buffer.extend(message["bytes"])
                    if len(audio_buffer) > websocket.app.state.max_audio_bytes:
                        raise ValueError("音频超过大小限制")
                    await websocket.send_json({"type": "recording", "buffer_size": len(audio_buffer)})
                elif message.get("text"):
                    control = json.loads(message["text"]).get("type")
                    if control == "start":
                        audio_buffer.clear()
                    elif control == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif control in {"stop", "demo"}:
                        await websocket.send_json({"type": "processing", "meeting_id": meeting_id})
                        mode = MeetingMode.DEMO if control == "demo" else MeetingMode.REAL
                        row = await process_meeting(websocket.app, meeting_id, user_id, mode=mode,
                                                    audio_data=bytes(audio_buffer) if control == "stop" else b"")
                        result = public_result(row)
                        for field in ("transcript", "summary", "actions", "unresolved", "followup"):
                            if result.get(field) is not None:
                                await websocket.send_json({"type": field, "data": result[field]})
                        await websocket.send_json({"type": "failed" if row["status"] == "failed" else "completed",
                            "meeting_id": meeting_id, "mode": row["mode"], "status": row["status"],
                            "errors": result.get("errors", []), "index_status": row["index_status"]})
                        audio_buffer.clear()
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("Meeting websocket failed: {} ({})", meeting_id, type(exc).__name__)
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
                await websocket.close(code=1011)
            except Exception:
                pass

    return app


app = create_app()
