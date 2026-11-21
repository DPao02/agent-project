"""FastAPI入口：统一管理MySQL、Milvus、工具和LangGraph生命周期。"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Overwrite
from pydantic import BaseModel, ConfigDict, Field
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from agents.supervisor import create_supervisor_graph
from api.business_routes import router as business_router
from business.database import Database
from business.service import BusinessError, BusinessService
from mcp.mcp_server import MCPToolServer, create_default_tools
from memory.long_term import LongTermMemory
from memory.short_term import SessionBusy, ShortTermMemory, StoredMessage
from memory.identity import session_scope
from business.task_state import TaskState
from memory.working_memory import WorkingMemory
from tracing.otel_config import AgentMetrics, init_tracer

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[1]
# 保持当前本地开发的配置优先级。部署环境可改为直接注入环境变量。
load_dotenv(PROJECT_DIR / ".env", override=True)

working_memory = WorkingMemory()
short_term_memory = ShortTermMemory(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "1800")),
    allow_memory_fallback=os.getenv("MEMORY_ALLOW_FALLBACK", "true").lower() == "true",
)
metrics = AgentMetrics()
graph = None
mcp_server: MCPToolServer | None = None
long_term_memory: LongTermMemory | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, mcp_server, long_term_memory
    database = None
    app.state.business_service = None
    try:
        init_tracer(service_name=os.getenv("OTEL_SERVICE_NAME", "smart-cs-multi-agent"),
                    otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
        # 启动只检查数据库，不自动建库/改表；先执行init_database.py。
        database = await asyncio.to_thread(Database.from_env)
        await asyncio.to_thread(database.check_ready)
        service = BusinessService(database.sessions)

        long_term_memory = await asyncio.to_thread(LongTermMemory)
        count = await asyncio.to_thread(
            long_term_memory.load_knowledge_base, str(PROJECT_DIR / "knowledge_base"),
        )
        if count == 0:
            raise RuntimeError("知识库中未提取到可用文本，请检查PDF或TXT文件")
        await asyncio.to_thread(long_term_memory.save)

        # HTTP与工具注入同一个业务服务；Graph注入同一个工具服务。
        mcp_server = create_default_tools(MCPToolServer(), service, long_term_memory)
        graph = create_supervisor_graph(
            working_memory=working_memory, short_term_memory=short_term_memory,
            long_term_memory=long_term_memory, tool_server=mcp_server,
        )
        app.state.business_service = service
        logger.info("MySQL业务服务与Milvus知识库就绪，共处理%s个文本块", count)
        yield
    finally:
        graph = None
        mcp_server = None
        app.state.business_service = None
        try:
            if long_term_memory is not None:
                await asyncio.to_thread(long_term_memory.close)
        finally:
            long_term_memory = None
            try:
                if database is not None:
                    await asyncio.to_thread(database.close)
            finally:
                await short_term_memory.close()


app = FastAPI(title="智能客服多Agent系统", version="1.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(business_router)


@app.exception_handler(BusinessError)
async def business_error_handler(request, exc: BusinessError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc), "code": exc.code})


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request, exc: SQLAlchemyError):
    logger.error("业务数据库操作失败", exc_info=(type(exc), exc, exc.__traceback__))
    return JSONResponse(status_code=503, content={"detail": "业务数据库暂时不可用，请查看服务端日志"})


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=8000)
    user_id: str = Field(default="anonymous", min_length=1, max_length=64)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    # 同一条消息的网络重试沿用同一请求号；与跨多轮会话的session_id不同。
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=8, max_length=64)


class ChatResponse(BaseModel):
    response: str
    session_id: str
    request_id: str
    intent: str
    compliance_passed: bool
    task: TaskState | None = None
    memory_backend: str


@app.exception_handler(SessionBusy)
async def session_busy_handler(request, exc: SessionBusy):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(RedisError)
async def redis_error_handler(request, exc: RedisError):
    logger.error("会话存储异常", exc_info=(type(exc), exc, exc.__traceback__))
    return JSONResponse(status_code=503, content={"detail": "会话存储暂时不可用，请稍后重试；结果不明确时请先查询已提交的工单。"})


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if graph is None:
        raise HTTPException(503, "系统初始化中")
    session_id = request.session_id or str(uuid.uuid4())
    async with short_term_memory.session(session_id, user_id=request.user_id) as lease:
        snapshot = await lease.load()
        fingerprint = hashlib.sha256(request.message.encode("utf-8")).hexdigest()
        cached = next((r for r in snapshot.replies if r["request_id"] == request.request_id), None)
        if cached:
            if cached["fingerprint"] != fingerprint:
                raise HTTPException(409, "同一request_id不能用于不同消息，请为新一轮使用新编号")
            return ChatResponse.model_validate({**cached["response"], "memory_backend": lease.backend})

        messages = [(HumanMessage if m.role == "user" else AIMessage)(content=m.content, id=m.id)
                    for m in snapshot.messages]
        messages.append(HumanMessage(content=request.message, id=request.request_id))
        # Redis快照是权威来源，不能让过期后仍留在工作缓存的旧上下文复活。
        working_memory.clear(session_id, user_id=request.user_id)
        working_memory.update(session_id, {**snapshot.context, "task": snapshot.task.model_dump() if snapshot.task else None},
                              user_id=request.user_id)
        initial_state = {
            # 每轮以持久快照重建消息，覆盖旧检查点，避免重复拼接及过期历史复活。
            "messages": Overwrite(value=messages),
            "user_id": request.user_id, "session_id": session_id, "request_id": request.request_id,
            "task": snapshot.task.model_dump() if snapshot.task else None,
            "intent": "", "sub_results": {}, "compliance_passed": True, "rewritten_query": "",
            "final_response": "", "current_agent": "", "retry_count": 0,
        }
        config = {"configurable": {"thread_id": session_scope(request.user_id, session_id)}}
        try:
            # 租约180秒，图执行最多120秒；超时后不提交半份会话快照。
            async with asyncio.timeout(120):
                result = await graph.ainvoke(initial_state, config=config)
        except TimeoutError as exc:
            raise HTTPException(504, "处理超时。已提交的业务请先查询核实，再重试同一请求。") from exc
        except Exception as exc:
            logger.exception("聊天执行失败，request_id=%s", request.request_id)
            raise HTTPException(500, "处理失败，请查看API终端日志；已提交的工单请先查询核实") from exc

        response = ChatResponse(
            response=result.get("final_response", "系统处理异常，请稍后重试"),
            session_id=session_id, request_id=request.request_id, intent=result.get("intent", "unknown"),
            compliance_passed=result.get("compliance_passed", True), task=result.get("task"), memory_backend=lease.backend,
        )
        snapshot.messages.extend([
            StoredMessage(role="user", content=request.message, id=request.request_id),
            StoredMessage(role="assistant", content=response.response[:16000], id="reply-" + request.request_id),
        ])
        snapshot.task = response.task
        snapshot.context["last_intent"] = response.intent
        if result.get("rewritten_query"):
            snapshot.context["last_rag_query"] = result["rewritten_query"]
        snapshot.replies.append({"request_id": request.request_id, "fingerprint": fingerprint,
                                "response": response.model_dump(mode="json")})
        await lease.save(snapshot)
        working_memory.update(session_id, {**snapshot.context, "task": response.task.model_dump() if response.task else None},
                              user_id=request.user_id)
        return response


@app.get("/api/history/{session_id}")
async def get_history(session_id: str, user_id: str = Query(default="anonymous", min_length=1, max_length=64)):
    return {"user_id": user_id, "session_id": session_id,
            "messages": await short_term_memory.get_history(session_id, user_id=user_id)}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str, user_id: str = Query(default="anonymous", min_length=1, max_length=64)):
    async with short_term_memory.session(session_id, user_id=user_id) as lease:
        snapshot = await lease.load()
        return {"user_id": user_id, "session_id": session_id, "memory_backend": lease.backend,
                "task": snapshot.task.model_dump() if snapshot.task else None, "message_count": len(snapshot.messages)}


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str, user_id: str = Query(default="anonymous", min_length=1, max_length=64)):
    await short_term_memory.clear(session_id, user_id=user_id)
    working_memory.clear(session_id, user_id=user_id)
    return {"detail": "会话和未提交任务已清空；不会删除数据库订单或工单"}


@app.get("/api/tools")
async def list_tools():
    if mcp_server is None:
        raise HTTPException(503, "工具服务尚未就绪")
    return {"tools": mcp_server.list_tools()}


class ToolRequest(BaseModel):
    name: str = Field(min_length=1)
    arguments: dict = Field(default_factory=dict)


@app.post("/api/tools/call")
async def call_tool(request: ToolRequest):
    if mcp_server is None:
        raise HTTPException(503, "工具服务尚未就绪")
    result = await mcp_server.call_tool(request.name, request.arguments)
    return {"success": result.success, "result": result.result,
            "error": result.error, "duration_ms": result.duration_ms}


@app.get("/api/metrics")
async def get_metrics():
    return {"agent_metrics": metrics.get_summary(),
            "tool_call_log": mcp_server.get_call_log(last_n=20) if mcp_server else []}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=os.getenv("HOST", "0.0.0.0"),
                port=int(os.getenv("PORT", "8000")), reload=True)
