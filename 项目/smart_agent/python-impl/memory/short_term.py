"""短期记忆：原子保存对话和任务快照，按用户+会话隔离。

Redis不可连接时允许显式降级内存；接口会告知后端，不能声称此时能跨重启恢复。
一旦本次请求使用Redis，读写失败就报错，不在中途切换到另一份空内存。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal
from weakref import WeakValueDictionary

from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import LockNotOwnedError, RedisError

from business.task_state import TaskState
from memory.identity import session_scope

logger = logging.getLogger(__name__)


class SessionBusy(Exception):
    pass


class StoredMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SessionSnapshot(BaseModel):
    messages: list[StoredMessage] = Field(default_factory=list)
    task: TaskState | None = None
    context: dict = Field(default_factory=dict)
    replies: list[dict] = Field(default_factory=list)


class MemorySession:
    """一次租约内固定使用同一后端，防止部分成功后切换到另一份状态。"""
    def __init__(self, owner: ShortTermMemory, scope: str, client: Redis | None):
        self.owner, self.scope, self.client = owner, scope, client
        self.backend = "redis" if client is not None else "memory"

    @property
    def key(self):
        return f"smartcs:v2:session:{self.scope}"

    async def load(self) -> SessionSnapshot:
        if self.client is not None:
            value = await self.client.get(self.key)
            return SessionSnapshot.model_validate_json(value) if value else SessionSnapshot()
        self.owner._prune()
        stored = self.owner._fallback_store.get(self.scope)
        return stored[1].model_copy(deep=True) if stored else SessionSnapshot()

    async def save(self, snapshot: SessionSnapshot) -> None:
        # max_turns表示用户/助手对话对数，不再误当成消息条数。
        snapshot.messages = snapshot.messages[-self.owner.max_turns * 2:]
        snapshot.replies = snapshot.replies[-10:]
        if self.client is not None:
            # 对话、任务字段、请求回执在一个SET里更新，并同时刷新TTL。
            await self.client.set(self.key, snapshot.model_dump_json(), ex=self.owner.ttl_seconds)
        else:
            self.owner._prune()
            if self.scope not in self.owner._fallback_store and len(self.owner._fallback_store) >= 1000:
                self.owner._fallback_store.pop(next(iter(self.owner._fallback_store)))
            self.owner._fallback_store[self.scope] = (
                time.monotonic() + self.owner.ttl_seconds, snapshot.model_copy(deep=True),
            )


class ShortTermMemory:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", max_turns: int = 20,
                 ttl_seconds: int = 1800, allow_memory_fallback: bool = True):
        if max_turns < 1 or ttl_seconds < 1:
            raise ValueError("max_turns和ttl_seconds必须为正整数")
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self.allow_memory_fallback = allow_memory_fallback
        self._redis_url = redis_url
        self._redis = None
        self._connection_lock = asyncio.Lock()
        self._retry_after = 0.0
        self._fallback_store = {}
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    def _prune(self):
        for key in [k for k, (deadline, _) in self._fallback_store.items() if deadline <= time.monotonic()]:
            self._fallback_store.pop(key, None)

    async def _get_redis(self):
        if self._redis_url == "memory://":
            return None
        if self._redis is not None:
            return self._redis
        # 首次并发访问只创建一个连接池，避免多个客户端覆盖引用后泄漏连接。
        async with self._connection_lock:
            return await self._connect_redis()

    async def _connect_redis(self):
        if self._redis is not None:
            return self._redis
        if time.monotonic() < self._retry_after:
            if not self.allow_memory_fallback:
                raise RedisError("Redis暂时不可用")
            return None
        client = Redis.from_url(self._redis_url, decode_responses=True,
                                socket_connect_timeout=2, socket_timeout=2)
        try:
            await client.ping()
        except (RedisError, OSError):
            await client.aclose()
            self._retry_after = time.monotonic() + 30
            if not self.allow_memory_fallback:
                raise
            logger.warning("Redis不可用，短期记忆降级为进程内存；重启后不能恢复")
            return None
        self._redis = client
        return client

    @asynccontextmanager
    async def session(self, session_id: str, *, user_id: str = "anonymous"):
        scope = session_scope(user_id, session_id)
        self._prune()
        # 降级期间已开始的会话保持使用原快照，不因Redis恢复突然丢掉草稿。
        # 新会话仍可重新尝试Redis；内存会话在TTL结束/清空后再切换。
        client = None if scope in self._fallback_store else await self._get_redis()
        if client is not None:
            # 同一用户/会话只能有一个办理请求。API图调用限制120秒，租约180秒。
            lock = client.lock(f"smartcs:v2:lock:{scope}", timeout=180,
                               blocking_timeout=0, thread_local=False)
            if not await lock.acquire(blocking=False):
                raise SessionBusy("本会话有请求正在处理，请等待完成后再发送")
            try:
                yield MemorySession(self, scope, client)
            finally:
                try:
                    await lock.release()
                except LockNotOwnedError:
                    logger.error("会话锁已失效，不能释放其他请求的锁")
        else:
            lock = self._locks.get(scope)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[scope] = lock
            if lock.locked():
                raise SessionBusy("本会话有请求正在处理，请稍后重试")
            async with lock:
                yield MemorySession(self, scope, None)

    async def get_snapshot(self, session_id: str, *, user_id: str = "anonymous") -> SessionSnapshot:
        async with self.session(session_id, user_id=user_id) as lease:
            return await lease.load()

    async def add_message(self, session_id: str, role: str, content: str, *, user_id: str = "anonymous"):
        # 保留供其他调用者使用的兼容接口；聊天API使用一次性快照提交。
        async with self.session(session_id, user_id=user_id) as lease:
            snapshot = await lease.load()
            snapshot.messages.append(StoredMessage(role=role, content=content[:16000]))
            await lease.save(snapshot)

    async def get_history(self, session_id: str, last_n: int | None = None, *, user_id: str = "anonymous") -> list:
        snapshot = await self.get_snapshot(session_id, user_id=user_id)
        messages = snapshot.messages[-last_n:] if last_n else snapshot.messages
        return [message.model_dump() for message in messages]

    async def clear(self, session_id: str, *, user_id: str = "anonymous"):
        async with self.session(session_id, user_id=user_id) as lease:
            if lease.client is not None:
                await lease.client.delete(lease.key)
            self._fallback_store.pop(lease.scope, None)

    async def get_context_window(self, session_id: str, max_tokens: int = 4000, *, user_id: str = "anonymous"):
        history = await self.get_history(session_id, user_id=user_id)
        return "\n".join(f"{m['role']}: {m['content']}" for m in history)[-max_tokens * 2:]

    async def close(self):
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        self._connection_lock = asyncio.Lock()
