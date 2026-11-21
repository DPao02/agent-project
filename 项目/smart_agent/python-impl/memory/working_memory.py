"""工作记忆：进程内的会话上下文缓存，使用用户+会话联合隔离。"""

import threading
import time
from copy import deepcopy
from datetime import datetime, timezone

from memory.identity import session_scope


class WorkingMemory:
    def __init__(self, max_entries_per_session: int = 50, ttl_seconds: int = 1800, max_sessions: int = 1000):
        self._store = {}
        self._context = {}
        self._expires = {}
        self._lock = threading.Lock()
        self._max_entries = max_entries_per_session
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions

    def _prune(self):
        expired = [key for key, deadline in self._expires.items() if deadline <= time.monotonic()]
        for key in expired:
            self._drop(key)

    def _drop(self, key):
        self._store.pop(key, None)
        self._context.pop(key, None)
        self._expires.pop(key, None)

    def update(self, session_id: str, data: dict, *, user_id: str = "anonymous") -> None:
        key = session_scope(user_id, session_id)
        with self._lock:
            self._prune()
            if key not in self._context and len(self._context) >= self.max_sessions:
                self._drop(next(iter(self._context)))
            entries = self._store.setdefault(key, [])
            entries.append({"timestamp": datetime.now(timezone.utc).isoformat(), "data": deepcopy(data)})
            self._store[key] = entries[-self._max_entries:]
            self._context.setdefault(key, {}).update(deepcopy(data))
            self._expires[key] = time.monotonic() + self.ttl_seconds

    def get_context(self, session_id: str, *, user_id: str = "anonymous") -> dict:
        with self._lock:
            self._prune()
            return deepcopy(self._context.get(session_scope(user_id, session_id), {}))

    def get_history(self, session_id: str, last_n: int = 10, *, user_id: str = "anonymous") -> list:
        with self._lock:
            self._prune()
            return deepcopy(self._store.get(session_scope(user_id, session_id), [])[-last_n:])

    def clear(self, session_id: str, *, user_id: str = "anonymous") -> None:
        with self._lock:
            self._drop(session_scope(user_id, session_id))

    def export_for_persistence(self, session_id: str, *, user_id: str = "anonymous") -> dict:
        return {"user_id": user_id, "session_id": session_id,
                "context": self.get_context(session_id, user_id=user_id),
                "history": self.get_history(session_id, user_id=user_id)}
