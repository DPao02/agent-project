"""跨进程原子请求计数；请求发出前计数，失败也计入，不自动重置授权上限。"""

import sqlite3
from contextlib import closing
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class RequestBudget:
    def __init__(self, path: Path, limit: int = 1200):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, ceiling INTEGER NOT NULL, used INTEGER NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS counts (provider TEXT PRIMARY KEY, used INTEGER NOT NULL)")
            connection.execute("INSERT OR IGNORE INTO budget VALUES (1, ?, 0)", (limit,))

    def charge(self, provider: str):
        with closing(sqlite3.connect(self.path, timeout=10)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            ceiling, used = connection.execute("SELECT ceiling, used FROM budget WHERE id=1").fetchone()
            if used >= ceiling:
                raise BudgetExceeded(f"已达到本轮{ceiling}次请求上限，未继续外发请求")
            connection.execute("UPDATE budget SET used=used+1 WHERE id=1")
            connection.execute("INSERT INTO counts VALUES (?,1) ON CONFLICT(provider) DO UPDATE SET used=used+1", (provider,))

    def summary(self):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            ceiling, used = connection.execute("SELECT ceiling, used FROM budget WHERE id=1").fetchone()
            counts = dict(connection.execute("SELECT provider, used FROM counts").fetchall())
            return {"limit": ceiling, "attempted_calls": used, "by_provider": counts}
