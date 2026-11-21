"""MySQL 连接池与表结构：模块导入时不连接、不建表。"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text, URL
from sqlalchemy import UniqueConstraint, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utc_now() -> datetime:
    """MySQL DATETIME 统一存 UTC，输出时再标注时区。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


# InnoDB 支持事务和外键；二进制排序规则让用户标识、请求号区分大小写。
TABLE_OPTIONS = {
    "mysql_engine": "InnoDB", "mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_bin",
}


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = dict(TABLE_OPTIONS)

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="paid")
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    items: Mapped[list[OrderItem]] = relationship(
        cascade="all, delete-orphan", lazy="selectin", order_by="OrderItem.id",
    )


class OrderItem(Base):
    __tablename__ = "order_items"
    __table_args__ = dict(TABLE_OPTIONS)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.order_id"), index=True)
    product_name: Mapped[str] = mapped_column(String(200))
    quantity: Mapped[int] = mapped_column(Integer)
    unit_price: Mapped[Decimal] = mapped_column(Numeric(10, 2))


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (
        UniqueConstraint("user_id", "request_id", name="uq_ticket_user_request"),
        dict(TABLE_OPTIONS),
    )

    ticket_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    order_id: Mapped[str | None] = mapped_column(ForeignKey("orders.order_id"), index=True)
    ticket_type: Mapped[str] = mapped_column(String(32))
    priority: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32), default="created")
    summary: Mapped[str] = mapped_column(String(200))
    details: Mapped[str] = mapped_column(Text)
    request_id: Mapped[str] = mapped_column(String(64))
    request_hash: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class TicketEvent(Base):
    """流转事件与工单修改在同一事务提交，避免只有状态没有过程。"""
    __tablename__ = "ticket_events"
    __table_args__ = dict(TABLE_OPTIONS)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(ForeignKey("tickets.ticket_id"), index=True)
    from_status: Mapped[str | None] = mapped_column(String(32))
    to_status: Mapped[str] = mapped_column(String(32))
    note: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class Database:
    def __init__(self, engine: Engine):
        self.engine = engine
        # 共享 Engine 和工厂；每次操作在自己的线程内创建独立 Session。
        self.sessions = sessionmaker(bind=engine, expire_on_commit=False)

    @classmethod
    def from_env(cls) -> Database:
        names = ["MYSQL_HOST", "MYSQL_PORT", "MYSQL_DATABASE", "MYSQL_USER", "MYSQL_PASSWORD"]
        missing = [name for name in names if not os.getenv(name)]
        if missing:
            raise ValueError("请在 python-impl/.env 配置：" + ", ".join(missing))
        # URL.create 自动处理密码中的 @ 等字符，不手工拼接连接串。
        url = URL.create(
            "mysql+pymysql", username=os.environ["MYSQL_USER"],
            password=os.environ["MYSQL_PASSWORD"], host=os.environ["MYSQL_HOST"],
            port=int(os.environ["MYSQL_PORT"]), database=os.environ["MYSQL_DATABASE"],
            query={"charset": "utf8mb4"},
        )
        return cls(create_engine(
            url, pool_pre_ping=True, pool_recycle=1800, pool_size=5, max_overflow=5,
            connect_args={"connect_timeout": 10, "read_timeout": 15, "write_timeout": 15},
            hide_parameters=True,
        ))

    def create_tables(self) -> None:
        """显式初始化脚本调用；不会删除已有表，也不会迁移旧表字段。"""
        Base.metadata.create_all(self.engine)

    def check_ready(self) -> None:
        """API 启动时检查四张表；数据库不可用时不回退到假数据。"""
        with self.sessions() as session:
            for model in (Order, OrderItem, Ticket, TicketEvent):
                session.execute(select(model).limit(1)).first()

    def close(self) -> None:
        self.engine.dispose()
