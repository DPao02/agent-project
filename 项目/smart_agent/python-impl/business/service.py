"""唯一业务入口：HTTP 和工具共用此服务，模型不生成 SQL。"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from business.database import Order, OrderItem, Ticket, TicketEvent, utc_now
from business.schemas import OrderCreate, OrderUpdate, Owner, TicketCreate, TicketUpdate


class BusinessError(Exception):
    def __init__(self, message: str, status_code: int = 400, code: str = "invalid_operation"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


ORDER_TRANSITIONS = {
    "pending_payment": {"paid", "cancelled"}, "paid": {"shipped", "cancelled"},
    "shipped": {"completed"}, "completed": set(), "cancelled": set(),
}
TICKET_TRANSITIONS = {
    "created": {"processing", "escalated", "closed"},
    "processing": {"pending_review", "resolved", "escalated", "closed"},
    "pending_review": {"processing", "resolved", "escalated", "closed"},
    "escalated": {"processing", "pending_review", "resolved", "closed"},
    "resolved": {"processing", "closed"}, "closed": set(),
}


def timestamp(value):
    return value.replace(tzinfo=timezone.utc).isoformat()


def order_dict(order: Order) -> dict:
    return {
        "order_id": order.order_id, "user_id": order.user_id, "status": order.status,
        "amount": str(order.amount), "currency": "CNY", "version": order.version,
        "items": [{"product_name": item.product_name, "quantity": item.quantity,
                   "unit_price": str(item.unit_price)} for item in order.items],
        "created_at": timestamp(order.created_at), "updated_at": timestamp(order.updated_at),
    }


def ticket_dict(ticket: Ticket) -> dict:
    return {
        "ticket_id": ticket.ticket_id, "user_id": ticket.user_id, "order_id": ticket.order_id,
        "type": ticket.ticket_type, "priority": ticket.priority, "status": ticket.status,
        "summary": ticket.summary, "details": ticket.details, "version": ticket.version,
        "created_at": timestamp(ticket.created_at), "updated_at": timestamp(ticket.updated_at),
    }


class BusinessService:
    def __init__(self, sessions: sessionmaker):
        self.sessions = sessions

    @staticmethod
    def _owner(user_id: str) -> str:
        return Owner(user_id=user_id).user_id

    @staticmethod
    def _check_version(record, expected_version: int) -> None:
        if record.version != expected_version:
            raise BusinessError("记录已被更新，请重新查询后再提交", 409, "version_conflict")

    @staticmethod
    def _check_transition(current: str, target: str, transitions: dict) -> None:
        if target != current and target not in transitions[current]:
            raise BusinessError(f"不允许从 {current} 变为 {target}", 409, "invalid_transition")

    def create_order(self, data: OrderCreate) -> dict:
        """录入订单及商品，不代表调用了真实支付或库存接口。"""
        amount = sum((item.unit_price * item.quantity for item in data.items), Decimal("0.00"))
        if amount > Decimal("9999999999.99"):
            raise BusinessError("订单总金额超出允许范围")
        try:
            # 订单头和所有商品一次提交，任一失败全部回滚。
            with self.sessions.begin() as session:
                order = Order(
                    order_id=data.order_id, user_id=data.user_id, status=data.status, amount=amount,
                    items=[OrderItem(**item.model_dump()) for item in data.items],
                )
                session.add(order)
                session.flush()
                result = order_dict(order)
            return result
        except IntegrityError as exc:
            raise BusinessError("订单号已存在，未重复创建", 409, "order_exists") from exc

    def query_orders(self, user_id: str, order_id: str = "", limit: int = 20) -> list[dict]:
        statement = select(Order).where(Order.user_id == self._owner(user_id))
        if order_id:
            statement = statement.where(Order.order_id == order_id)
        with self.sessions() as session:
            records = session.scalars(statement.order_by(
                Order.created_at.desc(), Order.order_id,
            ).limit(max(1, min(limit, 100)))).all()
            return [order_dict(record) for record in records]

    def update_order(self, order_id: str, data: OrderUpdate) -> dict:
        with self.sessions.begin() as session:
            # 行锁 + 版本检查，保护“读取 -> 校验 -> 写入”的完整过程。
            order = session.scalar(select(Order).where(
                Order.order_id == order_id, Order.user_id == data.user_id,
            ).with_for_update())
            if order is None:
                raise BusinessError("未找到当前用户的订单", 404, "order_not_found")
            self._check_version(order, data.expected_version)
            self._check_transition(order.status, data.status, ORDER_TRANSITIONS)
            if order.status != data.status:
                order.status = data.status
                order.version += 1
                order.updated_at = utc_now()
            session.flush()
            result = order_dict(order)
        return result

    def _replayed_ticket(self, data: TicketCreate, request_hash: str) -> dict | None:
        with self.sessions() as session:
            ticket = session.scalar(select(Ticket).where(
                Ticket.user_id == data.user_id, Ticket.request_id == data.request_id,
            ))
            if ticket is None:
                return None
            if ticket.request_hash != request_hash:
                raise BusinessError("此 request_id 已用于不同内容，请为新请求使用新编号", 409, "idempotency_conflict")
            return ticket_dict(ticket)

    def create_ticket(self, data: TicketCreate) -> dict:
        request_hash = hashlib.sha256(json.dumps(
            data.model_dump(exclude={"request_id"}), ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        existing = self._replayed_ticket(data, request_hash)
        if existing is not None:
            return {**existing, "replayed": True}
        try:
            with self.sessions.begin() as session:
                if data.order_id:
                    order = session.scalar(select(Order).where(
                        Order.order_id == data.order_id, Order.user_id == data.user_id,
                    ))
                    if order is None:
                        raise BusinessError("关联订单不存在或不属于当前用户", 404, "order_not_found")
                ticket = Ticket(
                    ticket_id="TK-" + uuid.uuid4().hex.upper(), user_id=data.user_id,
                    order_id=data.order_id, ticket_type=data.ticket_type, priority=data.priority,
                    summary=data.summary, details=data.details,
                    request_id=data.request_id, request_hash=request_hash,
                )
                session.add(ticket)
                session.flush()
                session.add(TicketEvent(ticket_id=ticket.ticket_id, from_status=None,
                                        to_status=ticket.status, note="创建工单"))
                result = ticket_dict(ticket)
            # 离开事务且提交成功之后，才允许反馈“创建成功”。
            return {**result, "replayed": False}
        except IntegrityError:
            # 并发请求可能同时通过前置查询，唯一约束负责最终兜底。
            existing = self._replayed_ticket(data, request_hash)
            if existing is not None:
                return {**existing, "replayed": True}
            raise

    def query_tickets(self, user_id: str, ticket_id: str = "", limit: int = 20) -> list[dict]:
        statement = select(Ticket).where(Ticket.user_id == self._owner(user_id))
        if ticket_id:
            statement = statement.where(Ticket.ticket_id == ticket_id)
        with self.sessions() as session:
            records = session.scalars(statement.order_by(
                Ticket.created_at.desc(), Ticket.ticket_id,
            ).limit(max(1, min(limit, 100)))).all()
            return [ticket_dict(record) for record in records]

    def update_ticket(self, ticket_id: str, data: TicketUpdate) -> dict:
        with self.sessions.begin() as session:
            ticket = session.scalar(select(Ticket).where(
                Ticket.ticket_id == ticket_id, Ticket.user_id == data.user_id,
            ).with_for_update())
            if ticket is None:
                raise BusinessError("未找到当前用户的工单", 404, "ticket_not_found")
            self._check_version(ticket, data.expected_version)
            previous = ticket.status
            if data.status is not None:
                self._check_transition(ticket.status, data.status, TICKET_TRANSITIONS)
                ticket.status = data.status
            if data.details is not None:
                ticket.details = data.details
            ticket.version += 1
            ticket.updated_at = utc_now()
            session.add(TicketEvent(
                ticket_id=ticket.ticket_id, from_status=previous, to_status=ticket.status,
                note="更新状态" if data.details is None else "更新工单内容或状态",
            ))
            session.flush()
            result = ticket_dict(ticket)
        return result

    def ticket_events(self, ticket_id: str, user_id: str) -> list[dict]:
        with self.sessions() as session:
            owner = session.scalar(select(Ticket.ticket_id).where(
                Ticket.ticket_id == ticket_id, Ticket.user_id == self._owner(user_id),
            ))
            if owner is None:
                raise BusinessError("未找到当前用户的工单", 404, "ticket_not_found")
            events = session.scalars(select(TicketEvent).where(
                TicketEvent.ticket_id == ticket_id,
            ).order_by(TicketEvent.id)).all()
            return [{"from_status": e.from_status, "to_status": e.to_status,
                     "note": e.note, "created_at": timestamp(e.created_at)} for e in events]
