"""工具适配层：所有业务操作转交同一个 BusinessService。"""

from __future__ import annotations

import asyncio

from business.schemas import Owner, TicketCreate, TicketUpdate
from business.service import BusinessService
from memory.long_term import LongTermMemory
from mcp.mcp_server import MCPToolServer


def register_business_tools(
    server: MCPToolServer, service: BusinessService, memory: LongTermMemory,
) -> MCPToolServer:
    @server.register(
        name="order_query", description="查询当前用户订单，未指定订单号时列出最近20条",
        category="order",
        input_schema={"type": "object", "properties": {
            "user_id": {"type": "string"}, "order_id": {"type": "string"},
        }, "required": ["user_id"], "additionalProperties": False},
    )
    async def order_query(user_id: str, order_id: str = "") -> dict:
        user_id = Owner(user_id=user_id).user_id
        orders = await asyncio.to_thread(service.query_orders, user_id, order_id)
        return {"orders": orders, "count": len(orders)}

    @server.register(
        name="ticket_create", description="创建实际工单，相同请求号和内容不会重复创建",
        category="ticket",
        input_schema={"type": "object", "properties": {
            "user_id": {"type": "string"}, "request_id": {"type": "string"},
            "title": {"type": "string"}, "description": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
            "category": {"type": "string", "enum": ["refund", "claim", "account_open", "account_change", "complaint", "general"]},
            "order_id": {"type": ["string", "null"]},
        }, "required": ["user_id", "request_id", "title", "description"], "additionalProperties": False},
    )
    async def ticket_create(
        user_id: str, request_id: str, title: str, description: str,
        priority: str = "medium", category: str = "general", order_id: str | None = None,
    ) -> dict:
        data = TicketCreate(user_id=user_id, request_id=request_id, summary=title,
                            details=description, priority=priority, ticket_type=category, order_id=order_id)
        return await asyncio.to_thread(service.create_ticket, data)

    @server.register(
        name="ticket_query", description="查询当前用户工单，未指定工单号时列出最近20条",
        category="ticket",
        input_schema={"type": "object", "properties": {
            "user_id": {"type": "string"}, "ticket_id": {"type": "string"},
        }, "required": ["user_id"], "additionalProperties": False},
    )
    async def ticket_query(user_id: str, ticket_id: str = "") -> dict:
        user_id = Owner(user_id=user_id).user_id
        tickets = await asyncio.to_thread(service.query_tickets, user_id, ticket_id)
        return {"tickets": tickets, "count": len(tickets)}

    @server.register(
        name="ticket_update", description="更新实际工单，需要查询得到的最新版本号",
        category="ticket",
        input_schema={"type": "object", "properties": {
            "user_id": {"type": "string"}, "ticket_id": {"type": "string"},
            "expected_version": {"type": "integer", "minimum": 1},
            "status": {"type": ["string", "null"], "enum": [None, "created", "processing", "pending_review", "resolved", "closed", "escalated"]},
            "details": {"type": ["string", "null"]},
        }, "required": ["user_id", "ticket_id", "expected_version"], "additionalProperties": False},
    )
    async def ticket_update(
        user_id: str, ticket_id: str, expected_version: int,
        status: str | None = None, details: str | None = None,
    ) -> dict:
        data = TicketUpdate(user_id=user_id, expected_version=expected_version, status=status, details=details)
        return await asyncio.to_thread(service.update_ticket, ticket_id, data)

    @server.register(
        name="knowledge_search", description="检索现有Milvus知识库，返回真实文档片段",
        category="knowledge",
        input_schema={"type": "object", "properties": {
            "query": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        }, "required": ["query"], "additionalProperties": False},
    )
    async def knowledge_search(query: str, top_k: int = 3) -> list[dict]:
        if not query.strip() or isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("query不能为空，top_k必须是1～20的整数")
        return await asyncio.to_thread(memory.search, query, top_k=top_k)

    # 沿用原有金额阈值规则，尚未对接真实外部风控服务。
    @server.register(
        name="risk_check", description="按金额阈值进行规则风险检查",
        category="compliance",
        input_schema={"type": "object", "properties": {
            "user_id": {"type": "string"}, "action": {"type": "string"}, "amount": {"type": "number"},
        }, "required": ["user_id", "action"], "additionalProperties": False},
    )
    async def risk_check(user_id: str, action: str, amount: float = 0.0) -> dict:
        level = "high" if amount > 50000 else "medium" if amount > 10000 else "low"
        return {"user_id": user_id, "action": action, "risk_level": level,
                "requires_manual_review": level == "high"}

    return server
