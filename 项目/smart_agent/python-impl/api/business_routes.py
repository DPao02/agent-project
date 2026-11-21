"""本地业务管理接口，与Agent工具共用同一个BusinessService。"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from business.schemas import OrderCreate, OrderUpdate, TicketCreate, TicketUpdate
from business.service import BusinessError, BusinessService

router = APIRouter(prefix="/api", tags=["业务数据"])


def service_from_app(request: Request) -> BusinessService:
    service = getattr(request.app.state, "business_service", None)
    if service is None:
        raise HTTPException(503, "业务数据库尚未就绪")
    return service


@router.post("/orders", status_code=201)
async def create_order(data: OrderCreate, service: BusinessService = Depends(service_from_app)):
    return await asyncio.to_thread(service.create_order, data)


@router.get("/orders")
async def list_orders(user_id: str = Query(min_length=1, max_length=64),
                      service: BusinessService = Depends(service_from_app)):
    return {"orders": await asyncio.to_thread(service.query_orders, user_id)}


@router.get("/orders/{order_id}")
async def get_order(order_id: str, user_id: str = Query(min_length=1, max_length=64),
                    service: BusinessService = Depends(service_from_app)):
    records = await asyncio.to_thread(service.query_orders, user_id, order_id)
    if not records:
        raise BusinessError("未找到当前用户的订单", 404, "order_not_found")
    return records[0]


@router.patch("/orders/{order_id}")
async def update_order(order_id: str, data: OrderUpdate,
                       service: BusinessService = Depends(service_from_app)):
    return await asyncio.to_thread(service.update_order, order_id, data)


@router.post("/tickets", status_code=201)
async def create_ticket(data: TicketCreate, service: BusinessService = Depends(service_from_app)):
    return await asyncio.to_thread(service.create_ticket, data)


@router.get("/tickets")
async def list_tickets(user_id: str = Query(min_length=1, max_length=64),
                       service: BusinessService = Depends(service_from_app)):
    return {"tickets": await asyncio.to_thread(service.query_tickets, user_id)}


@router.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str, user_id: str = Query(min_length=1, max_length=64),
                     service: BusinessService = Depends(service_from_app)):
    records = await asyncio.to_thread(service.query_tickets, user_id, ticket_id)
    if not records:
        raise BusinessError("未找到当前用户的工单", 404, "ticket_not_found")
    return records[0]


@router.patch("/tickets/{ticket_id}")
async def update_ticket(ticket_id: str, data: TicketUpdate,
                        service: BusinessService = Depends(service_from_app)):
    return await asyncio.to_thread(service.update_ticket, ticket_id, data)


@router.get("/tickets/{ticket_id}/events")
async def ticket_events(ticket_id: str, user_id: str = Query(min_length=1, max_length=64),
                        service: BusinessService = Depends(service_from_app)):
    return {"events": await asyncio.to_thread(service.ticket_events, ticket_id, user_id)}
