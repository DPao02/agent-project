"""HTTP、工具和业务服务共用的输入规范。"""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TicketType = Literal["refund", "claim", "account_open", "account_change", "complaint", "general"]
Priority = Literal["low", "medium", "high", "urgent"]
TicketStatus = Literal["created", "processing", "pending_review", "resolved", "closed", "escalated"]
OrderStatus = Literal["pending_payment", "paid", "shipped", "completed", "cancelled"]


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Owner(InputModel):
    # 本地演示的归属字段。上线时应从登录凭据取得，不能信任前端自报身份。
    user_id: str = Field(min_length=1, max_length=64)


class OrderItemInput(InputModel):
    product_name: str = Field(min_length=1, max_length=200)
    quantity: int = Field(ge=1, le=10000)
    unit_price: Decimal = Field(gt=0, max_digits=10, decimal_places=2)


class OrderCreate(Owner):
    order_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    items: list[OrderItemInput] = Field(min_length=1, max_length=50)
    status: OrderStatus = "paid"


class OrderUpdate(Owner):
    status: OrderStatus
    # 客户端先查询，带查询到的版本更新，防止覆盖其他请求的新数据。
    expected_version: int = Field(ge=1)


class TicketCreate(Owner):
    ticket_type: TicketType = "general"
    priority: Priority = "medium"
    summary: str = Field(min_length=1, max_length=200)
    details: str = Field(min_length=1, max_length=4000)
    order_id: str | None = Field(default=None, min_length=1, max_length=64)
    request_id: str = Field(min_length=8, max_length=64)


class TicketUpdate(Owner):
    status: TicketStatus | None = None
    details: str | None = Field(default=None, min_length=1, max_length=4000)
    expected_version: int = Field(ge=1)

    @model_validator(mode="after")
    def require_change(self):
        if self.status is None and self.details is None:
            raise ValueError("至少提供 status 或 details 中的一项")
        return self
