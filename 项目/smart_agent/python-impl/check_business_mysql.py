"""MySQL实机验收，手动执行后保留一笔CHECK订单、一张工单和两条事件。

不访问外部模型或Milvus，不删除已有记录。测试中途失败时，已提交的记录也保留。
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import uuid

from dotenv import load_dotenv

from business.database import Database
from business.schemas import OrderCreate, OrderUpdate, TicketCreate, TicketUpdate
from business.service import BusinessError, BusinessService


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
    suffix = uuid.uuid4().hex[:12].upper()
    order_id = "ORD-CHECK-" + suffix
    user_id = "mysql_check_user"
    database = Database.from_env()
    try:
        database.check_ready()
        service = BusinessService(database.sessions)
        order = service.create_order(OrderCreate(
            order_id=order_id, user_id=user_id,
            items=[{"product_name": "MySQL验收商品", "quantity": 2, "unit_price": "19.90"}],
        ))
        assert order["amount"] == "39.80"
        service.update_order(order_id, OrderUpdate(user_id=user_id, status="shipped", expected_version=1))
        data = TicketCreate(user_id=user_id, order_id=order_id, request_id="check-" + suffix,
                            ticket_type="refund", summary="MySQL验收工单", details="验证业务持久化")
        # 四个相同并发请求只能创建一张，检验数据库唯一约束兜底。
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(lambda _: service.create_ticket(data), range(4)))
        assert len({record["ticket_id"] for record in records}) == 1
        ticket_id = records[0]["ticket_id"]
        updated = service.update_ticket(ticket_id, TicketUpdate(
            user_id=user_id, expected_version=1, status="processing"))
        assert updated["status"] == "processing" and updated["version"] == 2
        assert len(service.ticket_events(ticket_id, user_id)) == 2
        assert service.query_orders("another_user", order_id) == []
        try:
            service.update_ticket(ticket_id, TicketUpdate(
                user_id=user_id, expected_version=1, status="closed"))
        except BusinessError as exc:
            assert exc.code == "version_conflict"
        else:
            raise AssertionError("旧版本不应更新成功")
    finally:
        database.close()

    # 重新建立连接池再读取，确认结果不依赖Python内存中的旧对象。
    database = Database.from_env()
    try:
        service = BusinessService(database.sessions)
        assert service.query_orders(user_id, order_id)[0]["status"] == "shipped"
        assert service.query_tickets(user_id, ticket_id)[0]["status"] == "processing"
        print("MySQL验收通过：创建、查询、更新、并发防重、版本冲突、重新连接后的数据保留")
        print("测试用户：", user_id)
        print("测试订单：", order_id)
        print("测试工单：", ticket_id)
    finally:
        database.close()


if __name__ == "__main__":
    main()
