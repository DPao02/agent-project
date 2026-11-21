"""临时SQLite真实SQL测试，不连接用户MySQL、Milvus或外部模型。

模型替身仅固定意图；HTTP、Graph、工具、数据库事务执行实际代码。
MySQL认证和并发另由check_business_mysql.py验收。
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, event, func, select

from business.database import Database, OrderItem
from business.schemas import OrderCreate, OrderUpdate, TicketCreate, TicketUpdate
from business.service import BusinessError, BusinessService

# 测试不向外部追踪服务发送数据。
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


class BusinessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "test.sqlite"
        self.database = self.open_database()
        self.database.create_tables()
        self.service = BusinessService(self.database.sessions)

    def open_database(self):
        engine = create_engine("sqlite:///" + self.path.as_posix(), connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def enforce_foreign_keys(connection, record):
            connection.execute("PRAGMA foreign_keys=ON")

        return Database(engine)

    def tearDown(self):
        self.database.close()
        self.directory.cleanup()

    def order(self):
        return self.service.create_order(OrderCreate(
            order_id="ORD-DEMO-1001", user_id="u001",
            items=[{"product_name": "商品甲", "quantity": 2, "unit_price": "19.90"},
                   {"product_name": "商品乙", "quantity": 1, "unit_price": "8.20"}],
        ))

    def ticket_input(self, **changes):
        values = dict(user_id="u001", request_id="request-demo-001", ticket_type="refund",
                      summary="退款申请", details="商品不合适，申请退款", order_id="ORD-DEMO-1001")
        values.update(changes)
        return TicketCreate(**values)

    def test_order_amount_and_owner_filter(self):
        self.order()
        order = self.service.query_orders("u001", "ORD-DEMO-1001")[0]
        self.assertEqual(order["amount"], "48.00")
        self.assertEqual(len(order["items"]), 2)
        self.assertEqual(self.service.query_orders("u002", "ORD-DEMO-1001"), [])

    def test_duplicate_order_rolls_back_items(self):
        self.order()
        with self.assertRaises(BusinessError):
            self.order()
        with self.database.sessions() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(OrderItem)), 2)

    def test_ticket_idempotency_and_conflict(self):
        self.order()
        first = self.service.create_ticket(self.ticket_input())
        retry = self.service.create_ticket(self.ticket_input())
        self.assertEqual(first["ticket_id"], retry["ticket_id"])
        self.assertTrue(retry["replayed"])
        with self.assertRaises(BusinessError) as conflict:
            self.service.create_ticket(self.ticket_input(details="不同的内容"))
        self.assertEqual(conflict.exception.code, "idempotency_conflict")
        self.assertEqual(len(self.service.query_tickets("u001")), 1)

    def test_cannot_attach_another_users_order(self):
        self.order()
        with self.assertRaises(BusinessError):
            self.service.create_ticket(self.ticket_input(user_id="u002"))
        self.assertEqual(self.service.query_tickets("u002"), [])

    def test_update_version_and_events(self):
        self.order()
        ticket = self.service.create_ticket(self.ticket_input())
        changed = self.service.update_ticket(ticket["ticket_id"], TicketUpdate(
            user_id="u001", expected_version=1, status="processing"))
        self.assertEqual(changed["version"], 2)
        self.assertEqual(changed["status"], "processing")
        with self.assertRaises(BusinessError) as stale:
            self.service.update_ticket(ticket["ticket_id"], TicketUpdate(
                user_id="u001", expected_version=1, status="closed"))
        self.assertEqual(stale.exception.code, "version_conflict")
        events = self.service.ticket_events(ticket["ticket_id"], "u001")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1]["from_status"], "created")
        self.assertEqual(events[1]["to_status"], "processing")

    def test_invalid_transition_does_not_partially_update(self):
        self.order()
        ticket = self.service.create_ticket(self.ticket_input())
        with self.assertRaises(BusinessError):
            self.service.update_ticket(ticket["ticket_id"], TicketUpdate(
                user_id="u001", expected_version=1, status="resolved", details="不应保存"))
        unchanged = self.service.query_tickets("u001", ticket["ticket_id"])[0]
        self.assertEqual(unchanged["version"], 1)
        self.assertNotEqual(unchanged["details"], "不应保存")
        self.assertEqual(len(self.service.ticket_events(ticket["ticket_id"], "u001")), 1)

    def test_order_update_and_reconnect_persistence(self):
        self.order()
        self.service.update_order("ORD-DEMO-1001", OrderUpdate(
            user_id="u001", expected_version=1, status="shipped"))
        self.database.close()
        self.database = self.open_database()
        self.database.check_ready()
        self.service = BusinessService(self.database.sessions)
        persisted = self.service.query_orders("u001", "ORD-DEMO-1001")[0]
        self.assertEqual(persisted["status"], "shipped")
        self.assertEqual(persisted["version"], 2)

    def test_cannot_update_another_users_ticket(self):
        self.order()
        ticket = self.service.create_ticket(self.ticket_input())
        with self.assertRaises(BusinessError):
            self.service.update_ticket(ticket["ticket_id"], TicketUpdate(
                user_id="u002", expected_version=1, status="closed"))
        self.assertEqual(self.service.query_tickets("u002", ticket["ticket_id"]), [])

    def test_http_graph_and_tools_share_database(self):
        from fastapi.testclient import TestClient
        from langchain_core.messages import AIMessage, HumanMessage
        from agents.supervisor import create_supervisor_graph as real_graph_builder
        from agents.ticket_handler import BusinessDecision
        from memory.short_term import ShortTermMemory

        # 禁止API导入时加载用户真实.env，测试配置只存在于当前进程。
        with patch("dotenv.load_dotenv", return_value=False):
            import api.main as api

        class FakeMemory:
            def load_knowledge_base(self, path): return 1
            def save(self): pass
            def close(self): pass
            def search(self, query, top_k=3): return []

        class Analyzer:
            async def ainvoke(self, messages):
                latest = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
                if "申请退款" in latest:
                    return BusinessDecision(action="ticket_create", order_id="ORD-DEMO-1001",
                                            ticket_type="refund", summary="退款申请", details="商品不合适")
                if "更新工单" in latest:
                    return BusinessDecision(action="ticket_update", ticket_id=latest.split()[-1], status="processing")
                if "信息不足" in latest:
                    return BusinessDecision(action="clarify", clarification="请补充信息")
                return BusinessDecision(action="order_query", order_id="ORD-DEMO-1001")

        class FakeModel:
            deny = False
            def with_structured_output(self, *args, **kwargs): return Analyzer()
            async def ainvoke(self, messages):
                if messages[0].content.startswith("你是一个金融合规审查Agent"):
                    return AIMessage(content=json.dumps({"passed": not self.deny, "risk_level": "low"}))
                return AIMessage(content="ticket_handler")

        model = FakeModel()

        def builder(**kwargs):
            return real_graph_builder(llm=model, **kwargs)

        with patch.object(api.Database, "from_env", return_value=self.database), \
             patch.object(api, "LongTermMemory", FakeMemory), \
             patch.object(api, "short_term_memory", ShortTermMemory(redis_url="memory://")), \
             patch.object(api, "init_tracer"), \
             patch.object(api, "create_supervisor_graph", side_effect=builder), \
             TestClient(api.app) as client:
            response = client.post("/api/orders", json={"order_id": "ORD-DEMO-1001", "user_id": "u001",
                "items": [{"product_name": "数据库商品", "quantity": 1, "unit_price": "88.50"}]})
            self.assertEqual(response.status_code, 201, response.text)
            changed = client.patch("/api/orders/ORD-DEMO-1001", json={
                "user_id": "u001", "expected_version": 1, "status": "shipped"})
            self.assertEqual(changed.status_code, 200, changed.text)
            tool = client.post("/api/tools/call", json={"name": "order_query", "arguments": {
                "user_id": "u001", "order_id": "ORD-DEMO-1001"}}).json()
            self.assertTrue(tool["success"], tool)
            self.assertEqual(tool["result"]["orders"][0]["status"], "shipped")
            chat = client.post("/api/chat", json={"user_id": "u001", "message": "查询订单 ORD-DEMO-1001"})
            self.assertEqual(chat.status_code, 200, chat.text)
            self.assertIn("数据库商品", chat.json()["response"])

            body = {"user_id": "u001", "session_id": "business-flow", "message": "申请退款 ORD-DEMO-1001", "request_id": "chat-request-001"}
            first = client.post("/api/chat", json=body)
            second = client.post("/api/chat", json=body)
            self.assertEqual(first.status_code, 200, first.text)
            self.assertEqual(second.status_code, 200, second.text)
            self.assertEqual(first.json()["task"]["stage"], "awaiting_confirmation")
            confirmed = client.post("/api/chat", json={**body, "message": "确认提交", "request_id": "chat-confirm-001"})
            self.assertEqual(confirmed.status_code, 200, confirmed.text)
            tickets = client.get("/api/tickets", params={"user_id": "u001"}).json()["tickets"]
            self.assertEqual(len(tickets), 1)
            ticket_id = tickets[0]["ticket_id"]
            changed = client.post("/api/chat", json={**body, "message": "更新工单 " + ticket_id, "request_id": "chat-update-001"})
            self.assertEqual(changed.status_code, 200, changed.text)
            changed = client.post("/api/chat", json={**body, "message": "确认提交", "request_id": "chat-update-confirm-001"})
            self.assertEqual(changed.status_code, 200, changed.text)
            record = client.get(f"/api/tickets/{ticket_id}", params={"user_id": "u001"}).json()
            self.assertEqual(record["status"], "processing")

            client.post("/api/chat", json={**body, "request_id": "chat-request-002"})
            model.deny = True
            denied = client.post("/api/chat", json={**body, "message": "确认提交", "request_id": "chat-confirm-002"})
            self.assertEqual(denied.status_code, 200, denied.text)
            self.assertFalse(denied.json()["compliance_passed"])
            self.assertIn("本次工单操作已保存", denied.json()["response"])
            self.assertNotIn("已转交人工", denied.json()["response"])

            before = len(self.service.query_tickets("u001"))
            client.post("/api/chat", json={"user_id": "u001", "message": "信息不足"})
            self.assertEqual(len(self.service.query_tickets("u001")), before)
            self.assertEqual(client.get(f"/api/tickets/{ticket_id}", params={"user_id": "u002"}).status_code, 404)
            log = client.get("/api/metrics").json()["tool_call_log"]
            self.assertIn("order_query", [record["tool"] for record in log])


class IntentParsingTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_model_output_does_not_create_ticket(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from langchain_core.messages import HumanMessage
        from agents.ticket_handler import TicketHandlerAgent

        analyzer = SimpleNamespace(ainvoke=AsyncMock(return_value=None))
        model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: analyzer)
        server = SimpleNamespace(call_tool=AsyncMock())
        agent = TicketHandlerAgent(model, server)
        result = await agent.process({
            "messages": [HumanMessage(content="帮我处理一下")],
            "user_id": "u001", "request_id": "empty-output-001",
        })
        server.call_tool.assert_not_awaited()
        self.assertIn("未能识别", result["sub_results"]["ticket_handler"])


if __name__ == "__main__":
    unittest.main()
