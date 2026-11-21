"""多轮任务与记忆隔离回归测试，不调用真实模型或写入用户MySQL。"""

import asyncio
import json
import os
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine

from business.database import Database
from business.schemas import OrderCreate
from business.service import BusinessService
from business.task_state import BusinessDecision, TaskState
from memory.context import recent_messages
from memory.identity import session_scope
from memory.short_term import SessionBusy, ShortTermMemory, StoredMessage
from memory.working_memory import WorkingMemory


class WorkingMemoryTests(unittest.TestCase):
    def test_user_and_session_isolation_and_no_delimiter_collision(self):
        memory = WorkingMemory()
        memory.update("same", {"task": {"order_id": "A"}}, user_id="u1")
        memory.update("same", {"task": {"order_id": "B"}}, user_id="u2")
        self.assertEqual(memory.get_context("same", user_id="u1")["task"]["order_id"], "A")
        self.assertEqual(memory.get_context("same", user_id="u2")["task"]["order_id"], "B")
        self.assertEqual(memory.get_context("another", user_id="u1"), {})
        self.assertNotEqual(session_scope("a:b", "c"), session_scope("a", "b:c"))
        copy = memory.get_context("same", user_id="u1")
        copy["task"]["order_id"] = "mutated"
        self.assertEqual(memory.get_context("same", user_id="u1")["task"]["order_id"], "A")

    def test_model_context_is_bounded(self):
        messages = [AIMessage(content="x" * 20000), HumanMessage(content="那运费呢")]
        result = recent_messages(messages, max_chars=100)
        self.assertEqual(result[-1].content, "那运费呢")
        self.assertLessEqual(sum(len(m.content) for m in result), 100)


class ShortMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_first_access_uses_one_redis_client(self):
        memory = ShortTermMemory(redis_url="redis://unused", allow_memory_fallback=False)
        client = AsyncMock()
        async def ping():
            await asyncio.sleep(0)
            return True
        client.ping.side_effect = ping
        with patch("memory.short_term.Redis.from_url", return_value=client) as factory:
            clients = await asyncio.gather(memory._get_redis(), memory._get_redis())
            factory.assert_called_once()
            self.assertIs(clients[0], clients[1])
        await memory.close()

    async def test_snapshot_history_task_isolation_and_lock(self):
        memory = ShortTermMemory(redis_url="memory://", max_turns=1)
        async with memory.session("same", user_id="u1") as lease:
            snapshot = await lease.load()
            snapshot.task = TaskState(action="ticket_create")
            snapshot.messages = [StoredMessage(role="user", content="old"),
                                 StoredMessage(role="assistant", content="old response"),
                                 StoredMessage(role="user", content="new")]
            await lease.save(snapshot)
            with self.assertRaises(SessionBusy):
                async with memory.session("same", user_id="u1"):
                    pass
            async with memory.session("same", user_id="u2") as another:
                self.assertIsNone((await another.load()).task)
        loaded = await memory.get_snapshot("same", user_id="u1")
        self.assertEqual(len(loaded.messages), 2)
        self.assertEqual(loaded.task.action, "ticket_create")
        self.assertEqual(await memory.get_history("other", user_id="u1"), [])

    async def test_fallback_expires(self):
        memory = ShortTermMemory(redis_url="memory://", ttl_seconds=1)
        await memory.add_message("s", "user", "temporary", user_id="u")
        await asyncio.sleep(1.05)
        self.assertEqual(await memory.get_history("s", user_id="u"), [])

    async def test_fallback_conversation_does_not_switch_to_empty_redis(self):
        memory = ShortTermMemory(redis_url="memory://")
        await memory.add_message("s", "user", "preserve", user_id="u")
        with patch.object(memory, "_get_redis", AsyncMock(side_effect=AssertionError("不应切换后端"))):
            self.assertEqual((await memory.get_history("s", user_id="u"))[0]["content"], "preserve")


class PolicyMemory:
    def load_knowledge_base(self, path): return 1
    def save(self): pass
    def close(self): pass
    def search(self, query, top_k=3):
        return [{"content": "退货运费按退款政策处理", "source": "refund_policy.pdf#page=1", "score": 0.9}]


class IncrementalAnalyzer:
    async def ainvoke(self, messages):
        text = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
        if text == "我要申请退款":
            return BusinessDecision(action="ticket_create", ticket_type="refund")
        if text == "重新开始退款":
            return BusinessDecision(action="ticket_create", ticket_type="refund", new_task=True)
        if text == "查询订单":
            return BusinessDecision(action="order_query")
        if text == "删除订单号":
            return BusinessDecision(action="continue", clear_fields=["order_id"])
        if text.startswith("订单号"):
            return BusinessDecision(action="continue", order_id=text.split()[-1])
        if text.startswith("原因"):
            return BusinessDecision(action="continue", details=text.split("：")[-1])
        if text.startswith("更新工单"):
            return BusinessDecision(action="ticket_update", ticket_id=text.split()[-1], status="processing")
        return BusinessDecision(action="continue")


class ConversationModel:
    def __init__(self):
        self.rewrite_inputs = []

    def with_structured_output(self, *args, **kwargs):
        return IncrementalAnalyzer()

    async def ainvoke(self, messages):
        first = messages[0].content
        if first.startswith("你是一个金融合规审查Agent"):
            return AIMessage(content=json.dumps({"passed": True, "risk_level": "low"}))
        if first.startswith("你是一个智能客服系统的Supervisor"):
            users = [m.content for m in messages if isinstance(m, HumanMessage) and not m.content.startswith("请分析用户")]
            return AIMessage(content="knowledge_rag" if "政策" in users[-1] or "运费" in users[-1] else "ticket_handler")
        if first.startswith("你是查询改写助手"):
            self.rewrite_inputs.append(messages)
            return AIMessage(content="退款政策中退货运费由谁承担")
        if first.startswith("你是一个文档相关性排序专家"):
            return AIMessage(content="0")
        return AIMessage(content="根据退款政策，退货运费按条款承担。")


class ConversationFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database = Database(create_engine("sqlite:///" + (Path(self.directory.name) / "business.sqlite").as_posix(),
                                              connect_args={"check_same_thread": False}))
        self.database.create_tables()
        self.service = BusinessService(self.database.sessions)
        for order_id in ("ORD-A", "ORD-B"):
            self.service.create_order(OrderCreate(order_id=order_id, user_id="u1", items=[
                {"product_name": order_id, "quantity": 1, "unit_price": "10.00"},
            ]))
        self.model = ConversationModel()
        self.memory = ShortTermMemory(redis_url="memory://")

    def tearDown(self):
        self.database.close()
        self.directory.cleanup()

    @contextmanager
    def client(self, memory=None):
        from fastapi.testclient import TestClient
        from agents.supervisor import create_supervisor_graph
        with patch("dotenv.load_dotenv", return_value=False):
            import api.main as api
        def builder(**kwargs):
            return create_supervisor_graph(llm=self.model, **kwargs)
        with patch.object(api.Database, "from_env", return_value=self.database), \
             patch.object(api, "LongTermMemory", PolicyMemory), \
             patch.object(api, "working_memory", WorkingMemory()), \
             patch.object(api, "short_term_memory", memory or self.memory), \
             patch.object(api, "init_tracer"), \
             patch.object(api, "create_supervisor_graph", side_effect=builder), \
             TestClient(api.app) as client:
            yield client

    def turn(self, client, text, user="u1", session="flow", request_id=None):
        response = client.post("/api/chat", json={"user_id": user, "session_id": session,
                               "message": text, "request_id": request_id or uuid.uuid4().hex})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def fill(self, client):
        first = self.turn(client, "我要申请退款")
        self.assertEqual(set(first["task"]["missing_fields"]), {"order_id", "details"})
        second = self.turn(client, "订单号是 ORD-A")
        self.assertEqual(second["task"]["task_id"], first["task"]["task_id"])
        self.assertEqual(second["task"]["missing_fields"], ["details"])
        ready = self.turn(client, "原因是：尺寸不合适")
        self.assertEqual(ready["task"]["stage"], "awaiting_confirmation")
        self.assertEqual(self.service.query_tickets("u1"), [])
        return ready

class ConversationTests(ConversationFixture):
    def test_supplement_modify_confirm_and_idempotent_retry(self):
        with self.client() as client:
            ready = self.fill(client)
            changed = self.turn(client, "订单号改成 ORD-B")
            self.assertEqual(changed["task"]["fields"]["details"], "尺寸不合适")
            self.assertEqual(changed["task"]["fields"]["order_id"], "ORD-B")
            done = self.turn(client, "确认提交", request_id="confirmation-001")
            self.assertEqual(done["task"]["stage"], "completed")
            retried = self.turn(client, "确认提交", request_id="confirmation-001")
            self.assertEqual(done, retried)
            records = self.service.query_tickets("u1")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["order_id"], "ORD-B")
            self.assertEqual(records[0]["details"], "尺寸不合适")
            self.assertEqual(ready["task"]["task_id"], done["task"]["task_id"])
            self.turn(client, "确认提交")
            self.assertEqual(len(self.service.query_tickets("u1")), 1)
            history = client.get("/api/history/flow", params={"user_id": "u1"}).json()["messages"]
            self.assertEqual(len(history), 12)  # 缓存重试没有重复追加消息。
            cancelled = self.turn(client, "取消")
            self.assertEqual(cancelled["task"]["stage"], "completed")
            self.assertEqual(len(self.service.query_tickets("u1")), 1)

    def test_cancel_then_start_has_no_old_fields(self):
        with self.client() as client:
            self.fill(client)
            cancelled = self.turn(client, "取消")
            self.assertEqual(cancelled["task"]["stage"], "cancelled")
            self.turn(client, "确认提交")
            self.assertEqual(self.service.query_tickets("u1"), [])
            fresh = self.turn(client, "我要申请退款")
            self.assertIsNone(fresh["task"]["fields"]["order_id"])
            self.assertIsNone(fresh["task"]["fields"]["details"])

    def test_clearing_slot_recomputes_missing_and_prevents_submit(self):
        with self.client() as client:
            self.fill(client)
            cleared = self.turn(client, "删除订单号")
            self.assertEqual(cleared["task"]["missing_fields"], ["order_id"])
            self.turn(client, "确认提交")
            self.assertEqual(self.service.query_tickets("u1"), [])

    def test_user_session_scope_and_reset_does_not_resurrect_checkpoint(self):
        with self.client() as client:
            self.fill(client)
            for user, session in (("u2", "flow"), ("u1", "other")):
                snapshot = client.get(f"/api/sessions/{session}", params={"user_id": user}).json()
                self.assertIsNone(snapshot["task"])
                self.assertEqual(snapshot["message_count"], 0)
                history = client.get(f"/api/history/{session}", params={"user_id": user}).json()["messages"]
                self.assertEqual(history, [])
            client.delete("/api/sessions/flow", params={"user_id": "u1"})
            response = self.turn(client, "确认提交")
            self.assertIsNone(response["task"])
            self.assertEqual(self.service.query_tickets("u1"), [])

    def test_rag_uses_prior_history_and_preserves_pending_task(self):
        with self.client() as client:
            ready = self.fill(client)
            policy = self.turn(client, "退款政策怎么规定？")
            followup = self.turn(client, "那运费呢？")
            self.assertEqual(policy["task"], ready["task"])
            self.assertEqual(followup["task"], ready["task"])
            last_context = self.model.rewrite_inputs[-1]
            self.assertTrue(any("退款政策怎么规定" in m.content for m in last_context))
            self.assertTrue(any("尺寸不合适" in m.content for m in last_context))
            self.assertEqual(self.service.query_tickets("u1"), [])

    def test_missing_order_does_not_list_all_without_permission(self):
        with self.client() as client:
            result = self.turn(client, "查询订单")
            self.assertEqual(result["task"]["missing_fields"], ["order_id"])
            self.assertNotIn('"ORD-A"', result["response"])

    def test_reusing_request_id_for_changed_message_is_conflict(self):
        with self.client() as client:
            self.turn(client, "我要申请退款", request_id="same-request-001")
            response = client.post("/api/chat", json={"user_id": "u1", "session_id": "flow",
                "message": "取消", "request_id": "same-request-001"})
            self.assertEqual(response.status_code, 409)

    def test_confirm_cannot_overwrite_database_change_after_preview(self):
        from business.schemas import TicketCreate, TicketUpdate
        record = self.service.create_ticket(TicketCreate(
            user_id="u1", order_id="ORD-A", ticket_type="refund", summary="退款",
            details="需要退款", request_id="external-ticket-001"))
        with self.client() as client:
            preview = self.turn(client, "更新工单 " + record["ticket_id"])
            self.assertEqual(preview["task"]["fields"]["expected_version"], 1)
            self.service.update_ticket(record["ticket_id"], TicketUpdate(
                user_id="u1", expected_version=1, status="escalated"))
            rejected = self.turn(client, "确认提交")
            self.assertEqual(rejected["task"]["stage"], "collecting")
            self.assertIsNone(rejected["task"]["fields"]["expected_version"])
            unchanged = self.service.query_tickets("u1", record["ticket_id"])[0]
            self.assertEqual(unchanged["status"], "escalated")
            self.assertEqual(unchanged["version"], 2)
            refreshed = self.turn(client, "确认提交")
            self.assertEqual(refreshed["task"]["stage"], "awaiting_confirmation")
            self.assertEqual(refreshed["task"]["fields"]["expected_version"], 2)


@unittest.skipUnless(os.getenv("TEST_REDIS_URL"), "设置TEST_REDIS_URL后执行真实Redis恢复/租约测试")
class RedisRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_client_recovers_task_and_isolates_owner(self):
        url = os.environ["TEST_REDIS_URL"]
        session = "test-" + uuid.uuid4().hex
        first = ShortTermMemory(redis_url=url, allow_memory_fallback=False)
        second = ShortTermMemory(redis_url=url, allow_memory_fallback=False)
        try:
            async with first.session(session, user_id="memory-test") as lease:
                snapshot = await lease.load()
                snapshot.task = TaskState(action="ticket_create")
                snapshot.task.fields.order_id = "ORD-RECOVER"
                snapshot.messages.append(StoredMessage(role="user", content="我要退款"))
                await lease.save(snapshot)
                with self.assertRaises(SessionBusy):
                    async with second.session(session, user_id="memory-test"):
                        pass
            await first.close()
            loaded = await second.get_snapshot(session, user_id="memory-test")
            self.assertEqual(loaded.task.fields.order_id, "ORD-RECOVER")
            self.assertEqual(loaded.messages[0].content, "我要退款")
            self.assertIsNone((await second.get_snapshot(session, user_id="someone-else")).task)
            await second.clear(session, user_id="memory-test")
        finally:
            await first.close()
            await second.close()


@unittest.skipUnless(os.getenv("TEST_REDIS_URL"), "设置TEST_REDIS_URL后执行真实Redis/API重建恢复测试")
class RedisAPIRecoveryTests(ConversationFixture):
    def test_rebuilt_graph_and_working_memory_resume_pending_task(self):
        url = os.environ["TEST_REDIS_URL"]
        session = "restart-test-" + uuid.uuid4().hex
        first_memory = ShortTermMemory(redis_url=url, allow_memory_fallback=False)
        second_memory = ShortTermMemory(redis_url=url, allow_memory_fallback=False)
        with self.client(first_memory) as client:
            first = self.turn(client, "我要申请退款", session=session)
            collected = self.turn(client, "订单号是 ORD-A", session=session)
            self.assertEqual(collected["memory_backend"], "redis")
        # 新客户端、新Graph和新工作缓存；仅Redis快照保留。
        with self.client(second_memory) as client:
            recovered = self.turn(client, "原因是：尺寸不合适", session=session)
            self.assertEqual(recovered["task"]["task_id"], first["task"]["task_id"])
            self.assertEqual(recovered["task"]["fields"]["order_id"], "ORD-A")
            self.assertEqual(recovered["task"]["stage"], "awaiting_confirmation")
            done = self.turn(client, "确认提交", session=session)
            self.assertEqual(done["task"]["stage"], "completed")
            self.assertEqual(len(self.service.query_tickets("u1")), 1)
            history = client.get(f"/api/history/{session}", params={"user_id": "u1"}).json()["messages"]
            self.assertEqual(len(history), 8)
            client.delete(f"/api/sessions/{session}", params={"user_id": "u1"})


if __name__ == "__main__":
    unittest.main()
