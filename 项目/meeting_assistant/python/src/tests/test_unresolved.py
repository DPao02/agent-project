"""验证未决事项提取、原文引用、图降级、存储、报告与HTTP/WebSocket输出。"""
import copy
import json
import os
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from src.agents.followup_agent import FollowUpAgent
from src.agents.transcription_agent import TranscriptionAgent
from src.agents.unresolved_agent import UnresolvedAgent
from src.integrations.feishu_client import FeishuClient
from src.integrations.milvus_client import MeetingVectorStore
from src.models.schemas import UnresolvedResult, create_initial_state
from src.storage import MeetingRecord, restore_state, serialize_state
from src.websocket.server import create_app
from test_meeting_features import DisabledVectors, pipeline, repository
from test_pipeline import model_result


_tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"})


def setUpModule():
    _tracing.start()


def tearDownModule():
    _tracing.stop()


def demo_state(meeting_id="unresolved-test"):
    state = create_initial_state(meeting_id, mode="demo")
    state["transcript"] = TranscriptionAgent._generate_demo_transcript(meeting_id)
    state["transcript_text"] = TranscriptionAgent._format_transcript_text(state["transcript"])
    return state


class FixedLLM:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def chat_json(self, messages, **kwargs):
        self.calls += 1
        return copy.deepcopy(self.result)


def issue_payload():
    return model_result('"issues"')


class UnresolvedAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_evidence_uses_actual_segment_identity_and_time(self):
        raw = issue_payload()
        raw["issues"][0]["evidence"][0].update(speaker_id="fake-user", start=999, end=1000)
        state = demo_state()
        before = copy.deepcopy(state)
        result = await UnresolvedAgent(FixedLLM(raw)).process(state)
        self.assertEqual(result["errors"], [])
        issue = result["unresolved"].issues[0]
        evidence = issue.evidence[0]
        self.assertEqual(evidence.speaker_id, "SPEAKER_03")
        self.assertEqual((evidence.start, evidence.end), (61.5, 70.2))
        self.assertEqual(state, before)
        self.assertEqual(set(result), {"unresolved", "errors"})

    async def test_duplicate_issues_and_quotes_are_merged(self):
        raw = issue_payload()
        second = copy.deepcopy(raw["issues"][0])
        second["evidence"].append({"segment_index": 10, "quote": "供应商选择留到下一次确认"})
        raw["issues"].append(second)
        result = await UnresolvedAgent(FixedLLM(raw)).process(demo_state())
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["unresolved"].issues), 1)
        self.assertEqual(len(result["unresolved"].issues[0].evidence), 2)

    async def test_no_issues_is_a_successful_empty_result(self):
        result = await UnresolvedAgent(FixedLLM({"issues": []})).process(demo_state())
        self.assertEqual(result["unresolved"].issues, [])
        self.assertEqual(result["errors"], [])
        text = FollowUpAgent._format_unresolved_markdown(result["unresolved"])
        self.assertIn("未发现", text)
        self.assertNotIn("失败", text)

    async def test_empty_transcript_does_not_call_model(self):
        llm = FixedLLM(issue_payload())
        result = await UnresolvedAgent(llm).process(create_initial_state("empty"))
        self.assertEqual(result["unresolved"].issues, [])
        self.assertEqual(llm.calls, 0)

    async def test_missing_structured_transcript_is_a_failure(self):
        state = create_initial_state("missing-segments")
        state["transcript_text"] = "Text without source segments"
        result = await UnresolvedAgent(FixedLLM(issue_payload())).process(state)
        self.assertIsNone(result["unresolved"])
        self.assertTrue(result["errors"])

    async def test_invalid_quotes_indexes_or_categories_fail_explicitly(self):
        variants = []
        for index in (-1, 200, True, "8"):
            raw = issue_payload()
            raw["issues"][0]["evidence"][0]["segment_index"] = index
            variants.append(raw)
        raw = issue_payload()
        raw["issues"][0]["evidence"][0]["quote"] = "这句话没有在会议中说过"
        variants.append(raw)
        raw = issue_payload()
        raw["issues"][0]["evidence"] = []
        variants.append(raw)
        raw = issue_payload()
        raw["issues"][0]["category"] = "sentiment"
        variants.append(raw)
        variants.extend([{}, {"issues": "not-a-list"}])
        for raw in variants:
            with self.subTest(raw=raw):
                result = await UnresolvedAgent(FixedLLM(raw)).process(demo_state())
                self.assertIsNone(result["unresolved"])
                self.assertEqual(len(result["errors"]), 1)
                self.assertIn("分析失败", FollowUpAgent._format_unresolved_markdown(result["unresolved"]))

    async def test_issue_ids_are_stable_within_meeting_and_isolated_between_meetings(self):
        agent = UnresolvedAgent(FixedLLM(issue_payload()))
        first = (await agent.process(demo_state("one")))["unresolved"].issues[0].issue_id
        again = (await agent.process(demo_state("one")))["unresolved"].issues[0].issue_id
        other = (await agent.process(demo_state("two")))["unresolved"].issues[0].issue_id
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    async def test_serialization_and_vector_chunks_preserve_evidence(self):
        state = demo_state()
        state.update(await UnresolvedAgent(FixedLLM(issue_payload())).process(state))
        saved = serialize_state(state)
        restored = restore_state(saved)
        self.assertEqual(restored["unresolved"], state["unresolved"])
        self.assertNotIn("insights", saved)
        chunks = [c for c in MeetingVectorStore.chunks(saved) if c["kind"] == "unresolved"]
        self.assertTrue(chunks)
        self.assertEqual((chunks[0]["start"], chunks[0]["end"]), (61.5, 70.2))
        self.assertIn("服务器供应商", chunks[0]["text"])


class UnresolvedApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_result_report_reanalysis_and_user_isolation(self):
        app = create_app(repository=repository(), pipeline=pipeline(), vectors=DisabledVectors())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                original = await client.post("/api/v1/meeting/unresolved-api/demo?user_id=alice")
                self.assertEqual(original.status_code, 200)
                body = original.json()
                self.assertEqual(body["status"], "completed")
                self.assertNotIn("insights", body)
                self.assertIn("/api/v1/meeting/{meeting_id}/unresolved", app.openapi()["paths"])
                self.assertNotIn("/api/v1/meeting/{meeting_id}/insights", app.openapi()["paths"])
                response = await client.get("/api/v1/meeting/unresolved-api/unresolved?user_id=alice")
                self.assertEqual(response.json(), body["unresolved"])
                report = await client.get(body["followup"]["report_url"])
                self.assertIn("## 未决事项", report.text)
                self.assertIn("服务器供应商还没有定下来", report.text)
                self.assertIn("61.5s", report.text)
                self.assertNotIn("会议洞察", report.text)
                denied = await client.get("/api/v1/meeting/unresolved-api/unresolved?user_id=bob")
                self.assertEqual(denied.status_code, 404)
                analyzed = await client.post("/api/v1/meeting/unresolved-api/analyze?user_id=alice")
                self.assertEqual(analyzed.json()["unresolved"], body["unresolved"])

    async def test_unresolved_failure_does_not_lose_summary_or_actions(self):
        class BranchFailure:
            async def chat_json(self, messages, **kwargs):
                if '"issues"' in messages[-1]["content"]:
                    raise RuntimeError("unresolved-unavailable")
                return model_result(messages[-1]["content"])
        app = create_app(repository=repository(), pipeline=pipeline(BranchFailure()), vectors=DisabledVectors())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                body = (await client.post("/api/v1/meeting/partial-unresolved/demo")).json()
                self.assertEqual(body["status"], "partial")
                self.assertTrue(body["summary"])
                self.assertTrue(body["actions"]["action_items"])
                self.assertIsNone(body["unresolved"])
                self.assertEqual(len(body["errors"]), 1)
                report = await client.get(body["followup"]["report_url"])
                self.assertIn("未决事项分析失败", report.text)
                self.assertEqual((await client.get("/api/v1/meeting/partial-unresolved/unresolved")).status_code, 409)

    async def test_old_meeting_reanalysis_generates_new_result_without_schema_change(self):
        repo = repository()
        app = create_app(repository=repo, pipeline=pipeline(), vectors=DisabledVectors())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                await client.post("/api/v1/meeting/legacy/demo")
                with repo.sessions.begin() as session:
                    row = session.get(MeetingRecord, "legacy")
                    old = copy.deepcopy(row.result)
                    old.pop("unresolved")
                    old["insights"] = {"meeting_id": "legacy", "efficiency_score": 8.0}
                    row.result = old
                response = await client.get("/api/v1/meeting/legacy/unresolved")
                self.assertEqual(response.status_code, 409)
                stored = repo.get("legacy")
                self.assertNotIn("insights", stored["result"])
                self.assertIsNone(restore_state(stored["result"])["unresolved"])
                analyzed = (await client.post("/api/v1/meeting/legacy/analyze")).json()
                self.assertEqual(analyzed["status"], "completed")
                self.assertTrue(analyzed["unresolved"]["issues"])

    async def test_feishu_card_uses_unresolved_section(self):
        sender = FeishuClient(webhook_url="https://example.test/hook")
        await sender._client.aclose()
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"code": 0})
        sender._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            self.assertTrue(await sender.send_meeting_summary("Test", "Summary", "Tasks", "Open questions"))
            text = json.dumps(requests[0], ensure_ascii=False)
            self.assertIn("未决事项", text)
            self.assertNotIn("会议洞察", text)
        finally:
            await sender.close()


class UnresolvedWebSocketTests(unittest.TestCase):
    def test_demo_emits_new_unresolved_event(self):
        app = create_app(repository=repository(), pipeline=pipeline(), vectors=DisabledVectors())
        with TestClient(app) as client:
            with client.websocket_connect("/ws/meeting/ws-unresolved?user_id=alice") as websocket:
                self.assertEqual(websocket.receive_json()["type"], "connected")
                websocket.send_json({"type": "demo"})
                events = []
                while True:
                    event = websocket.receive_json()
                    events.append(event)
                    if event["type"] in {"completed", "failed", "error"}:
                        break
                self.assertEqual(events[-1]["type"], "completed")
                kinds = [event["type"] for event in events]
                self.assertIn("unresolved", kinds)
                self.assertNotIn("insights", kinds)
                unresolved = next(event["data"] for event in events if event["type"] == "unresolved")
                self.assertTrue(unresolved["issues"])
