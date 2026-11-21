"""离线验证真实图和HTTP入口；不调用外部模型、任务平台或语音模型。"""

import asyncio
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from pydantic import ValidationError

# 离线测试不上传运行轨迹；调用方即使设置过追踪变量，也不影响本组测试。
_tracing_patch = patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"})


def setUpModule():
    _tracing_patch.start()


def tearDownModule():
    _tracing_patch.stop()

from src.agents.action_agent import ActionAgent
from src.agents.followup_agent import FollowUpAgent
from src.agents.unresolved_agent import UnresolvedAgent
from src.agents.summary_agent import SummaryAgent
from src.agents.transcription_agent import TranscriptionAgent
from src.config import load_environment
from src.graph.meeting_graph import compile_meeting_graph, run_meeting_pipeline
from src.integrations.deepseek_client import LLMClient
from src.models.schemas import MeetingStatus, TranscriptSegment, create_initial_state


class DisabledIntegration:
    """显式关闭同步，避免开发者本地.env意外触发真实外部写入。"""
    is_enabled = False


def model_result(prompt):
    if '"action_items"' in prompt:
        return {"action_items": [{"assignee": "SPEAKER_01", "task": "Prepare budget", "priority": "high"}]}
    if '"issues"' in prompt:
        return {"issues": [{"topic": "服务器供应商选择", "description": "供应商仍未确定",
                            "category": "decision", "priority": "medium",
                            "evidence": [{"segment_index": 8, "quote": "服务器供应商还没有定下来"}]}]}
    return {"title": "Budget meeting", "participants": ["SPEAKER_00", "SPEAKER_01"],
            "speaker_summaries": [{"speaker_id": "SPEAKER_00", "points": ["Review budget"]}],
            "topics": [], "decisions": ["Review budget"]}


class ParallelLLM:
    """三个分支必须同时到达，才能继续，用于验证图确实并行执行。"""
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0
        self.all_started = asyncio.Event()

    async def chat_json(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 3:
            self.all_started.set()
        await asyncio.wait_for(self.all_started.wait(), timeout=3)
        if self.fail:
            raise RuntimeError("offline-model-failure")
        return model_result(messages[-1]["content"])


class ModelsAndConfigTests(unittest.TestCase):
    def test_initial_states_have_independent_lists(self):
        first = create_initial_state("one")
        second = create_initial_state("two")
        first["errors"].append("first-error")
        self.assertEqual(second["errors"], [])
        self.assertEqual(first["status"], MeetingStatus.CREATED)

    def test_transcript_rejects_reversed_timestamps(self):
        with self.assertRaises(ValidationError):
            TranscriptSegment(text="hello", start=5, end=2)

    def test_config_precedence_and_blank_key_fallback(self):
        # 用临时文件验证配置优先级，不读取或输出开发者的真实密钥。
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            project = workspace / "meeting"
            project.mkdir()
            (workspace / "pyproject.toml").write_text("", encoding="utf-8")
            (workspace / ".env").write_text("DEEPSEEK_API_KEY=workspace-test\nDEEPSEEK_MODEL=workspace-model\nLANGSMITH_TRACING=true\n", encoding="utf-8")
            (project / ".env").write_text("DEEPSEEK_API_KEY=\nDEEPSEEK_MODEL=project-model\nDASHSCOPE_API_KEY=project-embedding-key\n", encoding="utf-8")
            with patch("src.config.PROJECT_DIR", project), patch.dict(os.environ, {}, clear=True):
                load_environment()
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "workspace-test")
                self.assertEqual(os.environ["DEEPSEEK_MODEL"], "project-model")
                self.assertNotIn("LANGSMITH_TRACING", os.environ)
            # 模拟SDK先把工作区旧值加载到os.environ：项目配置必须覆盖它。
            with patch("src.config.PROJECT_DIR", project), patch.dict(os.environ, {
                "DEEPSEEK_MODEL": "system-model", "DASHSCOPE_API_KEY": "preloaded-workspace-key",
            }, clear=True):
                load_environment()
                self.assertEqual(os.environ["DEEPSEEK_MODEL"], "project-model")
                self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "project-embedding-key")


class LLMClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_url_model_thinking_and_json(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]})

        async with LLMClient(api_key="test-key", model="test-model", base_url="https://mock.invalid/v1/", transport=httpx.MockTransport(handler)) as client:
            result = await client.chat_json([{"role": "user", "content": "Return JSON"}])
        self.assertEqual(result, {"ok": True})
        self.assertEqual(str(requests[0].url), "https://mock.invalid/v1/chat/completions")
        payload = json.loads(requests[0].content)
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertTrue(client._client.is_closed)

    async def test_auth_error_is_not_retried(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(401, json={"error": "test-auth-failure"})

        async with LLMClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(httpx.HTTPStatusError):
                await client.chat_json([{"role": "user", "content": "JSON"}])
        self.assertEqual(len(calls), 1)

    async def test_blank_and_truncated_responses_are_rejected(self):
        for content, finish in [("  ", "stop"), ('{"x":', "length")]:
            with self.subTest(finish=finish):
                def handler(request):
                    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]})
                async with LLMClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
                    with self.assertRaises(ValueError):
                        await client.chat_json([{"role": "user", "content": "JSON"}])


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def graph(self, llm):
        return compile_meeting_graph(llm_client=llm, jira_client=DisabledIntegration(), feishu_client=DisabledIntegration())

    async def test_parallel_success_joins_once_with_all_results(self):
        seen = []

        class InspectFollowup(FollowUpAgent):
            async def process(self, state):
                seen.append((state["summary"], state["actions"], state["unresolved"]))
                return await super().process(state)

        llm = ParallelLLM()
        with patch("src.graph.meeting_graph.FollowUpAgent", InspectFollowup):
            result = await self.graph(llm).ainvoke(create_initial_state("parallel-success", mode="demo"))
        self.assertEqual(llm.calls, 3)
        self.assertEqual(len(seen), 1)
        self.assertTrue(all(item is not None for item in seen[0]))
        self.assertEqual(result["summary"].title, "Budget meeting")
        self.assertEqual(len(result["actions"].action_items), 1)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["status"], MeetingStatus.COMPLETED)

    async def test_parallel_failures_merge_without_repeating_previous_errors(self):
        state = create_initial_state("parallel-failure", mode="demo")
        state["errors"] = ["previous-stage-error"]
        result = await self.graph(ParallelLLM(fail=True)).ainvoke(state)
        self.assertEqual(len(result["errors"]), 4)
        self.assertEqual(result["errors"].count("previous-stage-error"), 1)
        for name in ["SummaryAgent", "ActionAgent", "UnresolvedAgent"]:
            self.assertEqual(sum(error.startswith(name + ":") for error in result["errors"]), 1)
        self.assertIsNotNone(result["followup"])

    async def test_nodes_return_only_owned_fields_without_mutating_input(self):
        class LocalLLM:
            async def chat_json(self, messages, **kwargs):
                return model_result(messages[-1]["content"])

        state = create_initial_state("patch-check", mode="demo")
        transcribed = await TranscriptionAgent().process(state)
        self.assertIsNone(state["transcript"])
        state.update(transcribed)
        state["errors"] = ["previous"]
        before = copy.deepcopy(state)
        nodes = [
            (SummaryAgent(LocalLLM()), "summary"),
            (ActionAgent(LocalLLM(), DisabledIntegration(), DisabledIntegration()), "actions"),
            (UnresolvedAgent(LocalLLM()), "unresolved"),
        ]
        for node, field in nodes:
            update = await node.process(state)
            self.assertEqual(set(update), {field, "errors"})
            self.assertEqual(update["errors"], [])
            self.assertEqual(state, before)

    async def test_owned_clients_closed_when_pipeline_finishes(self):
        def handler(request):
            prompt = json.loads(request.content)["messages"][-1]["content"]
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(model_result(prompt))}}]})

        client = LLMClient(api_key="test-key", transport=httpx.MockTransport(handler))
        with patch("src.graph.meeting_graph.LLMClient", return_value=client):
            result = await run_meeting_pipeline("owned-client", mode="demo", jira_client=DisabledIntegration(), feishu_client=DisabledIntegration())
        self.assertEqual(result["errors"], [])
        self.assertTrue(client._client.is_closed)

    async def test_http_demo_serializes_complete_pipeline_result(self):
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
        from src.graph.meeting_graph import MeetingPipeline
        from src.storage import MeetingRepository
        from src.websocket.server import create_app

        class DisabledVectors:
            enabled = False
            def close(self):
                pass

        repo = MeetingRepository(create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool))
        worker = MeetingPipeline(llm_client=ParallelLLM(), jira_client=DisabledIntegration(), feishu_client=DisabledIntegration())
        app = create_app(repository=repo, pipeline=worker, vectors=DisabledVectors())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/api/v1/meeting/http-check/demo")
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(body["status"], "completed")
                self.assertEqual(body["errors"], [])
                for field in ["transcript", "summary", "actions", "unresolved", "followup"]:
                    self.assertIn(field, body)
                stored = await client.get("/api/v1/meeting/http-check/summary")
                self.assertEqual(stored.json()["title"], "Budget meeting")


if __name__ == "__main__":
    unittest.main()
