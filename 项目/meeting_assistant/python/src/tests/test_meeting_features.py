"""验证真实/演示边界、共享资源隔离以及会议持久化。"""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from src.agents.transcription_agent import TranscriptionAgent, TranscriptionConfig
from src.agents.summary_agent import SummaryAgent
from src.agents.action_agent import ActionAgent
from src.graph.meeting_graph import MeetingPipeline, compile_meeting_graph
from src.models.schemas import MeetingStatus, TranscriptResult, TranscriptSegment, create_initial_state
from src.storage import MeetingRepository, StorageError, serialize_state
from src.websocket.server import create_app
from test_pipeline import DisabledIntegration, ParallelLLM, model_result


_tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"})


def setUpModule():
    _tracing.start()


def tearDownModule():
    _tracing.stop()


class LocalLLM:
    async def chat_json(self, messages, **kwargs):
        return model_result(messages[-1]["content"])


class DisabledVectors:
    enabled = False
    def close(self):
        pass


class FailingVectors(DisabledVectors):
    enabled = True
    def index(self, record):
        raise RuntimeError("offline-index-failure")


def pipeline(llm=None, transcriber=None):
    return MeetingPipeline(llm_client=llm or LocalLLM(), jira_client=DisabledIntegration(),
                           feishu_client=DisabledIntegration(), transcription_agent=transcriber)


def repository(url="sqlite://"):
    kwargs = {"connect_args": {"check_same_thread": False}}
    if url == "sqlite://":
        kwargs["poolclass"] = StaticPool
    repo = MeetingRepository(create_engine(url, **kwargs))
    repo.initialize()
    return repo


class RealModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_labels_are_actual_speaker_ids(self):
        class NamedLLM:
            async def chat_json(self, *args, **kwargs):
                return {"title": "Test", "participants": ["Alice"],
                        "topics": [{"title": "Budget", "participants": ["Alice", "SPEAKER_00", "SPEAKER_99"]}],
                        "speaker_summaries": [{"speaker_id": "Alice", "points": ["Wrong identity"]},
                                              {"speaker_id": "SPEAKER_00", "points": ["Review budget"]}]}
        transcript = TranscriptionAgent._generate_demo_transcript("id-summary")
        state = create_initial_state("id-summary", mode="demo", transcript=transcript)
        state["transcript_text"] = TranscriptionAgent._format_transcript_text(transcript)
        result = await SummaryAgent(NamedLLM()).process(state)
        summary = result["summary"]
        self.assertEqual(summary.participants, transcript.speaker_ids)
        self.assertEqual(summary.topics[0].participants, ["SPEAKER_00"])
        self.assertEqual([group.speaker_id for group in summary.speaker_summaries], transcript.speaker_ids)
        self.assertEqual(summary.speaker_summaries[0].points, ["Review budget"])

    async def test_action_preserves_explicit_names_without_speaker_mapping(self):
        class NamedLLM:
            async def chat_json(self, *args, **kwargs):
                return {"action_items": [
                    {"assignee": "Alice", "task": "Prepare budget"},
                    {"assignee": "SPEAKER_01", "task": "Prepare report"},
                    {"assignee": "SPEAKER_99", "task": "Review result"},
                    {"assignee": "Bob", "task": "Review budget"},
                    {"task": "Arrange next meeting"},
                    {"assignee": "我", "task": "Prepare slides"},
                ]}
        agent = ActionAgent(NamedLLM(), DisabledIntegration(), DisabledIntegration())
        items = await agent._extract_actions("SPEAKER_00: Alice负责预算。SPEAKER_01: 我来写报告。", ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual([item.assignee for item in items],
                         ["Alice", "SPEAKER_01", "未明确", "未明确", "未明确", "未明确"])
        self.assertEqual(items[0].task, "Prepare budget")
        self.assertNotIn("尚不能关联", items[0].context)

    async def test_no_audio_fails_in_real_mode_but_demo_is_explicit(self):
        worker = pipeline()
        real = await worker.run("real-empty")
        demo = await worker.run("explicit-demo", mode="demo")
        self.assertEqual(real["status"], MeetingStatus.FAILED)
        self.assertIsNone(real["transcript"])
        self.assertIsNone(real["summary"])
        self.assertTrue(real["errors"])
        self.assertEqual(demo["status"], MeetingStatus.COMPLETED)
        self.assertTrue(demo["transcript"].segments)

    async def test_backend_failure_never_reaches_llm(self):
        class MustNotCall:
            async def chat_json(self, *args, **kwargs):
                raise AssertionError("LLM called after transcription failed")
        transcriber = TranscriptionAgent()
        with patch.object(transcriber, "_lazy_init", side_effect=RuntimeError("asr-unavailable")):
            result = await pipeline(MustNotCall(), transcriber).run("failed-asr", b"fake-audio")
        self.assertEqual(result["status"], MeetingStatus.FAILED)
        self.assertIsNone(result["followup"])
        self.assertIn("asr-unavailable", result["errors"][0])

    async def test_shared_speech_model_loads_once_and_keeps_meetings_separate(self):
        counts = {"loads": 0, "align_loads": 0, "active": 0, "max_active": 0}
        guard = threading.Lock()

        class Model:
            def transcribe(self, audio, **kwargs):
                with guard:
                    counts["active"] += 1
                    counts["max_active"] = max(counts["max_active"], counts["active"])
                time.sleep(0.02)
                with guard:
                    counts["active"] -= 1
                return {"segments": [{"text": audio, "start": 0, "end": 1}]}

        def load_model(*args, **kwargs):
            counts["loads"] += 1
            return Model()

        def load_align(**kwargs):
            counts["align_loads"] += 1
            return object(), {}

        whisper = SimpleNamespace(load_model=load_model, load_align_model=load_align,
            load_audio=lambda path: Path(path).read_bytes().decode(),
            align=lambda segments, *args: {"segments": segments},
            assign_word_speakers=lambda speakers, result: {"segments": [{**seg, "speaker": "SPEAKER_00"} for seg in result["segments"]]})
        config = TranscriptionConfig(device="cpu")
        config.diarization_enabled, config.hf_token = True, "offline-token"
        transcriber = TranscriptionAgent(config)
        transcriber._diarize_pipeline = lambda audio: []
        first = create_initial_state("meeting-a", b"audio-A")
        second = create_initial_state("meeting-b", b"audio-B")
        with patch.dict("sys.modules", {"whisperx": whisper}):
            a, b = await asyncio.gather(transcriber.process(first), transcriber.process(second))
        self.assertEqual(counts["loads"], 1)
        self.assertEqual(counts["align_loads"], 1)
        self.assertEqual(counts["max_active"], 1)
        self.assertEqual(a["transcript"].meeting_id, "meeting-a")
        self.assertEqual(b["transcript"].meeting_id, "meeting-b")
        self.assertEqual(a["transcript"].segments[0].speaker, "SPEAKER_00")
        self.assertEqual(b["transcript"].segments[0].speaker, "SPEAKER_00")
        self.assertEqual(a["transcript"].segments[0].text, "audio-A")
        self.assertEqual(b["transcript"].segments[0].text, "audio-B")
        self.assertEqual(a["transcript"].segments[0].speaker_id, "SPEAKER_00")

    async def test_one_compiled_graph_handles_concurrent_meetings(self):
        with patch("src.graph.meeting_graph.compile_meeting_graph", wraps=compile_meeting_graph) as build:
            worker = pipeline(ParallelLLM())
            a, b = await asyncio.gather(worker.run("parallel-a", mode="demo"), worker.run("parallel-b", mode="demo"))
        self.assertEqual(build.call_count, 1)
        self.assertEqual(a["transcript"].meeting_id, "parallel-a")
        self.assertEqual(b["transcript"].meeting_id, "parallel-b")
        a["transcript"].segments[0].speaker = "Changed"
        self.assertNotEqual(b["transcript"].segments[0].speaker, "Changed")


class RepositoryTests(unittest.TestCase):
    def test_claim_serializes_same_meeting_and_enforces_owner(self):
        repo = repository()
        repo.create("one", "alice")
        reservation = repo.claim("one", "alice", "real")
        with self.assertRaises(StorageError):
            repo.claim("one", "alice", "real")
        with self.assertRaises(StorageError):
            repo.get("one", "bob")
        failed = create_initial_state("one")
        failed.update(status=MeetingStatus.FAILED, errors=["test-error"])
        repo.finish("one", "alice", reservation["run_token"], serialize_state(failed))
        self.assertEqual(repo.get("one", "alice")["status"], "failed")
        repo.close()

    def test_saved_results_survive_engine_recreation(self):
        with tempfile.TemporaryDirectory() as temp:
            url = "sqlite:///" + str(Path(temp) / "meetings.db")
            repo = repository(url)
            repo.create("persist", "alice")
            reserved = repo.claim("persist", "alice", "demo")
            state = create_initial_state("persist", mode="demo")
            state.update(status=MeetingStatus.TRANSCRIBED,
                         transcript=TranscriptionAgent._generate_demo_transcript("persist"))
            repo.finish("persist", "alice", reserved["run_token"], serialize_state(state))
            repo.close()
            reopened = repository(url)
            result = reopened.get("persist", "alice")["result"]
            self.assertEqual(result["transcript"]["meeting_id"], "persist")
            self.assertTrue(result["transcript"]["segments"])
            reopened.close()


class ApiPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_id_only_report_and_reanalysis_reuse_transcript(self):
        repo = repository()
        worker = pipeline()
        app = create_app(repository=repo, pipeline=worker, vectors=DisabledVectors())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                response = await client.post("/api/v1/meeting/id-demo/demo?user_id=alice")
                self.assertEqual(response.status_code, 200)
                old = response.json()
                speaker_ids = old["transcript"]["speaker_ids"]
                self.assertTrue(all(segment["speaker"] == segment["speaker_id"] for segment in old["transcript"]["segments"]))
                self.assertEqual(old["summary"]["participants"], speaker_ids)
                self.assertEqual([group["speaker_id"] for group in old["summary"]["speaker_summaries"]], speaker_ids)
                self.assertNotIn("speaker_map", old)
                self.assertNotIn("/api/v1/meeting/{meeting_id}/speakers", app.openapi()["paths"])
                # 不允许重新生成演示或再次执行ASR，必须复用保存的转写。
                with patch.object(worker.transcriber, "_generate_demo_transcript", side_effect=AssertionError("must reuse stored transcript")):
                    analyzed = await client.post("/api/v1/meeting/id-demo/analyze?user_id=alice")
                self.assertEqual(analyzed.json()["status"], "completed")
                self.assertEqual(analyzed.json()["transcript"]["speaker_ids"], speaker_ids)
                report = await client.get(analyzed.json()["followup"]["report_url"])
                self.assertEqual(report.status_code, 200)
                self.assertIn("text/markdown", report.headers["content-type"])
                self.assertGreater(len(report.text), 100)
                self.assertIn("SPEAKER_00", report.text)
                other = await client.get("/api/v1/meeting/id-demo/report?user_id=bob")
                self.assertEqual(other.status_code, 404)

    async def test_index_failure_preserves_mysql_result(self):
        repo = repository()
        app = create_app(repository=repo, pipeline=pipeline(), vectors=FailingVectors())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                response = await client.post("/api/v1/meeting/index-failed/demo")
                self.assertEqual(response.json()["status"], "completed")
                self.assertEqual(response.json()["index_status"], "failed")
                stored = await client.get("/api/v1/meeting/index-failed/summary")
                self.assertEqual(stored.status_code, 200)
                self.assertEqual(stored.json()["title"], "Budget meeting")

    async def test_empty_real_upload_is_persisted_as_failure(self):
        app = create_app(repository=repository(), pipeline=pipeline(), vectors=DisabledVectors())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                response = await client.post("/api/v1/meeting/real-empty/upload", files={"file": ("empty.wav", b"", "audio/wav")})
                self.assertEqual(response.json()["mode"], "real")
                self.assertEqual(response.json()["status"], "failed")
                self.assertIsNone(response.json()["transcript"])
                saved = await client.get("/api/v1/meeting/real-empty/report")
                self.assertEqual(saved.json()["status"], "failed")

    async def test_real_upload_saves_audio_even_when_asr_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            worker, repo = pipeline(), repository()
            app = create_app(repository=repo, pipeline=worker, vectors=DisabledVectors(), data_dir=temp)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                    with patch.object(worker.transcriber, "_lazy_init", side_effect=RuntimeError("asr-unavailable")):
                        response = await client.post("/api/v1/meeting/audio-saved/upload", files={"file": ("test.wav", b"saved-audio", "audio/wav")})
                    self.assertEqual(response.json()["status"], "failed")
                    record = repo.get("audio-saved")
                    self.assertEqual(Path(record["audio_path"]).read_bytes(), b"saved-audio")

    async def test_search_excludes_vectors_from_previous_revision(self):
        class Vectors(DisabledVectors):
            enabled = True
            def __init__(self):
                self.rows = []
            def index(self, record):
                self.rows.append({"meeting_id": record["meeting_id"], "user_id": record["user_id"],
                                  "revision": record["revision"], "text": "test-hit", "score": 1.0})
            def search(self, *args):
                return self.rows

        app = create_app(repository=repository(), pipeline=pipeline(), vectors=Vectors())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                first = (await client.post("/api/v1/meeting/revision-demo/demo")).json()
                hidden = (await client.post("/api/v1/meetings/search", json={"query": "test"})).json()
                self.assertEqual(hidden["matches"], [])  # 默认只检索真实会议。
                initial = (await client.post("/api/v1/meetings/search", json={"query": "test", "mode": "demo"})).json()
                self.assertEqual(len(initial["matches"]), 1)
                await asyncio.to_thread(app.state.repository.mark_index, "revision-demo", "anonymous", first["revision"], "pending")
                pending = (await client.post("/api/v1/meetings/search", json={"query": "test", "mode": "demo"})).json()
                self.assertEqual(pending["matches"], [])
                latest = (await client.post("/api/v1/meeting/revision-demo/analyze")).json()
                indexed = (await client.post("/api/v1/meetings/search", json={"query": "test", "mode": "demo"})).json()
                self.assertEqual(len(indexed["matches"]), 1)
                self.assertEqual(indexed["matches"][0]["revision"], latest["revision"])


if __name__ == "__main__":
    unittest.main()
