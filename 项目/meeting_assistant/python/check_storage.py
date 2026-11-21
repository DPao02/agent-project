"""连接现有MySQL/Milvus验收；默认使用测试向量，不调用云模型。

--live-embedding 改用百炼真实Embedding（会产生API调用费用），不调用DeepSeek。
默认测试集合与正式集合隔离；所有验收会议归属__storage_check__用户。
"""
import argparse
import asyncio
import hashlib
import os
import uuid

import httpx

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from src.config import load_environment
from src.graph.meeting_graph import MeetingPipeline
from src.integrations.milvus_client import MeetingVectorStore
from src.storage import MeetingRepository
from src.websocket.server import create_app


class DisabledIntegration:
    is_enabled = False


class CheckLLM:
    async def chat_json(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if '"action_items"' in prompt:
            return {"action_items": [{"assignee": "SPEAKER_01", "task": "整理预算方案", "priority": "high"}]}
        if '"issues"' in prompt:
            return {"issues": [{"topic": "服务器供应商选择", "description": "供应商仍未确定",
                                "category": "decision", "priority": "medium",
                                "evidence": [{"segment_index": 8, "quote": "服务器供应商还没有定下来"}]}]}
        return {"title": "CHECK-预算评审会议", "participants": ["SPEAKER_00", "SPEAKER_01"],
                "speaker_summaries": [{"speaker_id": "SPEAKER_00", "points": ["审阅预算方案"]}],
                "decisions": ["整理预算方案"]}


async def main(live):
    load_environment()
    dimension = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

    def test_vectors(texts):
        # 只用于连通和数据隔离验收，不用于衡量语义检索效果。
        return [[(hashlib.sha256((text + str(i)).encode()).digest()[0] + 1) / 256
                 for i in range(dimension)] for text in texts]

    vectors = MeetingVectorStore(embedder=None if live else test_vectors,
        collection=None if live else f"meeting_assistant_storage_check_{dimension}")
    repo = MeetingRepository.from_env()
    worker = MeetingPipeline(llm_client=CheckLLM(), jira_client=DisabledIntegration(), feishu_client=DisabledIntegration())
    meeting_id, owner = "CHECK-" + uuid.uuid4().hex[:12], "__storage_check__"
    app = create_app(repository=repo, pipeline=worker, vectors=vectors)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://check") as client:
            response = await client.post(f"/api/v1/meeting/{meeting_id}/demo", params={"user_id": owner})
            response.raise_for_status()
            result = response.json()
            assert result["status"] == "completed", result["errors"]
            assert result["index_status"] == "indexed", result["index_error"]
            assert result["unresolved"]["issues"], "Unresolved issues were not saved"
            unresolved = await client.get(f"/api/v1/meeting/{meeting_id}/unresolved", params={"user_id": owner})
            unresolved.raise_for_status()
            assert unresolved.json()["issues"][0]["evidence"][0]["start"] == 61.5
            report = await client.get(result["followup"]["report_url"])
            assert report.status_code == 200 and len(report.text) > 100
            search = await client.post("/api/v1/meetings/search", json={"query": "谁负责整理预算方案", "user_id": owner, "meeting_id": meeting_id, "mode": "demo"})
            search.raise_for_status()
            assert search.json()["matches"], "No matches returned"
            assert all(segment["speaker"] == segment["speaker_id"] for segment in result["transcript"]["segments"])
            analyzed = await client.post(f"/api/v1/meeting/{meeting_id}/analyze", params={"user_id": owner})
            analyzed.raise_for_status()
            latest = analyzed.json()
            assert latest["index_status"] == "indexed", latest["index_error"]
            search = await client.post("/api/v1/meetings/search", json={"query": "预算方案", "user_id": owner, "meeting_id": meeting_id, "mode": "demo"})
            search.raise_for_status()
            assert search.json()["matches"]
            assert all(hit["revision"] == latest["revision"] for hit in search.json()["matches"])
            hits = vectors._client.query(vectors.collection,
                filter=f'meeting_id == "{meeting_id}" and kind == "unresolved"',
                output_fields=["kind", "start", "end", "revision"])
            assert hits and all(hit["start"] == 61.5 and hit["revision"] == latest["revision"] for hit in hits)
            print("Meeting:", meeting_id)
            print("MySQL saved; report downloaded; Milvus indexed and searched.")
            print("Speaker IDs preserved; unresolved evidence saved/indexed; only latest revision searched.")
            print("Collection:", vectors.collection)
            print("Embedding:", "live Bailian" if live else "local test vectors")
    # 关闭原连接池，创建新连接，验证查询不依赖进程内meeting_results字典。
    reopened = MeetingRepository.from_env()
    try:
        saved = await asyncio.to_thread(reopened.get, meeting_id, owner)
        assert saved["result"]["summary"]["title"] == "CHECK-预算评审会议"
        assert saved["result"]["unresolved"]["issues"][0]["evidence"][0]["start"] == 61.5
        print("MySQL reopened and result restored successfully.")
    finally:
        reopened.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-embedding", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.live_embedding))
