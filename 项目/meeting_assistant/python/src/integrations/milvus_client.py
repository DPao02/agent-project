"""Milvus保存可重建的会议语义索引；完整结果和当前版本以MySQL为准。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading

from openai import OpenAI
from pymilvus import DataType, MilvusClient

from ..config import load_environment


class MeetingVectorStore:
    def __init__(self, *, client=None, embedder=None, dimension=None, collection=None):
        load_environment()
        self.enabled = os.getenv("MILVUS_ENABLED", "true").lower() == "true"
        self.dimension = dimension or int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
        self.collection = collection or os.getenv("MILVUS_COLLECTION", "meeting_assistant_v4_1024_v1")
        self._client, self._embedder = client, embedder
        self._embedding_client = None
        self._lock = threading.RLock()
        self._ready = False

    def _connect(self):
        if self._ready:
            return
        if self._client is None:
            self._client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"),
                                        token=os.getenv("MILVUS_TOKEN", ""), timeout=30)
        if not self._client.has_collection(self.collection):
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
            for name, length in [("id", 64), ("meeting_id", 64), ("user_id", 64),
                                 ("kind", 32), ("text", 65535), ("embedding_model", 128)]:
                schema.add_field(name, DataType.VARCHAR, max_length=length, is_primary=name == "id")
            schema.add_field("revision", DataType.INT64)
            schema.add_field("start", DataType.FLOAT)
            schema.add_field("end", DataType.FLOAT)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dimension)
            indexes = MilvusClient.prepare_index_params()
            indexes.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
            self._client.create_collection(self.collection, schema=schema, index_params=indexes,
                                           consistency_level="Strong")
        description = self._client.describe_collection(self.collection)
        fields = {field["name"]: field for field in description["fields"]}
        if int(fields.get("vector", {}).get("params", {}).get("dim", 0)) != self.dimension:
            raise ValueError("会议Milvus集合维度不匹配，请配置独立集合")
        self._ready = True

    def _embed(self, texts):
        if self._embedder is not None:
            vectors = self._embedder(texts)
        else:
            if self._embedding_client is None:
                key = os.getenv("DASHSCOPE_API_KEY", "")
                if not key:
                    raise ValueError("会议语义索引需要DASHSCOPE_API_KEY")
                self._embedding_client = OpenAI(api_key=key, timeout=30, max_retries=2,
                    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
            vectors = []
            for offset in range(0, len(texts), 10):
                response = self._embedding_client.embeddings.create(model=self.model, input=texts[offset:offset + 10],
                                                                     dimensions=self.dimension, encoding_format="float")
                vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        if len(vectors) != len(texts):
            raise ValueError("Embedding返回条数不匹配")
        normalized = []
        for vector in vectors:
            if len(vector) != self.dimension or not all(math.isfinite(x) for x in vector):
                raise ValueError("Embedding返回无效维度或数值")
            norm = math.sqrt(sum(x * x for x in vector))
            if norm == 0:
                raise ValueError("Embedding返回零向量")
            normalized.append([x / norm for x in vector])
        return normalized

    @staticmethod
    def chunks(result):
        # 按转写顺序拼接到约1000字，保留可回到原音频的起止时间。
        chunks, lines, size, start, end = [], [], 0, 0.0, 0.0
        for segment in (result.get("transcript") or {}).get("segments", []):
            line = f"[{segment['speaker_id']}] {segment['text']}"
            if lines and size + len(line) > 1000:
                chunks.append({"kind": "transcript", "text": "\n".join(lines), "start": start, "end": end})
                lines, size = [], 0
            if not lines:
                start = segment["start"]
            lines.append(line)
            size += len(line)
            end = segment["end"]
        if lines:
            chunks.append({"kind": "transcript", "text": "\n".join(lines), "start": start, "end": end})
        for kind in ("summary", "actions"):
            if result.get(kind):
                text = json.dumps(result[kind], ensure_ascii=False)
                chunks.extend({"kind": kind, "text": text[i:i + 1000], "start": 0.0, "end": 0.0}
                              for i in range(0, len(text), 900))
        # 每项未决问题单独索引，并沿用原文证据时间，检索后能定位到录音片段。
        for issue in (result.get("unresolved") or {}).get("issues", []):
            evidence = issue.get("evidence", [])
            text = json.dumps(issue, ensure_ascii=False)
            chunks.extend({"kind": "unresolved", "text": text[i:i + 1000],
                           "start": min(e["start"] for e in evidence) if evidence else 0.0,
                           "end": max(e["end"] for e in evidence) if evidence else 0.0}
                          for i in range(0, len(text), 900))
        return chunks

    def index(self, record):
        with self._lock:
            self._connect()
            chunks = self.chunks(record["result"])
            if not chunks:
                return 0
            vectors = self._embed([chunk["text"] for chunk in chunks])
            rows = []
            for position, (chunk, vector) in enumerate(zip(chunks, vectors)):
                identity = f"{record['user_id']}:{record['meeting_id']}:{record['revision']}:{self.model}:{position}"
                rows.append({**chunk, "id": hashlib.sha256(identity.encode()).hexdigest(),
                             "meeting_id": record["meeting_id"], "user_id": record["user_id"],
                             "revision": record["revision"], "embedding_model": self.model, "vector": vector})
            for offset in range(0, len(rows), 100):
                self._client.upsert(self.collection, data=rows[offset:offset + 100])
            # 只有本版本全部写入后才清理本会议旧版本，不影响其他会议或客服集合。
            condition = (f"meeting_id == {json.dumps(record['meeting_id'])} and user_id == {json.dumps(record['user_id'])} "
                         f"and revision < {record['revision']}")
            self._client.delete(self.collection, filter=condition)
            return len(rows)

    def search(self, query, user_id, top_k=5, meeting_id=None):
        with self._lock:
            self._connect()
            vector = self._embed([query])[0]
            condition = f"user_id == {json.dumps(user_id)} and embedding_model == {json.dumps(self.model)}"
            if meeting_id:
                condition += f" and meeting_id == {json.dumps(meeting_id)}"
            result = self._client.search(self.collection, data=[vector], anns_field="vector", filter=condition,
                limit=min(top_k * 5, 100), search_params={"metric_type": "COSINE", "params": {}},
                output_fields=["meeting_id", "user_id", "revision", "kind", "text", "start", "end"],
                consistency_level="Strong")
            return [{**hit["entity"], "score": float(hit["distance"])} for hit in result[0]] if result else []

    def close(self):
        if self._client is not None:
            self._client.close()
        if self._embedding_client is not None:
            self._embedding_client.close()
