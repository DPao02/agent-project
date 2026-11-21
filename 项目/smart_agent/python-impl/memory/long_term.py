"""
知识库向量存储：百炼 Embedding + Milvus。

主要职责：
1. 读取 PDF/TXT 并分块；
2. 调用百炼生成真实语义向量；
3. 将向量、正文、来源、页码写入 Milvus；
4. 为 RAG Agent 提供文档检索接口。

本模块不负责生成客服答案；答案仍由 DeepSeek 生成。
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from langchain_openai import OpenAIEmbeddings
from pymilvus import DataType, MilvusClient
from memory.corpus import document_id, knowledge_chunks


class LongTermMemory:
    """向 RAG Agent 提供统一的知识库写入、检索接口。"""

    def __init__(self, embedding_max_retries: int = 2):
        # 聊天模型与 Embedding 使用不同配置，避免混用密钥。
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-v4"
        )
        self.embedding_dim = int(
            os.getenv("EMBEDDING_DIMENSIONS", "1024")
        )
        self.collection_name = os.getenv(
            "MILVUS_COLLECTION", "smart_cs_kb_v4_1024_v1"
        )
        self.timeout = float(os.getenv("MILVUS_TIMEOUT", "30"))

        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("DASHSCOPE_BASE_URL")

        if not api_key or not base_url:
            raise ValueError(
                "请在项目 .env 中配置 "
                "DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL"
            )

        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=api_key,
            base_url=base_url,
            dimensions=self.embedding_dim,
            # 将原始文本发送给百炼，不预先转成 OpenAI token。
            check_embedding_ctx_length=False,
            # text-embedding-v4 单批最多接收 10 条文本。
            chunk_size=10,
            model_kwargs={"encoding_format": "float"},
            request_timeout=30,
            max_retries=embedding_max_retries,
        )

        # 连接已经运行的 Milvus 服务。
        self.client = MilvusClient(
            uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"),
            token=os.getenv("MILVUS_TOKEN", ""),
            timeout=self.timeout,
        )

        try:
            self._ensure_collection()
        except Exception:
            # 初始化失败时释放连接，同时保留原始异常。
            self.client.close()
            raise

    def _ensure_collection(self) -> None:
        """创建或检查集合，确保字段、维度和索引符合要求。"""
        if not self.client.has_collection(
            collection_name=self.collection_name
        ):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )

            # 固定 ID 用于识别同一文档块，避免重复导入。
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=64,
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim,
            )
            schema.add_field(
                field_name="content",
                datatype=DataType.VARCHAR,
                max_length=65535,
            )
            schema.add_field(
                field_name="source",
                datatype=DataType.VARCHAR,
                max_length=2048,
            )
            schema.add_field(
                field_name="metadata",
                datatype=DataType.JSON,
            )
            schema.add_field(
                field_name="embedding_model",
                datatype=DataType.VARCHAR,
                max_length=128,
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                consistency_level="Strong",
                timeout=self.timeout,
            )

        # 集合已经存在时，检查是否误用了其他练习项目的集合。
        description = self.client.describe_collection(
            collection_name=self.collection_name,
            timeout=self.timeout,
        )
        fields = {
            field["name"]: field
            for field in description["fields"]
        }

        required_fields = {
            "id",
            "vector",
            "content",
            "source",
            "metadata",
            "embedding_model",
        }
        if not required_fields.issubset(fields):
            raise ValueError(
                "已有集合字段不匹配，请修改 MILVUS_COLLECTION，"
                "为当前项目使用新集合"
            )

        actual_dim = int(fields["vector"]["params"]["dim"])
        if actual_dim != self.embedding_dim:
            raise ValueError(
                f"集合维度为 {actual_dim}，"
                f"模型配置为 {self.embedding_dim}，"
                "请使用新集合重新导入"
            )

        index_names = self.client.list_indexes(
            collection_name=self.collection_name,
            field_name="vector",
        )

        if not index_names:
            # 初版让 Milvus 自动选择索引，使用余弦相似度。
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_name="vector_index",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params,
                timeout=self.timeout,
            )
        else:
            index_info = self.client.describe_index(
                collection_name=self.collection_name,
                index_name=index_names[0],
                timeout=self.timeout,
            )
            metric = index_info.get("metric_type")
            if metric and metric.upper() != "COSINE":
                raise ValueError(
                    "已有向量索引不是 COSINE，请使用新集合"
                )

        # 集合加载完成后才能进行向量检索。
        self.client.load_collection(
            collection_name=self.collection_name,
            timeout=self.timeout,
        )

    def _embed_text(
        self,
        text: str,
        *,
        is_query: bool,
    ) -> list[float]:
        """生成向量，校验维度和数值，再归一化。"""
        if is_query:
            values = self.embeddings.embed_query(text)
        else:
            values = self.embeddings.embed_documents([text])[0]

        vector = np.asarray(values, dtype=np.float32)

        if vector.shape != (self.embedding_dim,):
            raise ValueError(
                f"模型返回形状为 {vector.shape}，"
                f"预期为 ({self.embedding_dim},)"
            )

        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm == 0:
            raise ValueError("Embedding 返回了无效向量")

        return (vector / norm).tolist()

    def add_document(
        self,
        content: str,
        source: str = "",
        metadata: dict | None = None,
    ) -> str:
        """写入一个文档块；相同文档块已存在时直接复用。"""
        content = content.strip()
        if not content:
            raise ValueError("文档内容不能为空")

        # 模型、来源、内容共同决定 ID。
        # 同一份知识库重复启动时使用相同 ID。
        doc_id = document_id(self.embedding_model, source, content)

        existing = self.client.get(
            collection_name=self.collection_name,
            ids=[doc_id],
            output_fields=["id"],
            consistency_level="Strong",
            timeout=self.timeout,
        )
        if existing:
            return doc_id

        vector = self._embed_text(content, is_query=False)

        # 按主键插入或更新，正文和向量保存在同一条记录中。
        self.client.upsert(
            collection_name=self.collection_name,
            data=[{
                "id": doc_id,
                "vector": vector,
                "content": content,
                "source": source,
                "metadata": metadata or {},
                "embedding_model": self.embedding_model,
            }],
            timeout=self.timeout,
        )
        return doc_id

    def add_documents_batch(
        self,
        documents: list[dict],
    ) -> list[str]:
        """保留原项目的批量入口，逐个处理文档块。"""
        return [
            self.add_document(
                content=doc.get("content", ""),
                source=doc.get("source", ""),
                metadata=doc.get("metadata", {}),
            )
            for doc in documents
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """将问题向量化，检索并返回现有 RAG 所需的文档结构。"""
        query = query.strip()
        if not query or top_k <= 0:
            return []

        query_vector = self.embed_query(query)
        return self.search_by_vector(query_vector, top_k)

    def embed_query(self, query: str) -> list[float]:
        """评估器可以单独计时/缓存向量化；性能测试会禁用缓存。"""
        return self._embed_text(query, is_query=True)

    def search_by_vector(self, query_vector: list[float], top_k: int = 5,
                         document_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """可限制固定语料ID，防止Milvus中旧文档污染离线评估的候选集合。"""
        if top_k <= 0 or document_ids == []:
            return []

        # 相同维度不代表不同模型的向量可以混用。
        model_filter = (
            f"embedding_model == {json.dumps(self.embedding_model)}"
        )
        if document_ids is not None:
            model_filter += f" and id in {json.dumps(document_ids)}"

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="vector",
            filter=model_filter,
            limit=top_k,
            search_params={
                "metric_type": "COSINE",
                "params": {},
            },
            output_fields=["content", "source", "metadata"],
            consistency_level="Strong",
            timeout=self.timeout,
        )

        documents = []
        for hit in results[0] if results else []:
            entity = hit["entity"]
            documents.append({
                "id": hit["id"],
                "content": entity["content"],
                "source": entity["source"],
                "metadata": entity.get("metadata", {}),
                # COSINE 度量下，数值越大越相似。
                "score": float(hit["distance"]),
            })

        return documents

    def load_knowledge_base(self, kb_dir: str) -> int:
        """读取 PDF/TXT，保留来源和页码，再分块导入。"""
        documents = knowledge_chunks(kb_dir, self.embedding_model)
        for document in documents:
            self.add_document(document["content"], document["source"], document["metadata"])
        for filename, count in Counter(d["metadata"]["file"] for d in documents).items():
            print(f"{filename}：处理了 {count} 个文本块")
        return len(documents)

    def save(self) -> None:
        """
        在批量导入结束后触发一次 flush。

        Milvus 由服务端管理持久化，不再生成本地 FAISS 文件。
        此方法不需要在每次查询后调用。
        """
        self.client.flush(
            collection_name=self.collection_name,
            timeout=60,
        )

    def close(self) -> None:
        """关闭应用时释放 Milvus 客户端连接。"""
        self.client.close()
