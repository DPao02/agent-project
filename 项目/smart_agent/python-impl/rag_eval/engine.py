"""可复现的检索消融实验。质量可缓存，延迟实验强制不读/写结果缓存。"""

import asyncio
import inspect
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from agents.knowledge_rag import KnowledgeRAGAgent, QUERY_REWRITE_PROMPT
from rag.providers import DashScopeReranker
from rag.retrieval import LexicalIndex, TermDictionary, reciprocal_rank_fusion
from rag_eval.dataset import DATA, digest
from rag_eval.budget import RequestBudget


@dataclass(frozen=True)
class Configuration:
    name: str
    retriever: str = "hybrid"
    candidates: int = 50
    rerank: bool = False
    dictionary: bool = False
    rewrite: bool = False
    rerank_provider: str = "dashscope"
    output_k: int = 10


CONFIGURATIONS = [
    Configuration("bm25", retriever="bm25"),
    Configuration("vector", retriever="vector"),
    # 额外测量当前线上真实策略；最终只有3条，不能与Top10输出假装完全同预算。
    Configuration("current_online", retriever="vector", candidates=5, rerank=True, rewrite=True,
                  rerank_provider="llm", output_k=3),
    Configuration("hybrid_rrf"),
    Configuration("hybrid_rerank_50", rerank=True),
    Configuration("hybrid_rerank_50_dict", rerank=True, dictionary=True),
    Configuration("hybrid_rerank_50_rewrite", rerank=True, rewrite=True),
    Configuration("hybrid_rerank_50_dict_rewrite", rerank=True, dictionary=True, rewrite=True),
    Configuration("hybrid_rerank_20_dict_rewrite", candidates=20, rerank=True, dictionary=True, rewrite=True),
]


class JsonCache:
    def __init__(self, directory: Path):
        self.directory = directory
        self.locks = {}

    async def compute(self, key: dict, function, enabled: bool):
        started = time.perf_counter()
        if not enabled:
            return await function(), False, (time.perf_counter() - started) * 1000
        name = digest(key)
        lock = self.locks.setdefault(name, asyncio.Lock())
        async with lock:
            path = self.directory / (name + ".json")
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8")), True, (time.perf_counter() - started) * 1000
            value = await function()
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
            return value, False, (time.perf_counter() - started) * 1000


class EvaluationEngine:
    def __init__(self, memory, llm, documents: list[dict], cache_directory: Path):
        self.memory, self.documents = memory, documents
        self.ids = [document["id"] for document in documents]
        self.by_id = {document["id"]: document for document in documents}
        self.terms = TermDictionary.load(DATA / "terms.json")
        self.lexical = LexicalIndex(documents)
        self.domain_lexical = LexicalIndex(documents, self.terms)
        self.reranker = DashScopeReranker()
        self.query_agent = KnowledgeRAGAgent(llm, memory)
        self.cache = JsonCache(cache_directory)
        self.budget = RequestBudget(cache_directory.parent / "provider_budget.sqlite", limit=1200)
        self.corpus_hash = digest(documents)
        self.prompt_hash = digest(inspect.getsource(KnowledgeRAGAgent.rewrite_query) + QUERY_REWRITE_PROMPT)
        self.model = llm.model_name

    def validate_index(self):
        existing = self.memory.client.get(collection_name=self.memory.collection_name, ids=self.ids,
                                           output_fields=["id", "content"], timeout=self.memory.timeout)
        lookup = {doc["id"]: doc["content"] for doc in existing}
        if any(lookup.get(doc["id"]) != doc["content"] for doc in self.documents):
            raise ValueError("固定语料与Milvus不一致。先运行check_rag.py导入当前PDF，再重新准备评测；不混入旧文档")

    async def search(self, query: str, config: Configuration, use_cache: bool = True) -> dict:
        start = time.perf_counter()
        timings, hits = {}, {}
        used_query = self.terms.normalize(query) if config.dictionary else query
        if config.rewrite:
            async def rewrite():
                self.budget.charge("deepseek_rewrite")
                result = await self.query_agent.rewrite_query(used_query)
                if not result or len(result) > 1000:
                    raise ValueError("查询改写返回空值或异常长文本")
                return result
            used_query, hits["rewrite"], timings["rewrite_ms"] = await self.cache.compute(
                {"stage": "rewrite", "model": self.model, "prompt": self.prompt_hash, "query": used_query}, rewrite, use_cache)
            if config.dictionary:
                used_query = self.terms.normalize(used_query)

        limit = min(config.candidates, len(self.documents))
        lexical = []
        if config.retriever in {"bm25", "hybrid"}:
            tick = time.perf_counter()
            lexical = (self.domain_lexical if config.dictionary else self.lexical).search(used_query, limit)
            timings["bm25_ms"] = (time.perf_counter() - tick) * 1000

        dense = []
        if config.retriever in {"vector", "hybrid"}:
            async def embed():
                self.budget.charge("dashscope_embedding")
                return await asyncio.to_thread(self.memory.embed_query, used_query)
            vector, hits["embedding"], timings["embedding_ms"] = await self.cache.compute(
                {"stage": "embedding", "model": self.memory.embedding_model, "dim": self.memory.embedding_dim,
                 "query": used_query}, embed, use_cache)
            tick = time.perf_counter()
            dense = await asyncio.to_thread(self.memory.search_by_vector, vector, limit, self.ids)
            timings["milvus_ms"] = (time.perf_counter() - tick) * 1000

        tick = time.perf_counter()
        if config.retriever == "hybrid":
            candidates = reciprocal_rank_fusion([lexical, dense], limit, k=60)
        else:
            candidates = lexical if config.retriever == "bm25" else dense
        timings["fusion_ms"] = (time.perf_counter() - tick) * 1000
        actual = len(candidates)
        if config.rerank:
            async def rerank():
                if config.rerank_provider == "llm":
                    self.budget.charge("deepseek_current_rerank")
                    ranked = await self.query_agent.rerank_documents(used_query, candidates, top_k=config.output_k)
                    if len({d["id"] for d in ranked}) != len(ranked):
                        raise ValueError("线上LLM重排返回重复文档，不能按正确排序评估")
                    return [{"id": d["id"]} for d in ranked]
                self.budget.charge("dashscope_rerank")
                ranked = await self.reranker.rank(used_query, candidates, top_n=config.output_k)
                return [{"id": d["id"], "rerank_score": d["rerank_score"]} for d in ranked]
            ranking, hits["rerank"], timings["rerank_ms"] = await self.cache.compute({
                "stage": "rerank", "provider": config.rerank_provider,
                "model": self.model if config.rerank_provider == "llm" else self.reranker.model,
                "url": self.reranker.url,
                "llm_prompt": digest(inspect.getsource(KnowledgeRAGAgent.rerank_documents)) if config.rerank_provider == "llm" else None,
                "corpus": self.corpus_hash, "query": used_query,
                "ordered_candidates": [d["id"] for d in candidates], "top_n": config.output_k,
            }, rerank, use_cache)
        else:
            ranking = [{"id": d["id"], "retrieval_score": d["score"]} for d in candidates[:config.output_k]]
        return {"configuration": config.name, "query": query, "used_query": used_query,
                "ranked_ids": [d["id"] for d in ranking], "ranking": ranking,
                "candidate_ids": [d["id"] for d in candidates],
                "requested_candidates": config.candidates, "actual_candidates": actual,
                "candidate_clipped": config.candidates > len(self.documents),
                "stage_timings_ms": timings, "cache_hits": hits,
                "total_ms": (time.perf_counter() - start) * 1000, "error": None}

    async def close(self):
        await self.reranker.close()


def configuration_metadata() -> list[dict]:
    return [asdict(config) for config in CONFIGURATIONS]
