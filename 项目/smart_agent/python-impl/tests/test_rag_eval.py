"""离线评估的指标、数据隔离、缓存、预算回归；不请求任何模型API。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from memory.corpus import document_id, knowledge_chunks
from rag.retrieval import LexicalIndex, TermDictionary, reciprocal_rank_fusion
from rag_eval.budget import BudgetExceeded, RequestBudget
from rag_eval.dataset import ARTIFACTS, DATA, digest, import_review, read_jsonl, validate
from rag_eval.engine import JsonCache
from rag_eval.metrics import aggregate, latency_summary, retrieval_metrics


class MetricTests(unittest.TestCase):
    def test_recall_hit_and_mrr_have_distinct_denominators(self):
        result = retrieval_metrics(["x", "y", "a"], ["a", "b"])
        self.assertEqual(result["recall5"], 0.5)
        self.assertEqual(result["hit5"], 1.0)
        self.assertAlmostEqual(result["mrr10"], 1 / 3)

    def test_mrr_cutoff_and_empty_relevance(self):
        result = retrieval_metrics([str(i) for i in range(11)], ["10"])
        self.assertEqual(result["mrr10"], 0)
        self.assertIsNone(retrieval_metrics(["x"], [])["recall5"])

    def test_duplicate_ranking_is_not_silently_improved(self):
        with self.assertRaises(ValueError):
            retrieval_metrics(["x", "x", "a"], ["a"])

    def test_unanswerable_is_excluded_but_failed_answerable_is_zero(self):
        rows = [{"group_id": "one", "recall5": 1, "mrr10": 1, "hit5": 1, "error": None},
                {"group_id": "two", "recall5": 0, "mrr10": 0, "hit5": 0, "error": "HTTP error"},
                {"group_id": "three", "recall5": None, "mrr10": None, "hit5": None, "error": None}]
        result = aggregate(rows)
        self.assertEqual(result["answerable_queries"], 2)
        self.assertEqual(result["recall5"], 0.5)
        self.assertEqual(result["errors"], 1)

    def test_latency_rejects_cached_samples_and_counts_failures(self):
        with self.assertRaises(ValueError):
            latency_summary([{"cache_hits": {"embedding": True}}])
        result = latency_summary([{"cache_hits": {}, "error": None, "actual_candidates": 18, "total_ms": 100},
                                  {"cache_hits": {}, "error": "failure", "actual_candidates": 0, "total_ms": 500}])
        self.assertEqual(result["failures"], 1)
        self.assertEqual(result["p95_ms"], 100)


class RetrievalTests(unittest.TestCase):
    def test_rrf_fuses_ranks_instead_of_raw_scores(self):
        result = reciprocal_rank_fusion([
            [{"id": "a", "score": 1000000}, {"id": "b", "score": 1}],
            [{"id": "b", "score": 0.001}, {"id": "c", "score": 0.0001}],
        ], limit=3)
        self.assertEqual(result[0]["id"], "b")
        self.assertAlmostEqual(result[0]["score"], 1 / 61 + 1 / 62)

    def test_unknown_bm25_terms_do_not_pad_zero_score_documents(self):
        documents = [{"id": "a", "content": "退款需要订单号"}, {"id": "b", "content": "物流签收"},
                     {"id": "c", "content": "商品质量检测报告"}]
        index = LexicalIndex(documents)
        self.assertEqual(index.search("quantum_unseen_987654", 10), [])

    def test_domain_dictionary_is_idempotent_and_keeps_semantic_distinctions(self):
        terms = TermDictionary.load(DATA / "terms.json")
        text = "七日无理由售后单退钱"
        normalized = terms.normalize(text)
        self.assertEqual(terms.normalize(normalized), normalized)
        self.assertEqual(terms.normalize("确认收货"), "确认收货")

    def test_shared_chunk_ids_match_existing_hash_contract(self):
        import hashlib
        expected = hashlib.sha256("model\npolicy.txt\n退款说明".encode()).hexdigest()
        self.assertEqual(document_id("model", "policy.txt", " 退款说明 "), expected)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "policy.txt").write_text("退款说明", encoding="utf-8")
            document = knowledge_chunks(path, "model")[0]
            self.assertEqual(document["id"], expected)
            self.assertIsNone(document["metadata"]["page"])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.documents = read_jsonl(ARTIFACTS / "corpus.jsonl")
        self.queries = read_jsonl(ARTIFACTS / "queries.draft.jsonl")
        self.manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))

    def test_one_hundred_drafts_and_disjoint_intent_groups(self):
        self.assertEqual(len(self.queries), 100)
        self.assertEqual(sum(q["answerable"] for q in self.queries), 90)
        self.assertEqual(sum(q["split"] == "test" for q in self.queries), 20)
        self.assertTrue(all(q["annotation_status"] == "draft" and not q["reviewer"] for q in self.queries))
        validate(self.documents, self.queries, self.manifest, allow_draft=True)

    def test_formal_evaluation_rejects_unreviewed_labels(self):
        with self.assertRaises(ValueError):
            validate(self.documents, self.queries, self.manifest, allow_draft=False)

    def test_unknown_relevance_and_corpus_drift_are_rejected(self):
        bad = [dict(self.queries[0], relevant_ids=["unknown"])]
        with self.assertRaises(ValueError):
            validate(self.documents, bad, self.manifest, allow_draft=True)
        with self.assertRaises(ValueError):
            validate([*self.documents, dict(self.documents[0], id="changed")], self.queries, self.manifest, True)
        with self.assertRaises(ValueError):
            validate(self.documents, [dict(self.queries[0], corpus_sha256="old-corpus")], self.manifest, True)

    def test_review_requires_a_named_reviewer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "review.csv"
            with (ARTIFACTS / "annotation_review.csv").open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                rows, fields = list(reader), reader.fieldnames
            rows[0]["reviewed"] = "true"
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                import_review(path, ARTIFACTS / "queries.draft.jsonl", Path(temporary) / "out.jsonl", ARTIFACTS / "corpus.jsonl")


class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_latency_bypasses_persistent_result_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = JsonCache(Path(temporary))
            calls = []
            async def function():
                calls.append(1)
                return len(calls)
            first, first_hit, _ = await cache.compute({"query": "one"}, function, True)
            second, second_hit, _ = await cache.compute({"query": "one"}, function, True)
            third, third_hit, _ = await cache.compute({"query": "one"}, function, False)
            self.assertEqual((first, second, third), (1, 1, 2))
            self.assertEqual((first_hit, second_hit, third_hit), (False, True, False))


class BudgetTests(unittest.TestCase):
    def test_budget_is_persistent_and_stops_before_extra_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "budget.sqlite"
            first = RequestBudget(path, limit=2)
            first.charge("embedding")
            second = RequestBudget(path, limit=1000)  # 新进程不能静默抬高已存在上限。
            second.charge("rerank")
            with self.assertRaises(BudgetExceeded):
                second.charge("rewrite")
            self.assertEqual(second.summary()["attempted_calls"], 2)
            self.assertEqual(second.summary()["limit"], 2)


if __name__ == "__main__":
    unittest.main()
