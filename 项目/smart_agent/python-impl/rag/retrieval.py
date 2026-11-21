"""中文BM25与RRF。融合名次，不直接相加不同分布的BM25/COSINE分数。"""

import json
import logging
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

jieba.setLogLevel(logging.WARNING)


class TermDictionary:
    def __init__(self, entries: list[dict] | None = None):
        self.entries = entries or []
        self.mapping = {alias: entry["canonical"] for entry in self.entries for alias in entry["aliases"]
                        if alias != entry["canonical"]}
        self.pattern = re.compile("|".join(re.escape(s) for s in sorted(self.mapping, key=len, reverse=True))) if self.mapping else None

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["entries"])

    def normalize(self, text: str) -> str:
        return self.pattern.sub(lambda match: self.mapping[match.group()], text) if self.pattern else text


class LexicalIndex:
    def __init__(self, documents: list[dict], terms: TermDictionary | None = None):
        if not documents:
            raise ValueError("BM25语料不能为空")
        self.documents = documents
        self.terms = terms or TermDictionary()
        self.tokenizer = jieba.Tokenizer()
        for entry in self.terms.entries:
            self.tokenizer.add_word(entry["canonical"])
        tokens = [self.tokenize(doc["content"]) or ["__empty__"] for doc in documents]
        self.index = BM25Okapi(tokens, k1=1.5, b=0.75)

    def tokenize(self, text: str) -> list[str]:
        # 不删除“不、未”等否定词，避免政策条件被反转。
        return [token.lower() for token in self.tokenizer.cut(self.terms.normalize(text))
                if re.search(r"[\u4e00-\u9fffA-Za-z0-9]", token)]

    def search(self, query: str, limit: int) -> list[dict]:
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = self.index.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.documents[i]["id"]))
        # 没有匹配的文档不能仅为凑够TopK而进入词法结果。
        return [{**self.documents[i], "score": float(scores[i])} for i in order[:limit] if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[dict]], limit: int, k: int = 60) -> list[dict]:
    if k <= 0:
        raise ValueError("RRF平滑参数k必须大于0")
    scores, documents = {}, {}
    for ranking in rankings:
        seen = set()
        for position, document in enumerate(ranking, 1):
            doc_id = document["id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            documents[doc_id] = document
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
    ids = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:limit]
    return [{**documents[doc_id], "score": scores[doc_id]} for doc_id in ids]
