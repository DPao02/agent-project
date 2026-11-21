"""固定语料、证据锚点和人工复核入口。AI草案不能冒充人工标注。"""

import csv
import hashlib
import json
import re
from pathlib import Path

from memory.corpus import CHUNK_SETTINGS, knowledge_chunks

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / "evaluation_artifacts"
DATA = Path(__file__).resolve().parent / "data"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def file_fingerprints(directory: Path) -> dict:
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}}


def prepare(model: str, dimensions: int, output: Path = ARTIFACTS, force: bool = False):
    if (output / "annotation_review.csv").exists() and not force:
        raise FileExistsError("复核表已存在，不覆盖人工工作。先备份，确需重建再使用prepare --force")
    documents = knowledge_chunks(PROJECT / "knowledge_base", model)
    corpus_hash = digest(documents)
    topics = json.loads((DATA / "topics.json").read_text(encoding="utf-8"))
    positives = sorted((t["id"] for t in topics if t["evidence"]), key=lambda value: digest(value))
    negatives = sorted((t["id"] for t in topics if not t["evidence"]), key=lambda value: digest(value))
    # 按意图组切分，两个同义问法不会分到开发集和测试集两边。
    heldout = set(positives[:9] + negatives[:1])
    rows = []
    for topic in topics:
        evidence, relevant = [], set()
        for filename, page, quote in topic["evidence"]:
            matched = [d["id"] for d in documents if d["metadata"]["file"] == filename
                       and d["metadata"]["page"] == page and compact(quote) in compact(d["content"])]
            if not matched:
                raise ValueError(f"证据没有落入任何当前文档块：{topic['id']} / {filename} / {page} / {quote}")
            relevant.update(matched)
            evidence.append({"file": filename, "page": page, "quote": quote, "chunk_ids": matched})
        for variant, query in enumerate(topic["questions"], 1):
            rows.append({"qid": f"q{len(rows) + 1:03d}", "group_id": topic["id"], "variant": variant,
                         "corpus_sha256": corpus_hash,
                         "split": "test" if topic["id"] in heldout else "dev", "query": query,
                         "answerable": bool(evidence), "expected_answer": topic["answer"],
                         "relevant_ids": sorted(relevant), "evidence": evidence,
                         "unanswerable_reason": topic.get("unanswerable_reason"),
                         "annotation_status": "draft", "reviewer": ""})
    if len(rows) != 100:
        raise ValueError("当前草案应恰好100条")
    manifest = {"schema_version": 1, "embedding_model": model, "embedding_dimensions": dimensions,
                "chunk_settings": CHUNK_SETTINGS, "document_count": len(documents),
                "corpus_sha256": digest(documents), "terms_sha256": digest(json.loads((DATA / "terms.json").read_text(encoding="utf-8"))),
                "files": file_fingerprints(PROJECT / "knowledge_base"),
                "queries": len(rows), "answerable": sum(q["answerable"] for q in rows),
                "annotation_status": "assistant_authored_draft_not_human_verified",
                "split_strategy": "80 dev / 20 test; disjoint intent groups; 90 answerable + 10 unanswerable"}
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "corpus.jsonl", documents)
    write_jsonl(output / "queries.draft.jsonl", rows)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "annotation_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["qid", "group_id", "split", "query", "answerable", "expected_answer", "relevant_ids", "evidence", "reviewed", "reviewer", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ["qid", "group_id", "split", "query", "answerable", "expected_answer"]} | {
                "relevant_ids": json.dumps(row["relevant_ids"]), "evidence": json.dumps(row["evidence"], ensure_ascii=False),
                "reviewed": "false", "reviewer": "", "notes": "待人工检查相关块是否完整，不能仅因检索命中就判定正确"})
    return manifest


def validate(documents: list[dict], queries: list[dict], manifest: dict, allow_draft: bool):
    if digest(documents) != manifest["corpus_sha256"]:
        raise ValueError("语料指纹改变，请重新准备快照和复核标签")
    ids = {d["id"] for d in documents}
    if len(ids) != len(documents) or len({q["qid"] for q in queries}) != len(queries):
        raise ValueError("文档或查询ID重复")
    groups = {}
    for query in queries:
        if query.get("corpus_sha256") != manifest["corpus_sha256"]:
            raise ValueError(f"查询标注属于另一个语料版本：{query['qid']}，需重新复核")
        if not set(query["relevant_ids"]).issubset(ids):
            raise ValueError(f"未知相关块ID：{query['qid']}")
        if bool(query["relevant_ids"]) != query["answerable"]:
            raise ValueError(f"answerable和相关块标注不一致：{query['qid']}")
        groups.setdefault(query["group_id"], set()).add(query["split"])
        if not allow_draft and (query["annotation_status"] != "human_reviewed" or not query["reviewer"].strip()):
            raise ValueError("标注尚未人工复核；预评测使用--allow-draft，正式评测请先review-import")
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("同一意图组跨越dev/test，存在同义问题泄漏")


def import_review(csv_path: Path, original_path: Path, output: Path, corpus_path: Path):
    original = {row["qid"]: row for row in read_jsonl(original_path)}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        edits = list(csv.DictReader(stream))
    if len(edits) != len(original) or {row["qid"] for row in edits} != set(original):
        raise ValueError("复核CSV必须保持全部qid，不得缺失或重复")
    result = []
    for edit in edits:
        row = dict(original[edit["qid"]])
        reviewed = edit["reviewed"].strip().lower() in {"true", "yes", "1"}
        if reviewed and not edit["reviewer"].strip():
            raise ValueError("已复核记录必须填写reviewer")
        row.update(query=edit["query"], answerable=edit["answerable"].strip().lower() in {"true", "yes", "1"},
                   expected_answer=edit["expected_answer"], relevant_ids=json.loads(edit["relevant_ids"]),
                   annotation_status="human_reviewed" if reviewed else "draft", reviewer=edit["reviewer"], review_notes=edit["notes"])
        result.append(row)
    manifest = json.loads((corpus_path.parent / "manifest.json").read_text(encoding="utf-8"))
    validate(read_jsonl(corpus_path), result, manifest, allow_draft=True)
    write_jsonl(output, result)
    return sum(row["annotation_status"] == "human_reviewed" for row in result)
