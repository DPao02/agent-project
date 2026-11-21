"""用法见--help；只生成JSON/JSONL/CSV/TXT，不生成Markdown。"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

from rag_eval.dataset import ARTIFACTS, DATA, PROJECT, digest, file_fingerprints, import_review, prepare, read_jsonl, validate, write_jsonl
from rag_eval.metrics import aggregate, latency_summary, retrieval_metrics


def parser():
    cli = argparse.ArgumentParser(description="RAG离线评估：草案与人工复核严格区分；质量缓存与真实延迟严格区分")
    sub = cli.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate", "smoke", "run", "latency", "review-import", "summary"):
        command = sub.add_parser(name)
        command.add_argument("--data-dir", type=Path, default=ARTIFACTS)
        if name == "prepare":
            command.add_argument("--force", action="store_true", help="重建前自行备份人工复核表")
        elif name == "summary":
            command.add_argument("--quality-dir", type=Path)
            command.add_argument("--latency-dir", type=Path)
        elif name == "review-import":
            command.add_argument("--csv", type=Path, required=True)
            command.add_argument("--output", type=Path)
        else:
            command.add_argument("--dataset", type=Path)
            command.add_argument("--allow-draft", action="store_true")
            if name == "run":
                command.add_argument("--workers", type=int, default=3)
                command.add_argument("--limit", type=int, default=0, help="仅调试用，正式报告应为全部100条")
                command.add_argument("--methods", nargs="+", help="选择配置名称，省略为全部9组")
            if name == "latency":
                command.add_argument("--queries", type=int, default=10)
                command.add_argument("--repeats", type=int, default=3)
                command.add_argument("--warmup", type=int, default=2)
                command.add_argument("--allow-capped", action="store_true", help="语料不足50时仅测实际候选，不声称Top50对比有效")
    return cli


def safe_error(exc: Exception) -> str:
    # 第三方AuthenticationError可能包含Key片段，不把完整异常文本写进评测报告。
    from rag.providers import RerankError
    return str(exc) if isinstance(exc, RerankError) else f"{type(exc).__name__}(HTTP={getattr(exc, 'status_code', 'n/a')})"


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_data(args):
    directory = args.data_dir
    documents = read_jsonl(directory / "corpus.jsonl")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    queries = read_jsonl(args.dataset or directory / "queries.draft.jsonl")
    validate(documents, queries, manifest, args.allow_draft)
    if file_fingerprints(PROJECT / "knowledge_base") != manifest["files"]:
        raise ValueError("当前知识库文件与评估快照不同，请建立新语料版本并复核标签")
    if digest(json.loads((DATA / "terms.json").read_text(encoding="utf-8"))) != manifest["terms_sha256"]:
        raise ValueError("词典指纹变化，请建立新版本快照，不在同一实验中静默更换词典")
    return documents, queries, manifest


def result_directory(kind: str, parent: Path):
    path = parent / "results" / (kind + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    path.mkdir(parents=True, exist_ok=False)
    return path


async def build_engine(documents, manifest, directory):
    from langchain_openai import ChatOpenAI
    from memory.long_term import LongTermMemory
    from rag_eval.engine import EvaluationEngine
    # 本轮用户明确授权的外发目的地；不能通过替换URL扩展授权范围。
    if urlsplit(os.environ["DASHSCOPE_BASE_URL"]).hostname != "dashscope.aliyuncs.com":
        raise ValueError("当前百炼地址超出本轮已确认目的地范围")
    if urlsplit(os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).hostname != "api.deepseek.com":
        raise ValueError("当前DeepSeek地址超出本轮已确认目的地范围")
    if os.getenv("RERANK_URL") and urlsplit(os.environ["RERANK_URL"]).hostname != "dashscope.aliyuncs.com":
        raise ValueError("当前Rerank地址超出本轮已确认目的地范围")
    memory = await asyncio.to_thread(LongTermMemory, embedding_max_retries=0)
    try:
        if memory.embedding_model != manifest["embedding_model"] or memory.embedding_dim != manifest["embedding_dimensions"]:
            raise ValueError("当前Embedding配置与固定语料不一致")
        llm = ChatOpenAI(model=os.getenv("MODEL_NAME", "deepseek-flash"),
                        api_key=os.environ["DEEPSEEK_API_KEY"],
                        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                        temperature=0, timeout=30, max_retries=0,
                        extra_body={"thinking": {"type": "disabled"}})
        engine = EvaluationEngine(memory, llm, documents, directory / "cache")
        await asyncio.to_thread(engine.validate_index)
        return engine
    except Exception:
        await asyncio.to_thread(memory.close)
        raise


async def retrieve_record(engine, query, config, cached):
    started = time.perf_counter()
    try:
        result = await engine.search(query["query"], config, use_cache=cached)
    except Exception as exc:
        result = {"configuration": config.name, "query": query["query"], "used_query": None,
                  "ranked_ids": [], "ranking": [], "candidate_ids": [], "requested_candidates": config.candidates,
                  "actual_candidates": 0, "candidate_clipped": config.candidates > len(engine.documents),
                  "cache_hits": {}, "stage_timings_ms": {}, "error": safe_error(exc),
                  "total_ms": (time.perf_counter() - started) * 1000}
    gold = set(query["relevant_ids"])
    return {**result, "qid": query["qid"], "group_id": query["group_id"], "split": query["split"],
            "annotation_status": query["annotation_status"], "relevant_ids": query["relevant_ids"],
            "candidate_recall": len(set(result["candidate_ids"]) & gold) / len(gold) if gold else None,
            "missed_at_recall": sorted(gold - set(result["candidate_ids"])),
            "missed_after_ranking": sorted((gold & set(result["candidate_ids"])) - set(result["ranked_ids"][:5])),
            **retrieval_metrics(result["ranked_ids"], query["relevant_ids"])}


def report_metadata(engine, manifest, queries):
    reviewed = sum(q["annotation_status"] == "human_reviewed" for q in queries)
    return {"label_status": "human_reviewed" if reviewed == len(queries) else "PROVISIONAL_NOT_HUMAN_VERIFIED",
            "human_reviewed_queries": reviewed, "evaluated_queries": len(queries),
            "dataset_sha256": digest(queries), "manifest": manifest,
            "collection": engine.memory.collection_name, "embedding_model": engine.memory.embedding_model,
            "rerank_model": engine.reranker.model, "rewrite_model": engine.model,
            "code_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [PROJECT / "rag/retrieval.py", PROJECT / "rag/providers.py", PROJECT / "rag_eval/engine.py",
                             PROJECT / "rag_eval/metrics.py", PROJECT / "rag_eval/budget.py", PROJECT / "agents/knowledge_rag.py"]},
            "metric_definition": "Macro document Recall@5; first relevant rank MRR@10; Hit@5 separately. Empty qrels excluded, not scored as zero.",
            "unanswerable_scope": "10条无答案样本不参与Recall/MRR；本实验不验证生成答案或拒答准确率",
            "candidate_ablation_feasible": len(engine.documents) >= 50,
            "request_budget": engine.budget.summary()}


async def quality(args, engine, queries, manifest):
    from rag_eval.engine import CONFIGURATIONS, configuration_metadata
    selected = queries[:args.limit] if args.limit else queries
    configs = [c for c in CONFIGURATIONS if not args.methods or c.name in args.methods]
    if not configs or (args.methods and set(args.methods) - {c.name for c in configs}):
        raise ValueError("存在未知配置名称")
    semaphore = asyncio.Semaphore(max(1, args.workers))
    completed = 0
    records = []
    async def evaluate(query):
        nonlocal completed
        async with semaphore:
            for config in configs:
                records.append(await retrieve_record(engine, query, config, cached=True))
            completed += 1
            if completed % 10 == 0 or completed == len(selected):
                print(f"完成 {completed}/{len(selected)} 条查询", flush=True)
    await asyncio.gather(*(evaluate(query) for query in selected))
    records.sort(key=lambda row: (row["qid"], row["configuration"]))
    directory = result_directory("quality", args.data_dir)
    summaries = []
    for config in configs:
        for split in ("all", "dev", "test"):
            subset = [r for r in records if r["configuration"] == config.name and (split == "all" or r["split"] == split)]
            summaries.append({"configuration": config.name, "split": split, **aggregate(subset),
                              "candidate_counts": sorted({r["actual_candidates"] for r in subset if not r["error"]})})
    report = {**report_metadata(engine, manifest, selected), "configurations": configuration_metadata(),
              "quality_run_uses_cache": True, "p95_from_quality_run": None,
              "warning": "不得将质量缓存命中的耗时当作线上延迟；18块语料时Top50和Top20不是两个不同实验",
              "summaries": summaries}
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(directory / "per_query.jsonl", records)
    failures = [r for r in records if r["error"] or (r["recall5"] is not None and r["recall5"] < 1)]
    write_jsonl(directory / "error_analysis.jsonl", failures)
    save_csv(directory / "summary.csv", [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                                          for k, v in row.items()} for row in summaries])
    # 供人工复核查看不同方法的完整候选池，不据此自动修改相关性标签。
    pools = []
    for query in selected:
        ids = sorted({doc_id for row in records if row["qid"] == query["qid"] for doc_id in row["candidate_ids"]})
        pools.append({"qid": query["qid"], "query": query["query"], "draft_relevant_ids": query["relevant_ids"],
                      "pooled_documents": [engine.by_id[doc_id] for doc_id in ids]})
    write_jsonl(directory / "review_pools.jsonl", pools)
    print(report["label_status"])
    for row in summaries:
        if row["split"] == "all":
            print(f"{row['configuration']}: Recall@5={row['recall5']:.4f}, MRR@10={row['mrr10']:.4f}, Hit@5={row['hit5']:.4f}, errors={row['errors']}")
    print("结果目录:", directory)


async def benchmark(args, engine, queries, manifest):
    from rag_eval.engine import CONFIGURATIONS
    if len(engine.documents) < 50 and not args.allow_capped:
        raise ValueError(f"仅{len(engine.documents)}个块，不能有效比较Top50/Top20。可扩充真实语料，或--allow-capped仅测实际候选")
    configs = [c for c in CONFIGURATIONS if c.name in {"hybrid_rerank_50_dict_rewrite", "hybrid_rerank_20_dict_rewrite"}]
    # 实际候选数相同就只测一次，不把随机网络波动包装成剪枝收益。
    unique = {}
    for config in configs:
        unique.setdefault(min(config.candidates, len(engine.documents)), config)
    configs = list(unique.values())
    groups = {}
    for query in queries:
        if query["answerable"]:
            groups.setdefault(query["group_id"], query)
    selected = list(groups.values())
    random.Random(20261115).shuffle(selected)
    selected = selected[:args.queries]
    if not selected or args.repeats < 1:
        raise ValueError("延迟测试需要至少一个问题及一次重复")
    print(f"实际语料{len(engine.documents)}块；{len(selected)}个意图、{args.repeats}次重复；不使用应用结果缓存", flush=True)
    for index in range(args.warmup):
        warmup = await retrieve_record(engine, selected[index % len(selected)], configs[index % len(configs)], cached=False)
        if warmup["error"]:
            raise RuntimeError("预热失败：" + warmup["error"])
    jobs = [(query, config, repeat) for repeat in range(args.repeats) for query in selected for config in configs]
    random.Random(20261115).shuffle(jobs)
    samples = []
    for index, (query, config, repeat) in enumerate(jobs, 1):
        samples.append({**await retrieve_record(engine, query, config, cached=False), "repeat": repeat})
        if index % 5 == 0 or index == len(jobs):
            print(f"延迟样本 {index}/{len(jobs)}", flush=True)
    summaries = {config.name: latency_summary([s for s in samples if s["configuration"] == config.name]) for config in configs}
    directory = result_directory("latency", args.data_dir)
    report = {**report_metadata(engine, manifest, queries), "warmup_excluded": args.warmup,
              "scope": "词典+改写+Embedding+Milvus+BM25+RRF+Rerank；不含答案生成、意图路由与合规",
              "application_result_cache": "disabled", "concurrency": 1,
              "requested_candidates": [50, 20], "effective_candidates": sorted(unique),
              "top50_vs_top20_comparison_valid": len(engine.documents) >= 50,
              "summaries": summaries}
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(directory / "samples.jsonl", samples)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print("结果目录:", directory)


async def run_online(args, documents, queries, manifest):
    engine = await build_engine(documents, manifest, args.data_dir)
    try:
        if args.command == "smoke":
            from rag_eval.engine import CONFIGURATIONS
            for config in (CONFIGURATIONS[0], CONFIGURATIONS[-2]):
                record = await retrieve_record(engine, queries[0], config, cached=True)
                print(json.dumps({key: record[key] for key in ["configuration", "used_query", "actual_candidates", "recall5", "mrr10", "error"]}, ensure_ascii=False))
                if record["error"]:
                    raise RuntimeError("模型/检索连通性测试失败")
        elif args.command == "run":
            await quality(args, engine, queries, manifest)
        else:
            await benchmark(args, engine, queries, manifest)
    finally:
        await engine.close()
        await asyncio.to_thread(engine.memory.close)


def main():
    args = parser().parse_args()
    load_dotenv(PROJECT / ".env", override=True)
    # 离线评估不自动向外部追踪平台发送整个标注集。
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    if args.command == "prepare":
        report = prepare(os.getenv("EMBEDDING_MODEL", "text-embedding-v4"), int(os.getenv("EMBEDDING_DIMENSIONS", "1024")), args.data_dir, args.force)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "review-import":
        count = import_review(args.csv, args.data_dir / "queries.draft.jsonl", args.output or args.data_dir / "queries.reviewed.jsonl", args.data_dir / "corpus.jsonl")
        print(f"已导入，人工复核记录数：{count}；未复核记录仍为draft")
    elif args.command == "summary":
        from rag_eval.summarize import summarize
        summarize(args.data_dir, args.quality_dir, args.latency_dir)
        print("已生成results_summary.csv / results_summary.txt / latest_results.json")
    else:
        documents, queries, manifest = load_data(args)
        if args.command == "validate":
            print(f"数据校验通过：{len(queries)}条，{len(documents)}个块；正式/草案状态已检查")
        else:
            asyncio.run(run_online(args, documents, queries, manifest))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("评估未完成：", type(exc).__name__, str(exc) if isinstance(exc, (ValueError, FileExistsError)) else "请检查模型权限、网络及配置；未把失败伪装成成功")
        raise SystemExit(1)
