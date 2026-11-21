"""合并已完成的质量与延迟结果，不访问模型，不生成Markdown。"""

import csv
import json
from pathlib import Path


def summarize(data_directory: Path, quality_directory: Path | None = None, latency_directory: Path | None = None):
    quality_directory = quality_directory or sorted((data_directory / "results").glob("quality_*"))[-1]
    latency_directory = latency_directory or sorted((data_directory / "results").glob("latency_*"))[-1]
    quality = json.loads((quality_directory / "report.json").read_text(encoding="utf-8"))
    latency = json.loads((latency_directory / "report.json").read_text(encoding="utf-8"))
    if quality["manifest"]["corpus_sha256"] != latency["manifest"]["corpus_sha256"]:
        raise ValueError("质量与耗时测试的语料版本不同，不能合并")
    configs = {row["name"]: row for row in quality["configurations"]}
    rows = []
    for row in quality["summaries"]:
        if row["split"] != "all":
            continue
        config = configs[row["configuration"]]
        timing = latency["summaries"].get(row["configuration"], {})
        rows.append({"configuration": row["configuration"], "label_status": quality["label_status"],
            "answerable_queries": row["answerable_queries"], "recall_at_5": row["recall5"],
            "mrr_at_10": row["mrr10"], "hit_at_5": row["hit5"], "api_errors": row["errors"],
            "requested_candidates": config["candidates"], "actual_candidates": json.dumps(row["candidate_counts"]),
            "returned_k": config["output_k"], "retrieval_p95_ms": timing.get("p95_ms"),
            "rerank_p95_ms": timing.get("stage_p95_ms", {}).get("rerank_ms"),
            "latency_samples": timing.get("attempts"),
            "top50_top20_comparison_valid": latency["top50_vs_top20_comparison_valid"]})
    with (data_directory / "results_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pointer = {"quality_report": str((quality_directory / "report.json").relative_to(data_directory)),
               "per_query": str((quality_directory / "per_query.jsonl").relative_to(data_directory)),
               "error_analysis": str((quality_directory / "error_analysis.jsonl").relative_to(data_directory)),
               "review_pools": str((quality_directory / "review_pools.jsonl").relative_to(data_directory)),
               "latency_report": str((latency_directory / "report.json").relative_to(data_directory)),
               "latency_samples": str((latency_directory / "samples.jsonl").relative_to(data_directory)),
               "annotation_review": "annotation_review.csv", "results_summary": "results_summary.csv"}
    (data_directory / "latest_results.json").write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["RAG检索预评测结果", "状态：" + quality["label_status"],
             f"查询总数：{quality['evaluated_queries']}；人工复核：{quality['human_reviewed_queries']}。",
             "90条可回答样本参与Recall/MRR，10条无答案样本单独保留；未评估生成答案正确率或拒答率。",
             "以下全量结果是描述性预评测；开发集/测试集按意图组隔离，细分结果与区间在quality report.json。",
             "Recall@5是相关块覆盖率，MRR@10是首个相关块的倒数排名均值，Hit@5是至少命中一个块。", ""]
    for row in rows:
        lines.append(f"{row['configuration']}: Recall@5={row['recall_at_5']:.4%}, MRR@10={row['mrr_at_10']:.4f}, Hit@5={row['hit_at_5']:.4%}")
    lines.extend(["", "当前线上策略召回5条后输出3条；受控消融其他方法输出10条，不能把全部差额归因于一个因素。",
                  f"知识库只有{quality['manifest']['document_count']}块：请求Top50和Top20都被上限裁剪。",
                  "Top50与Top20有效对比：" + str(latency["top50_vs_top20_comparison_valid"]),
                  "延迟范围：" + latency["scope"],
                  "延迟实验不读写应用结果缓存；排除预热，串行执行。供应商内部缓存不可控。"])
    for name, timing in latency["summaries"].items():
        lines.append(f"{name}: {timing['successes']}/{timing['attempts']}成功, actual={timing['actual_candidates']}, P95={timing['p95_ms']/1000:.3f}s, Rerank P95={timing['stage_p95_ms']['rerank_ms']/1000:.3f}s")
    lines.extend(["", "人工复核：编辑annotation_review.csv的相关块、reviewed和reviewer；不要按检索是否命中反向修改标签。",
                  "python -m rag_eval review-import --csv evaluation_artifacts/annotation_review.csv",
                  "python -m rag_eval run --dataset evaluation_artifacts/queries.reviewed.jsonl",
                  "未复核样本会阻止正式模式运行。草案结果不代表100条人工标注或真实业务准确率。"])
    (data_directory / "results_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
