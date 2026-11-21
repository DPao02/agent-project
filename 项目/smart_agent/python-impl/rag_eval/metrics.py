"""文档级Recall和首个相关文档倒数排名；无答案样本不伪装成召回失败。"""

import numpy as np


def retrieval_metrics(ranked_ids: list[str], relevant_ids: list[str]) -> dict:
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("排序结果包含重复ID，不能删除重复后人为抬高MRR")
    relevant = set(relevant_ids)
    if not relevant:
        return {"recall5": None, "mrr10": None, "hit5": None}
    ranked = ranked_ids
    recall = len(set(ranked[:5]) & relevant) / len(relevant)
    first = next((position for position, doc_id in enumerate(ranked[:10], 1) if doc_id in relevant), None)
    return {"recall5": recall, "mrr10": 1.0 / first if first else 0.0, "hit5": float(bool(set(ranked[:5]) & relevant))}


def aggregate(rows: list[dict]) -> dict:
    scored = [row for row in rows if row["recall5"] is not None]
    result = {"queries": len(rows), "answerable_queries": len(scored), "unanswerable_queries": len(rows) - len(scored),
              "errors": sum(bool(row.get("error")) for row in rows)}
    for metric in ("recall5", "mrr10", "hit5"):
        result[metric] = float(np.mean([row[metric] for row in scored])) if scored else None
    # 按意图组bootstrap，避免把同义的两个问题当作完全独立样本。
    groups = sorted({row["group_id"] for row in scored})
    if groups:
        means = np.array([[np.mean([r[m] for r in scored if r["group_id"] == group])
                           for m in ("recall5", "mrr10")] for group in groups])
        samples = means[np.random.default_rng(20261115).integers(0, len(groups), size=(1000, len(groups)))].mean(axis=1)
        for index, metric in enumerate(("recall5", "mrr10")):
            result[metric + "_ci95"] = np.percentile(samples[:, index], [2.5, 97.5]).tolist()
    return result


def latency_summary(samples: list[dict]) -> dict:
    success = [s for s in samples if not s.get("error")]
    if any(any(s.get("cache_hits", {}).values()) for s in samples):
        raise ValueError("延迟样本出现结果缓存，不能用来计算线上P95")
    timings = [s["total_ms"] for s in success]
    stages = sorted({stage for sample in success for stage in sample.get("stage_timings_ms", {})})
    return {"attempts": len(samples), "successes": len(success), "failures": len(samples) - len(success),
            "p50_ms": float(np.percentile(timings, 50)) if timings else None,
            "p95_ms": float(np.percentile(timings, 95)) if timings else None,
            "mean_ms": float(np.mean(timings)) if timings else None,
            "actual_candidates": sorted({s["actual_candidates"] for s in success}),
            "stage_p95_ms": {stage: float(np.percentile([s["stage_timings_ms"][stage] for s in success
                                                         if stage in s.get("stage_timings_ms", {})], 95)) for stage in stages}}
