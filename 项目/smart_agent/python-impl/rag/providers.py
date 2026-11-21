"""百炼专用Rerank客户端；错误显式上报，不默默回退后仍宣称启用了重排。"""

import os
from urllib.parse import urlsplit

import httpx


class RerankError(RuntimeError):
    pass


class DashScopeReranker:
    def __init__(self):
        base = urlsplit(os.environ["DASHSCOPE_BASE_URL"])
        # 保留原Key所在地域/工作空间的host，也允许显式配置完整地址。
        self.url = os.getenv("RERANK_URL", f"{base.scheme}://{base.netloc}/api/v1/services/rerank/text-rerank/text-rerank")
        self.model = os.getenv("RERANK_MODEL", "qwen3-rerank")
        self.client = httpx.AsyncClient(timeout=30, headers={"Authorization": "Bearer " + os.environ["DASHSCOPE_API_KEY"]})

    async def rank(self, query: str, documents: list[dict], top_n: int = 10) -> list[dict]:
        if not documents:
            return []
        response = await self.client.post(self.url, json={
            "model": self.model, "input": {"query": query, "documents": [d["content"] for d in documents]},
            "parameters": {"top_n": min(top_n, len(documents)), "return_documents": False},
        })
        try:
            body = response.json()
        except ValueError as exc:
            raise RerankError(f"排序接口返回非JSON响应，HTTP {response.status_code}") from exc
        if response.is_error or "output" not in body:
            # 不输出请求头、Key或整段HTTP对象。
            raise RerankError(f"Rerank HTTP {response.status_code}, code={body.get('code', 'unknown')}, request_id={body.get('request_id', '')}")
        results, seen = [], set()
        for item in body["output"]["results"]:
            index = item["index"]
            if not isinstance(index, int) or not 0 <= index < len(documents) or index in seen:
                raise RerankError("排序接口返回重复或越界索引")
            seen.add(index)
            results.append({**documents[index], "rerank_score": float(item["relevance_score"])})
        if len(results) != min(top_n, len(documents)):
            raise RerankError("排序接口返回结果数量与请求不符")
        return results

    async def close(self):
        await self.client.aclose()
