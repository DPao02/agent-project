"""DeepSeek异步客户端：读取.env配置，供所有Agent共用。

chat/chat_json负责模型请求和JSON解析，所有调用保留在这一个客户端中。
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..config import load_environment


def _is_retryable(exc: BaseException) -> bool:
    """只重试网络故障、限流和服务端故障；密钥/参数错误立即反馈。"""
    return isinstance(exc, httpx.TransportError) or (
        isinstance(exc, httpx.HTTPStatusError)
        and (exc.response.status_code == 429 or exc.response.status_code >= 500)
    )


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        load_environment()
        key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY", "")
        if not key.strip():
            raise ValueError("请在 meeting_assistant/.env 或工作区 .env 配置 DEEPSEEK_API_KEY")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=120.0,
            headers={"Authorization": f"Bearer {key}"},
            transport=transport,
        )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # 使用对象格式，避免 thinking="disabled" 触发服务端422。
            "thinking": {"type": "disabled"},
        }
        if response_format is not None:
            payload["response_format"] = response_format
        response = await self._client.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("DeepSeek 响应缺少 choices")
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise ValueError("DeepSeek 输出被截断，请增加 max_tokens 或缩短会议文本")
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("DeepSeek 返回空内容")
        return content

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        text = await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        # JSON模式下直接解析，格式错误交给Agent的现有异常分支处理。
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("DeepSeek JSON输出必须是对象")
        return result

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
