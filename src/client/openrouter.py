from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from src.config import Settings

from .key_pool import KeyPool

logger = logging.getLogger(__name__)

# 模块级 KeyPool 缓存，相同 key 集合共享同一个池
_pool_cache: dict[tuple[str, ...], KeyPool] = {}
_pool_lock = threading.Lock()


def _get_pool(settings: Settings) -> KeyPool:
    """获取或创建与 settings.api_keys 对应的 KeyPool（线程安全）."""
    cache_key = tuple(settings.api_keys)
    with _pool_lock:
        if cache_key not in _pool_cache:
            _pool_cache[cache_key] = KeyPool(list(settings.api_keys))
        return _pool_cache[cache_key]


class OpenRouterClient:
    """OpenRouter API 客户端."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool = _get_pool(settings)

    def create_chat_completion(self, messages: list[dict[str, object]]) -> dict[str, Any]:
        payload = {
            "model": self._settings.model,
            "messages": messages,
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"{self._settings.base_url.rstrip('/')}/chat/completions"

        last_error: Exception | None = None
        for attempt in range(self._settings.max_retries + 1):
            api_key = self._pool.acquire()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self._settings.site_url or "https://localhost",
                "X-Title": self._settings.app_name,
            }
            request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
            try:
                logger.debug("API 请求  model=%s  attempt=%d  key=...%s",
                             self._settings.model, attempt + 1, api_key[-6:])
                with urllib.request.urlopen(request, timeout=self._settings.timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    logger.debug("API 响应  id=%s", result.get("id", "?"))
                    return result
            except urllib.error.HTTPError as exc:
                detail = _read_error_detail(exc)
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                logger.warning("API HTTP 错误 %d  attempt=%d  key=...%s: %s",
                               exc.code, attempt + 1, api_key[-6:], detail[:200])
                self._pool.report_failure(api_key)
                if attempt >= self._settings.max_retries:
                    break
                time.sleep(self._settings.retry_delay * (2 ** attempt))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning("API 网络错误  attempt=%d  key=...%s: %s",
                               attempt + 1, api_key[-6:], exc)
                self._pool.report_failure(api_key)
                if attempt >= self._settings.max_retries:
                    break
                time.sleep(self._settings.retry_delay * (2 ** attempt))

        raise RuntimeError(f"OpenRouter request failed: {last_error}")


def _read_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        payload = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        payload = ""
    return payload or exc.reason


# ─────────────────────────────────────────────────────────────────────────────
# 响应解析工具函数
# ─────────────────────────────────────────────────────────────────────────────


def extract_text_content(payload: dict[str, Any]) -> str:
    """从 OpenRouter chat completion 响应中提取纯文本内容."""
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("OpenRouter response does not contain choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise ValueError("OpenRouter response does not contain message content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part.strip())
    raise ValueError("Unsupported message content format")


def extract_usage(raw_response: dict[str, object]) -> dict[str, object]:
    """从 OpenRouter 响应中提取 usage/cost 数据."""
    usage: dict[str, object] = {}
    api_usage = raw_response.get("usage")
    if isinstance(api_usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in api_usage:
                usage[key] = api_usage[key]
    gen_id = raw_response.get("id")
    if gen_id:
        usage["generation_id"] = gen_id
    return usage
