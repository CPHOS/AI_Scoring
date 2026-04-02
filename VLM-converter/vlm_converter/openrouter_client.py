from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .config import Settings


class OpenRouterClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def create_chat_completion(self, messages: list[dict[str, object]]) -> dict[str, Any]:
        payload = {
            "model": self._settings.model,
            "messages": messages,
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"{self._settings.base_url.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._settings.site_url or "https://localhost",
            "X-Title": self._settings.app_name,
        }

        last_error: Exception | None = None
        for attempt in range(self._settings.max_retries + 1):
            request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self._settings.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = _read_error_detail(exc)
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                if attempt >= self._settings.max_retries:
                    break
                time.sleep(min(2 ** attempt, 4))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self._settings.max_retries:
                    break
                time.sleep(min(2 ** attempt, 4))

        raise RuntimeError(f"OpenRouter request failed: {last_error}")


def _read_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        payload = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        payload = ""
    return payload or exc.reason
