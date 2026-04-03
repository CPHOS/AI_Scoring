from __future__ import annotations

from typing import Any


def extract_text_content(payload: dict[str, Any]) -> str:
    """Extract the plain text content from an OpenRouter chat completion response."""
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("OpenRouter response does not contain choices")

    message = choices[0].get("message") or {}
    raw_content = message.get("content")
    if not raw_content:
        raise ValueError("OpenRouter response does not contain message content")

    return _flatten_content(raw_content)


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part.strip())
    raise ValueError("Unsupported message content format")
