"""解析 LLM 的评分 JSON 响应."""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from .types import ItemScore, ScoringItem


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_response(
    payload: dict[str, Any],
    primary_items: list[ScoringItem],
) -> list[ItemScore]:
    """Parse the OpenRouter response payload into a list of ItemScore."""
    raw_text = _extract_text(payload)
    scores_data = _parse_json(raw_text)
    return _map_scores(scores_data, primary_items)


def _extract_text(payload: dict[str, Any]) -> str:
    """Extract plain text from an OpenRouter chat completion response."""
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
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    raise ValueError("Unsupported message content format")


def _parse_json(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from potentially decorated text."""
    # Try direct parse first
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fence
    fence_match = _CODE_FENCE_RE.search(stripped)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object
    obj_match = _JSON_OBJECT_RE.search(stripped)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot parse JSON from LLM response: {stripped[:200]}")


def _map_scores(
    data: dict[str, Any],
    primary_items: list[ScoringItem],
) -> list[ItemScore]:
    """Map parsed JSON scores to ItemScore objects."""
    scores_dict = data.get("scores", data)
    if not isinstance(scores_dict, dict):
        raise ValueError("Expected 'scores' to be a JSON object")

    result: list[ItemScore] = []
    for item in primary_items:
        raw = scores_dict.get(item.item_id)
        if raw is None:
            print(
                f"  warning: item {item.item_id} not found in LLM response, scoring 0",
                file=sys.stderr,
            )
            result.append(ItemScore(
                item_id=item.item_id,
                item_type=item.item_type,
                score=0,
                max_score=item.max_score,
                reasoning="未在模型输出中找到该项评分",
            ))
            continue

        if isinstance(raw, dict):
            score = int(raw.get("score", 0))
            reasoning = str(raw.get("reasoning", ""))
        elif isinstance(raw, (int, float)):
            score = int(raw)
            reasoning = ""
        else:
            score = 0
            reasoning = f"Unexpected format: {raw}"

        # Clamp score to valid range
        if score < 0:
            score = 0
        if score > item.max_score:
            print(
                f"  warning: item {item.item_id} score {score} exceeds max {item.max_score}, clamping",
                file=sys.stderr,
            )
            score = item.max_score

        result.append(ItemScore(
            item_id=item.item_id,
            item_type=item.item_type,
            score=score,
            max_score=item.max_score,
            reasoning=reasoning,
        ))

    return result
