"""Recognize 模块 Markdown 输出."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.model.types import TranscriptionResult


def build_transcription_markdown(result: TranscriptionResult, warnings: list[str]) -> str:
    """Build markdown output with YAML frontmatter."""
    frontmatter = _build_frontmatter(result, warnings)
    return f"{frontmatter}\n\n{result.transcription}\n"


def _build_frontmatter(result: TranscriptionResult, warnings: list[str]) -> str:
    lines = ["---"]
    lines.append(f"source_files: {json.dumps(result.source_files, ensure_ascii=False)}")
    lines.append(f'request_id: "{result.request_id}"')
    lines.append(f'model: "{result.model}"')
    lines.append(f'provider: "{result.provider}"')
    lines.append(f'timestamp: "{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}"')

    usage = result.usage
    if usage:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in usage:
                lines.append(f"{key}: {usage[key]}")
        if "generation_id" in usage:
            lines.append(f'generation_id: "{usage["generation_id"]}"')

    if warnings:
        lines.append(f"warnings: {json.dumps(warnings, ensure_ascii=False)}")

    lines.append("---")
    return "\n".join(lines)
