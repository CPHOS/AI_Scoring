"""生成判卷结果的 Markdown 输出."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .types import JudgingResult


def build_judging_markdown(result: JudgingResult, verbosity: int = 1) -> str:
    """Build a Markdown document with YAML frontmatter from judging results."""
    frontmatter = _build_frontmatter(result)
    body = _build_body(result, verbosity) if verbosity >= 1 else ""
    if body:
        return f"{frontmatter}\n\n{body}\n"
    return f"{frontmatter}\n"


def _build_frontmatter(result: JudgingResult) -> str:
    lines = ["---"]
    lines.append(f'student_source: "{result.student_source}"')
    lines.append(f'standard_source: "{result.standard_source}"')
    if result.problem_title:
        lines.append(f'problem_title: "{result.problem_title}"')
    lines.append(f'model: "{result.model}"')
    lines.append(f'provider: "{result.provider}"')
    lines.append(f'timestamp: "{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}"')
    lines.append(f"total_score: {result.total_score}")
    lines.append(f"max_score: {result.max_score}")

    # Per-item scores
    lines.append("scores:")
    for item in result.item_scores:
        lines.append(f'  "{item.item_id}": {{score: {item.score}, max: {item.max_score}}}')

    # Usage
    usage = result.usage
    if usage:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in usage:
                lines.append(f"{key}: {usage[key]}")
        if "generation_id" in usage:
            lines.append(f'generation_id: "{usage["generation_id"]}"')

    lines.append("---")
    return "\n".join(lines)


def _build_body(result: JudgingResult, verbosity: int) -> str:
    """Build the detailed scoring body text."""
    lines: list[str] = []
    lines.append(f"# 评分结果：{result.problem_title}")
    lines.append("")
    lines.append(f"**总分：{result.total_score} / {result.max_score}**")
    lines.append("")

    # Group by sub_question if we have that info — for now just list items
    lines.append("## 逐项评分")
    lines.append("")

    for item in result.item_scores:
        type_label = "方程" if item.item_type == "equation" else "文字/讨论"
        status = "✓" if item.score == item.max_score else ("✗" if item.score == 0 else "△")
        lines.append(f"- {status} **{item.item_id}** [{type_label}]: {item.score}/{item.max_score}")
        if verbosity >= 1 and item.reasoning:
            lines.append(f"  - {item.reasoning}")
        lines.append("")

    return "\n".join(lines)
