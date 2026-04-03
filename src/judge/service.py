"""Judge 模块编排层 — 单次判卷 + CLI."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.config import load_judge_settings
from src.client.openrouter import OpenRouterClient

from .answer_parser import parse_scoring_rubric
from .output import build_judging_markdown
from .prompt_builder import build_judge_messages
from .response_parser import parse_judge_response
from .types import JudgingResult


_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def run_judging(
    student_path: Path,
    standard_path: Path,
    output_path: Path | None = None,
    verbosity: int = 1,
) -> JudgingResult:
    """Score a student's transcription against a .tex answer key."""
    # 1. Parse rubric from .tex
    rubric = parse_scoring_rubric(standard_path)

    # 2. Read student markdown and strip frontmatter
    student_md = student_path.read_text(encoding="utf-8")
    student_text = _strip_frontmatter(student_md)

    # 3. Build messages and call LLM, with retry on parse failure
    settings = load_judge_settings()
    client = OpenRouterClient(settings)
    messages = build_judge_messages(rubric, student_text, verbosity)

    primary_items = [it for it in rubric.items if not it.alternative]
    last_error: Exception | None = None

    for attempt in range(settings.max_retries + 1):
        raw_response = client.create_chat_completion(messages)
        try:
            item_scores = parse_judge_response(raw_response, primary_items)
            break
        except ValueError as exc:
            last_error = exc
            if attempt < settings.max_retries:
                print(
                    f"  Retrying judge parse (attempt {attempt + 1}): {exc}",
                    file=sys.stderr,
                )
                continue
            raise RuntimeError(
                f"Failed to parse judge response after {settings.max_retries + 1} attempts: {last_error}"
            ) from last_error

    # 5. Compute totals
    total_score = sum(it.score for it in item_scores)
    max_score = sum(it.max_score for it in primary_items)

    # 6. Extract usage
    usage = _extract_usage(raw_response)

    result = JudgingResult(
        total_score=total_score,
        max_score=max_score,
        item_scores=item_scores,
        student_source=student_path.name,
        standard_source=standard_path.name,
        model=settings.model,
        provider="openrouter",
        usage=usage,
        problem_title=rubric.problem_title,
    )

    # 7. Write output
    if output_path is not None:
        md_content = build_judging_markdown(result, verbosity)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_content, encoding="utf-8")
        print(f"  Score: {total_score}/{max_score} → {output_path}", file=sys.stderr)

    return result


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (--- ... ---) if present."""
    return _FRONTMATTER_RE.sub("", text, count=1).strip()


def _extract_usage(raw_response: dict[str, object]) -> dict[str, object]:
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_judge_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a student transcription against a .tex answer key",
    )
    parser.add_argument(
        "--student", required=True,
        help="Path to student transcription .md file (recognize output)",
    )
    parser.add_argument(
        "--standard", required=True,
        help="Path to standard answer .tex file",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for the judging result .md file",
    )
    parser.add_argument(
        "--verbosity", type=int, default=1, choices=[0, 1, 2],
        help="Detail level: 0=scores only, 1=brief comments, 2=full reasoning (default: 1)",
    )
    return parser


def judge_cli_main(argv: list[str] | None = None) -> int:
    parser = build_judge_arg_parser()
    args = parser.parse_args(argv)
    try:
        student_path = Path(args.student).expanduser().resolve()
        standard_path = Path(args.standard).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve() if args.output else None

        result = run_judging(student_path, standard_path, output_path, args.verbosity)
        md = build_judging_markdown(result, args.verbosity)
        print(md)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0
