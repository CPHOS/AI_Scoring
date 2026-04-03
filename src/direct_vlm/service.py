"""DirectVLM 模块编排层 — 单次/批量直接 VLM 判卷 + CLI."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_direct_vlm_settings
from src.client.openrouter import OpenRouterClient
from src.judge.answer_parser import parse_scoring_rubric
from src.judge.output import build_judging_markdown
from src.judge.response_parser import parse_judge_response
from src.judge.types import JudgingResult
from src.recognize.input_processing import load_inputs

from .prompt_builder import build_direct_vlm_messages


SUPPORTED_BATCH_SUFFIXES = {".jpg", ".jpeg", ".png", ".pdf"}


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
# Core
# ─────────────────────────────────────────────────────────────────────────────

def run_direct_judging(
    student_paths: list[Path],
    standard_path: Path,
    output_path: Path | None = None,
    verbosity: int = 1,
) -> JudgingResult:
    """直接用 VLM 读取手写图片并按标准答案评分（一步完成）."""
    # 1. Parse rubric
    rubric = parse_scoring_rubric(standard_path)
    primary_items = [it for it in rubric.items if not it.alternative]

    # 2. Load images
    assets = load_inputs(student_paths)

    # 3. Build messages (images + rubric in one prompt)
    messages = build_direct_vlm_messages(assets, rubric, verbosity)

    # 4. Call VLM with retry on parse failure
    settings = load_direct_vlm_settings()
    client = OpenRouterClient(settings)
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
                    f"  Retrying DirectVLM parse (attempt {attempt + 1}): {exc}",
                    file=sys.stderr,
                )
                continue
            raise RuntimeError(
                f"Failed to parse DirectVLM response after {settings.max_retries + 1} attempts: {last_error}"
            ) from last_error

    # 5. Compute totals
    total_score = sum(it.score for it in item_scores)
    max_score = sum(it.max_score for it in primary_items)

    # 6. Extract usage
    usage = _extract_usage(raw_response)

    student_label = ", ".join(p.name for p in student_paths)
    result = JudgingResult(
        total_score=total_score,
        max_score=max_score,
        item_scores=item_scores,
        student_source=student_label,
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


def run_direct_batch_judging(
    input_dir: Path,
    standard_path: Path,
    output_dir: Path | None = None,
    verbosity: int = 1,
) -> dict[str, object]:
    """批量处理：对目录中的每张图片分别做直接 VLM 判卷."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    files = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_BATCH_SUFFIXES],
        key=lambda item: item.name.lower(),
    )
    if not files:
        raise ValueError(f"No supported input files found in directory: {input_dir}")

    # 默认输出到图片同目录
    effective_output_dir = output_dir if output_dir is not None else input_dir
    effective_output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    total_prompt = 0
    total_completion = 0
    total_tokens = 0

    for path in files:
        image_id = path.name
        md_name = path.stem + "_direct.md"
        md_path = effective_output_dir / md_name

        print(f"  Processing (DirectVLM): {image_id} ...", file=sys.stderr)
        try:
            result = run_direct_judging(
                [path], standard_path, output_path=md_path, verbosity=verbosity,
            )
            usage = result.usage
            entry: dict[str, object] = {
                "id": image_id,
                "score": result.total_score,
                "max_score": result.max_score,
                "output_file": md_name,
            }
            if usage:
                entry["usage"] = usage
                total_prompt += int(usage.get("prompt_tokens", 0))
                total_completion += int(usage.get("completion_tokens", 0))
                total_tokens += int(usage.get("total_tokens", 0))
            results.append(entry)

        except Exception as exc:
            failures.append({"id": image_id, "error": str(exc)})

    summary: dict[str, object] = {
        "mode": "direct_vlm",
        "source_directory": str(input_dir),
        "standard_answer": str(standard_path),
        "result_count": len(results),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
    }
    if total_tokens:
        summary["total_usage"] = {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
        }

    summary_path = effective_output_dir / "_direct_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Summary: {summary_path}", file=sys.stderr)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_direct_vlm_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DirectVLM: 直接用 VLM 读取手写答卷图片并评分（一步完成）",
    )
    parser.add_argument(
        "inputs", nargs="*",
        help="One or two student answer image/PDF paths",
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing student answer image/PDF files to batch process",
    )
    parser.add_argument(
        "--standard", required=True,
        help="Path to standard answer .tex file",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path: a .md file (single mode) or a directory (batch mode, default: same as input-dir)",
    )
    parser.add_argument(
        "--verbosity", type=int, default=1, choices=[0, 1, 2],
        help="Detail level: 0=scores only, 1=brief comments, 2=full reasoning (default: 1)",
    )
    return parser


def direct_vlm_cli_main(argv: list[str] | None = None) -> int:
    parser = build_direct_vlm_arg_parser()
    args = parser.parse_args(argv)
    try:
        standard_path = Path(args.standard).expanduser().resolve()

        if args.input_dir:
            if args.inputs:
                raise ValueError("Use either positional inputs or --input-dir, not both")
            output_dir = Path(args.output).expanduser().resolve() if args.output else None
            summary = run_direct_batch_judging(
                Path(args.input_dir).expanduser().resolve(),
                standard_path,
                output_dir=output_dir,
                verbosity=args.verbosity,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            if not args.inputs:
                raise ValueError("Provide one or two input image files, or use --input-dir")
            input_paths = [Path(item).expanduser().resolve() for item in args.inputs]
            if args.output:
                output_path = Path(args.output).expanduser().resolve()
            else:
                # 默认输出到第一张图片同目录
                output_path = input_paths[0].parent / (input_paths[0].stem + "_direct.md")

            result = run_direct_judging(
                input_paths, standard_path, output_path=output_path, verbosity=args.verbosity,
            )
            md = build_judging_markdown(result, args.verbosity)
            print(md)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0
