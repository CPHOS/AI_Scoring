"""Judge 模块编排层 — 单次/批量判卷 + 直接 VLM 判卷 + CLI."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from src.pipeline import Pipeline, PipelineContext, PipelineMode

from .output import build_judging_markdown
from .types import JudgingResult

logger = logging.getLogger(__name__)


_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

SUPPORTED_BATCH_SUFFIXES = {".jpg", ".jpeg", ".png", ".pdf"}


# ─────────────────────────────────────────────────────────────────────────────
# 文本评分（两步：先转录后评分）
# ─────────────────────────────────────────────────────────────────────────────


def run_judging(
    student_path: Path,
    standard_path: Path,
    output_path: Path | None = None,
    verbosity: int = 1,
    strictness: int = 1,
) -> JudgingResult:
    """Score a student's transcription against a .tex answer key."""
    student_md = student_path.read_text(encoding="utf-8")
    student_text = _strip_frontmatter(student_md)

    ctx = PipelineContext(
        mode=PipelineMode.JUDGE,
        student_text=student_text,
        student_source=student_path.name,
        standard_path=standard_path,
        output_path=output_path,
        verbosity=verbosity,
        strictness=strictness,
    )
    result = Pipeline().run(ctx)
    if result.error:
        raise result.error
    return result.judging_result


# ─────────────────────────────────────────────────────────────────────────────
# 直接 VLM 评分（一步：VLM 直读图片 + 评分）
# ─────────────────────────────────────────────────────────────────────────────


def run_direct_judging(
    student_paths: list[Path],
    standard_path: Path,
    output_path: Path | None = None,
    verbosity: int = 1,
    strictness: int = 1,
) -> JudgingResult:
    """直接用 VLM 读取手写图片并按标准答案评分（一步完成）."""
    ctx = PipelineContext(
        mode=PipelineMode.DIRECT,
        input_paths=student_paths,
        standard_path=standard_path,
        output_path=output_path,
        verbosity=verbosity,
        strictness=strictness,
    )
    result = Pipeline().run(ctx)
    if result.error:
        raise result.error
    return result.judging_result


def run_direct_batch_judging(
    input_dir: Path,
    standard_path: Path,
    output_dir: Path | None = None,
    verbosity: int = 1,
    strictness: int = 1,
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
                [path], standard_path, output_path=md_path,
                verbosity=verbosity, strictness=strictness,
            )
            usage = result.usage
            entry: dict[str, object] = {
                "id": image_id,
                "score": result.total_score,
                "max_score": result.max_score,
                "duration_seconds": result.duration_seconds,
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
    logger.info("批量评分完成  summary=%s", summary_path)

    return summary


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (--- ... ---) if present."""
    return _FRONTMATTER_RE.sub("", text, count=1).strip()


# ─────────────────────────────────────────────────────────────────────────────
# CLI — judge
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
    parser.add_argument(
        "--strictness", type=int, default=1, choices=[0, 1, 2],
        help="Strictness: 0=lenient, 1=mixed (default), 2=strict",
    )
    return parser


def judge_cli_main(argv: list[str] | None = None) -> int:
    parser = build_judge_arg_parser()
    args = parser.parse_args(argv)
    try:
        student_path = Path(args.student).expanduser().resolve()
        standard_path = Path(args.standard).expanduser().resolve()
        output_path = Path(args.output).expanduser().resolve() if args.output else None

        result = run_judging(
            student_path, standard_path, output_path,
            args.verbosity, strictness=args.strictness,
        )
        md = build_judging_markdown(result, args.verbosity)
        print(md)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI — direct
# ─────────────────────────────────────────────────────────────────────────────

def build_direct_arg_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--strictness", type=int, default=1, choices=[0, 1, 2],
        help="Strictness: 0=lenient, 1=mixed (default), 2=strict",
    )
    return parser


def direct_cli_main(argv: list[str] | None = None) -> int:
    parser = build_direct_arg_parser()
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
                strictness=args.strictness,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            if not args.inputs:
                raise ValueError("Provide one or two input image files, or use --input-dir")
            input_paths = [Path(item).expanduser().resolve() for item in args.inputs]
            if args.output:
                output_path = Path(args.output).expanduser().resolve()
            else:
                output_path = input_paths[0].parent / (input_paths[0].stem + "_direct.md")

            result = run_direct_judging(
                input_paths, standard_path, output_path=output_path,
                verbosity=args.verbosity, strictness=args.strictness,
            )
            md = build_judging_markdown(result, args.verbosity)
            print(md)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0
