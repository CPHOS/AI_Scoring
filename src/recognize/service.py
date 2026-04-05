from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.model.types import TranscriptionResult
from src.pipeline import Pipeline, PipelineContext, PipelineMode

from .output import build_transcription_markdown
from .validation import validate_transcription


SUPPORTED_BATCH_SUFFIXES = {".jpg", ".jpeg", ".png", ".pdf"}


# ─────────────────────────────────────────────────────────────────────────────
# Core transcription
# ─────────────────────────────────────────────────────────────────────────────

def run_transcription(
    input_paths: list[Path],
    output_path: Path | None = None,
) -> TranscriptionResult:
    ctx = PipelineContext(
        mode=PipelineMode.RECOGNIZE,
        input_paths=input_paths,
        output_path=output_path,
    )
    result = Pipeline().run(ctx)
    if result.error:
        raise result.error
    return result.transcription_result


def run_batch_transcription(
    input_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, object]:
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

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    total_prompt = 0
    total_completion = 0
    total_tokens = 0

    for path in files:
        image_id = path.name
        md_name = path.stem + ".md"
        md_path = output_dir / md_name if output_dir else None

        print(f"  Processing: {image_id} ...", file=sys.stderr)
        try:
            result = run_transcription([path], output_path=md_path)
            usage = result.usage
            entry: dict[str, object] = {
                "id": image_id,
                "request_id": result.request_id,
                "output_file": md_name if md_path else None,
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
        "source_directory": str(input_dir),
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

    if output_dir is not None:
        summary_path = output_dir / "_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Summary: {summary_path}", file=sys.stderr)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe handwritten solutions into LaTeX-enriched Markdown",
    )
    parser.add_argument("inputs", nargs="*", help="One or two input image/PDF paths")
    parser.add_argument("--input-dir", help="Directory containing image/PDF files to process one by one")
    parser.add_argument("-o", "--output", help="Output path: a .md file (single mode) or a directory (batch mode)")
    return parser


def cli_main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        if args.input_dir:
            if args.inputs:
                raise ValueError("Use either positional inputs or --input-dir, not both")
            output_dir = Path(args.output).expanduser().resolve() if args.output else None
            summary = run_batch_transcription(
                Path(args.input_dir).expanduser().resolve(),
                output_dir=output_dir,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            if not args.inputs:
                raise ValueError("Provide one or two input files, or use --input-dir")
            input_paths = [Path(item).expanduser().resolve() for item in args.inputs]
            output_path = Path(args.output).expanduser().resolve() if args.output else None
            result = run_transcription(input_paths, output_path=output_path)
            warnings = validate_transcription(result.transcription)
            print(build_transcription_markdown(result, warnings))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0
