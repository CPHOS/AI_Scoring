from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_settings
from src.client.openrouter import OpenRouterClient
from src.model.types import TranscriptionResult

from .input_processing import load_inputs
from .latex_normalizer import normalize_transcription
from .prompt_builder import build_messages
from .request_id import build_request_id
from .response_parser import extract_text_content
from .validation import validate_transcription


SUPPORTED_BATCH_SUFFIXES = {".jpg", ".jpeg", ".png", ".pdf"}


def _extract_usage(raw_response: dict[str, object]) -> dict[str, object]:
    """Extract usage/cost data from an OpenRouter API response."""
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
# Markdown generation
# ─────────────────────────────────────────────────────────────────────────────

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


def _build_markdown(result: TranscriptionResult, warnings: list[str]) -> str:
    frontmatter = _build_frontmatter(result, warnings)
    return f"{frontmatter}\n\n{result.transcription}\n"


# ─────────────────────────────────────────────────────────────────────────────
# Core transcription
# ─────────────────────────────────────────────────────────────────────────────

def run_transcription(
    input_paths: list[Path],
    output_path: Path | None = None,
) -> TranscriptionResult:
    settings = load_settings()
    assets = load_inputs(input_paths)
    request_id = build_request_id(input_paths)
    client = OpenRouterClient(settings)
    messages = build_messages(assets)
    raw_response = client.create_chat_completion(messages)
    raw_text = extract_text_content(raw_response)
    normalized = normalize_transcription(raw_text)
    usage = _extract_usage(raw_response)

    result = TranscriptionResult(
        request_id=request_id,
        transcription=normalized,
        source_files=[a.filename for a in assets],
        model=settings.model,
        provider="openrouter",
        usage=usage,
    )

    warnings = validate_transcription(result.transcription)
    if warnings:
        for w in warnings:
            print(f"  warning: {w}", file=sys.stderr)

    if output_path is not None:
        md_content = _build_markdown(result, warnings)
        output_path.write_text(md_content, encoding="utf-8")

    return result


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
            print(_build_markdown(result, warnings))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0
