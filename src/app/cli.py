"""CLI 入口 — 统一子命令路由、日志初始化、全局异常处理."""

from __future__ import annotations

import argparse
import sys

from src.log import setup_logging


def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-scoring",
        description="CPHOS AI 自动阅卷系统",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="日志级别: 不带 -v = WARNING, -v = INFO, -vv = DEBUG",
    )

    sub = parser.add_subparsers(dest="command")

    # ── recognize ──
    rec = sub.add_parser("recognize", help="卷面识别（手写 → LaTeX Markdown）")
    rec.add_argument("inputs", nargs="*", help="输入图片/PDF 路径")
    rec.add_argument("--input-dir", help="输入目录（批量处理）")
    rec.add_argument("-o", "--output", help="输出路径（.md 或目录）")

    # ── judge ──
    jud = sub.add_parser("judge", help="文本评分（转录文本 → 评分）")
    jud.add_argument("--student", required=True, help="学生转录 .md 文件路径")
    jud.add_argument("--standard", required=True, help="标准答案 .tex 文件路径")
    jud.add_argument("-o", "--output", help="评分结果输出路径")
    jud.add_argument(
        "--verbosity", type=int, default=1, choices=[0, 1, 2],
        help="Detail level: 0=scores only, 1=brief, 2=full (default: 1)",
    )
    jud.add_argument(
        "--strictness", type=int, default=1, choices=[0, 1, 2],
        help="Strictness: 0=lenient, 1=mixed, 2=strict (default: 1)",
    )

    # ── direct ──
    drc = sub.add_parser("direct", help="直接评分（图片 → 评分，一步完成）")
    drc.add_argument("inputs", nargs="*", help="学生答卷图片/PDF 路径")
    drc.add_argument("--input-dir", help="输入目录（批量处理）")
    drc.add_argument("--standard", required=True, help="标准答案 .tex 文件路径")
    drc.add_argument("-o", "--output", help="输出路径（.md 或目录）")
    drc.add_argument(
        "--verbosity", type=int, default=1, choices=[0, 1, 2],
        help="Detail level: 0=scores only, 1=brief, 2=full (default: 1)",
    )
    drc.add_argument(
        "--strictness", type=int, default=1, choices=[0, 1, 2],
        help="Strictness: 0=lenient, 1=mixed, 2=strict (default: 1)",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# 子命令调度
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_recognize(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from src.recognize.output import build_transcription_markdown
    from src.recognize.service import run_batch_transcription, run_transcription
    from src.recognize.validation import validate_transcription

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
    return 0


def _cmd_judge(args: argparse.Namespace) -> int:
    from pathlib import Path

    from src.judge.output import build_judging_markdown
    from src.judge.service import run_judging

    student_path = Path(args.student).expanduser().resolve()
    standard_path = Path(args.standard).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    result = run_judging(
        student_path, standard_path, output_path,
        args.verbosity, strictness=args.strictness,
    )
    md = build_judging_markdown(result, args.verbosity)
    print(md)
    return 0


def _cmd_direct(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from src.judge.output import build_judging_markdown
    from src.judge.service import run_direct_batch_judging, run_direct_judging

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
    return 0


_DISPATCH = {
    "recognize": _cmd_recognize,
    "judge": _cmd_judge,
    "direct": _cmd_direct,
}


def main() -> int:
    parser = _build_root_parser()
    args = parser.parse_args()

    # 日志初始化
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return 1

    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
