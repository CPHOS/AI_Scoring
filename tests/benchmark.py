#!/usr/bin/env python3
"""基准测试脚本 — 调用 AI 批改并与人工阅卷结果进行对比统计。

用法:
    uv run python tests/benchmark.py                        # 跑所有 cases
    uv run python tests/benchmark.py --cases 2 6            # 只跑指定 case
    uv run python tests/benchmark.py --skip-scoring         # 跳过 AI 评分，只统计已有结果
    uv run python tests/benchmark.py --mode recognize+judge # 两步模式
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
CASES_DIR = TESTS_DIR / "cases"
RESULTS_DIR = TESTS_DIR / "results"

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".pdf"}


# ─────────────────────────────────────────────────────────────────────────────
# AI 评分（返回分数 + 耗时）
# ─────────────────────────────────────────────────────────────────────────────


def _scoring_entry(score: int, duration: float = 0.0) -> dict:
    return {"score": score, "duration": round(duration, 2)}


def run_ai_scoring_direct(
    case_dir: Path, output_dir: Path, verbosity: int, strictness: int,
    images: list[Path] | None = None,
    max_workers: int = 1,
) -> dict[str, dict]:
    """对一个 case 目录执行 direct 模式批改，返回 {filename: {score, duration}}。

    传入 *images* 可指定仅评分的图片列表（用于采样模式）。
    *max_workers* > 1 时启用并发。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from src.judge.answer_parser import parse_scoring_rubric
    from src.judge.service import run_direct_judging

    standard_path = case_dir / "main.tex"
    if images is None:
        images = sorted(
            [p for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES],
            key=lambda f: f.name.lower(),
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 只解析一次评分标准
    rubric = parse_scoring_rubric(standard_path)

    def _score(img: Path) -> tuple[str, dict]:
        md_path = output_dir / f"{img.stem}_direct.md"
        print(f"    Scoring: {img.name} ...", file=sys.stderr)
        try:
            result = run_direct_judging(
                [img], standard_path, output_path=md_path,
                verbosity=verbosity, strictness=strictness,
                rubric=rubric,
            )
            return img.name, _scoring_entry(result.total_score, result.duration_seconds)
        except Exception as exc:
            print(f"    ERROR: {img.name}: {exc}", file=sys.stderr)
            return img.name, _scoring_entry(-1)

    ai_results: dict[str, dict] = {}
    if max_workers <= 1:
        for img in images:
            name, entry = _score(img)
            ai_results[name] = entry
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_score, img): img for img in images}
            for fut in as_completed(futures):
                name, entry = fut.result()
                ai_results[name] = entry
    return ai_results


def run_ai_scoring_two_step(
    case_dir: Path, output_dir: Path, verbosity: int, strictness: int,
    images: list[Path] | None = None,
    max_workers: int = 1,
) -> dict[str, dict]:
    """两步模式：先 recognize 再 judge，返回 {filename: {score, duration}}。

    *max_workers* > 1 时启用并发。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from src.judge.service import run_judging
    from src.recognize.service import run_transcription

    standard_path = case_dir / "main.tex"
    if images is None:
        images = sorted(
            [p for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES],
            key=lambda f: f.name.lower(),
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    def _score(img: Path) -> tuple[str, dict]:
        recog_path = output_dir / f"{img.stem}_recog.md"
        judge_path = output_dir / f"{img.stem}_judge.md"
        t0 = time.perf_counter()
        print(f"    Recognize: {img.name} ...", file=sys.stderr)
        try:
            run_transcription([img], output_path=recog_path)
        except Exception as exc:
            print(f"    RECOG ERROR: {img.name}: {exc}", file=sys.stderr)
            return img.name, _scoring_entry(-1)
        print(f"    Judge: {img.name} ...", file=sys.stderr)
        try:
            result = run_judging(
                recog_path, standard_path, output_path=judge_path,
                verbosity=verbosity, strictness=strictness,
            )
            elapsed = time.perf_counter() - t0
            return img.name, _scoring_entry(result.total_score, elapsed)
        except Exception as exc:
            print(f"    JUDGE ERROR: {img.name}: {exc}", file=sys.stderr)
            return img.name, _scoring_entry(-1)

    ai_results: dict[str, dict] = {}
    if max_workers <= 1:
        for img in images:
            name, entry = _score(img)
            ai_results[name] = entry
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_score, img): img for img in images}
            for fut in as_completed(futures):
                name, entry = fut.result()
                ai_results[name] = entry
    return ai_results


# ─────────────────────────────────────────────────────────────────────────────
# 已有结果加载
# ─────────────────────────────────────────────────────────────────────────────


def load_existing_ai_scores(output_dir: Path) -> dict[str, dict]:
    """从已有的 _direct.md / _judge.md 文件中读取 total_score 和 duration。"""

    results: dict[str, dict] = {}
    md_files = sorted(output_dir.glob("*_direct.md")) or sorted(output_dir.glob("*_judge.md"))
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        m = re.search(r'student_source:\s*"([^"]+)"', text)
        ts = re.search(r"total_score:\s*(\d+)", text)
        dur = re.search(r"duration_seconds:\s*([\d.]+)", text)
        if m and ts:
            results[m.group(1)] = _scoring_entry(
                int(ts.group(1)),
                float(dur.group(1)) if dur else 0.0,
            )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 统计分析
# ─────────────────────────────────────────────────────────────────────────────


def compute_statistics(
    human_scores: dict[str, int],
    ai_results: dict[str, dict],
) -> dict[str, object]:
    """计算 AI 与人工评分的统计指标。"""
    filenames = sorted(set(human_scores) & set(ai_results))
    # 排除评分失败的项
    valid = [(f, human_scores[f], ai_results[f]) for f in filenames if ai_results[f]["score"] >= 0]

    if not valid:
        return {"n": 0, "error": "no valid score pairs"}

    n = len(valid)
    diffs = [entry["score"] - h for _, h, entry in valid]
    abs_diffs = [abs(d) for d in diffs]
    durations = [entry["duration"] for _, _, entry in valid]

    mean_diff = sum(diffs) / n
    mae = sum(abs_diffs) / n
    rmse = math.sqrt(sum(d * d for d in diffs) / n)
    max_abs_diff = max(abs_diffs)
    avg_duration = sum(durations) / n if durations else 0.0

    # 各项指标
    per_file: list[dict[str, object]] = []
    for f in filenames:
        h = human_scores[f]
        entry = ai_results[f]
        a = entry["score"]
        per_file.append({
            "file": f,
            "human": h,
            "ai": a if a >= 0 else "FAILED",
            "diff": a - h if a >= 0 else None,
            "duration": entry["duration"],
        })

    return {
        "n": n,
        "failed": sum(1 for f in filenames if ai_results[f]["score"] < 0),
        "mean_diff": round(mean_diff, 2),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "max_abs_diff": max_abs_diff,
        "avg_duration": round(avg_duration, 2),
        "total_duration": round(sum(durations), 2),
        "details": per_file,
    }


def print_report(case_name: str, stats: dict[str, object]) -> None:
    """打印单个 case 的统计报告。"""
    print(f"\n{'=' * 60}")
    print(f"Case: {case_name}  (n={stats['n']}, failed={stats.get('failed', 0)})")
    print(f"{'=' * 60}")

    if stats["n"] == 0:
        print("  No valid score pairs.")
        return

    print(f"  Mean Diff (AI - Human): {stats['mean_diff']:+.2f}")
    print(f"  MAE:                    {stats['mae']:.2f}")
    print(f"  RMSE:                   {stats['rmse']:.2f}")
    print(f"  Max |Diff|:             {stats['max_abs_diff']}")
    print(f"  Avg Duration:           {stats['avg_duration']:.2f}s")
    print()
    print(f"  {'File':<20} {'Human':>6} {'AI':>6} {'Diff':>6} {'Time':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for d in stats["details"]:
        ai_str = str(d["ai"]) if d["ai"] != "FAILED" else "FAIL"
        diff_str = f"{d['diff']:+g}" if d["diff"] is not None else "  -"
        dur_str = f"{d['duration']:.1f}s" if d["duration"] else "-"
        print(f"  {d['file']:<20} {d['human']:>6} {ai_str:>6} {diff_str:>6} {dur_str:>8}")


def print_overall_summary(all_stats: dict[str, dict[str, object]]) -> None:
    """打印跨 case 的汇总。"""
    total_n = 0
    total_failed = 0
    total_sq_diff = 0.0
    total_abs_diff = 0.0
    total_diff = 0.0
    total_dur = 0.0

    for stats in all_stats.values():
        n = stats["n"]
        if n == 0:
            continue
        total_n += n
        total_failed += stats.get("failed", 0)
        total_diff += stats["mean_diff"] * n
        total_abs_diff += stats["mae"] * n
        total_sq_diff += stats["rmse"] ** 2 * n
        total_dur += stats.get("total_duration", 0)

    print(f"\n{'=' * 60}")
    print(f"Overall Summary  (total samples={total_n}, total failed={total_failed})")
    print(f"{'=' * 60}")
    if total_n > 0:
        print(f"  Mean Diff:  {total_diff / total_n:+.2f}")
        print(f"  MAE:        {total_abs_diff / total_n:.2f}")
        print(f"  RMSE:       {math.sqrt(total_sq_diff / total_n):.2f}")
        print(f"  Total Time: {total_dur:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Report 生成
# ─────────────────────────────────────────────────────────────────────────────


def _setup_plt():
    """初始化 matplotlib 并返回 plt 模块，失败返回 None。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib 未安装，跳过图表生成", file=sys.stderr)
        return None

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "STSong",
        "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def generate_case_scatter(
    case_name: str,
    stats: dict[str, object],
    max_score: int,
    output_path: Path,
) -> Path | None:
    """生成单个 case 的散点图（原始分数），返回图片路径。"""
    plt = _setup_plt()
    if plt is None:
        return None

    details = stats.get("details", [])
    humans = [d["human"] for d in details if d["ai"] != "FAILED"]
    ais = [d["ai"] for d in details if d["ai"] != "FAILED"]
    if not humans:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(humans, ais, alpha=0.7, edgecolors="k", linewidths=0.5, s=60)

    ax.plot([0, max_score], [0, max_score], "r--", linewidth=1, label="y = x")
    ax.set_xlim(-0.5, max_score + 0.5)
    ax.set_ylim(-0.5, max_score + 0.5)

    ax.set_xlabel("人工评分", fontsize=12)
    ax.set_ylabel("AI 评分", fontsize=12)
    ax.set_title(f"Case {case_name}（满分 {max_score}）", fontsize=14)
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_overall_scatter(
    all_stats: dict[str, dict[str, object]],
    case_max_scores: dict[str, int],
    output_path: Path,
) -> Path | None:
    """生成所有 case 合并的归一化散点图（分数 / 题目总分），返回图片路径。"""
    plt = _setup_plt()
    if plt is None:
        return None

    from matplotlib.lines import Line2D
    _filled = [m for m in Line2D.filled_markers if isinstance(m, str)]

    case_data: dict[str, tuple[list[float], list[float]]] = {}
    for case_name, stats in all_stats.items():
        ms = case_max_scores.get(case_name, 1)
        h_list: list[float] = []
        a_list: list[float] = []
        for d in stats.get("details", []):
            if d["ai"] != "FAILED" and d["diff"] is not None:
                h_list.append(d["human"] / ms)
                a_list.append(d["ai"] / ms)
        if h_list:
            case_data[case_name] = (h_list, a_list)

    if not case_data:
        return None

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, (case_name, (hs, ais)) in enumerate(case_data.items()):
        marker = _filled[i % len(_filled)]
        ax.scatter(hs, ais, alpha=0.7, edgecolors="k", linewidths=0.5,
                   s=60, marker=marker, label=f"Case {case_name}")

    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="y = x (完美一致)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel("人工评分（归一化）", fontsize=13)
    ax.set_ylabel("AI 评分（归一化）", fontsize=13)
    ax.set_title("AI 评分 vs 人工评分（归一化）", fontsize=15)
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_report(
    all_stats: dict[str, dict[str, object]],
    case_max_scores: dict[str, int],
    case_plots: dict[str, Path | None],
    mode: str,
    strictness: int,
    report_path: Path,
    overall_plot_path: Path | None,
    max_samples: int | None = None,
) -> None:
    """生成 report.md 汇总报告。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []

    lines.append("# 基准测试报告")
    lines.append("")
    lines.append(f"- **生成时间**: {now}")
    lines.append(f"- **评分模式**: `{mode}`")
    lines.append(f"- **严厉程度**: {strictness}")
    lines.append(f"- **测试用例数**: {len(all_stats)}")
    if max_samples is not None:
        lines.append(f"- **最大采样数**: {max_samples}")
    lines.append("")

    # 归一化散点图
    if overall_plot_path and overall_plot_path.exists():
        rel = overall_plot_path.name
        lines.append("## 总体散点图（归一化）")
        lines.append("")
        lines.append(f"![AI vs Human (归一化)]({rel})")
        lines.append("")

    # 汇总统计
    total_n = 0
    total_failed = 0
    total_diff = 0.0
    total_abs = 0.0
    total_sq = 0.0
    total_dur = 0.0

    for stats in all_stats.values():
        n = stats["n"]
        if n == 0:
            continue
        total_n += n
        total_failed += stats.get("failed", 0)
        total_diff += stats["mean_diff"] * n
        total_abs += stats["mae"] * n
        total_sq += stats["rmse"] ** 2 * n
        total_dur += stats.get("total_duration", 0)

    lines.append("## 整体统计")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|:---|---:|")
    lines.append(f"| 总样本数 | {total_n} |")
    lines.append(f"| 失败数 | {total_failed} |")
    if total_n > 0:
        lines.append(f"| Mean Diff (AI − Human) | {total_diff / total_n:+.2f} |")
        lines.append(f"| MAE | {total_abs / total_n:.2f} |")
        lines.append(f"| RMSE | {math.sqrt(total_sq / total_n):.2f} |")
        lines.append(f"| 平均耗时 | {total_dur / total_n:.2f}s |")
        lines.append(f"| 总耗时 | {total_dur:.1f}s |")
    lines.append("")

    # 逐题统计
    for case_name, stats in all_stats.items():
        ms = case_max_scores.get(case_name, 0)
        lines.append(f"## Case: {case_name}（满分 {ms}）")
        lines.append("")

        # 插入单 case 散点图
        cp = case_plots.get(case_name)
        if cp and cp.exists():
            lines.append(f"![Case {case_name}]({cp.name})")
            lines.append("")

        if stats["n"] == 0:
            lines.append("无有效评分对。")
            lines.append("")
            continue

        lines.append("| 指标 | 值 |")
        lines.append("|:---|---:|")
        lines.append(f"| 样本数 | {stats['n']} |")
        lines.append(f"| 失败数 | {stats.get('failed', 0)} |")
        lines.append(f"| Mean Diff | {stats['mean_diff']:+.2f} |")
        lines.append(f"| MAE | {stats['mae']:.2f} |")
        lines.append(f"| RMSE | {stats['rmse']:.2f} |")
        lines.append(f"| Max |Diff| | {stats['max_abs_diff']} |")
        lines.append(f"| 平均耗时 | {stats.get('avg_duration', 0):.2f}s |")
        lines.append("")

        # 明细表
        lines.append("| 文件 | 人工 | AI | 差值 | 耗时 |")
        lines.append("|:---|---:|---:|---:|---:|")
        for d in stats.get("details", []):
            ai_str = str(d["ai"]) if d["ai"] != "FAILED" else "FAIL"
            diff_str = f"{d['diff']:+g}" if d["diff"] is not None else "-"
            dur_str = f"{d['duration']:.1f}s" if d.get("duration") else "-"
            lines.append(f"| {d['file']} | {d['human']} | {ai_str} | {diff_str} | {dur_str} |")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {report_path}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="AI 阅卷基准测试")
    parser.add_argument(
        "--cases", nargs="*",
        help="要测试的 case 目录名（默认: 全部）",
    )
    parser.add_argument(
        "--mode", choices=["direct", "recognize+judge"], default="direct",
        help="评分模式 (default: direct)",
    )
    parser.add_argument(
        "--verbosity", type=int, default=1, choices=[0, 1, 2],
        help="评分详细程度 (default: 1)",
    )
    parser.add_argument(
        "--strictness", type=int, default=1, choices=[0, 1, 2],
        help="严厉程度: 0=宽松, 1=混合(default), 2=严格",
    )
    parser.add_argument(
        "--skip-scoring", action="store_true",
        help="跳过 AI 评分，只统计 results/ 下的已有结果",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, metavar="N",
        help="从所有测试图片中等概率随机抽取最多 N 个样本",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, metavar="N",
        help="并发评分线程数 (default: 1, 串行)",
    )
    args = parser.parse_args()

    from src.log import setup_logging
    setup_logging(args.verbosity)

    # 确定要跑的 case 列表
    if args.cases:
        case_dirs = [CASES_DIR / c for c in args.cases]
        for d in case_dirs:
            if not d.is_dir():
                print(f"Case directory not found: {d}", file=sys.stderr)
                return 1
    else:
        case_dirs = sorted(
            [d for d in CASES_DIR.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

    all_stats: dict[str, dict[str, object]] = {}
    case_max_scores: dict[str, int] = {}

    # ── 采样逻辑 ──
    # 收集所有 (case_dir, image) 对，按 --max-samples 随机抽取
    case_images: dict[str, list[Path]] | None = None
    if args.max_samples is not None and not args.skip_scoring:
        import random
        all_pairs: list[tuple[Path, Path]] = []  # (case_dir, image)
        for case_dir in case_dirs:
            if not (case_dir / "score.json").exists():
                continue
            imgs = sorted(
                [p for p in case_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES],
                key=lambda f: f.name.lower(),
            )
            all_pairs.extend((case_dir, img) for img in imgs)

        if args.max_samples < len(all_pairs):
            sampled = random.sample(all_pairs, args.max_samples)
            print(f"  Sampled {args.max_samples}/{len(all_pairs)} images", file=sys.stderr)
        else:
            sampled = all_pairs
            print(f"  --max-samples={args.max_samples} >= total {len(all_pairs)}, using all", file=sys.stderr)

        case_images = {}
        for case_dir_s, img in sampled:
            case_images.setdefault(case_dir_s.name, []).append(img)
        # 确保每个 case 内无重复图片
        for cn, imgs in case_images.items():
            names = [i.name for i in imgs]
            if len(names) != len(set(names)):
                raise RuntimeError(f"Case {cn} 中存在重复采样图片: {names}")

    for case_dir in case_dirs:
        case_name = case_dir.name
        score_json = case_dir / "score.json"
        if not score_json.exists():
            print(f"Skipping {case_name}: no score.json", file=sys.stderr)
            continue

        # 获取题目满分
        main_tex = case_dir / "main.tex"
        if main_tex.exists():
            from src.judge.answer_parser import parse_scoring_rubric
            rubric = parse_scoring_rubric(main_tex)
            case_max_scores[case_name] = rubric.total_score
        else:
            case_max_scores[case_name] = 0

        human_scores: dict[str, int] = json.loads(score_json.read_text(encoding="utf-8"))
        output_dir = RESULTS_DIR / case_name

        if args.skip_scoring:
            print(f"  Loading existing results for case {case_name} ...", file=sys.stderr)
            ai_results = load_existing_ai_scores(output_dir)
        else:
            # 如果启用了采样，传入抽到的图片子集
            imgs = case_images.get(case_name) if case_images is not None else None
            if case_images is not None and imgs is None:
                # 该 case 没有被抽到任何图片，跳过
                continue
            print(f"  Running AI scoring for case {case_name} ({args.mode}) ...", file=sys.stderr)
            if args.mode == "direct":
                ai_results = run_ai_scoring_direct(
                    case_dir, output_dir, args.verbosity, args.strictness, imgs,
                    max_workers=args.concurrency,
                )
            else:
                ai_results = run_ai_scoring_two_step(
                    case_dir, output_dir, args.verbosity, args.strictness, imgs,
                    max_workers=args.concurrency,
                )

        if not ai_results:
            print(f"  No AI scores for case {case_name}", file=sys.stderr)
            continue

        stats = compute_statistics(human_scores, ai_results)
        all_stats[case_name] = stats

        # 保存原始结果 JSON
        output_dir.mkdir(parents=True, exist_ok=True)
        result_json = {
            "case": case_name,
            "mode": args.mode,
            "human_scores": human_scores,
            "ai_scores": {k: v for k, v in ai_results.items()},
            "statistics": {k: v for k, v in stats.items() if k != "details"},
        }
        (output_dir / "_benchmark.json").write_text(
            json.dumps(result_json, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        print_report(case_name, stats)

    if len(all_stats) > 1:
        print_overall_summary(all_stats)

    # 生成散点图 + report.md
    if all_stats:
        # 每个 case 单独散点图
        case_plots: dict[str, Path | None] = {}
        for case_name, stats in all_stats.items():
            ms = case_max_scores.get(case_name, 0)
            if ms > 0 and stats["n"] > 0:
                plot_path = RESULTS_DIR / f"scatter_{case_name}.png"
                case_plots[case_name] = generate_case_scatter(
                    case_name, stats, ms, plot_path,
                )
            else:
                case_plots[case_name] = None

        # 所有 case 归一化散点图
        overall_plot = generate_overall_scatter(
            all_stats, case_max_scores, RESULTS_DIR / "scatter.png",
        )

        generate_report(
            all_stats, case_max_scores, case_plots,
            args.mode, args.strictness,
            RESULTS_DIR / "report.md", overall_plot,
            max_samples=args.max_samples,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
