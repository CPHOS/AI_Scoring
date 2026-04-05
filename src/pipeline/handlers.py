"""Pipeline 各阶段处理器."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .states import PipelineMode, PipelineState

if TYPE_CHECKING:
    from .context import PipelineContext

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────────────────────

def handle_init(ctx: PipelineContext) -> None:
    """校验上下文是否满足当前模式的前置条件."""
    logger.debug("INIT 校验  mode=%s", ctx.mode.value)
    if ctx.mode == PipelineMode.RECOGNIZE:
        if not ctx.input_paths:
            raise ValueError("input_paths required for recognize mode")
    elif ctx.mode == PipelineMode.JUDGE:
        if ctx.student_text is None:
            raise ValueError("student_text required for judge mode")
        if ctx.standard_path is None:
            raise ValueError("standard_path required for judge mode")
    elif ctx.mode == PipelineMode.DIRECT:
        if not ctx.input_paths:
            raise ValueError("input_paths required for direct mode")
        if ctx.standard_path is None:
            raise ValueError("standard_path required for direct mode")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD_INPUT
# ─────────────────────────────────────────────────────────────────────────────

def handle_load_input(ctx: PipelineContext) -> None:
    """加载并编码输入文件（图片/PDF）."""
    from src.recognize.input_processing import load_inputs

    logger.info("加载输入文件: %s", [p.name for p in ctx.input_paths])
    ctx.assets = load_inputs(ctx.input_paths)
    logger.info("已加载 %d 个资源", len(ctx.assets))


# ─────────────────────────────────────────────────────────────────────────────
# PARSE_RUBRIC
# ─────────────────────────────────────────────────────────────────────────────

def handle_parse_rubric(ctx: PipelineContext) -> None:
    """解析 .tex 标准答案文件."""
    from src.judge.answer_parser import parse_scoring_rubric

    logger.info("解析标准答案: %s", ctx.standard_path.name)
    ctx.rubric = parse_scoring_rubric(ctx.standard_path)
    logger.info("评分标准: %s  共 %d 个评分点  满分 %d",
                ctx.rubric.problem_title, len(ctx.rubric.items), ctx.rubric.total_score)


# ─────────────────────────────────────────────────────────────────────────────
# RECOGNIZE
# ─────────────────────────────────────────────────────────────────────────────

def handle_recognize(ctx: PipelineContext) -> None:
    """VLM 转录手写答卷."""
    from src.client.openrouter import OpenRouterClient, extract_text_content, extract_usage
    from src.config import get_settings
    from src.model.types import TranscriptionResult
    from src.recognize.latex_normalizer import normalize_transcription
    from src.recognize.prompt_builder import build_messages
    from src.recognize.request_id import build_request_id
    from src.recognize.validation import validate_transcription

    settings = get_settings("default")
    client = OpenRouterClient(settings)
    messages = build_messages(ctx.assets)
    logger.info("VLM 转录请求  model=%s", settings.model)
    raw_response = client.create_chat_completion(messages)
    raw_text = extract_text_content(raw_response)
    normalized = normalize_transcription(raw_text)

    ctx.transcription_result = TranscriptionResult(
        request_id=build_request_id(ctx.input_paths),
        transcription=normalized,
        source_files=[a.filename for a in ctx.assets],
        model=settings.model,
        provider="openrouter",
        usage=extract_usage(raw_response),
    )
    ctx.raw_response = raw_response
    ctx.warnings = validate_transcription(normalized)
    if ctx.warnings:
        for w in ctx.warnings:
            logger.warning("转录校验: %s", w)


# ─────────────────────────────────────────────────────────────────────────────
# JUDGE
# ─────────────────────────────────────────────────────────────────────────────

def handle_judge(ctx: PipelineContext) -> None:
    """评分（支持文本输入和直接图片输入两种模式）."""
    import time
    from datetime import datetime, timezone

    from src.client.openrouter import OpenRouterClient, extract_usage
    from src.config import get_settings
    from src.judge.prompt_builder import build_judge_messages
    from src.judge.response_parser import parse_judge_response
    from src.judge.types import JudgingResult

    is_direct = ctx.mode == PipelineMode.DIRECT
    profile = "direct" if is_direct else "judge"
    settings = get_settings(profile)
    client = OpenRouterClient(settings)
    mode_label = "DirectVLM" if is_direct else "Judge"

    submitted_at = datetime.now(timezone.utc)
    logger.info("%s 评分请求  model=%s  strictness=%d", mode_label, settings.model, ctx.strictness)

    if is_direct:
        messages = build_judge_messages(
            ctx.rubric, student_text=None, assets=ctx.assets,
            verbosity=ctx.verbosity, strictness=ctx.strictness,
        )
    else:
        messages = build_judge_messages(
            ctx.rubric, student_text=ctx.student_text,
            verbosity=ctx.verbosity, strictness=ctx.strictness,
        )

    primary_items = [it for it in ctx.rubric.items if not it.alternative]
    last_error: Exception | None = None
    t0 = time.perf_counter()

    for attempt in range(settings.max_retries + 1):
        raw_response = client.create_chat_completion(messages)
        try:
            item_scores = parse_judge_response(raw_response, primary_items)
            break
        except ValueError as exc:
            last_error = exc
            if attempt < settings.max_retries:
                logger.warning("%s 解析重试 (attempt %d): %s", mode_label, attempt + 1, exc)
                continue
            raise RuntimeError(
                f"Failed to parse response after {settings.max_retries + 1} attempts: {last_error}"
            ) from last_error

    elapsed = time.perf_counter() - t0
    completed_at = datetime.now(timezone.utc)

    total_score = sum(it.score for it in item_scores)
    max_score = sum(it.max_score for it in primary_items)
    usage = extract_usage(raw_response)

    student_label = (
        ", ".join(p.name for p in ctx.input_paths) if ctx.input_paths
        else ctx.student_source
    )

    ctx.judging_result = JudgingResult(
        total_score=total_score,
        max_score=max_score,
        item_scores=item_scores,
        student_source=student_label,
        standard_source=ctx.standard_path.name,
        model=settings.model,
        provider="openrouter",
        usage=usage,
        problem_title=ctx.rubric.problem_title,
        submitted_at=submitted_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        completed_at=completed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        duration_seconds=round(elapsed, 2),
    )
    ctx.raw_response = raw_response
    logger.info("%s 评分完成  %d/%d  耗时 %.2fs", mode_label, total_score, max_score, elapsed)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def handle_output(ctx: PipelineContext) -> None:
    """将结果写入文件（仅当 output_path 已指定时）."""
    if ctx.output_path is None:
        return

    if ctx.mode == PipelineMode.RECOGNIZE:
        from src.recognize.output import build_transcription_markdown

        md_content = build_transcription_markdown(ctx.transcription_result, ctx.warnings)
    else:
        from src.judge.output import build_judging_markdown

        md_content = build_judging_markdown(ctx.judging_result, ctx.verbosity)

    ctx.output_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.output_path.write_text(md_content, encoding="utf-8")

    if ctx.mode != PipelineMode.RECOGNIZE:
        result = ctx.judging_result
        logger.info("结果输出  %d/%d → %s", result.total_score, result.max_score, ctx.output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 处理器注册表
# ─────────────────────────────────────────────────────────────────────────────

HANDLERS: dict[PipelineState, object] = {
    PipelineState.INIT: handle_init,
    PipelineState.LOAD_INPUT: handle_load_input,
    PipelineState.PARSE_RUBRIC: handle_parse_rubric,
    PipelineState.RECOGNIZE: handle_recognize,
    PipelineState.JUDGE: handle_judge,
    PipelineState.OUTPUT: handle_output,
}
