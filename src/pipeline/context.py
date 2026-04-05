"""Pipeline 上下文 — 在各阶段之间传递数据."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .states import PipelineMode, PipelineState

if TYPE_CHECKING:
    from src.judge.types import JudgingResult, ScoringRubric
    from src.model.types import InputAsset, TranscriptionResult


@dataclass
class PipelineContext:
    """可变上下文包，贯穿整个 pipeline 生命周期."""

    mode: PipelineMode

    # ── 输入 ──
    input_paths: list[Path] = field(default_factory=list)
    student_text: str | None = None  # judge 模式：已转录的学生答案
    student_source: str = ""  # 学生文件名（用于输出元数据）
    standard_path: Path | None = None
    output_path: Path | None = None
    verbosity: int = 1
    strictness: int = 1

    # ── 中间结果 ──
    assets: list[InputAsset] | None = None
    rubric: ScoringRubric | None = None
    raw_response: dict[str, Any] | None = None

    # ── 最终结果 ──
    transcription_result: TranscriptionResult | None = None
    judging_result: JudgingResult | None = None
    warnings: list[str] = field(default_factory=list)

    # ── 状态机 ──
    state: PipelineState = PipelineState.INIT
    error: Exception | None = None
