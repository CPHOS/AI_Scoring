"""Pipeline 状态与模式定义."""

from __future__ import annotations

from enum import Enum, auto


class PipelineState(Enum):
    """流程状态."""

    INIT = auto()
    LOAD_INPUT = auto()
    PARSE_RUBRIC = auto()
    RECOGNIZE = auto()
    JUDGE = auto()
    OUTPUT = auto()
    DONE = auto()
    ERROR = auto()


class PipelineMode(Enum):
    """流程模式."""

    RECOGNIZE = "recognize"
    JUDGE = "judge"
    DIRECT = "direct"


# 每种模式下的状态转移表: current_state → next_state
TRANSITIONS: dict[PipelineMode, dict[PipelineState, PipelineState]] = {
    PipelineMode.RECOGNIZE: {
        PipelineState.INIT: PipelineState.LOAD_INPUT,
        PipelineState.LOAD_INPUT: PipelineState.RECOGNIZE,
        PipelineState.RECOGNIZE: PipelineState.OUTPUT,
        PipelineState.OUTPUT: PipelineState.DONE,
    },
    PipelineMode.JUDGE: {
        PipelineState.INIT: PipelineState.PARSE_RUBRIC,
        PipelineState.PARSE_RUBRIC: PipelineState.JUDGE,
        PipelineState.JUDGE: PipelineState.OUTPUT,
        PipelineState.OUTPUT: PipelineState.DONE,
    },
    PipelineMode.DIRECT: {
        PipelineState.INIT: PipelineState.LOAD_INPUT,
        PipelineState.LOAD_INPUT: PipelineState.PARSE_RUBRIC,
        PipelineState.PARSE_RUBRIC: PipelineState.JUDGE,
        PipelineState.JUDGE: PipelineState.OUTPUT,
        PipelineState.OUTPUT: PipelineState.DONE,
    },
}
