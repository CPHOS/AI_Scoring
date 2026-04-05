"""判卷模块公用数据类型."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScoringItem:
    """单个评分点."""

    item_type: str  # "equation" | "text"
    item_id: str  # e.g. "1", "7*"
    max_score: int
    context: str  # 期望内容（方程文本或讨论段落）
    sub_question: str  # 所属子题，e.g. "1", "2.1"
    alternative: bool = False  # 是否为多解中的非首选解法
    solution_index: int = 0  # 所属解法编号，0 为第一种


@dataclass(frozen=True)
class ScoringRubric:
    """完整评分标准."""

    problem_title: str
    total_score: int
    problem_statement: str
    solution_text: str
    items: list[ScoringItem]


@dataclass
class ItemScore:
    """单项评分结果."""

    item_id: str
    item_type: str
    score: int
    max_score: int
    reasoning: str = ""


@dataclass
class JudgingResult:
    """完整判卷结果."""

    total_score: int
    max_score: int
    item_scores: list[ItemScore] = field(default_factory=list)
    student_source: str = ""
    standard_source: str = ""
    model: str = ""
    provider: str = "openrouter"
    usage: dict[str, Any] = field(default_factory=dict)
    problem_title: str = ""
    submitted_at: str = ""     # ISO 8601 提交时间
    completed_at: str = ""     # ISO 8601 完成时间
    duration_seconds: float = 0.0  # 批阅耗时（秒）
