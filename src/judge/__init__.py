"""判卷模块 — 标准答案 + 学生转录/图片 → 评分."""

from .service import run_direct_judging, run_judging

__all__ = ["run_judging", "run_direct_judging"]
