"""DirectVLM 模块 — 直接用 VLM 一步完成卷面识别 + 评分."""

from .service import run_direct_judging

__all__ = ["run_direct_judging"]
