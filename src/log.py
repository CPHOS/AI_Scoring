"""统一日志配置."""

from __future__ import annotations

import logging
import sys


def setup_logging(verbosity: int = 0) -> None:
    """初始化根日志，verbosity: 0=WARNING, 1=INFO, 2=DEBUG."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger("src")
    root.setLevel(level)
    # 避免重复添加 handler
    if not root.handlers:
        root.addHandler(handler)
