"""Pipeline 状态机运行器."""

from __future__ import annotations

import logging
import time

from .context import PipelineContext
from .handlers import HANDLERS
from .states import TRANSITIONS, PipelineState

logger = logging.getLogger(__name__)


class Pipeline:
    """按模式驱动状态转移，依次执行各阶段处理器."""

    def run(self, ctx: PipelineContext) -> PipelineContext:
        transitions = TRANSITIONS[ctx.mode]
        logger.info("Pipeline 启动  mode=%s", ctx.mode.value)

        while ctx.state != PipelineState.DONE:
            handler = HANDLERS.get(ctx.state)
            if handler is None:
                ctx.error = RuntimeError(f"No handler for state {ctx.state}")
                ctx.state = PipelineState.ERROR
                break

            state_name = ctx.state.name
            logger.debug("进入阶段 %s", state_name)
            t0 = time.perf_counter()

            try:
                handler(ctx)
            except Exception as exc:
                logger.error("阶段 %s 失败: %s", state_name, exc)
                ctx.error = exc
                ctx.state = PipelineState.ERROR
                break

            elapsed = time.perf_counter() - t0
            logger.info("阶段 %s 完成  耗时 %.2fs", state_name, elapsed)

            next_state = transitions.get(ctx.state)
            if next_state is None:
                ctx.error = RuntimeError(
                    f"No transition from {ctx.state} in mode {ctx.mode}"
                )
                ctx.state = PipelineState.ERROR
                break

            ctx.state = next_state

        if ctx.state == PipelineState.ERROR:
            logger.error("Pipeline 异常终止: %s", ctx.error)
        else:
            logger.info("Pipeline 完成  mode=%s", ctx.mode.value)

        return ctx
