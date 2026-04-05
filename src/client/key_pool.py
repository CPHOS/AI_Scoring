"""线程安全的 API Key 轮询池."""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class KeyPool:
    """管理多个 API Key 的线程安全轮询池.

    - ``acquire()`` 按轮询顺序返回下一个可用 key。
    - ``report_failure(key)`` 将 *key* 移到队尾，下次优先使用其他 key。
    - 所有操作均为线程安全。
    """

    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise ValueError("至少需要一个 API key")
        # 去重并保持顺序
        seen: set[str] = set()
        unique: list[str] = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                unique.append(k)
        self._keys = unique
        self._index = 0
        self._lock = threading.Lock()
        logger.debug("KeyPool 初始化: %d 个 key", len(self._keys))

    @property
    def size(self) -> int:
        return len(self._keys)

    def acquire(self) -> str:
        """获取下一个 key（轮询）."""
        with self._lock:
            key = self._keys[self._index % len(self._keys)]
            self._index += 1
            return key

    def report_failure(self, key: str) -> None:
        """报告某个 key 失败，将其移到队列末尾以便优先使用其他 key."""
        with self._lock:
            if key in self._keys and len(self._keys) > 1:
                self._keys.remove(key)
                self._keys.append(key)
                logger.info("Key ...%s 失败，已移至队尾", key[-6:])
