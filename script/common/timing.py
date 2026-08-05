"""简单计时工具。"""

from __future__ import annotations

import time

__all__ = ["Timer"]


class Timer:
    """Context manager measuring wall-clock seconds: ``with Timer() as t: ...``."""

    def __init__(self) -> None:
        self._start: float | None = None
        self.seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.seconds = time.perf_counter() - self._start
        return False
