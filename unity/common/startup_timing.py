"""Opt-in timing logs for cold-start and first-reply diagnosis."""

from __future__ import annotations

from contextlib import contextmanager
import os
from time import perf_counter
from typing import Iterator

_TRUE_VALUES = {"1", "true", "yes", "on", "debug"}


def startup_timing_enabled() -> bool:
    """Return True when detailed startup timing logs should be emitted."""

    return os.environ.get("UNITY_STARTUP_TIMING", "").strip().lower() in _TRUE_VALUES


def log_startup_timing(logger, message: str, *args) -> None:
    """Emit a gated startup timing log line."""

    if startup_timing_enabled():
        logger.info(message, *args)


@contextmanager
def startup_timing(logger, phase: str, *details: str) -> Iterator[None]:
    """Measure a phase when ``UNITY_STARTUP_TIMING`` is enabled."""

    if not startup_timing_enabled():
        yield
        return

    suffix = f" {' '.join(details)}" if details else ""
    start = perf_counter()
    logger.info("⏱️ [StartupTiming] %s started%s", phase, suffix)
    try:
        yield
    finally:
        logger.info(
            "⏱️ [StartupTiming] %s completed in %.2fs%s",
            phase,
            perf_counter() - start,
            suffix,
        )
