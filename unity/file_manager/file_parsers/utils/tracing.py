from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, Optional, TypeVar

from unity.file_manager.file_parsers.types.contracts import (
    FileParseTrace,
    ParseError,
    StepStatus,
    StepTrace,
)

T = TypeVar("T")


@contextmanager
def traced_step(
    trace: FileParseTrace,
    *,
    name: str,
    counters: Optional[Dict[str, int]] = None,
) -> Iterator[StepTrace]:
    """
    Context manager that records a StepTrace and appends it to a FileParseTrace.

    Usage
    -----
    ```python
    with traced_step(trace, name="convert") as step:
        ...
        step.counters["tables"] = 3
    ```
    """
    step = StepTrace(
        name=name,
        status=StepStatus.SUCCESS,
        counters=dict(counters or {}),
    )
    t0 = time.perf_counter()
    try:
        yield step
    except Exception as e:
        step.status = StepStatus.FAILED
        step.error = ParseError(
            code="step_failed",
            message=str(e),
            exception_type=type(e).__name__,
        )
        raise
    finally:
        step.duration_ms = (time.perf_counter() - t0) * 1000.0
        trace.steps.append(step)


def trace_degraded(
    step: StepTrace,
    *,
    code: str,
    message: str,
    exc: Optional[BaseException] = None,
) -> None:
    """Mark an already-created StepTrace as degraded (best-effort failure)."""
    step.status = StepStatus.DEGRADED
    step.error = ParseError(
        code=code,
        message=message,
        exception_type=type(exc).__name__ if exc is not None else None,
    )


def safe_call(step: StepTrace, fn: Callable[[], T], *, code: str, default: T) -> T:
    """
    Run `fn` and if it fails, mark the step as degraded and return `default`.

    This is the preferred pattern for non-critical enrichment steps (e.g. summaries)
    where we still want to produce a valid FileParseResult.
    """
    try:
        return fn()
    except Exception as e:
        trace_degraded(step, code=code, message=str(e), exc=e)
        return default
