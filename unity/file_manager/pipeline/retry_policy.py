from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Literal

FailureKind = Literal[
    "retry_exhausted",
    "non_retryable",
    "deadline_exceeded",
]


@dataclass(frozen=True)
class RetryDecision:
    should_retry: bool
    failure_kind: FailureKind | None = None


class ResilientRequestPolicy:
    """Typed retry policy with deadlines, backoff, and jitter."""

    def __init__(
        self,
        *,
        max_retries: int = 3,
        retry_delay_seconds: float = 3.0,
        backoff_multiplier: float = 2.0,
        max_backoff_seconds: float | None = 60.0,
        jitter_ratio: float = 0.1,
        deadline_seconds: float | None = None,
        retry_mode: Literal["all_errors", "transient_only"] = "transient_only",
    ) -> None:
        self.max_retries = max(int(max_retries), 0)
        self.retry_delay_seconds = max(float(retry_delay_seconds), 0.0)
        self.backoff_multiplier = max(float(backoff_multiplier), 1.0)
        self.max_backoff_seconds = (
            None
            if max_backoff_seconds is None
            else max(float(max_backoff_seconds), 0.0)
        )
        self.jitter_ratio = max(float(jitter_ratio), 0.0)
        self.deadline_seconds = (
            None if deadline_seconds is None else max(float(deadline_seconds), 0.0)
        )
        self.retry_mode = retry_mode

    @classmethod
    def from_config(cls, config) -> "ResilientRequestPolicy":
        return cls(
            max_retries=getattr(config, "max_retries", 3),
            retry_delay_seconds=getattr(config, "retry_delay_seconds", 3.0),
            backoff_multiplier=getattr(config, "backoff_multiplier", 2.0),
            max_backoff_seconds=getattr(config, "max_backoff_seconds", 60.0),
            jitter_ratio=getattr(config, "jitter_ratio", 0.1),
            deadline_seconds=getattr(config, "deadline_seconds", None),
            retry_mode=getattr(config, "retry_mode", "transient_only"),
        )

    def check_retry(
        self,
        exc: BaseException,
        *,
        attempt_index: int,
        started_at: float,
    ) -> RetryDecision:
        now = time.perf_counter()
        if (
            self.deadline_seconds is not None
            and (now - started_at) >= self.deadline_seconds
        ):
            return RetryDecision(False, "deadline_exceeded")
        if attempt_index >= self.max_retries:
            return RetryDecision(False, "retry_exhausted")
        if self.retry_mode == "transient_only" and not is_retryable_exception(exc):
            return RetryDecision(False, "non_retryable")
        return RetryDecision(True, None)

    def compute_delay(self, *, attempt_index: int) -> float:
        delay = self.retry_delay_seconds * (self.backoff_multiplier**attempt_index)
        if self.max_backoff_seconds is not None:
            delay = min(delay, self.max_backoff_seconds)
        if delay <= 0 or self.jitter_ratio <= 0:
            return delay
        jitter_span = delay * self.jitter_ratio
        jitter = random.uniform(-jitter_span, jitter_span)
        return max(delay + jitter, 0.0)


def is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, OSError):
        return True

    text = str(exc).strip().lower()
    if not text:
        return False

    return any(
        token in text
        for token in (
            "timed out",
            "timeout",
            "temporarily unavailable",
            "temporary failure",
            "connection reset",
            "connection aborted",
            "connection refused",
            "broken pipe",
            "rate limit",
            "too many requests",
            "429",
            "408",
            "500",
            "502",
            "503",
            "504",
        )
    )
