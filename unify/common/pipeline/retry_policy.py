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

    _KNOWN_FIELDS = frozenset(
        {
            "max_retries",
            "retry_delay_seconds",
            "backoff_multiplier",
            "max_backoff_seconds",
            "jitter_ratio",
            "deadline_seconds",
            "retry_mode",
        },
    )

    @classmethod
    def from_config(cls, config) -> "ResilientRequestPolicy":
        raw = (
            config.model_dump()
            if hasattr(config, "model_dump")
            else {k: v for k, v in vars(config).items() if not k.startswith("_")}
        )
        return cls(**{k: v for k, v in raw.items() if k in cls._KNOWN_FIELDS})

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


def _status_code_of(exc: BaseException) -> int | None:
    """The HTTP status an exception carries, if it carries one.

    Read from the exception rather than from its rendered message. Several
    client libraries expose it under different names, and a response object is
    the most reliable of them, so all three are tried before giving up.
    """
    for attribute in ("status_code", "status", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    return None


def is_retryable_exception(exc: BaseException) -> bool:
    """Whether retrying *exc* could plausibly succeed.

    Decided from the exception's type and status where either is available, and
    only then from its text. Reading the status matters because the two answers
    disagree in both directions: a deterministic 400 was retried to the cap
    because no token in its message looked permanent, while any error whose text
    merely contained "500" -- a URL, a row count, an id -- was retried as though
    the server had faulted.

    A refusal the caller caused will refuse identically every time. Retrying it
    is not merely wasted work: the message is never acked, so it is redelivered,
    and one bad request can occupy a worker fleet indefinitely.
    """
    # A live-attempt collision is neither a fault nor retryable work: another
    # attempt already holds the lease and is making progress. Retrying into it
    # only contends, and treating it as a permanent failure discards work that is
    # about to succeed -- so it is reported as its own thing and the caller
    # yields.
    if type(exc).__name__ == "DuplicateLiveAttempt":
        return False

    status = _status_code_of(exc)
    if status is not None:
        # 408 and 429 are the two client-side codes worth retrying; every other
        # 4xx states something about the request that a repeat cannot change.
        if 400 <= status < 500:
            return status in (408, 429)
        return status >= 500

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
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "internal server error",
        )
    )
    # Bare status numbers used to live in this list. They were the only way to
    # spot a server fault before the status was read from the exception, and
    # they matched anything else that happened to contain the digits -- a row
    # count of 500, an id ending 502, a URL with a port. Phrases cannot collide
    # that way, and the status check above is now the reliable path.
