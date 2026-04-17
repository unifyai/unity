"""Targeted resilience tests for ResilientRequestPolicy.

Validates retry decisions, backoff computation, deadline enforcement,
failure classification, and from_config field filtering.
"""

from __future__ import annotations

import time

import pytest

from unity.common.pipeline import (
    ResilientRequestPolicy,
    is_retryable_exception,
)


class TestRetryDecisions:
    """Verify that check_retry produces the correct decision for each scenario."""

    def test_allows_retry_on_first_transient_failure(self):
        policy = ResilientRequestPolicy(max_retries=3, retry_delay_seconds=1.0)
        decision = policy.check_retry(
            ConnectionError("reset"),
            attempt_index=0,
            started_at=time.perf_counter(),
        )
        assert decision.should_retry is True
        assert decision.failure_kind is None

    def test_exhausts_retries_at_max(self):
        policy = ResilientRequestPolicy(max_retries=2)
        decision = policy.check_retry(
            ConnectionError("reset"),
            attempt_index=2,
            started_at=time.perf_counter(),
        )
        assert decision.should_retry is False
        assert decision.failure_kind == "retry_exhausted"

    def test_rejects_non_retryable_in_transient_only_mode(self):
        policy = ResilientRequestPolicy(
            max_retries=5,
            retry_mode="transient_only",
        )
        decision = policy.check_retry(
            ValueError("bad input"),
            attempt_index=0,
            started_at=time.perf_counter(),
        )
        assert decision.should_retry is False
        assert decision.failure_kind == "non_retryable"

    def test_retries_any_error_in_all_errors_mode(self):
        policy = ResilientRequestPolicy(
            max_retries=5,
            retry_mode="all_errors",
        )
        decision = policy.check_retry(
            ValueError("bad input"),
            attempt_index=0,
            started_at=time.perf_counter(),
        )
        assert decision.should_retry is True

    def test_deadline_exceeded_stops_retry(self):
        policy = ResilientRequestPolicy(
            max_retries=10,
            deadline_seconds=0.001,
        )
        started = time.perf_counter() - 1.0
        decision = policy.check_retry(
            ConnectionError("reset"),
            attempt_index=0,
            started_at=started,
        )
        assert decision.should_retry is False
        assert decision.failure_kind == "deadline_exceeded"

    def test_deadline_not_exceeded_allows_retry(self):
        policy = ResilientRequestPolicy(
            max_retries=10,
            deadline_seconds=60.0,
        )
        decision = policy.check_retry(
            ConnectionError("reset"),
            attempt_index=0,
            started_at=time.perf_counter(),
        )
        assert decision.should_retry is True

    def test_zero_max_retries_never_retries(self):
        policy = ResilientRequestPolicy(max_retries=0)
        decision = policy.check_retry(
            ConnectionError("reset"),
            attempt_index=0,
            started_at=time.perf_counter(),
        )
        assert decision.should_retry is False
        assert decision.failure_kind == "retry_exhausted"


class TestBackoffComputation:
    """Verify exponential backoff with jitter and ceiling."""

    def test_exponential_growth(self):
        policy = ResilientRequestPolicy(
            retry_delay_seconds=1.0,
            backoff_multiplier=2.0,
            jitter_ratio=0.0,
        )
        assert policy.compute_delay(attempt_index=0) == 1.0
        assert policy.compute_delay(attempt_index=1) == 2.0
        assert policy.compute_delay(attempt_index=2) == 4.0
        assert policy.compute_delay(attempt_index=3) == 8.0

    def test_max_backoff_ceiling(self):
        policy = ResilientRequestPolicy(
            retry_delay_seconds=1.0,
            backoff_multiplier=10.0,
            max_backoff_seconds=5.0,
            jitter_ratio=0.0,
        )
        assert policy.compute_delay(attempt_index=0) == 1.0
        assert policy.compute_delay(attempt_index=1) == 5.0
        assert policy.compute_delay(attempt_index=5) == 5.0

    def test_jitter_stays_within_bounds(self):
        policy = ResilientRequestPolicy(
            retry_delay_seconds=10.0,
            backoff_multiplier=1.0,
            jitter_ratio=0.1,
        )
        for _ in range(100):
            delay = policy.compute_delay(attempt_index=0)
            assert 9.0 <= delay <= 11.0, f"Delay {delay} outside jitter bounds"

    def test_zero_delay_produces_zero(self):
        policy = ResilientRequestPolicy(
            retry_delay_seconds=0.0,
            jitter_ratio=0.0,
        )
        assert policy.compute_delay(attempt_index=0) == 0.0


class TestRetryableClassification:
    """Verify is_retryable_exception correctly classifies error types."""

    @pytest.mark.parametrize(
        "exc",
        [
            TimeoutError("timed out"),
            ConnectionError("connection reset"),
            ConnectionRefusedError("refused"),
            ConnectionResetError("reset by peer"),
            OSError("network unreachable"),
        ],
    )
    def test_transient_exceptions_are_retryable(self, exc):
        assert is_retryable_exception(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("bad value"),
            TypeError("wrong type"),
            KeyError("missing"),
            RuntimeError("unexpected"),
        ],
    )
    def test_logic_exceptions_are_not_retryable(self, exc):
        assert is_retryable_exception(exc) is False

    @pytest.mark.parametrize(
        "message",
        [
            "HTTP 429 too many requests",
            "rate limit exceeded",
            "HTTP 503 service unavailable",
            "HTTP 502 bad gateway",
            "HTTP 500 internal server error",
            "request timed out after 30s",
            "connection reset by peer",
            "temporarily unavailable",
        ],
    )
    def test_retryable_error_messages(self, message):
        assert is_retryable_exception(RuntimeError(message)) is True

    def test_empty_error_message_is_not_retryable(self):
        assert is_retryable_exception(RuntimeError("")) is False


class TestFromConfig:
    """Verify from_config filters unknown fields and builds correctly."""

    def test_filters_unknown_fields(self):
        class FakeConfig:
            def model_dump(self):
                return {
                    "max_retries": 5,
                    "retry_delay_seconds": 2.0,
                    "fail_fast": True,
                    "unknown_field": "garbage",
                }

        policy = ResilientRequestPolicy.from_config(FakeConfig())
        assert policy.max_retries == 5
        assert policy.retry_delay_seconds == 2.0

    def test_works_with_vars_fallback(self):
        class PlainConfig:
            def __init__(self):
                self.max_retries = 7
                self.backoff_multiplier = 3.0
                self.fail_fast = True

        policy = ResilientRequestPolicy.from_config(PlainConfig())
        assert policy.max_retries == 7
        assert policy.backoff_multiplier == 3.0
