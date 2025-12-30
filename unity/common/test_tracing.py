"""
Test Tracing Support
====================

Provides OpenTelemetry trace context propagation for pytest tests.

When enabled, each test gets a unique trace_id that propagates to all HTTP
calls, allowing correlation between pytest logs and Orchestra API traces.

The trace_id appears in:
- Unity pytest logs: "TRACE_ID=<trace_id>"
- Orchestra trace files: trace_id in filename and content

This enables deterministic alignment of test runs with their API calls,
even during parallel test execution.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

# Module-level state
_TRACER_INITIALIZED = False
_TRACER = None


def _is_tracing_enabled() -> bool:
    """Check if test tracing is enabled via environment variable."""
    return os.environ.get("UNITY_TEST_TRACING", "true").lower() in ("true", "1", "yes")


def _initialize_tracer() -> None:
    """Initialize OpenTelemetry tracer and HTTP client instrumentation.

    This sets up:
    1. A TracerProvider with cryptographically random IDs
    2. HTTPXClientInstrumentor to auto-inject traceparent into httpx requests
    3. AioHTTPClientInstrumentor for aiohttp requests (used by unillm)

    Called once on first use.
    """
    global _TRACER_INITIALIZED, _TRACER

    if _TRACER_INITIALIZED:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.id_generator import IdGenerator
        import secrets

        # Custom ID generator using secrets module for cryptographically random IDs
        # This avoids interference from tests that seed Python's random module
        class SecureIdGenerator(IdGenerator):
            def generate_span_id(self) -> int:
                return secrets.randbits(64)

            def generate_trace_id(self) -> int:
                return secrets.randbits(128)

        # Create a tracer provider with service name and secure ID generator
        resource = Resource.create({"service.name": "unity-tests"})
        provider = TracerProvider(resource=resource, id_generator=SecureIdGenerator())
        trace.set_tracer_provider(provider)

        _TRACER = trace.get_tracer("unity.tests")

        # Instrument HTTP clients to propagate trace context
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            HTTPXClientInstrumentor().instrument()
            logger.debug("HTTPXClientInstrumentor enabled for trace propagation")
        except ImportError:
            logger.debug("opentelemetry-instrumentation-httpx not available")

        try:
            from opentelemetry.instrumentation.aiohttp_client import (
                AioHTTPClientInstrumentor,
            )

            AioHTTPClientInstrumentor().instrument()
            logger.debug("AioHTTPClientInstrumentor enabled for trace propagation")
        except ImportError:
            logger.debug("opentelemetry-instrumentation-aiohttp-client not available")

        _TRACER_INITIALIZED = True
        logger.debug("OpenTelemetry test tracing initialized")

    except ImportError as e:
        logger.debug(f"OpenTelemetry not available, test tracing disabled: {e}")
        _TRACER_INITIALIZED = True  # Don't retry
        _TRACER = None


def get_tracer():
    """Get the test tracer, initializing if needed."""
    if not _TRACER_INITIALIZED:
        _initialize_tracer()
    return _TRACER


@contextmanager
def trace_test(
    test_name: str,
) -> Generator[tuple[str | None, "Span" | None], None, None]:
    """Context manager that wraps a test in a NEW trace (not a child span).

    Each test gets its own unique trace_id, which propagates to all HTTP calls
    made during the test.

    Args:
        test_name: The name of the test (e.g., "test_create_contact")

    Yields:
        Tuple of (trace_id, span) - trace_id is a 32-char hex string, or None if tracing disabled

    Example:
        with trace_test("test_foo") as (trace_id, span):
            if trace_id:
                print(f"TRACE_ID={trace_id}")
            # Run test code - all HTTP calls will propagate trace_id
    """
    if not _is_tracing_enabled():
        yield None, None
        return

    tracer = get_tracer()
    if tracer is None:
        yield None, None
        return

    # Use a fresh context to ensure we get a NEW trace, not a child of any existing span
    from opentelemetry import context
    from opentelemetry.trace import set_span_in_context

    # Create a clean context with no parent span
    clean_ctx = context.Context()

    # Start the span in the clean context
    span = tracer.start_span(f"test:{test_name}", context=clean_ctx)
    token = context.attach(set_span_in_context(span, clean_ctx))

    try:
        trace_id = f"{span.get_span_context().trace_id:032x}"
        yield trace_id, span
    finally:
        span.end()
        context.detach(token)


def get_current_trace_id() -> str | None:
    """Get the current trace_id if inside an active span.

    Returns:
        32-character hex trace_id, or None if no active span
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            return f"{ctx.trace_id:032x}"
    except Exception:
        pass
    return None
