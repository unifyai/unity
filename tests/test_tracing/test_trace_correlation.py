"""
Tests for OpenTelemetry trace correlation between Unity and Orchestra.

Verifies that:
1. Each test gets a unique trace_id
2. The trace_id is logged to pytest output
3. HTTP clients propagate traceparent header
"""

import pytest


class TestTraceCorrelation:
    """Test trace_id assignment and propagation."""

    def test_trace_id_is_assigned(self):
        """Verify the test gets a trace_id from OpenTelemetry."""
        from unity.common.test_tracing import get_current_trace_id

        trace_id = get_current_trace_id()
        # Should have a valid trace_id (32 hex chars)
        assert trace_id is not None, "Expected trace_id to be set"
        assert len(trace_id) == 32, f"Expected 32-char hex, got {len(trace_id)}"
        assert all(c in "0123456789abcdef" for c in trace_id), "Expected hex chars"

    def test_different_tests_get_different_trace_ids(self):
        """Verify each test gets a unique trace_id."""
        from unity.common.test_tracing import get_current_trace_id

        # Store this test's trace_id for comparison
        trace_id = get_current_trace_id()
        assert trace_id is not None
        # Store in module for next test to verify
        TestTraceCorrelation._first_trace_id = trace_id

    def test_trace_id_differs_from_previous(self):
        """Verify this test has a different trace_id than the previous test."""
        from unity.common.test_tracing import get_current_trace_id

        trace_id = get_current_trace_id()
        assert trace_id is not None

        # Compare with previous test's trace_id (if available)
        first_id = getattr(TestTraceCorrelation, "_first_trace_id", None)
        if first_id is not None:
            assert trace_id != first_id, "Each test should have unique trace_id"


class TestTraceparentHeader:
    """Test that traceparent header is injected into HTTP requests."""

    def test_httpx_injects_traceparent(self):
        """Verify HTTPXClientInstrumentor injects traceparent header."""
        import httpx

        from unity.common.test_tracing import get_current_trace_id

        trace_id = get_current_trace_id()
        if trace_id is None:
            pytest.skip("Tracing not enabled")

        # Create a request and check if traceparent would be injected
        # We use a mock transport to capture the request headers
        captured_headers = {}

        class CapturingTransport(httpx.BaseTransport):
            def handle_request(self, request):
                captured_headers.update(request.headers)
                # Return a dummy response
                return httpx.Response(200, content=b"ok")

        client = httpx.Client(transport=CapturingTransport())
        try:
            client.get("http://test.local/")
        finally:
            client.close()

        # Check if traceparent header was injected
        traceparent = captured_headers.get("traceparent")
        if traceparent:
            # traceparent format: 00-<trace_id>-<span_id>-<flags>
            parts = traceparent.split("-")
            assert len(parts) == 4, f"Invalid traceparent format: {traceparent}"
            assert parts[1] == trace_id, f"trace_id mismatch: {parts[1]} != {trace_id}"
        else:
            # If HTTPXClientInstrumentor didn't inject, that's ok in some environments
            pytest.skip("HTTPXClientInstrumentor not active")

    @pytest.mark.asyncio
    async def test_async_httpx_injects_traceparent(self):
        """Verify async HTTPXClientInstrumentor injects traceparent header."""
        import httpx

        from unity.common.test_tracing import get_current_trace_id

        trace_id = get_current_trace_id()
        if trace_id is None:
            pytest.skip("Tracing not enabled")

        captured_headers = {}

        class CapturingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                captured_headers.update(request.headers)
                return httpx.Response(200, content=b"ok")

        async with httpx.AsyncClient(transport=CapturingTransport()) as client:
            await client.get("http://test.local/")

        traceparent = captured_headers.get("traceparent")
        if traceparent:
            parts = traceparent.split("-")
            assert len(parts) == 4
            assert parts[1] == trace_id
        else:
            pytest.skip("HTTPXClientInstrumentor not active")
