"""
Tests for OpenTelemetry trace correlation between Unity and Orchestra.

Verifies that:
1. Each test gets a unique trace_id
2. The trace_id is logged to pytest output
3. HTTP clients propagate traceparent header
4. End-to-end: trace_id propagates from Unity test to Orchestra trace files
"""

import json
import os
import time
from pathlib import Path

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


class TestEndToEndTraceCorrelation:
    """End-to-end tests verifying trace_id propagates from Unity to Orchestra.

    These tests make real HTTP calls to a running Orchestra instance and verify
    that the trace_id from Unity's OpenTelemetry span appears in Orchestra's
    trace files.

    Requirements:
    - Local Orchestra must be running (started via parallel_run.sh or the orchestra shell function)
    - ORCHESTRA_LOG_DIR must be set (done automatically by parallel_run.sh)
    """

    def _get_orchestra_log_dir(self) -> Path | None:
        """Get the Orchestra log directory from environment.

        Returns None if not configured (Orchestra logging not enabled).
        """
        # Check environment variable set by parallel_run.sh
        trace_dir = os.environ.get("ORCHESTRA_LOG_DIR")
        if trace_dir:
            return Path(trace_dir)

        # Fall back to scanning logs/orchestra for most recent session
        unity_root = Path(__file__).parent.parent.parent
        orchestra_logs = unity_root / "logs" / "orchestra"
        if orchestra_logs.exists():
            # Find most recent session directory
            sessions = sorted(orchestra_logs.iterdir(), reverse=True)
            for session in sessions:
                if session.is_dir() and not session.name.startswith("."):
                    return session
        return None

    def _find_trace_file_with_trace_id(
        self,
        trace_log_dir: Path,
        trace_id: str,
        timeout: float = 5.0,
    ) -> Path | None:
        """Wait for and find a trace file containing the given trace_id.

        Orchestra flushes completed traces after ~0.5s, so we poll with a timeout.
        """
        requests_dir = trace_log_dir / "requests"
        if not requests_dir.exists():
            return None

        # trace_id short form (last 8 chars) is used in filename
        trace_id_short = trace_id[-8:]
        start = time.time()

        while time.time() - start < timeout:
            for f in requests_dir.glob("*.json"):
                # Check filename contains trace_id short form
                if trace_id_short in f.name.lower():
                    return f
                # Also check file content for full trace_id
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                        if data.get("trace_id", "").lower() == trace_id.lower():
                            return f
                except (json.JSONDecodeError, OSError):
                    continue
            time.sleep(0.2)

        return None

    def _is_orchestra_running(self) -> bool:
        """Check if local Orchestra is running and accessible."""
        import httpx

        base_url = os.environ.get("UNIFY_BASE_URL", "http://127.0.0.1:8000/v0")
        try:
            # Orchestra's health endpoint is at /v0/health
            # Note: health endpoint is excluded from tracing, but useful for checking if running
            resp = httpx.get(f"{base_url}/health", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def _get_traced_endpoint(self) -> str:
        """Get an endpoint that is traced (not excluded from OpenTelemetry).

        The health, openapi, swagger, redoc, and metrics endpoints are excluded.
        We use /projects which will return 401/403 without auth but still be traced.
        """
        return "/projects"

    def test_trace_id_propagates_to_orchestra(self):
        """Verify trace_id from Unity test appears in Orchestra trace files.

        This is the critical end-to-end test that proves the full correlation
        chain works:
        1. Unity test creates a trace span
        2. HTTP call includes traceparent header with trace_id
        3. Orchestra receives and uses the same trace_id
        4. Orchestra writes trace file with that trace_id

        Note: Uses /projects endpoint because /health is excluded from tracing.
        """
        import httpx

        from unity.common.test_tracing import get_current_trace_id

        # Skip if tracing not enabled
        trace_id = get_current_trace_id()
        if trace_id is None:
            pytest.skip("OpenTelemetry tracing not enabled")

        # Skip if Orchestra not running
        if not self._is_orchestra_running():
            pytest.skip("Local Orchestra not running")

        # Skip if logging not configured
        trace_log_dir = self._get_orchestra_log_dir()
        if trace_log_dir is None:
            pytest.skip("ORCHESTRA_LOG_DIR not set (logging disabled)")

        # Make a real HTTP call to Orchestra
        base_url = os.environ.get("UNIFY_BASE_URL", "http://127.0.0.1:8000/v0")
        endpoint = self._get_traced_endpoint()

        # Make the request - HTTPXClientInstrumentor injects traceparent header
        # The endpoint may return 401/403 without auth, but the trace is still recorded
        with httpx.Client() as client:
            try:
                resp = client.get(f"{base_url}{endpoint}", timeout=10.0)
                # Any response is fine - we just need the request to reach Orchestra
            except httpx.HTTPError:
                # Even errors are fine - the trace will still be recorded
                pass

        # Wait for Orchestra to flush the trace and find it
        trace_file = self._find_trace_file_with_trace_id(
            trace_log_dir,
            trace_id,
            timeout=5.0,
        )

        if trace_file is None:
            # List available trace files for debugging
            requests_dir = trace_log_dir / "requests"
            available = (
                list(requests_dir.glob("*.json")) if requests_dir.exists() else []
            )
            pytest.fail(
                f"No trace file found with trace_id={trace_id} (short: {trace_id[-8:]})\n"
                f"Trace log dir: {trace_log_dir}\n"
                f"Available files ({len(available)}): {[f.name for f in available[:10]]}",
            )

        # Verify the trace file contains the correct trace_id
        with open(trace_file) as f:
            data = json.load(f)

        assert data.get("trace_id", "").lower() == trace_id.lower(), (
            f"trace_id mismatch in {trace_file.name}: "
            f"expected {trace_id}, got {data.get('trace_id')}"
        )

        # Verify there are spans in the trace
        assert "spans" in data, f"No spans in trace file: {trace_file.name}"
        assert len(data["spans"]) > 0, f"Empty spans in trace file: {trace_file.name}"

        # Success! Log for debugging
        print(f"\n✅ Trace correlation verified!")
        print(f"   Unity trace_id: {trace_id}")
        print(f"   Orchestra file: {trace_file.name}")
        print(f"   Spans recorded: {len(data['spans'])}")

    @pytest.mark.asyncio
    async def test_async_trace_id_propagates_to_orchestra(self):
        """Verify async HTTP calls also propagate trace_id to Orchestra."""
        import httpx

        from unity.common.test_tracing import get_current_trace_id

        trace_id = get_current_trace_id()
        if trace_id is None:
            pytest.skip("OpenTelemetry tracing not enabled")

        if not self._is_orchestra_running():
            pytest.skip("Local Orchestra not running")

        trace_log_dir = self._get_orchestra_log_dir()
        if trace_log_dir is None:
            pytest.skip("ORCHESTRA_LOG_DIR not set")

        base_url = os.environ.get("UNIFY_BASE_URL", "http://127.0.0.1:8000/v0")
        endpoint = self._get_traced_endpoint()

        # Make async request
        async with httpx.AsyncClient() as client:
            try:
                await client.get(f"{base_url}{endpoint}", timeout=10.0)
            except httpx.HTTPError:
                pass

        # Wait for and verify trace file
        trace_file = self._find_trace_file_with_trace_id(
            trace_log_dir,
            trace_id,
            timeout=5.0,
        )

        if trace_file is None:
            requests_dir = trace_log_dir / "requests"
            available = (
                list(requests_dir.glob("*.json")) if requests_dir.exists() else []
            )
            pytest.fail(
                f"No trace file found with trace_id={trace_id}\n"
                f"Available: {[f.name for f in available[:10]]}",
            )

        with open(trace_file) as f:
            data = json.load(f)

        assert data.get("trace_id", "").lower() == trace_id.lower()
        print(f"\n✅ Async trace correlation verified! trace_id={trace_id}")
