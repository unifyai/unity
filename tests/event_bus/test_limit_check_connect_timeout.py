"""Tests for shared HTTP client in spending limit checks.

Verifies that limit checks use a shared httpx.AsyncClient with connection
pooling, so that TCP+TLS connections are reused across checks. This prevents
ConnectTimeout under event loop congestion (e.g., during video calls).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from unillm.limit_hooks import LimitCheckRequest


def _patch_context(*, org_id=None):
    """Patch API key, base URL, and SESSION_DETAILS for limit checks."""

    class _Ctx:
        def __init__(self):
            self._stack = []

        def __enter__(self):
            import contextlib

            stack = contextlib.ExitStack()
            stack.enter_context(
                patch("unity.spending_limits._get_api_key", return_value="test-key"),
            )
            stack.enter_context(
                patch(
                    "unity.spending_limits._get_base_url",
                    return_value="http://test/v0",
                ),
            )
            mock_session = stack.enter_context(
                patch("unity.session_details.SESSION_DETAILS"),
            )
            mock_session.assistant.agent_id = 1
            mock_session.user_id = "user_1"
            mock_session.org_id = org_id
            mock_session.assistant.timezone = "UTC"
            self._stack.append(stack)
            return mock_session

        def __exit__(self, *exc):
            self._stack.pop().close()

    return _Ctx()


def _make_mock_client():
    """Create a mock httpx.AsyncClient that returns valid spend responses."""
    mock = MagicMock(spec=httpx.AsyncClient)
    mock.is_closed = False

    resp = MagicMock()
    resp.json.return_value = {
        "cumulative_spend": 5.0,
        "limit": 100.0,
        "credit_balance": 50.0,
    }
    resp.raise_for_status = MagicMock()
    resp.status_code = 200

    mock.get = AsyncMock(return_value=resp)
    mock.post = AsyncMock(return_value=resp)
    return mock


class TestSharedHttpClient:
    """Verifies that limit checks reuse a single shared httpx.AsyncClient
    rather than creating a new one per request."""

    @pytest.mark.asyncio
    async def test_single_client_for_personal_context(self):
        """Personal context (assistant + user) should use ONE shared client,
        not create two separate clients."""
        import unity.spending_limits as sl

        mock_client = _make_mock_client()

        # Reset module-level client so _get_http_client creates a fresh one
        sl._http_client = None

        with _patch_context():
            with patch.object(
                sl,
                "_get_http_client",
                return_value=mock_client,
            ) as mock_getter:
                await sl.check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        # _get_http_client is called once per check (assistant + user = 2),
        # but it returns the SAME client object both times
        assert mock_getter.call_count == 2
        # Both checks used the same client instance
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_single_client_for_org_context(self):
        """Org context (assistant + member + org) should use ONE shared client,
        not create three separate clients."""
        import unity.spending_limits as sl

        mock_client = _make_mock_client()
        sl._http_client = None

        with _patch_context(org_id=789):
            with patch.object(
                sl,
                "_get_http_client",
                return_value=mock_client,
            ):
                await sl.check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        # All three checks used the same client instance
        assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_concurrent_calls_share_client(self):
        """Multiple concurrent limit checks should all share the same client.

        5 concurrent LLM calls should NOT create 10 separate connections —
        they should all reuse a single pooled client.
        """
        import unity.spending_limits as sl

        mock_client = _make_mock_client()
        sl._http_client = None

        with _patch_context():
            with patch.object(
                sl,
                "_get_http_client",
                return_value=mock_client,
            ) as mock_getter:
                tasks = [
                    sl.check_spending_limits_callback(
                        LimitCheckRequest(model="gpt-4", endpoint="test"),
                    )
                    for _ in range(5)
                ]
                results = await asyncio.gather(*tasks)

        assert all(r.allowed for r in results)
        # 5 calls × 2 checks = 10 calls to _get_http_client,
        # but all return the same client object
        assert mock_getter.call_count == 10
        # All 10 HTTP GETs went through the SAME client (connection pool)
        assert mock_client.get.call_count == 10

    @pytest.mark.asyncio
    async def test_get_http_client_reuses_existing(self):
        """_get_http_client should return the same instance on repeated calls."""
        import unity.spending_limits as sl

        sl._http_client = None

        client1 = sl._get_http_client()
        client2 = sl._get_http_client()
        client3 = sl._get_http_client()

        assert client1 is client2
        assert client2 is client3

        # Cleanup
        await client1.aclose()
        sl._http_client = None

    @pytest.mark.asyncio
    async def test_get_http_client_recreates_if_closed(self):
        """_get_http_client should create a new client if the previous one was closed."""
        import unity.spending_limits as sl

        sl._http_client = None

        client1 = sl._get_http_client()
        await client1.aclose()

        client2 = sl._get_http_client()

        assert client1 is not client2
        assert not client2.is_closed

        # Cleanup
        await client2.aclose()
        sl._http_client = None

    @pytest.mark.asyncio
    async def test_get_http_client_recreates_on_event_loop_change(self):
        """_get_http_client should create a new client when the event loop changes.

        Without this, an httpx.AsyncClient created on loop A will carry
        asyncio.Event / asyncio.Lock objects bound to loop A. Using that
        client on loop B raises:
            RuntimeError: <asyncio.locks.Event ...> is bound to a different event loop
        """
        import unity.spending_limits as sl

        sl._http_client = None
        sl._http_client_loop = None

        client1 = sl._get_http_client()
        assert sl._http_client is client1
        assert sl._http_client_loop is asyncio.get_running_loop()

        # Simulate a different event loop (e.g., new test run or loop recreation)
        fake_loop = asyncio.new_event_loop()
        try:
            sl._http_client_loop = fake_loop
            client2 = sl._get_http_client()

            assert client2 is not client1
            assert sl._http_client_loop is asyncio.get_running_loop()
        finally:
            fake_loop.close()
            await client1.aclose()
            await client2.aclose()
            sl._http_client = None
            sl._http_client_loop = None

    @pytest.mark.asyncio
    async def test_no_async_context_manager_per_request(self):
        """The check functions should NOT use `async with httpx.AsyncClient()`.

        This is the pattern that caused the production ConnectTimeout — each
        check created and tore down its own client. Verify it no longer happens
        by ensuring httpx.AsyncClient is never instantiated during a check.
        """
        import unity.spending_limits as sl

        mock_client = _make_mock_client()
        sl._http_client = None

        instantiation_count = 0
        original_init = httpx.AsyncClient.__init__

        with _patch_context():
            with patch.object(sl, "_get_http_client", return_value=mock_client):
                with patch("httpx.AsyncClient") as mock_cls:
                    mock_cls.return_value = mock_client

                    await sl.check_spending_limits_callback(
                        LimitCheckRequest(model="gpt-4", endpoint="test"),
                    )

                    # httpx.AsyncClient() should NOT have been called directly
                    # (it's only called inside _get_http_client, which we patched)
                    assert mock_cls.call_count == 0, (
                        f"httpx.AsyncClient was instantiated {mock_cls.call_count} time(s) "
                        "during a limit check. The shared client should be used instead."
                    )
