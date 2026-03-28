"""Tests for shared admin client in spending limit checks.

Verifies that limit checks use a shared ``unify.AsyncAdminClient`` with
connection pooling and retries, so that connections are reused across checks.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unillm.limit_hooks import LimitCheckRequest


def _patch_context(*, org_id=None):
    """Patch API key and SESSION_DETAILS for limit checks."""

    class _Ctx:
        def __init__(self):
            self._stack = []

        def __enter__(self):
            import contextlib

            stack = contextlib.ExitStack()
            stack.enter_context(
                patch("unity.spending_limits._get_api_key", return_value="test-key"),
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


_SPEND_DATA = {
    "cumulative_spend": 5.0,
    "limit": 100.0,
    "credit_balance": 50.0,
}


def _make_mock_client():
    """Create a mock AsyncAdminClient that returns valid spend responses."""
    mock = MagicMock()
    mock.closed = False
    mock.get_assistant_spend = AsyncMock(return_value=_SPEND_DATA)
    mock.get_user_spend = AsyncMock(return_value=_SPEND_DATA)
    mock.get_member_spend = AsyncMock(return_value=_SPEND_DATA)
    mock.get_org_spend = AsyncMock(return_value=_SPEND_DATA)
    mock.notify_limit_reached = AsyncMock(return_value={"notified": False})
    return mock


class TestSharedAdminClient:
    """Verifies that limit checks reuse a single shared AsyncAdminClient
    rather than creating a new one per request."""

    @pytest.mark.asyncio
    async def test_single_client_for_personal_context(self):
        """Personal context (assistant + user) should use ONE shared client."""
        import unity.spending_limits as sl

        mock_client = _make_mock_client()
        sl._admin_client = None

        with _patch_context():
            with patch.object(
                sl,
                "_get_admin_client",
                return_value=mock_client,
            ) as mock_getter:
                await sl.check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert mock_getter.call_count == 2
        mock_client.get_assistant_spend.assert_called_once()
        mock_client.get_user_spend.assert_called_once()

    @pytest.mark.asyncio
    async def test_single_client_for_org_context(self):
        """Org context (assistant + member + org) should use ONE shared client."""
        import unity.spending_limits as sl

        mock_client = _make_mock_client()
        sl._admin_client = None

        with _patch_context(org_id=789):
            with patch.object(
                sl,
                "_get_admin_client",
                return_value=mock_client,
            ):
                await sl.check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        mock_client.get_assistant_spend.assert_called_once()
        mock_client.get_member_spend.assert_called_once()
        mock_client.get_org_spend.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_calls_share_client(self):
        """Multiple concurrent limit checks should all share the same client.

        5 concurrent LLM calls should NOT create 10 separate connections —
        they should all reuse a single pooled client.
        """
        import unity.spending_limits as sl

        mock_client = _make_mock_client()
        sl._admin_client = None

        with _patch_context():
            with patch.object(
                sl,
                "_get_admin_client",
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
        assert mock_getter.call_count == 10
        assert mock_client.get_assistant_spend.call_count == 5
        assert mock_client.get_user_spend.call_count == 5

    @pytest.mark.asyncio
    async def test_get_admin_client_reuses_existing(self):
        """_get_admin_client should return the same instance on repeated calls."""
        import unity.spending_limits as sl

        mock_client = MagicMock()
        mock_client.closed = False
        sl._admin_client = mock_client

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            client1 = sl._get_admin_client()
            client2 = sl._get_admin_client()
            client3 = sl._get_admin_client()

        assert client1 is client2
        assert client2 is client3
        assert client1 is mock_client

        sl._admin_client = None

    @pytest.mark.asyncio
    async def test_get_admin_client_recreates_if_closed(self):
        """_get_admin_client should create a new client if the previous one is closed."""
        import unity.spending_limits as sl

        mock_client = MagicMock()
        mock_client.closed = True
        sl._admin_client = mock_client

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            new_client = sl._get_admin_client()

        assert new_client is not mock_client
        assert sl._admin_client is new_client

        sl._admin_client = None

    @pytest.mark.asyncio
    async def test_no_raw_httpx_usage_during_check(self):
        """Spending limit checks should use AsyncAdminClient, not raw httpx."""
        import unity.spending_limits as sl

        assert not hasattr(sl, "_get_http_client"), (
            "_get_http_client still exists — spending_limits should use "
            "AsyncAdminClient via _get_admin_client instead."
        )

        mock_client = _make_mock_client()
        sl._admin_client = None

        with _patch_context():
            with patch.object(sl, "_get_admin_client", return_value=mock_client):
                with patch("unify.async_admin.AsyncAdminClient") as mock_cls:
                    await sl.check_spending_limits_callback(
                        LimitCheckRequest(model="gpt-4", endpoint="test"),
                    )

                    assert mock_cls.call_count == 0, (
                        f"AsyncAdminClient was instantiated {mock_cls.call_count} time(s) "
                        "during a limit check. The shared client should be used instead."
                    )
