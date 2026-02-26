"""Behavioral tests for the credit balance guard (Option 1).

These tests verify that the spending limit callback correctly blocks LLM calls
when the billing account's credit balance is zero or negative, and allows them
when balance is positive — regardless of how the credits were spent.

The credit balance guard piggybacks on the existing spending limit HTTP calls
to Orchestra, adding zero extra latency.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from unillm.limit_hooks import LimitCheckRequest, LimitType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spend_response(
    *,
    cumulative_spend: float = 10.0,
    limit: float | None = 100.0,
    credit_balance: float | None = 50.0,
    limit_set_at: str | None = None,
):
    """Build a mock Orchestra spend-endpoint response."""
    resp = MagicMock()
    data = {"cumulative_spend": cumulative_spend, "limit": limit}
    if credit_balance is not None:
        data["credit_balance"] = credit_balance
    if limit_set_at is not None:
        data["limit_set_at"] = limit_set_at
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _patch_context(*, org_id=None):
    """Context manager that patches API key, base URL, and SESSION_DETAILS."""

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


def _patch_http(get_side_effect):
    """Patch httpx.AsyncClient with a custom GET handler."""

    class _Ctx:
        def __init__(self):
            self._stack = []

        def __enter__(self):
            import contextlib

            stack = contextlib.ExitStack()
            mock_client = stack.enter_context(patch("httpx.AsyncClient"))
            mock_instance = AsyncMock()
            if callable(get_side_effect) and not isinstance(get_side_effect, MagicMock):
                mock_instance.get = get_side_effect
            else:
                mock_instance.get.return_value = get_side_effect
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance
            self._stack.append(stack)
            return mock_instance

        def __exit__(self, *exc):
            self._stack.pop().close()

    return _Ctx()


# ---------------------------------------------------------------------------
# Core behaviour: credit balance blocks / allows
# ---------------------------------------------------------------------------


class TestCreditBalanceBlocking:
    """The callback must block when credit_balance ≤ 0 and allow when > 0."""

    @pytest.mark.asyncio
    async def test_zero_balance_blocks(self):
        """A zero credit balance should deny the request."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(credit_balance=0.0)
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False
        assert "insufficient credits" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_negative_balance_blocks(self):
        """A negative credit balance should deny the request."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(credit_balance=-42.50)
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False
        assert "insufficient credits" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_tiny_positive_balance_allows(self):
        """Even $0.01 remaining should let the call through."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(credit_balance=0.01)
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_large_positive_balance_allows(self):
        """A healthy balance should let the call through."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(credit_balance=1000.0)
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("balance", [0, -0.01, -100, -1e6])
    async def test_various_non_positive_balances_all_block(self, balance):
        """Any non-positive balance should block."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(credit_balance=balance)
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False


# ---------------------------------------------------------------------------
# Interaction with spending caps
# ---------------------------------------------------------------------------


class TestCreditBalanceVsSpendingCap:
    """Credit balance and spending cap checks should coexist correctly."""

    @pytest.mark.asyncio
    async def test_under_cap_but_no_credits_blocks(self):
        """Spending under cap but zero credits → blocked by credit check."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(
            cumulative_spend=10.0,
            limit=100.0,
            credit_balance=0.0,
        )
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False
        assert "insufficient credits" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_over_cap_with_credits_blocks_on_cap(self):
        """Spending over cap with remaining credits → blocked by spending cap."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(
            cumulative_spend=150.0,
            limit=100.0,
            credit_balance=500.0,
        )
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False
        assert "spending limit exceeded" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_over_cap_and_no_credits_blocks_on_cap_first(self):
        """Both cap exceeded and zero credits → cap takes precedence (checked first)."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(
            cumulative_spend=200.0,
            limit=100.0,
            credit_balance=0.0,
        )
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False
        # Spending cap is checked before credit balance in the loop
        assert "spending limit exceeded" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_no_cap_set_but_no_credits_blocks(self):
        """No spending cap set (unlimited) but zero credits → blocked."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(
            cumulative_spend=999.0,
            limit=None,  # No cap
            credit_balance=0.0,
        )
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False
        assert "insufficient credits" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_no_cap_set_with_credits_allows(self):
        """No spending cap set and positive credits → allowed."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(
            cumulative_spend=999.0,
            limit=None,
            credit_balance=10.0,
        )
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is True


# ---------------------------------------------------------------------------
# Org context: credit balance from org billing account
# ---------------------------------------------------------------------------


class TestOrgCreditBalance:
    """Credit balance check must work in organization context."""

    @pytest.mark.asyncio
    async def test_org_zero_balance_blocks(self):
        """Zero credits in org billing account blocks all members."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(
            cumulative_spend=5.0,
            limit=100.0,
            credit_balance=0.0,
        )
        with _patch_context(org_id=789):
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False
        assert "insufficient credits" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_org_positive_balance_allows(self):
        """Positive credits in org billing account allows all members."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(
            cumulative_spend=5.0,
            limit=100.0,
            credit_balance=50.0,
        )
        with _patch_context(org_id=789):
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_org_member_limit_exceeded_but_credits_remain(self):
        """Member cap exceeded blocks even if org has credits."""
        from unity.spending_limits import check_spending_limits_callback

        async def route_by_url(url, *args, **kwargs):
            if "members" in url:
                return _make_spend_response(
                    cumulative_spend=300.0,
                    limit=200.0,
                    credit_balance=500.0,
                )
            return _make_spend_response(
                cumulative_spend=5.0,
                limit=1000.0,
                credit_balance=500.0,
            )

        with _patch_context(org_id=789):
            with _patch_http(route_by_url):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False
        assert result.limit_type == LimitType.MEMBER


# ---------------------------------------------------------------------------
# Backward compatibility: missing credit_balance field
# ---------------------------------------------------------------------------


class TestMissingCreditBalance:
    """When Orchestra doesn't return credit_balance, calls should be allowed
    (backward compat / fail-open)."""

    @pytest.mark.asyncio
    async def test_no_credit_balance_in_response_allows(self):
        """Old Orchestra without credit_balance → allowed (fail-open)."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(credit_balance=None)
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_all_endpoints_fail_still_allows(self):
        """If every endpoint errors out, fail-open: no credit_balance to check."""
        from unity.spending_limits import check_spending_limits_callback

        async def failing_get(url, *args, **kwargs):
            raise httpx.TimeoutException("Timeout")

        with _patch_context():
            with _patch_http(failing_get):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is True


# ---------------------------------------------------------------------------
# Partial endpoint failures
# ---------------------------------------------------------------------------


class TestPartialEndpointFailures:
    """If some endpoints fail but one succeeds, its credit_balance is used."""

    @pytest.mark.asyncio
    async def test_assistant_endpoint_fails_user_provides_balance(self):
        """Assistant endpoint fails, but user endpoint returns zero balance → blocked."""
        from unity.spending_limits import check_spending_limits_callback

        call_count = 0

        async def mixed_get(url, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "assistant" in url:
                raise httpx.TimeoutException("Timeout")
            return _make_spend_response(
                cumulative_spend=5.0,
                limit=100.0,
                credit_balance=0.0,
            )

        with _patch_context():
            with _patch_http(mixed_get):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False
        assert "insufficient credits" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_user_endpoint_fails_assistant_provides_balance(self):
        """User endpoint fails, but assistant endpoint returns positive balance → allowed."""
        from unity.spending_limits import check_spending_limits_callback

        async def mixed_get(url, *args, **kwargs):
            if "user" in url and "assistant" not in url:
                raise httpx.TimeoutException("Timeout")
            return _make_spend_response(
                cumulative_spend=5.0,
                limit=100.0,
                credit_balance=25.0,
            )

        with _patch_context():
            with _patch_http(mixed_get):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is True


# ---------------------------------------------------------------------------
# Non-LLM spending drains credits
# ---------------------------------------------------------------------------


class TestNonLlmSpendingDrainsCredits:
    """The primary scenario Option 1 addresses: credits exhausted by non-LLM
    costs (photo gen, assistant creation, etc.) should still block LLM calls."""

    @pytest.mark.asyncio
    async def test_credits_drained_by_external_costs_blocks_llm(self):
        """Credits at $0 due to photo generation → LLM call blocked.

        The spending cap is well under limit, but the billing account is empty.
        This is the scenario that a client-side flag (Option 3) could never catch.
        """
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(
            cumulative_spend=2.0,  # Only $2 of LLM spend
            limit=1000.0,  # Huge cap — not the issue
            credit_balance=0.0,  # Drained by non-LLM costs
        )
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False
        assert "insufficient credits" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_credits_refilled_after_drain_allows(self):
        """Credits refilled (auto-recharge) after drain → LLM call allowed."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(
            cumulative_spend=2.0,
            limit=1000.0,
            credit_balance=50.0,  # Auto-recharge kicked in
        )
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is True


# ---------------------------------------------------------------------------
# Reason message format
# ---------------------------------------------------------------------------


class TestReasonMessage:
    """The denial reason should include the credit balance for debugging."""

    @pytest.mark.asyncio
    async def test_reason_includes_balance_amount(self):
        """Reason should show the actual balance."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(credit_balance=-12.34)
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False
        assert "-12.34" in result.reason

    @pytest.mark.asyncio
    async def test_reason_includes_zero_balance(self):
        """Reason should show $0.00 for zero balance."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(credit_balance=0.0)
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False
        assert "0.00" in result.reason


# ---------------------------------------------------------------------------
# Concurrent stress tests
# ---------------------------------------------------------------------------


class TestConcurrentCreditChecks:
    """Multiple simultaneous LLM calls should all see the credit balance."""

    @pytest.mark.asyncio
    async def test_many_concurrent_calls_all_blocked_when_empty(self):
        """20 concurrent limit checks with zero balance → all denied."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(credit_balance=0.0)

        async def slow_get(url, *args, **kwargs):
            await asyncio.sleep(0.01)  # simulate network
            return resp

        with _patch_context():
            with _patch_http(slow_get):
                tasks = [
                    check_spending_limits_callback(
                        LimitCheckRequest(model="gpt-4", endpoint="test"),
                    )
                    for _ in range(20)
                ]
                results = await asyncio.gather(*tasks)

        assert all(not r.allowed for r in results)
        assert all("insufficient credits" in r.reason.lower() for r in results)

    @pytest.mark.asyncio
    async def test_many_concurrent_calls_all_allowed_when_positive(self):
        """20 concurrent limit checks with positive balance → all allowed."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(credit_balance=100.0)

        async def slow_get(url, *args, **kwargs):
            await asyncio.sleep(0.01)
            return resp

        with _patch_context():
            with _patch_http(slow_get):
                tasks = [
                    check_spending_limits_callback(
                        LimitCheckRequest(model="gpt-4", endpoint="test"),
                    )
                    for _ in range(20)
                ]
                results = await asyncio.gather(*tasks)

        assert all(r.allowed for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_calls_see_balance_change(self):
        """Simulate balance dropping to zero mid-burst: later calls see the change."""
        from unity.spending_limits import check_spending_limits_callback

        call_counter = 0

        async def depleting_get(url, *args, **kwargs):
            nonlocal call_counter
            call_counter += 1
            # First 5 calls see positive balance, rest see zero
            if call_counter <= 10:  # 2 endpoints per call × 5 calls = 10
                balance = 50.0
            else:
                balance = 0.0
            return _make_spend_response(credit_balance=balance)

        with _patch_context():
            with _patch_http(depleting_get):
                # Serial calls to control ordering
                results = []
                for _ in range(10):
                    r = await check_spending_limits_callback(
                        LimitCheckRequest(model="gpt-4", endpoint="test"),
                    )
                    results.append(r)

        # First 5 should be allowed, rest blocked
        allowed = [r.allowed for r in results]
        assert all(allowed[:5])
        assert not any(allowed[5:])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and unusual conditions."""

    @pytest.mark.asyncio
    async def test_balance_exactly_zero_blocks(self):
        """Balance of exactly $0.00 should block."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(credit_balance=0.0)
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_balance_epsilon_above_zero_allows(self):
        """A very small positive balance (1e-10) should still allow."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(credit_balance=1e-10)
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_balance_very_large_negative_blocks(self):
        """Even a hugely negative balance should block."""
        from unity.spending_limits import check_spending_limits_callback

        resp = _make_spend_response(credit_balance=-1_000_000.0)
        with _patch_context():
            with _patch_http(resp):
                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_credit_balance_none_vs_zero(self):
        """None credit_balance (missing) should allow; 0.0 should block."""
        from unity.spending_limits import check_spending_limits_callback

        # None → allow
        resp_none = _make_spend_response(credit_balance=None)
        with _patch_context():
            with _patch_http(resp_none):
                result_none = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        # 0.0 → block
        resp_zero = _make_spend_response(credit_balance=0.0)
        with _patch_context():
            with _patch_http(resp_zero):
                result_zero = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result_none.allowed is True
        assert result_zero.allowed is False

    @pytest.mark.asyncio
    async def test_no_api_key_skips_all_checks(self):
        """When no API key is set, all checks are skipped → allowed."""
        from unity.spending_limits import check_spending_limits_callback

        with patch("unity.spending_limits._get_api_key", return_value=None):
            result = await check_spending_limits_callback(
                LimitCheckRequest(model="gpt-4", endpoint="test"),
            )

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_missing_session_context_skips_all_checks(self):
        """When SESSION_DETAILS has no assistant/user, checks are skipped → allowed."""
        from unity.spending_limits import check_spending_limits_callback

        with patch("unity.spending_limits._get_api_key", return_value="key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = None
                mock_session.user_id = None
                mock_session.org_id = None

                result = await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert result.allowed is True


# ---------------------------------------------------------------------------
# Latency: credit check adds no extra HTTP calls
# ---------------------------------------------------------------------------


class TestZeroLatencyOverhead:
    """The credit balance check must not add additional HTTP round-trips."""

    @pytest.mark.asyncio
    async def test_personal_context_makes_exactly_two_http_calls(self):
        """Personal context: assistant + user = 2 HTTP GETs. Credit check is free."""
        from unity.spending_limits import check_spending_limits_callback

        call_count = 0

        async def counting_get(url, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_spend_response(credit_balance=0.0)

        with _patch_context(org_id=None):
            with _patch_http(counting_get):
                await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert call_count == 2  # assistant + user, no extra call for credits

    @pytest.mark.asyncio
    async def test_org_context_makes_exactly_three_http_calls(self):
        """Org context: assistant + member + org = 3 HTTP GETs. Credit check is free."""
        from unity.spending_limits import check_spending_limits_callback

        call_count = 0

        async def counting_get(url, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_spend_response(credit_balance=0.0)

        with _patch_context(org_id=789):
            with _patch_http(counting_get):
                await check_spending_limits_callback(
                    LimitCheckRequest(model="gpt-4", endpoint="test"),
                )

        assert call_count == 3  # assistant + member + org, no extra call
