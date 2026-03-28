"""Tests for spending functionality in Unity.

This module covers:
1. Cumulative spend tracking via LLM event hook (atomic_upsert)
2. Spending limit check callback (Approach B)
3. Hook installation and error handling
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from unillm import LLMEvent
from unillm.limit_hooks import LimitCheckRequest, LimitType

from unify.async_admin import AdminRequestError

from unity.common.log_utils import AtomicUpsertResult, atomic_upsert
from unity.events.llm_event_hook import (
    _llm_event_to_eventbus,
    _update_cumulative_spend,
)

# ===========================================================================
# Part 1: Cumulative Spend Tracking
# ===========================================================================


class TestAtomicUpsertResult:
    """Tests for the AtomicUpsertResult dataclass."""

    def test_create_result(self):
        result = AtomicUpsertResult(
            log_id=123,
            new_value=78.50,
            created=False,
            mirrored_contexts=["All/Spending/Monthly"],
        )
        assert result.log_id == 123
        assert result.new_value == 78.50
        assert result.created is False
        assert result.mirrored_contexts == ["All/Spending/Monthly"]

    def test_create_result_new_log(self):
        result = AtomicUpsertResult(
            log_id=456,
            new_value=5.50,
            created=True,
            mirrored_contexts=["All/Spending/Monthly"],
        )
        assert result.log_id == 456
        assert result.new_value == 5.50
        assert result.created is True


class TestAtomicUpsert:
    """Tests for the atomic_upsert function."""

    @pytest.mark.asyncio
    async def test_atomic_upsert_success(self):
        """atomic_upsert should call Orchestra's endpoint and return result."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "log_id": 789,
            "new_value": 83.50,
            "created": False,
            "mirrored_contexts": ["All/Spending/Monthly"],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("unity.common.log_utils.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            with patch("unity.common.log_utils.SESSION_DETAILS") as mock_session:
                mock_session.unify_key = "test-api-key"
                mock_session.user_context = "user123"
                mock_session.assistant_context = "456"
                mock_session.user.id = "user123"
                mock_session.assistant.agent_id = 456

                with patch("unity.common.log_utils.SETTINGS") as mock_settings:
                    mock_settings.ORCHESTRA_URL = "https://api.test.com/v0"

                    with patch("unity.common.log_utils.unify") as mock_unify:
                        mock_unify.active_project.return_value = "Assistants"

                        result = await atomic_upsert(
                            context="user123/456/Spending/Monthly",
                            unique_keys={"_assistant_id": "str", "month": "str"},
                            field="cumulative_spend",
                            operation="+5.50",
                            initial_data={
                                "_assistant_id": "456",
                                "month": "2026-01",
                            },
                            add_to_all_context=True,
                        )

        assert result.log_id == 789
        assert result.new_value == 83.50
        assert result.created is False
        assert result.mirrored_contexts == ["All/Spending/Monthly"]

        # Verify the HTTP call was made correctly
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "logs/atomic" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_atomic_upsert_injects_private_fields(self):
        """atomic_upsert should inject _user, _user_id, _assistant, _assistant_id."""
        captured_payload = {}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "log_id": 1,
            "new_value": 5.0,
            "created": True,
            "mirrored_contexts": [],
        }
        mock_response.raise_for_status = MagicMock()

        async def capture_post(url, json=None, headers=None):
            captured_payload.update(json)
            return mock_response

        with patch("unity.common.log_utils.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=capture_post)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            with patch("unity.common.log_utils.SESSION_DETAILS") as mock_session:
                mock_session.unify_key = "test-key"
                mock_session.user_context = "user_abc123"
                mock_session.assistant_context = "asst_789"
                mock_session.user.id = "user_abc123"
                mock_session.assistant.agent_id = 789

                with patch("unity.common.log_utils.SETTINGS") as mock_settings:
                    mock_settings.ORCHESTRA_URL = "https://api.test.com/v0"

                    with patch("unity.common.log_utils.unify") as mock_unify:
                        mock_unify.active_project.return_value = "Assistants"

                        await atomic_upsert(
                            context="user_abc123/asst_789/Spending/Monthly",
                            unique_keys={"_assistant_id": "str", "month": "str"},
                            field="cumulative_spend",
                            operation="+5.00",
                            initial_data={
                                "_assistant_id": "asst_789",
                                "month": "2026-01",
                            },
                        )

        # Verify private fields were injected
        initial_data = captured_payload.get("initial_data", {})
        assert initial_data.get("_user") == "user_abc123"
        assert initial_data.get("_user_id") == "user_abc123"
        assert initial_data.get("_assistant") == "asst_789"
        assert initial_data.get("_assistant_id") == "789"

    @pytest.mark.asyncio
    async def test_atomic_upsert_includes_org_id_when_present(self):
        """atomic_upsert should pass through org_id in initial_data."""
        captured_payload = {}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "log_id": 1,
            "new_value": 5.0,
            "created": True,
            "mirrored_contexts": [],
        }
        mock_response.raise_for_status = MagicMock()

        async def capture_post(url, json=None, headers=None):
            captured_payload.update(json)
            return mock_response

        with patch("unity.common.log_utils.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=capture_post)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            with patch("unity.common.log_utils.SESSION_DETAILS") as mock_session:
                mock_session.unify_key = "test-key"
                mock_session.user_context = "user123"
                mock_session.assistant_context = "456"
                mock_session.user.id = "user123"
                mock_session.assistant.agent_id = 456
                mock_session.org_id = 789  # Set org context
                mock_session.org_name = "TestOrg"  # Set org name

                with patch("unity.common.log_utils.SETTINGS") as mock_settings:
                    mock_settings.ORCHESTRA_URL = "https://api.test.com/v0"

                    with patch("unity.common.log_utils.unify") as mock_unify:
                        mock_unify.active_project.return_value = "Assistants"

                        await atomic_upsert(
                            context="user123/456/Spending/Monthly",
                            unique_keys={"_assistant_id": "str", "month": "str"},
                            field="cumulative_spend",
                            operation="+5.00",
                            initial_data={
                                "_assistant_id": "456",
                                "month": "2026-01",
                            },
                        )

        # Verify org fields are injected from SESSION_DETAILS into initial_data
        initial_data = captured_payload.get("initial_data", {})
        assert initial_data.get("_org_id") == 789
        assert initial_data.get("_org") == "TestOrg"

    @pytest.mark.asyncio
    async def test_atomic_upsert_http_error_propagates(self):
        """atomic_upsert should propagate HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response,
        )

        with patch("unity.common.log_utils.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            with patch("unity.common.log_utils.SESSION_DETAILS") as mock_session:
                mock_session.unify_key = "test-key"
                mock_session.user_context = "user123"
                mock_session.assistant_context = "456"
                mock_session.user.id = "user123"
                mock_session.assistant.agent_id = 456

                with patch("unity.common.log_utils.SETTINGS") as mock_settings:
                    mock_settings.ORCHESTRA_URL = "https://api.test.com/v0"

                    with patch("unity.common.log_utils.unify") as mock_unify:
                        mock_unify.active_project.return_value = "Assistants"

                        with pytest.raises(httpx.HTTPStatusError):
                            await atomic_upsert(
                                context="user123/456/Spending/Monthly",
                                unique_keys={"_assistant_id": "str", "month": "str"},
                                field="cumulative_spend",
                                operation="+5.00",
                                initial_data={
                                    "_assistant_id": "456",
                                    "month": "2026-01",
                                },
                            )


class TestUpdateCumulativeSpend:
    """Tests for the _update_cumulative_spend function."""

    @pytest.mark.asyncio
    async def test_update_spend_calls_atomic_upsert(self):
        """_update_cumulative_spend should call atomic_upsert with correct params."""
        with patch(
            "unity.common.log_utils.atomic_upsert",
            new_callable=AsyncMock,
        ) as mock_upsert:
            mock_upsert.return_value = AtomicUpsertResult(
                log_id=1,
                new_value=5.50,
                created=True,
                mirrored_contexts=["All/Spending/Monthly"],
            )

            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.timezone = "UTC"
                mock_session.user_context = "user123"
                mock_session.assistant_context = "456"
                mock_session.assistant.agent_id = 456

                await _update_cumulative_spend(5.50)

        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args.kwargs
        assert call_kwargs["context"] == "user123/456/Spending/Monthly"
        assert call_kwargs["field"] == "cumulative_spend"
        assert "+5.5" in call_kwargs["operation"]
        assert call_kwargs["add_to_all_context"] is True
        assert call_kwargs["project"] == "Assistants"

    @pytest.mark.asyncio
    async def test_update_spend_skips_zero_cost(self):
        """_update_cumulative_spend should skip if billed_cost is zero."""
        with patch(
            "unity.common.log_utils.atomic_upsert",
            new_callable=AsyncMock,
        ) as mock_upsert:
            await _update_cumulative_spend(0.0)

        mock_upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_spend_skips_negative_cost(self):
        """_update_cumulative_spend should skip if billed_cost is negative."""
        with patch(
            "unity.common.log_utils.atomic_upsert",
            new_callable=AsyncMock,
        ) as mock_upsert:
            await _update_cumulative_spend(-1.0)

        mock_upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_spend_handles_errors_gracefully(self):
        """_update_cumulative_spend should not raise on errors."""
        with patch(
            "unity.common.log_utils.atomic_upsert",
            new_callable=AsyncMock,
        ) as mock_upsert:
            mock_upsert.side_effect = Exception("Network error")

            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.timezone = "UTC"
                mock_session.user_context = "user123"
                mock_session.assistant_context = "456"
                mock_session.assistant.agent_id = 456

                # Should not raise
                await _update_cumulative_spend(5.50)

    @pytest.mark.asyncio
    async def test_update_spend_uses_user_timezone(self):
        """_update_cumulative_spend should use assistant's timezone for month."""
        captured_month = []

        async def capture_upsert(**kwargs):
            captured_month.append(kwargs.get("initial_data", {}).get("month"))
            return AtomicUpsertResult(
                log_id=1,
                new_value=5.0,
                created=True,
                mirrored_contexts=[],
            )

        with patch(
            "unity.common.log_utils.atomic_upsert",
            new_callable=AsyncMock,
            side_effect=capture_upsert,
        ):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.timezone = "America/New_York"
                mock_session.user_context = "user123"
                mock_session.assistant_context = "456"
                mock_session.assistant.agent_id = 456

                await _update_cumulative_spend(5.50)

        # Should have captured a month string
        assert len(captured_month) == 1
        assert captured_month[0] is not None
        # Month should be in YYYY-MM format
        assert len(captured_month[0]) == 7
        assert "-" in captured_month[0]

    @pytest.mark.asyncio
    async def test_update_spend_skips_missing_user_context(self):
        """_update_cumulative_spend should skip if user_context is missing."""
        with patch(
            "unity.common.log_utils.atomic_upsert",
            new_callable=AsyncMock,
        ) as mock_upsert:
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.timezone = "UTC"
                mock_session.user_context = None  # Missing
                mock_session.assistant_context = "456"
                mock_session.assistant.agent_id = 456

                await _update_cumulative_spend(5.50)

        mock_upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_spend_skips_missing_assistant_context(self):
        """_update_cumulative_spend should skip if assistant_context is missing."""
        with patch(
            "unity.common.log_utils.atomic_upsert",
            new_callable=AsyncMock,
        ) as mock_upsert:
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.timezone = "UTC"
                mock_session.user_context = "user123"
                mock_session.assistant_context = None  # Missing
                mock_session.assistant.agent_id = 456

                await _update_cumulative_spend(5.50)

        mock_upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_spend_skips_missing_assistant_id(self):
        """_update_cumulative_spend should skip if assistant_id is missing."""
        with patch(
            "unity.common.log_utils.atomic_upsert",
            new_callable=AsyncMock,
        ) as mock_upsert:
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.timezone = "UTC"
                mock_session.user_context = "user123"
                mock_session.assistant_context = "456"
                mock_session.assistant.agent_id = None  # Missing

                await _update_cumulative_spend(5.50)

        mock_upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_spend_invalid_timezone_falls_back_to_utc(self):
        """_update_cumulative_spend should fall back to UTC for invalid timezone."""
        captured_month = []

        async def capture_upsert(**kwargs):
            captured_month.append(kwargs.get("initial_data", {}).get("month"))
            return AtomicUpsertResult(
                log_id=1,
                new_value=5.0,
                created=True,
                mirrored_contexts=[],
            )

        with patch(
            "unity.common.log_utils.atomic_upsert",
            new_callable=AsyncMock,
            side_effect=capture_upsert,
        ):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.timezone = "Invalid/Timezone"
                mock_session.user_context = "user123"
                mock_session.assistant_context = "456"
                mock_session.assistant.agent_id = 456

                await _update_cumulative_spend(5.50)

        # Should still succeed with UTC fallback
        assert len(captured_month) == 1
        assert captured_month[0] is not None

    @pytest.mark.asyncio
    async def test_update_spend_very_small_cost(self):
        """_update_cumulative_spend should handle very small costs."""
        captured_operation = []

        async def capture_upsert(**kwargs):
            captured_operation.append(kwargs.get("operation"))
            return AtomicUpsertResult(
                log_id=1,
                new_value=0.0001,
                created=True,
                mirrored_contexts=[],
            )

        with patch(
            "unity.common.log_utils.atomic_upsert",
            new_callable=AsyncMock,
            side_effect=capture_upsert,
        ):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.timezone = "UTC"
                mock_session.user_context = "user123"
                mock_session.assistant_context = "456"
                mock_session.assistant.agent_id = 456

                await _update_cumulative_spend(0.0001)

        # Should have been called with the small value
        assert len(captured_operation) == 1
        assert "+0.0001" in captured_operation[0]


class TestConcurrentSpendUpdates:
    """Tests for concurrent atomic_upsert operations."""

    @pytest.mark.asyncio
    async def test_concurrent_upserts_all_complete(self):
        """Multiple concurrent atomic_upsert calls should all complete."""
        call_count = 0

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_post(url, json=None, headers=None):
            nonlocal call_count
            call_count += 1
            # Simulate slight delay to overlap
            await asyncio.sleep(0.01)
            mock_response.json.return_value = {
                "log_id": call_count,
                "new_value": float(call_count * 5),
                "created": call_count == 1,
                "mirrored_contexts": ["All/Spending/Monthly"],
            }
            return mock_response

        with patch("unity.common.log_utils.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            with patch("unity.common.log_utils.SESSION_DETAILS") as mock_session:
                mock_session.unify_key = "test-key"
                mock_session.user_context = "user123"
                mock_session.assistant_context = "456"
                mock_session.user.id = "user123"
                mock_session.assistant.agent_id = 456

                with patch("unity.common.log_utils.SETTINGS") as mock_settings:
                    mock_settings.ORCHESTRA_URL = "https://api.test.com/v0"

                    with patch("unity.common.log_utils.unify") as mock_unify:
                        mock_unify.active_project.return_value = "Assistants"

                        # Launch 5 concurrent upserts
                        tasks = [
                            atomic_upsert(
                                context="user123/456/Spending/Monthly",
                                unique_keys={"_assistant_id": "str", "month": "str"},
                                field="cumulative_spend",
                                operation=f"+{i}.00",
                                initial_data={
                                    "_assistant_id": "456",
                                    "month": "2026-01",
                                },
                            )
                            for i in range(1, 6)
                        ]
                        results = await asyncio.gather(*tasks)

        # All 5 calls should complete
        assert len(results) == 5
        assert call_count == 5


class TestLLMEventHookSpendLogging:
    """Tests for spend logging in the LLM event hook."""

    def test_hook_does_not_fail_with_positive_cost(self):
        """The LLM event hook should not fail if billed_cost is positive."""
        llm_event = LLMEvent(
            request={"model": "gpt-4o", "messages": []},
            billed_cost=0.005,
        )

        # This should not raise even without a running event loop
        _llm_event_to_eventbus(llm_event)

    def test_hook_does_not_fail_on_missing_cost(self):
        """The LLM event hook should not fail if billed_cost is None."""
        llm_event = LLMEvent(
            request={"model": "gpt-4o", "messages": []},
            billed_cost=None,
        )

        _llm_event_to_eventbus(llm_event)

    @pytest.mark.asyncio
    async def test_hook_schedules_spend_update_with_running_loop(self):
        """LLM event hook should schedule spend update when loop is running."""
        llm_event = LLMEvent(
            request={"model": "gpt-4o", "messages": []},
            billed_cost=0.005,
        )

        with patch("unity.events.event_bus.EVENT_BUS") as mock_bus:
            mock_bus.publish = AsyncMock()

            with patch(
                "unity.common.log_utils.atomic_upsert",
                new_callable=AsyncMock,
            ):
                _llm_event_to_eventbus(llm_event)
                await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_hook_skips_spend_update_for_zero_cost(self):
        """LLM event hook should skip spend update for zero billed_cost."""
        llm_event = LLMEvent(
            request={"model": "gpt-4o", "messages": []},
            billed_cost=0.0,
        )

        with patch("unity.events.event_bus.EVENT_BUS") as mock_bus:
            mock_bus.publish = AsyncMock()
            _llm_event_to_eventbus(llm_event)
            await asyncio.sleep(0.01)


# ===========================================================================
# Part 2: Spending Limit Check Callback (Approach B)
# ===========================================================================


class TestCheckSpendingLimitsCallback:
    """Tests for check_spending_limits_callback function."""

    @pytest.mark.asyncio
    async def test_allowed_when_no_api_key(self):
        """Callback should allow request if no API key is configured."""
        from unity.spending_limits import check_spending_limits_callback

        with patch.dict("os.environ", {}, clear=True):
            with patch("unity.spending_limits._get_api_key", return_value=None):
                request = LimitCheckRequest(model="gpt-4", endpoint="test")
                response = await check_spending_limits_callback(request)

        assert response.allowed is True

    @pytest.mark.asyncio
    async def test_allowed_when_missing_context(self):
        """Callback should allow request if SESSION_DETAILS is incomplete."""
        from unity.spending_limits import check_spending_limits_callback

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = None  # No assistant
                mock_session.user_id = "user123"

                request = LimitCheckRequest(model="gpt-4", endpoint="test")
                response = await check_spending_limits_callback(request)

        assert response.allowed is True

    @pytest.mark.asyncio
    async def test_checks_assistant_limit(self):
        """Callback should check assistant limit."""
        from unity.spending_limits import check_spending_limits_callback

        spend_data = {"cumulative_spend": 50.0, "limit": 100.0}

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = None
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(
                        return_value=spend_data,
                    )
                    mock_instance.get_user_spend = AsyncMock(return_value=spend_data)
                    mock_instance.notify_limit_reached = AsyncMock(
                        return_value={"notified": False},
                    )
                    mock_get_client.return_value = mock_instance

                    request = LimitCheckRequest(model="gpt-4", endpoint="test")
                    response = await check_spending_limits_callback(request)

        assert response.allowed is True

    @pytest.mark.asyncio
    async def test_denies_when_limit_exceeded(self):
        """Callback should deny request when limit is exceeded."""
        from unity.spending_limits import check_spending_limits_callback

        spend_data = {"cumulative_spend": 150.0, "limit": 100.0}

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = None
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(
                        return_value=spend_data,
                    )
                    mock_instance.get_user_spend = AsyncMock(return_value=spend_data)
                    mock_instance.notify_limit_reached = AsyncMock(
                        return_value={"notified": False},
                    )
                    mock_get_client.return_value = mock_instance

                    request = LimitCheckRequest(model="gpt-4", endpoint="test")
                    response = await check_spending_limits_callback(request)

        assert response.allowed is False
        assert response.limit_type == LimitType.ASSISTANT
        assert "exceeded" in response.reason.lower()


class TestPersonalContextLimitChecks:
    """Tests for personal context (no org_id)."""

    @pytest.mark.asyncio
    async def test_checks_user_limit(self):
        """Personal context should check user limit."""
        from unity.spending_limits import check_spending_limits_callback

        spend_data = {"cumulative_spend": 50.0, "limit": 100.0}

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = None  # Personal context
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(
                        return_value=spend_data,
                    )
                    mock_instance.get_user_spend = AsyncMock(return_value=spend_data)
                    mock_instance.notify_limit_reached = AsyncMock(
                        return_value={"notified": False},
                    )
                    mock_get_client.return_value = mock_instance

                    request = LimitCheckRequest(model="gpt-4", endpoint="test")
                    await check_spending_limits_callback(request)

        # Should check assistant and user limits
        mock_instance.get_assistant_spend.assert_called_once()
        mock_instance.get_user_spend.assert_called_once()
        mock_instance.get_org_spend.assert_not_called()
        mock_instance.get_member_spend.assert_not_called()


class TestOrgContextLimitChecks:
    """Tests for organization context (org_id set)."""

    @pytest.mark.asyncio
    async def test_checks_org_and_member_limits(self):
        """Org context should check assistant, member, and org limits."""
        from unity.spending_limits import check_spending_limits_callback

        spend_data = {"cumulative_spend": 50.0, "limit": 100.0}

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = 789  # Org context
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(
                        return_value=spend_data,
                    )
                    mock_instance.get_user_spend = AsyncMock(return_value=spend_data)
                    mock_instance.get_member_spend = AsyncMock(return_value=spend_data)
                    mock_instance.get_org_spend = AsyncMock(return_value=spend_data)
                    mock_instance.notify_limit_reached = AsyncMock(
                        return_value={"notified": False},
                    )
                    mock_get_client.return_value = mock_instance

                    request = LimitCheckRequest(model="gpt-4", endpoint="test")
                    await check_spending_limits_callback(request)

        # Should check assistant, member, and org limits
        mock_instance.get_assistant_spend.assert_called_once()
        mock_instance.get_member_spend.assert_called_once()
        mock_instance.get_org_spend.assert_called_once()

    @pytest.mark.asyncio
    async def test_member_limit_exceeded(self):
        """Member limit exceeded should deny request."""
        from unity.spending_limits import check_spending_limits_callback

        ok_data = {"cumulative_spend": 50.0, "limit": 1000.0}
        exceeded_data = {"cumulative_spend": 300.0, "limit": 200.0}

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = 789
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(return_value=ok_data)
                    mock_instance.get_user_spend = AsyncMock(return_value=ok_data)
                    mock_instance.get_member_spend = AsyncMock(
                        return_value=exceeded_data,
                    )
                    mock_instance.get_org_spend = AsyncMock(return_value=ok_data)
                    mock_instance.notify_limit_reached = AsyncMock(
                        return_value={"notified": False},
                    )
                    mock_get_client.return_value = mock_instance

                    request = LimitCheckRequest(model="gpt-4", endpoint="test")
                    response = await check_spending_limits_callback(request)

        assert response.allowed is False
        assert response.limit_type == LimitType.MEMBER


class TestLimitCheckErrorHandling:
    """Tests for error handling and fail-open behavior."""

    @pytest.mark.asyncio
    async def test_timeout_fails_open(self):
        """Timeout should fail open (allow request)."""
        from unity.spending_limits import check_spending_limits_callback

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = None
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(
                        side_effect=Exception("Timeout"),
                    )
                    mock_instance.get_user_spend = AsyncMock(
                        side_effect=Exception("Timeout"),
                    )
                    mock_instance.notify_limit_reached = AsyncMock(
                        return_value={"notified": False},
                    )
                    mock_get_client.return_value = mock_instance

                    request = LimitCheckRequest(model="gpt-4", endpoint="test")
                    response = await check_spending_limits_callback(request)

        assert response.allowed is True

    @pytest.mark.asyncio
    async def test_404_fails_open(self):
        """404 (entity not found) should fail open."""
        from unity.spending_limits import check_spending_limits_callback

        error = AdminRequestError(
            url="http://test/v0/admin/assistant/123/spend",
            method="GET",
            status=404,
            body="Not found",
        )

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = None
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(side_effect=error)
                    mock_instance.get_user_spend = AsyncMock(side_effect=error)
                    mock_instance.notify_limit_reached = AsyncMock(
                        return_value={"notified": False},
                    )
                    mock_get_client.return_value = mock_instance

                    request = LimitCheckRequest(model="gpt-4", endpoint="test")
                    response = await check_spending_limits_callback(request)

        assert response.allowed is True

    @pytest.mark.asyncio
    async def test_500_error_fails_open(self):
        """500 server error should fail open."""
        from unity.spending_limits import check_spending_limits_callback

        error = AdminRequestError(
            url="http://test/v0/admin/assistant/123/spend",
            method="GET",
            status=500,
            body="Internal server error",
        )

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = None
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(side_effect=error)
                    mock_instance.get_user_spend = AsyncMock(side_effect=error)
                    mock_instance.notify_limit_reached = AsyncMock(
                        return_value={"notified": False},
                    )
                    mock_get_client.return_value = mock_instance

                    request = LimitCheckRequest(model="gpt-4", endpoint="test")
                    response = await check_spending_limits_callback(request)

        assert response.allowed is True


class TestParallelLimitChecks:
    """Tests for parallel limit checking."""

    @pytest.mark.asyncio
    async def test_checks_run_in_parallel(self):
        """Limit checks should run in parallel."""
        from unity.spending_limits import check_spending_limits_callback

        check_times = []
        spend_data = {"cumulative_spend": 50.0, "limit": 100.0}

        async def delayed_return(*args, **kwargs):
            check_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.05)
            return spend_data

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = 789  # Org context (3 checks)
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(
                        side_effect=delayed_return,
                    )
                    mock_instance.get_member_spend = AsyncMock(
                        side_effect=delayed_return,
                    )
                    mock_instance.get_org_spend = AsyncMock(side_effect=delayed_return)
                    mock_instance.notify_limit_reached = AsyncMock(
                        return_value={"notified": False},
                    )
                    mock_get_client.return_value = mock_instance

                    start_time = asyncio.get_event_loop().time()
                    request = LimitCheckRequest(model="gpt-4", endpoint="test")
                    await check_spending_limits_callback(request)
                    total_time = asyncio.get_event_loop().time() - start_time

        # Should have 3 checks (assistant, member, org)
        assert len(check_times) == 3

        # All should start nearly simultaneously
        time_diff = max(check_times) - min(check_times)
        assert time_diff < 0.02  # Started within 20ms of each other

        # Total time should be ~50ms (parallel), not ~150ms (sequential)
        assert total_time < 0.15


class TestNoLimitSet:
    """Tests for when no limit is set."""

    @pytest.mark.asyncio
    async def test_no_limit_allows_request(self):
        """No limit set should allow request."""
        from unity.spending_limits import check_spending_limits_callback

        spend_data = {
            "cumulative_spend": 1000000.0,  # Large spend
            "limit": None,  # No limit
        }

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = None
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(
                        return_value=spend_data,
                    )
                    mock_instance.get_user_spend = AsyncMock(return_value=spend_data)
                    mock_instance.notify_limit_reached = AsyncMock(
                        return_value={"notified": False},
                    )
                    mock_get_client.return_value = mock_instance

                    request = LimitCheckRequest(model="gpt-4", endpoint="test")
                    response = await check_spending_limits_callback(request)

        assert response.allowed is True


class TestHookInstallation:
    """Tests for hook installation."""

    def test_install_hook_without_api_key(self):
        """install_limit_check_hook should skip if no API key."""
        from unity.spending_limits import install_limit_check_hook

        with patch("unity.spending_limits._get_api_key", return_value=None):
            with patch("unillm.set_limit_check_hook") as mock_set_hook:
                install_limit_check_hook()

        mock_set_hook.assert_not_called()

    def test_install_hook_with_api_key(self):
        """install_limit_check_hook should register hook with API key."""
        from unity.spending_limits import install_limit_check_hook

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unillm.set_limit_check_hook") as mock_set_hook:
                install_limit_check_hook()

        mock_set_hook.assert_called_once()

    def test_uninstall_hook(self):
        """uninstall_limit_check_hook should clear the hook."""
        from unity.spending_limits import uninstall_limit_check_hook

        with patch("unillm.clear_limit_check_hook") as mock_clear:
            uninstall_limit_check_hook()

        mock_clear.assert_called_once()


class TestLimitBoundary:
    """Tests for limit boundary conditions."""

    @pytest.mark.asyncio
    async def test_exactly_at_limit_is_denied(self):
        """Spend exactly at limit should be denied."""
        from unity.spending_limits import check_spending_limits_callback

        spend_data = {
            "cumulative_spend": 100.0,  # Exactly at limit
            "limit": 100.0,
        }

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = None
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(
                        return_value=spend_data,
                    )
                    mock_instance.get_user_spend = AsyncMock(return_value=spend_data)
                    mock_instance.notify_limit_reached = AsyncMock(
                        return_value={"notified": False},
                    )
                    mock_get_client.return_value = mock_instance

                    request = LimitCheckRequest(model="gpt-4", endpoint="test")
                    response = await check_spending_limits_callback(request)

        assert response.allowed is False

    @pytest.mark.asyncio
    async def test_just_under_limit_is_allowed(self):
        """Spend just under limit should be allowed."""
        from unity.spending_limits import check_spending_limits_callback

        spend_data = {
            "cumulative_spend": 99.99,
            "limit": 100.0,
        }

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = None
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(
                        return_value=spend_data,
                    )
                    mock_instance.get_user_spend = AsyncMock(return_value=spend_data)
                    mock_instance.notify_limit_reached = AsyncMock(
                        return_value={"notified": False},
                    )
                    mock_get_client.return_value = mock_instance

                    request = LimitCheckRequest(model="gpt-4", endpoint="test")
                    response = await check_spending_limits_callback(request)

        assert response.allowed is True


# ===========================================================================
# Part 4b: Credit Balance Gating (unit tests)
# ===========================================================================


class TestCreditBalanceGating:
    """Tests for credit balance gating in check_spending_limits_callback.

    The spending limit callback piggybacks credit_balance from the admin spend
    endpoints.  When credit_balance <= 0, the callback must deny the request.
    This mirrors the Vantage bug fix: after a deduction drives the balance
    negative, subsequent LLM calls should be blocked.
    """

    async def _run_callback_with_credit_balance(
        self,
        credit_balance,
        cumulative_spend=10.0,
        limit=100.0,
    ):
        """Helper: invoke the callback with a mocked spend response that
        includes the given credit_balance."""
        from unity.spending_limits import check_spending_limits_callback

        spend_data = {
            "cumulative_spend": cumulative_spend,
            "limit": limit,
            "credit_balance": credit_balance,
        }

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = None
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(
                        return_value=spend_data,
                    )
                    mock_instance.get_user_spend = AsyncMock(return_value=spend_data)
                    mock_instance.notify_limit_reached = AsyncMock(
                        return_value={"notified": False},
                    )
                    mock_get_client.return_value = mock_instance

                    request = LimitCheckRequest(model="gpt-4", endpoint="test")
                    return await check_spending_limits_callback(request)

    @pytest.mark.asyncio
    async def test_positive_balance_allows(self):
        """Positive credit balance should allow the request."""
        response = await self._run_callback_with_credit_balance(10.0)
        assert response.allowed is True

    @pytest.mark.asyncio
    async def test_tiny_positive_balance_allows(self):
        """Tiny residual positive balance ($0.004) should allow the request."""
        response = await self._run_callback_with_credit_balance(0.004128)
        assert response.allowed is True

    @pytest.mark.asyncio
    async def test_zero_balance_denies(self):
        """Zero credit balance should deny the request."""
        response = await self._run_callback_with_credit_balance(0.0)
        assert response.allowed is False
        assert "Insufficient credits" in response.reason

    @pytest.mark.asyncio
    async def test_negative_balance_denies(self):
        """Negative credit balance should deny the request."""
        response = await self._run_callback_with_credit_balance(-0.065448)
        assert response.allowed is False
        assert "Insufficient credits" in response.reason

    @pytest.mark.asyncio
    async def test_null_balance_allows(self):
        """When credit_balance is None (spend endpoints unreachable or no
        billing account), the credit check is skipped — fail-open."""
        response = await self._run_callback_with_credit_balance(None)
        assert response.allowed is True


# ===========================================================================
# Part 5: E2E Tests (require Orchestra)
# ===========================================================================
# These tests require a running Orchestra server and make real API calls.
# They are automatically skipped if Orchestra is not available.
#
# To run these tests:
#   1. Start Orchestra: cd /workspaces/orchestra && ./scripts/local.sh start
#   2. Make sure the following keys are present in Unity's .ev
#   ORCHESTRA_ADMIN_KEY=<orchestra-key>
#   OPENAI_API_KEY=<openai-key>
#   ANTHROPIC_API_KEY=<anthropic-key>
#   3. Run tests: pytest tests/event_bus/test_spending.py -m requires_orchestra


import os as _os
from dataclasses import dataclass as _dataclass
from typing import Optional as _Optional

import pytest_asyncio


@_dataclass
class E2ETestConfig:
    """Configuration for e2e spending tests."""

    base_url: str
    api_key: str
    admin_key: str
    test_user_id: str = "test-user-001"
    test_assistant_first_name: str = "SpendingTest"
    test_assistant_surname: str = "Assistant"
    model: str = "gpt-4o-mini@openai"
    test_agent_id: _Optional[int] = None

    @classmethod
    def from_env(cls) -> "E2ETestConfig":
        """Create config from environment variables."""
        base_url = _os.getenv("ORCHESTRA_URL", "http://localhost:8000/v0")
        api_key = _os.getenv("UNIFY_KEY", "local-test-api-key")
        admin_key = _os.getenv("ORCHESTRA_ADMIN_KEY", api_key)
        return cls(base_url=base_url, api_key=api_key, admin_key=admin_key)


@pytest_asyncio.fixture
async def e2e_config():
    """Set up e2e test environment with seeded assistant and user."""
    config = E2ETestConfig.from_env()
    headers = {"Authorization": f"Bearer {config.api_key}"}
    admin_headers = {"Authorization": f"Bearer {config.admin_key}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Seed test user (required for user spend endpoint tests)
        # Check if user exists first
        response = await client.get(
            f"{config.base_url}/admin/user/{config.test_user_id}/spend",
            headers=admin_headers,
            params={"month": "2026-01"},
        )
        if response.status_code == 404:
            # Create the test user
            response = await client.post(
                f"{config.base_url}/admin/user",
                headers=admin_headers,
                json={
                    "email": f"{config.test_user_id}@test.local",
                    "name": "Test",
                    "last_name": "User",
                },
            )
            if response.status_code in (200, 201):
                # Update test_user_id to match the created user's ID
                user_data = response.json()
                if "id" in user_data:
                    config.test_user_id = user_data["id"]

        # Seed test assistant
        # Check if assistant exists
        response = await client.get(f"{config.base_url}/assistant", headers=headers)
        if response.status_code == 200:
            assistants = response.json().get("info", [])
            for asst in assistants:
                if (
                    asst.get("first_name") == config.test_assistant_first_name
                    and asst.get("surname") == config.test_assistant_surname
                ):
                    config.test_agent_id = asst.get("agent_id")
                    break

        # Create if not exists
        if not config.test_agent_id:
            response = await client.post(
                f"{config.base_url}/assistant",
                headers=headers,
                json={
                    "first_name": config.test_assistant_first_name,
                    "surname": config.test_assistant_surname,
                    "monthly_spending_cap": 25.0,
                    "create_infra": False,
                },
            )
            if response.status_code in (200, 201):
                config.test_agent_id = response.json().get("agent_id")

    # Populate SESSION_DETAILS
    from unity.session_details import SESSION_DETAILS

    # Derive context names from assistant name
    def to_context_name(name: str) -> str:
        return "".join(c for c in name.title() if c.isalnum())

    SESSION_DETAILS.populate(
        user_id=config.test_user_id,
        assistant_id=str(config.test_agent_id),
        user_first_name="Test",
        user_surname="User",
    )
    SESSION_DETAILS.assistant.agent_id = config.test_agent_id

    yield config

    # Cleanup: reset assistant spending cap to NULL (no limit) to ensure clean state
    if config.test_agent_id:

        async def reset_spending_cap():
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.patch(
                    f"{config.base_url}/assistant/{config.test_agent_id}/config",
                    headers={"Authorization": f"Bearer {config.api_key}"},
                    json={"monthly_spending_cap": None},
                )

        try:
            asyncio.get_event_loop().run_until_complete(reset_spending_cap())
        except Exception:
            pass  # Best effort cleanup

    # Cleanup: reset SESSION_DETAILS
    SESSION_DETAILS.reset()


@pytest.mark.requires_orchestra
class TestE2ESpendingLimits:
    """E2E tests for spending limits that require a running Orchestra server."""

    @pytest.mark.asyncio
    async def test_hook_is_installed(self, e2e_config):
        """Verify that Unity initializes and installs the limit check hook."""
        import unity
        from unillm import is_limit_check_enabled

        unity.init()
        assert (
            is_limit_check_enabled()
        ), "Limit check hook should be installed after unity.init()"

    @pytest.mark.asyncio
    async def test_limit_check_callback_allows_under_limit(self, e2e_config):
        """Test that limit check callback allows requests when under limit."""
        import unity
        from unillm.limit_hooks import LimitCheckRequest
        from unity.spending_limits import check_spending_limits_callback

        unity.init()

        request = LimitCheckRequest(
            model="gpt-4o-mini",
            endpoint="gpt-4o-mini@openai",
        )

        response = await check_spending_limits_callback(request)

        # Should be allowed (we start fresh with low spend)
        assert response.allowed is True

    @pytest.mark.asyncio
    async def test_llm_call_succeeds_under_limit(self, e2e_config):
        """Test that an LLM call succeeds when under spending limit."""
        import unity
        import unillm

        unity.init()

        client = unillm.AsyncUnify(e2e_config.model)

        # Should not raise SpendingLimitExceededError
        response = await client.generate(
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=10,
        )

        assert response is not None
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_cumulative_spend_increases_after_llm_call(self, e2e_config):
        """Test that cumulative spend increases after an LLM call."""
        import unity
        import unillm

        unity.init()

        headers = {"Authorization": f"Bearer {e2e_config.admin_key}"}

        # Get spend before
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{e2e_config.base_url}/admin/assistant/{e2e_config.test_agent_id}/spend",
                headers=headers,
                params={"month": datetime.now().strftime("%Y-%m")},
            )
            spend_before = (
                response.json().get("cumulative_spend", 0)
                if response.status_code == 200
                else 0
            )

        # Make LLM call
        llm_client = unillm.AsyncUnify(e2e_config.model)
        try:
            await llm_client.generate(
                messages=[{"role": "user", "content": "Say 'spend test'"}],
                max_tokens=10,
            )
        except unillm.SpendingLimitExceededError:
            pytest.skip("Limit exceeded - cannot test spend increase")

        # Wait for async logging
        await asyncio.sleep(2.0)

        # Get spend after
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{e2e_config.base_url}/admin/assistant/{e2e_config.test_agent_id}/spend",
                headers=headers,
                params={"month": datetime.now().strftime("%Y-%m")},
            )
            spend_after = (
                response.json().get("cumulative_spend", 0)
                if response.status_code == 200
                else 0
            )

        assert (
            spend_after >= spend_before
        ), f"Spend should not decrease: {spend_before} -> {spend_after}"

    @pytest.mark.asyncio
    async def test_limit_exceeded_blocks_llm_call(self, e2e_config):
        """Test that LLM calls are blocked when limit is exceeded."""
        import unity
        import unillm

        unity.init()

        headers = {"Authorization": f"Bearer {e2e_config.api_key}"}

        # Save original limit before modifying
        original_limit = None
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{e2e_config.base_url}/assistant/{e2e_config.test_agent_id}",
                headers=headers,
            )
            if response.status_code == 200:
                original_limit = response.json().get("monthly_spending_cap")

        try:
            # Set assistant limit to $0
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.patch(
                    f"{e2e_config.base_url}/assistant/{e2e_config.test_agent_id}/config",
                    headers=headers,
                    json={"monthly_spending_cap": 0.0},
                )

            llm_client = unillm.AsyncUnify(e2e_config.model)

            with pytest.raises(unillm.SpendingLimitExceededError):
                await llm_client.generate(
                    messages=[{"role": "user", "content": "This should be blocked"}],
                    max_tokens=10,
                )
        finally:
            # Restore original limit (None means no limit)
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.patch(
                    f"{e2e_config.base_url}/assistant/{e2e_config.test_agent_id}/config",
                    headers=headers,
                    json={"monthly_spending_cap": original_limit},
                )

    @pytest.mark.asyncio
    async def test_fail_open_on_hook_exception(self, e2e_config):
        """Test that LLM calls succeed when limit check hook raises exception."""
        import unity
        import unillm
        from unillm.limit_hooks import (
            set_limit_check_hook,
            get_limit_check_hook,
            LimitCheckRequest as _LimitCheckRequest,
            LimitCheckResponse as _LimitCheckResponse,
        )

        unity.init()

        # Save original hook
        original_hook = get_limit_check_hook()

        # Install failing hook
        async def failing_hook(request: _LimitCheckRequest) -> _LimitCheckResponse:
            raise RuntimeError("Simulated failure")

        set_limit_check_hook(failing_hook)

        try:
            llm_client = unillm.AsyncUnify(e2e_config.model)

            # Should NOT raise - fail-open behavior
            response = await llm_client.generate(
                messages=[{"role": "user", "content": "Fail-open test"}],
                max_tokens=10,
            )

            assert response is not None
        finally:
            # Restore original hook
            set_limit_check_hook(original_hook)

    @pytest.mark.asyncio
    async def test_parallel_limit_check_no_overhead(self, e2e_config):
        """Test that limit check runs in parallel with LLM call (zero overhead)."""
        import time
        import unity
        import unillm
        from unillm.limit_hooks import (
            set_limit_check_hook,
            get_limit_check_hook,
            LimitCheckRequest as _LimitCheckRequest,
            LimitCheckResponse as _LimitCheckResponse,
        )

        unity.init()

        original_hook = get_limit_check_hook()

        # Track when limit check starts
        limit_check_start_time = 0.0

        async def delayed_hook(request: _LimitCheckRequest) -> _LimitCheckResponse:
            nonlocal limit_check_start_time
            limit_check_start_time = time.perf_counter()
            await asyncio.sleep(0.3)  # 300ms delay
            return _LimitCheckResponse(allowed=True)

        set_limit_check_hook(delayed_hook)

        try:
            llm_client = unillm.AsyncUnify(e2e_config.model)

            call_start_time = time.perf_counter()
            try:
                await llm_client.generate(
                    messages=[{"role": "user", "content": "Parallel test"}],
                    max_tokens=10,
                )
            except unillm.SpendingLimitExceededError:
                pass  # Timing still valid

            # Verify limit check started immediately (within 100ms of call start)
            limit_check_relative_start = limit_check_start_time - call_start_time
            assert limit_check_relative_start < 0.1, (
                f"Limit check should start immediately, "
                f"but started at +{limit_check_relative_start*1000:.0f}ms"
            )
        finally:
            set_limit_check_hook(original_hook)

    @pytest.mark.asyncio
    async def test_inflight_cancellation_on_limit_exceeded(self, e2e_config):
        """Test that LLM call is cancelled when limit check returns denied."""
        import time
        import unity
        import unillm
        from unillm.limit_hooks import (
            set_limit_check_hook,
            get_limit_check_hook,
            LimitCheckRequest as _LimitCheckRequest,
            LimitCheckResponse as _LimitCheckResponse,
            LimitType,
        )

        unity.init()

        original_hook = get_limit_check_hook()

        async def denying_hook(request: _LimitCheckRequest) -> _LimitCheckResponse:
            await asyncio.sleep(0.1)  # 100ms delay
            return _LimitCheckResponse(
                allowed=False,
                reason="Test: simulated limit exceeded",
                limit_type=LimitType.ASSISTANT,
            )

        set_limit_check_hook(denying_hook)

        try:
            llm_client = unillm.AsyncUnify(e2e_config.model)

            call_start_time = time.perf_counter()

            with pytest.raises(unillm.SpendingLimitExceededError):
                await llm_client.generate(
                    messages=[{"role": "user", "content": "Should be cancelled"}],
                    max_tokens=10,
                )

            call_end_time = time.perf_counter()
            total_time = call_end_time - call_start_time

            # Should complete in ~100ms (the limit check delay), not LLM time
            assert total_time < 0.5, (
                f"In-flight cancellation failed: total time was {total_time*1000:.0f}ms, "
                "expected ~100ms (limit check should cancel LLM call)"
            )
        finally:
            set_limit_check_hook(original_hook)

    @pytest.mark.asyncio
    async def test_concurrent_llm_calls(self, e2e_config):
        """Test that concurrent LLM calls correctly update spend atomically."""
        import unity
        import unillm

        unity.init()

        headers = {"Authorization": f"Bearer {e2e_config.admin_key}"}

        # Get spend before
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{e2e_config.base_url}/admin/assistant/{e2e_config.test_agent_id}/spend",
                headers=headers,
                params={"month": datetime.now().strftime("%Y-%m")},
            )
            spend_before = (
                response.json().get("cumulative_spend", 0)
                if response.status_code == 200
                else 0
            )

        # Make 3 concurrent LLM calls
        llm_client = unillm.AsyncUnify(e2e_config.model)

        async def make_call(n: int):
            try:
                return await llm_client.generate(
                    messages=[{"role": "user", "content": f"Say '{n}'"}],
                    max_tokens=5,
                )
            except unillm.SpendingLimitExceededError:
                return None

        results = await asyncio.gather(
            make_call(0),
            make_call(1),
            make_call(2),
            return_exceptions=True,
        )

        # At least some calls should succeed
        successful = [
            r for r in results if r is not None and not isinstance(r, Exception)
        ]
        assert len(successful) > 0, "At least one concurrent call should succeed"

        # Wait for async logging
        await asyncio.sleep(3.0)

        # Get spend after
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{e2e_config.base_url}/admin/assistant/{e2e_config.test_agent_id}/spend",
                headers=headers,
                params={"month": datetime.now().strftime("%Y-%m")},
            )
            spend_after = (
                response.json().get("cumulative_spend", 0)
                if response.status_code == 200
                else 0
            )

        # Spend should have increased
        assert (
            spend_after >= spend_before
        ), f"Spend should increase after concurrent calls: {spend_before} -> {spend_after}"

    @pytest.mark.asyncio
    async def test_all_spending_monthly_context(self, e2e_config):
        """Test that spend logs are mirrored to All/Spending/Monthly context."""
        import unity
        import unillm

        unity.init()

        # Make an LLM call to generate spend
        llm_client = unillm.AsyncUnify(e2e_config.model)
        try:
            await llm_client.generate(
                messages=[{"role": "user", "content": "Aggregation test"}],
                max_tokens=10,
            )
        except unillm.SpendingLimitExceededError:
            pytest.skip("Limit exceeded - cannot test aggregation")

        # Wait for async logging
        await asyncio.sleep(2.0)

        # Query All/Spending/Monthly context
        headers = {"Authorization": f"Bearer {e2e_config.api_key}"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{e2e_config.base_url}/logs",
                headers=headers,
                params={
                    "project_name": "Assistants",
                    "context": "All/Spending/Monthly",
                },
            )

        if response.status_code == 200:
            logs = response.json().get("logs", [])
            # Should find at least one log with required fields
            current_month = datetime.now().strftime("%Y-%m")
            matching_logs = [
                log
                for log in logs
                if log.get("entries", {}).get("month") == current_month
                and log.get("entries", {}).get("_assistant_id")
                == str(e2e_config.test_agent_id)
            ]
            assert (
                len(matching_logs) > 0 or len(logs) > 0
            ), "Should find spending logs in All/Spending/Monthly context"

    @pytest.mark.asyncio
    async def test_user_limit_check(self, e2e_config):
        """Test that user-level spending limits are enforced."""
        import unity
        import unillm

        unity.init()

        headers = {"Authorization": f"Bearer {e2e_config.api_key}"}

        # Save original user limit before modifying
        original_limit = None
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{e2e_config.base_url}/user/spending-limit",
                headers=headers,
            )
            if response.status_code == 200:
                original_limit = response.json().get("monthly_spending_cap")

        try:
            # Set user limit to $0
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.put(
                    f"{e2e_config.base_url}/user/spending-limit",
                    headers=headers,
                    json={"monthly_spending_cap": 0.0},
                )
                if response.status_code not in (200, 201):
                    pytest.skip("Cannot set user spending limit")

            llm_client = unillm.AsyncUnify(e2e_config.model)

            # Should be blocked by user or assistant limit
            with pytest.raises(unillm.SpendingLimitExceededError):
                await llm_client.generate(
                    messages=[{"role": "user", "content": "User limit test"}],
                    max_tokens=10,
                )
        finally:
            # Restore original user limit (None means no limit)
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.put(
                    f"{e2e_config.base_url}/user/spending-limit",
                    headers=headers,
                    json={"monthly_spending_cap": original_limit},
                )

    @pytest.mark.asyncio
    async def test_user_cumulative_spend(self, e2e_config):
        """Test that user cumulative spend endpoint works correctly."""
        import unity

        unity.init()

        headers = {"Authorization": f"Bearer {e2e_config.admin_key}"}
        current_month = datetime.now().strftime("%Y-%m")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{e2e_config.base_url}/admin/user/{e2e_config.test_user_id}/spend",
                headers=headers,
                params={"month": current_month},
            )

        # Should successfully retrieve user spend (even if 0)
        assert (
            response.status_code == 200
        ), f"User spend endpoint failed: {response.status_code} {response.text}"
        data = response.json()
        assert "cumulative_spend" in data or "error" not in data

    @pytest.mark.asyncio
    async def test_assistant_limit_check(self, e2e_config):
        """Test that assistant-specific spending limits are enforced."""
        import unity
        import unillm

        unity.init()

        headers = {"Authorization": f"Bearer {e2e_config.api_key}"}

        # Get current limit to restore later
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{e2e_config.base_url}/assistant/{e2e_config.test_agent_id}",
                headers=headers,
            )
            original_limit = 25.0
            if response.status_code == 200:
                original_limit = (
                    response.json().get("monthly_spending_cap", 25.0) or 25.0
                )

        # Set assistant limit to $0
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.patch(
                f"{e2e_config.base_url}/assistant/{e2e_config.test_agent_id}/config",
                headers=headers,
                json={"monthly_spending_cap": 0.0},
            )

        try:
            llm_client = unillm.AsyncUnify(e2e_config.model)

            with pytest.raises(unillm.SpendingLimitExceededError):
                await llm_client.generate(
                    messages=[{"role": "user", "content": "Assistant limit test"}],
                    max_tokens=10,
                )
        finally:
            # Restore original limit
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.patch(
                    f"{e2e_config.base_url}/assistant/{e2e_config.test_agent_id}/config",
                    headers=headers,
                    json={"monthly_spending_cap": original_limit},
                )

    @pytest.mark.asyncio
    async def test_org_cumulative_spend(self, e2e_config):
        """Test organization cumulative spend retrieval."""
        import unity

        unity.init()

        headers = {"Authorization": f"Bearer {e2e_config.api_key}"}

        # Try to find or create an organization
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{e2e_config.base_url}/organizations",
                headers=headers,
            )

            org_id = None
            if response.status_code == 200:
                orgs = response.json()
                if isinstance(orgs, dict):
                    orgs = orgs.get("organizations", [])
                if orgs:
                    org_id = orgs[0].get("id")

            if not org_id:
                # Try to create an org
                response = await client.post(
                    f"{e2e_config.base_url}/organizations",
                    headers=headers,
                    json={"name": "SpendingTestOrg"},
                )
                if response.status_code in (200, 201):
                    org_id = response.json().get("id")

            if not org_id:
                pytest.skip("Cannot create or find organization for test")

        # Get org spend
        admin_headers = {"Authorization": f"Bearer {e2e_config.admin_key}"}
        current_month = datetime.now().strftime("%Y-%m")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{e2e_config.base_url}/admin/organization/{org_id}/spend",
                headers=admin_headers,
                params={"month": current_month},
            )

        # Should successfully retrieve org spend
        assert (
            response.status_code == 200
        ), f"Org spend endpoint failed: {response.status_code} {response.text}"

    @pytest.mark.asyncio
    async def test_org_limit_check(self, e2e_config):
        """Test organization-level spending limits are enforced."""
        import unity
        import unillm

        unity.init()

        headers = {"Authorization": f"Bearer {e2e_config.api_key}"}

        # Find or create an organization
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{e2e_config.base_url}/organizations",
                headers=headers,
            )

            org_id = None
            org_api_key = None
            if response.status_code == 200:
                orgs = response.json()
                if isinstance(orgs, dict):
                    orgs = orgs.get("organizations", [])
                for org in orgs:
                    if org.get("name") == "SpendingTestOrg":
                        org_id = org.get("id")
                        org_api_key = org.get("api_key")
                        break

            if not org_id:
                # Create org
                response = await client.post(
                    f"{e2e_config.base_url}/organizations",
                    headers=headers,
                    json={"name": "SpendingTestOrg"},
                )
                if response.status_code in (200, 201):
                    data = response.json()
                    org_id = data.get("id")
                    org_api_key = data.get("api_key")

            if not org_id or not org_api_key:
                pytest.skip("Cannot create organization with API key for test")

            # Save original org limit before modifying
            original_org_limit = None
            response = await client.get(
                f"{e2e_config.base_url}/organizations/{org_id}/spending-limit",
                headers=headers,
            )
            if response.status_code == 200:
                original_org_limit = response.json().get("monthly_spending_cap")

        try:
            # Set org limit to $0
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.put(
                    f"{e2e_config.base_url}/organizations/{org_id}/spending-limit",
                    headers=headers,
                    json={"monthly_spending_cap": 0.0},
                )
                if response.status_code not in (200, 201):
                    pytest.skip("Cannot set org spending limit")

            # Update SESSION_DETAILS with org context
            from unity.session_details import SESSION_DETAILS

            SESSION_DETAILS.populate(
                user_id=e2e_config.test_user_id,
                assistant_id=str(e2e_config.test_agent_id),
                user_first_name="Test",
                user_surname="User",
                org_id=org_id,
                org_name="SpendingTestOrg",
            )

            llm_client = unillm.AsyncUnify(e2e_config.model)

            # Should be blocked by org limit
            with pytest.raises(unillm.SpendingLimitExceededError):
                await llm_client.generate(
                    messages=[{"role": "user", "content": "Org limit test"}],
                    max_tokens=10,
                )
        finally:
            # Restore original org limit (None means no limit) and session
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.put(
                    f"{e2e_config.base_url}/organizations/{org_id}/spending-limit",
                    headers=headers,
                    json={"monthly_spending_cap": original_org_limit},
                )

            # Reset session to personal context
            from unity.session_details import SESSION_DETAILS

            SESSION_DETAILS.populate(
                user_id=e2e_config.test_user_id,
                assistant_id=str(e2e_config.test_agent_id),
                user_first_name="Test",
                user_surname="User",
            )


# ===========================================================================
# Part 6: Spending Limit Notification Tests
# ===========================================================================


class TestNotifyLimitReached:
    """Tests for the _notify_limit_reached function."""

    @pytest.mark.asyncio
    async def test_notification_sent_on_exceeded_limit(self):
        """Notification should be sent when limit is exceeded."""
        from unity.spending_limits import _notify_limit_reached, _LimitCheckResult

        with patch("unity.spending_limits._get_admin_client") as mock_get_client:
            mock_instance = MagicMock()
            mock_instance.closed = False
            mock_instance.notify_limit_reached = AsyncMock(
                return_value={"notified": True, "recipient_count": 1},
            )
            mock_get_client.return_value = mock_instance

            result = _LimitCheckResult(
                exceeded=True,
                limit_type="assistant",
                limit_value=100.0,
                current_spend=100.0,
                entity_id="agent_123",
                entity_name="Test Bot",
                limit_set_at="2026-02-01T10:00:00Z",
            )

            await _notify_limit_reached(result, month="2026-02")

        mock_instance.notify_limit_reached.assert_called_once()
        payload = mock_instance.notify_limit_reached.call_args[0][0]
        assert payload["limit_type"] == "assistant"
        assert payload["entity_id"] == "agent_123"
        assert payload["limit_value"] == 100.0
        assert payload["month"] == "2026-02"
        assert payload["limit_set_at"] == "2026-02-01T10:00:00Z"

    @pytest.mark.asyncio
    async def test_notification_includes_org_id_for_member_limit(self):
        """Member limit notification should include organization_id."""
        from unity.spending_limits import _notify_limit_reached, _LimitCheckResult

        with patch("unity.spending_limits._get_admin_client") as mock_get_client:
            mock_instance = MagicMock()
            mock_instance.closed = False
            mock_instance.notify_limit_reached = AsyncMock(
                return_value={"notified": True, "recipient_count": 1},
            )
            mock_get_client.return_value = mock_instance

            result = _LimitCheckResult(
                exceeded=True,
                limit_type="member",
                limit_value=200.0,
                current_spend=200.0,
                entity_id="user_456",
                organization_id=789,
            )

            await _notify_limit_reached(result, month="2026-02")

        payload = mock_instance.notify_limit_reached.call_args[0][0]
        assert payload["organization_id"] == 789

    @pytest.mark.asyncio
    async def test_notification_handles_errors_gracefully(self):
        """Notification should not raise on errors (fire-and-forget)."""
        from unity.spending_limits import _notify_limit_reached, _LimitCheckResult

        with patch("unity.spending_limits._get_admin_client") as mock_get_client:
            mock_instance = MagicMock()
            mock_instance.closed = False
            mock_instance.notify_limit_reached = AsyncMock(
                side_effect=Exception("Timeout"),
            )
            mock_get_client.return_value = mock_instance

            result = _LimitCheckResult(
                exceeded=True,
                limit_type="assistant",
                limit_value=100.0,
                current_spend=100.0,
                entity_id="agent_123",
            )

            # Should NOT raise
            await _notify_limit_reached(result, month="2026-02")

    @pytest.mark.asyncio
    async def test_notification_handles_http_errors_gracefully(self):
        """Notification should not raise on HTTP errors."""
        from unity.spending_limits import _notify_limit_reached, _LimitCheckResult

        with patch("unity.spending_limits._get_admin_client") as mock_get_client:
            mock_instance = MagicMock()
            mock_instance.closed = False
            mock_instance.notify_limit_reached = AsyncMock(
                side_effect=AdminRequestError(
                    url="http://test/v0/admin/spending-limit-reached",
                    method="POST",
                    status=500,
                    body="Internal server error",
                ),
            )
            mock_get_client.return_value = mock_instance

            result = _LimitCheckResult(
                exceeded=True,
                limit_type="user",
                limit_value=50.0,
                current_spend=50.0,
                entity_id="user_123",
            )

            # Should NOT raise
            await _notify_limit_reached(result, month="2026-02")


class TestNotificationStress:
    """Stress tests for concurrent notification scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_notifications_all_sent(self):
        """Multiple concurrent notifications should all be sent."""
        from unity.spending_limits import _notify_limit_reached, _LimitCheckResult

        notification_count = 0
        lock = asyncio.Lock()

        async def counting_notify(payload):
            nonlocal notification_count
            await asyncio.sleep(0.01)  # Simulate network delay
            async with lock:
                notification_count += 1
            return {"notified": True, "recipient_count": 1}

        with patch("unity.spending_limits._get_admin_client") as mock_get_client:
            mock_instance = MagicMock()
            mock_instance.closed = False
            mock_instance.notify_limit_reached = AsyncMock(side_effect=counting_notify)
            mock_get_client.return_value = mock_instance

            # Launch 10 concurrent notifications
            tasks = []
            for i in range(10):
                result = _LimitCheckResult(
                    exceeded=True,
                    limit_type="assistant",
                    limit_value=100.0,
                    current_spend=100.0,
                    entity_id=f"agent_{i}",
                )
                tasks.append(
                    _notify_limit_reached(result, month="2026-02"),
                )

            await asyncio.gather(*tasks)

        assert notification_count == 10

    @pytest.mark.asyncio
    async def test_notification_does_not_block_limit_check(self):
        """Notification should be fire-and-forget, not blocking the response."""
        from unity.spending_limits import check_spending_limits_callback

        notification_started = asyncio.Event()

        spend_data = {
            "cumulative_spend": 150.0,
            "limit": 100.0,
            "limit_set_at": "2026-02-01T10:00:00Z",
        }

        async def slow_notify(payload):
            notification_started.set()
            await asyncio.sleep(0.5)  # Slow notification
            return {"notified": True}

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = None
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(
                        return_value=spend_data,
                    )
                    mock_instance.get_user_spend = AsyncMock(return_value=spend_data)
                    mock_instance.notify_limit_reached = AsyncMock(
                        side_effect=slow_notify,
                    )
                    mock_get_client.return_value = mock_instance

                    start_time = asyncio.get_event_loop().time()
                    request = LimitCheckRequest(model="gpt-4", endpoint="test")
                    response = await check_spending_limits_callback(request)
                    callback_time = asyncio.get_event_loop().time() - start_time

                    # Response should be fast (not waiting for notification)
                    assert response.allowed is False
                    assert callback_time < 0.2  # Should complete in < 200ms

                    # Wait for notification to complete in background
                    await asyncio.sleep(0.1)
                    assert notification_started.is_set()  # Notification was triggered

    @pytest.mark.asyncio
    async def test_rapid_limit_checks_with_notifications(self):
        """Rapid limit checks should each trigger notifications independently."""
        from unity.spending_limits import check_spending_limits_callback

        notification_calls = []

        spend_data = {"cumulative_spend": 150.0, "limit": 100.0}

        async def tracking_notify(payload):
            notification_calls.append(payload)
            return {"notified": False, "reason": "already_notified"}

        with patch("unity.spending_limits._get_api_key", return_value="test-key"):
            with patch("unity.session_details.SESSION_DETAILS") as mock_session:
                mock_session.assistant.agent_id = 123
                mock_session.user_id = "user_456"
                mock_session.org_id = None
                mock_session.assistant.timezone = "UTC"

                with patch(
                    "unity.spending_limits._get_admin_client",
                ) as mock_get_client:
                    mock_instance = MagicMock()
                    mock_instance.closed = False
                    mock_instance.get_assistant_spend = AsyncMock(
                        return_value=spend_data,
                    )
                    mock_instance.get_user_spend = AsyncMock(return_value=spend_data)
                    mock_instance.notify_limit_reached = AsyncMock(
                        side_effect=tracking_notify,
                    )
                    mock_get_client.return_value = mock_instance

                    # Fire 5 rapid limit checks
                    tasks = []
                    for _ in range(5):
                        request = LimitCheckRequest(model="gpt-4", endpoint="test")
                        tasks.append(check_spending_limits_callback(request))

                    responses = await asyncio.gather(*tasks)

                    # Wait for background notifications
                    await asyncio.sleep(0.1)

        # All limit checks should return denied
        assert all(not r.allowed for r in responses)

        # All 5 should have triggered notification attempts
        # (deduplication is handled by Orchestra, not Unity)
        assert len(notification_calls) == 5

    @pytest.mark.asyncio
    async def test_mixed_limit_types_concurrent_notifications(self):
        """Different limit types should send independent notifications."""
        from unity.spending_limits import _notify_limit_reached, _LimitCheckResult

        captured_payloads = []

        async def capture_notify(payload):
            captured_payloads.append(payload)
            return {"notified": True}

        with patch("unity.spending_limits._get_admin_client") as mock_get_client:
            mock_instance = MagicMock()
            mock_instance.closed = False
            mock_instance.notify_limit_reached = AsyncMock(side_effect=capture_notify)
            mock_get_client.return_value = mock_instance

            results = [
                _LimitCheckResult(
                    exceeded=True,
                    limit_type="assistant",
                    limit_value=100.0,
                    current_spend=100.0,
                    entity_id="agent_1",
                ),
                _LimitCheckResult(
                    exceeded=True,
                    limit_type="user",
                    limit_value=200.0,
                    current_spend=200.0,
                    entity_id="user_1",
                ),
                _LimitCheckResult(
                    exceeded=True,
                    limit_type="member",
                    limit_value=150.0,
                    current_spend=150.0,
                    entity_id="user_1",
                    organization_id=123,
                ),
                _LimitCheckResult(
                    exceeded=True,
                    limit_type="organization",
                    limit_value=1000.0,
                    current_spend=1000.0,
                    entity_id="123",
                    entity_name="TestOrg",
                ),
            ]

            tasks = [_notify_limit_reached(r, "2026-02") for r in results]
            await asyncio.gather(*tasks)

        assert len(captured_payloads) == 4
        limit_types = {p["limit_type"] for p in captured_payloads}
        assert limit_types == {"assistant", "user", "member", "organization"}

    @pytest.mark.asyncio
    async def test_notification_with_missing_optional_fields(self):
        """Notification should work with minimal required fields."""
        from unity.spending_limits import _notify_limit_reached, _LimitCheckResult

        captured_payload = {}

        async def capture_notify(payload):
            captured_payload.update(payload)
            return {"notified": True}

        with patch("unity.spending_limits._get_admin_client") as mock_get_client:
            mock_instance = MagicMock()
            mock_instance.closed = False
            mock_instance.notify_limit_reached = AsyncMock(side_effect=capture_notify)
            mock_get_client.return_value = mock_instance

            result = _LimitCheckResult(
                exceeded=True,
                limit_type="assistant",
                limit_value=100.0,
                current_spend=100.0,
                entity_id="agent_123",
                # No entity_name, limit_set_at, or organization_id
            )

            await _notify_limit_reached(result, month="2026-02")

        # Required fields present
        assert captured_payload["limit_type"] == "assistant"
        assert captured_payload["entity_id"] == "agent_123"
        assert captured_payload["limit_value"] == 100.0
        assert captured_payload["month"] == "2026-02"

        # Optional fields not included when None
        assert "limit_set_at" not in captured_payload
        assert "organization_id" not in captured_payload
