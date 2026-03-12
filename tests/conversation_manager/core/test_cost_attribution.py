"""
tests/conversation_manager/core/test_cost_attribution.py
=========================================================

Tests for per-user LLM cost attribution.

Covers the full chain: contact provisioning -> ContextVar threading through
the CM brain and act tool -> spending write path with per-user rows ->
concurrent isolation between parallel act calls.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)
from unity.conversation_manager.domains.contact_index import ContactIndex
from unity.conversation_manager.domains.notifications import NotificationBar
from unity.conversation_manager.types.mode import Mode
from unity.events.cost_attribution import COST_ATTRIBUTION

SUPERVISOR_UID = "supervisor_uid_999"
ALICE_UID = "alice_uid_aaa"
BOB_UID = "bob_uid_bbb"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _build_contact_index() -> ContactIndex:
    """ContactIndex with system contacts (have user_id + is_system), an external
    contact, and a non-org platform user contact."""
    ci = ContactIndex()
    ci.set_fallback_contacts(
        [
            {
                "contact_id": 2,
                "first_name": "Alice",
                "user_id": ALICE_UID,
                "is_system": True,
            },
            {
                "contact_id": 3,
                "first_name": "Bob",
                "user_id": BOB_UID,
                "is_system": True,
            },
            {"contact_id": 10, "first_name": "External"},  # no user_id, no is_system
            # Platform user from a different org — has a user_id but is NOT a
            # system contact in this assistant's org.
            {
                "contact_id": 20,
                "first_name": "OtherOrg",
                "user_id": "other_org_uid_zzz",
            },
        ],
    )
    return ci


def _mock_session_details(mock_sd, *, org_id=1):
    """Configure a SESSION_DETAILS mock with supervisor identity.

    Defaults to org context (org_id=1).  Pass org_id=None for personal account tests.
    """
    mock_sd.user.id = SUPERVISOR_UID
    mock_sd.user_context = "u1"
    mock_sd.assistant_context = "a1"
    mock_sd.assistant.agent_id = 42
    mock_sd.assistant.timezone = "UTC"
    mock_sd.is_initialized = True
    mock_sd.org_id = org_id
    mock_sd.org_name = "TestOrg" if org_id else ""
    mock_sd.unify_key = "test-key"


@pytest.fixture
def mock_cm():
    cm = MagicMock()
    cm.mode = Mode.TEXT
    cm.contact_index = _build_contact_index()
    cm.in_flight_actions = {}
    cm.completed_actions = {}
    cm.notifications_bar = NotificationBar()
    cm.chat_history = []
    cm._current_state_snapshot = None
    cm._current_snapshot_state = None
    cm._pending_steering_tasks = set()
    cm._initialized = asyncio.Event()
    cm._initialized.set()
    cm._session_logger = MagicMock()
    cm.request_llm_run = AsyncMock()
    cm.event_broker = MagicMock()
    cm.event_broker.publish = AsyncMock()
    cm.call_manager = MagicMock()
    return cm


@pytest.fixture
def brain_action_tools(mock_cm):
    with patch(
        "unity.conversation_manager.domains.brain_action_tools.get_event_broker",
    ) as mock_broker:
        mock_broker.return_value = MagicMock()
        mock_broker.return_value.publish = AsyncMock()
        yield ConversationManagerBrainActionTools(mock_cm)


def _make_fake_actor():
    """Actor mock that captures kwargs and records COST_ATTRIBUTION at call time."""
    captured: dict[str, Any] = {}

    async def fake_act(request, **kwargs):
        captured.update(kwargs)
        captured["_cost_attribution_at_call"] = COST_ATTRIBUTION.get()
        handle = MagicMock()
        handle.result = AsyncMock(return_value="done")
        handle.next_notification = AsyncMock(side_effect=asyncio.CancelledError)
        handle.next_clarification = AsyncMock(side_effect=asyncio.CancelledError)
        return handle

    actor = MagicMock()
    actor.act = fake_act
    return actor, captured


# ═════════════════════════════════════════════════════════════════════════════
# 1-2. act tool sets COST_ATTRIBUTION
# ═════════════════════════════════════════════════════════════════════════════


class TestActCostAttribution:

    @pytest.mark.asyncio
    async def test_act_sets_attribution_from_contact_user_id(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """act() resolves requesting_contact_id -> user_id and sets ContextVar."""
        actor, captured = _make_fake_actor()
        mock_cm.actor = actor

        with patch("unity.session_details.SESSION_DETAILS") as mock_sd:
            _mock_session_details(mock_sd)
            await brain_action_tools.act(
                query="find Alice's email",
                requesting_contact_id=2,
            )

        assert captured["_cost_attribution_at_call"] == [ALICE_UID]

    @pytest.mark.asyncio
    async def test_act_falls_back_to_supervisor_for_external_contact(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """External contacts (no user_id) fall back to supervisor."""
        actor, captured = _make_fake_actor()
        mock_cm.actor = actor

        with patch("unity.session_details.SESSION_DETAILS") as mock_sd:
            _mock_session_details(mock_sd)
            await brain_action_tools.act(
                query="check something",
                requesting_contact_id=10,
            )

        assert captured["_cost_attribution_at_call"] == [SUPERVISOR_UID]

    @pytest.mark.asyncio
    async def test_act_falls_back_for_non_org_platform_user(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """A contact with a user_id from another org (not is_system) falls back
        to supervisor — costs must not leak to a different organization."""
        actor, captured = _make_fake_actor()
        mock_cm.actor = actor

        with patch("unity.session_details.SESSION_DETAILS") as mock_sd:
            _mock_session_details(mock_sd)
            await brain_action_tools.act(
                query="look up something",
                requesting_contact_id=20,  # has user_id but NOT is_system
            )

        assert captured["_cost_attribution_at_call"] == [SUPERVISOR_UID]

    @pytest.mark.asyncio
    async def test_act_skips_attribution_for_personal_account(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """Personal accounts (no org) skip attribution entirely."""
        actor, captured = _make_fake_actor()
        mock_cm.actor = actor

        with patch("unity.session_details.SESSION_DETAILS") as mock_sd:
            _mock_session_details(mock_sd, org_id=None)
            await brain_action_tools.act(
                query="find Alice's email",
                requesting_contact_id=2,
            )

        # For personal accounts, COST_ATTRIBUTION should NOT be set by act().
        # It stays at whatever the ambient value is (None by default), and the
        # spending write path falls back to SESSION_DETAILS.user.id.
        assert captured["_cost_attribution_at_call"] is None or captured[
            "_cost_attribution_at_call"
        ] == [SUPERVISOR_UID]


# ═════════════════════════════════════════════════════════════════════════════
# 3-4. _run_llm sets COST_ATTRIBUTION from triggering_contact_id
# ═════════════════════════════════════════════════════════════════════════════


class TestRunLlmCostAttribution:
    """Test the attribution logic that _run_llm executes at its start.

    We replicate the exact logic from _run_llm rather than invoking the full
    method, which requires dozens of CM attributes that are irrelevant to
    cost attribution.
    """

    def _resolve_attribution(
        self,
        contact_index,
        trace_meta,
        supervisor_uid,
        *,
        org_id=1,
    ):
        """Replicate the attribution logic from _run_llm."""
        if org_id is None:
            return  # personal account — leave COST_ATTRIBUTION untouched
        triggering_contact_id = trace_meta.get("triggering_contact_id")
        attributed_user_id = None
        if triggering_contact_id is not None:
            contact = contact_index.get_contact(
                contact_id=triggering_contact_id,
            )
            if contact and contact.get("is_system"):
                attributed_user_id = contact.get("user_id")
        if attributed_user_id:
            COST_ATTRIBUTION.set([attributed_user_id])
        else:
            COST_ATTRIBUTION.set([supervisor_uid])

    def test_run_llm_attributes_to_triggering_contact(self):
        """Attribution resolves triggering contact's user_id."""
        ci = _build_contact_index()
        self._resolve_attribution(ci, {"triggering_contact_id": 2}, SUPERVISOR_UID)
        assert COST_ATTRIBUTION.get() == [ALICE_UID]

    def test_run_llm_falls_back_to_supervisor_when_no_trigger(self):
        """Attribution falls back to supervisor when no triggering contact."""
        ci = _build_contact_index()
        self._resolve_attribution(ci, {}, SUPERVISOR_UID)
        assert COST_ATTRIBUTION.get() == [SUPERVISOR_UID]

    def test_run_llm_falls_back_for_unknown_contact(self):
        """Attribution falls back to supervisor for unknown contact_id."""
        ci = _build_contact_index()
        self._resolve_attribution(ci, {"triggering_contact_id": 999}, SUPERVISOR_UID)
        assert COST_ATTRIBUTION.get() == [SUPERVISOR_UID]

    def test_run_llm_falls_back_for_external_contact(self):
        """External contact (no user_id) falls back to supervisor."""
        ci = _build_contact_index()
        self._resolve_attribution(ci, {"triggering_contact_id": 10}, SUPERVISOR_UID)
        assert COST_ATTRIBUTION.get() == [SUPERVISOR_UID]

    def test_run_llm_falls_back_for_non_org_platform_user(self):
        """Contact with user_id but NOT is_system falls back to supervisor."""
        ci = _build_contact_index()
        self._resolve_attribution(ci, {"triggering_contact_id": 20}, SUPERVISOR_UID)
        assert COST_ATTRIBUTION.get() == [SUPERVISOR_UID]

    def test_run_llm_skips_for_personal_account(self):
        """Personal accounts leave COST_ATTRIBUTION untouched."""
        token = COST_ATTRIBUTION.set(None)
        try:
            ci = _build_contact_index()
            self._resolve_attribution(
                ci,
                {"triggering_contact_id": 2},
                SUPERVISOR_UID,
                org_id=None,
            )
            assert COST_ATTRIBUTION.get() is None
        finally:
            COST_ATTRIBUTION.reset(token)


# ═════════════════════════════════════════════════════════════════════════════
# 5-6. _update_cumulative_spend reads ContextVar
# ═════════════════════════════════════════════════════════════════════════════


class TestSpendingWritePath:

    @pytest.mark.asyncio
    async def test_spend_uses_cost_attribution_contextvar(self):
        """_update_cumulative_spend reads COST_ATTRIBUTION and passes data_overrides."""
        from unity.events.llm_event_hook import _update_cumulative_spend

        token = COST_ATTRIBUTION.set([ALICE_UID])
        try:
            with (
                patch("unity.session_details.SESSION_DETAILS") as mock_sd,
                patch(
                    "unity.common.log_utils.atomic_upsert",
                    new_callable=AsyncMock,
                ) as mock_upsert,
            ):
                _mock_session_details(mock_sd)
                await _update_cumulative_spend(0.05)

            mock_upsert.assert_called_once()
            call_kwargs = mock_upsert.call_args.kwargs
            assert "_user_id" in call_kwargs["unique_keys"]
            assert call_kwargs["data_overrides"] == {"_user_id": ALICE_UID}
        finally:
            COST_ATTRIBUTION.reset(token)

    @pytest.mark.asyncio
    async def test_spend_falls_back_to_supervisor_when_contextvar_unset(self):
        """When COST_ATTRIBUTION is None, spend is attributed to supervisor."""
        from unity.events.llm_event_hook import _update_cumulative_spend

        token = COST_ATTRIBUTION.set(None)
        try:
            with (
                patch("unity.session_details.SESSION_DETAILS") as mock_sd,
                patch(
                    "unity.common.log_utils.atomic_upsert",
                    new_callable=AsyncMock,
                ) as mock_upsert,
            ):
                _mock_session_details(mock_sd)
                await _update_cumulative_spend(0.05)

            mock_upsert.assert_called_once()
            call_kwargs = mock_upsert.call_args.kwargs
            assert call_kwargs["data_overrides"] == {"_user_id": SUPERVISOR_UID}
        finally:
            COST_ATTRIBUTION.reset(token)


# ═════════════════════════════════════════════════════════════════════════════
# 6b. _llm_event_to_eventbus includes _attributed_user_id in payload
# ═════════════════════════════════════════════════════════════════════════════


class TestLLMEventPayloadAttribution:
    """Verify that LLM event payloads carry _attributed_user_id so the console
    usage chart can filter by the user who triggered the call."""

    def test_payload_uses_cost_attribution_when_set(self):
        """When COST_ATTRIBUTION is set (org member), _attributed_user_id
        should reflect the attributed user, not the supervisor."""
        from unillm import LLMEvent
        from unity.events.llm_event_hook import _llm_event_to_eventbus
        from unity.events.types.llm import LLMPayload

        token = COST_ATTRIBUTION.set([ALICE_UID])
        try:
            with patch("unity.session_details.SESSION_DETAILS") as mock_sd:
                _mock_session_details(mock_sd)

                # Capture the payload by intercepting Event creation
                with patch("unity.events.event_bus.Event") as MockEvent:
                    llm_event = LLMEvent(
                        request={"model": "gpt-4o", "messages": []},
                        billed_cost=0.01,
                    )
                    _llm_event_to_eventbus(llm_event)

                    MockEvent.assert_called_once()
                    payload = MockEvent.call_args.kwargs["payload"]
                    assert isinstance(payload, LLMPayload)
                    assert payload.model_dump()["_attributed_user_id"] == ALICE_UID
        finally:
            COST_ATTRIBUTION.reset(token)

    def test_payload_falls_back_to_supervisor_when_unset(self):
        """When COST_ATTRIBUTION is None (personal account / no override),
        _attributed_user_id should fall back to SESSION_DETAILS.user.id."""
        from unillm import LLMEvent
        from unity.events.llm_event_hook import _llm_event_to_eventbus
        from unity.events.types.llm import LLMPayload

        token = COST_ATTRIBUTION.set(None)
        try:
            with patch("unity.session_details.SESSION_DETAILS") as mock_sd:
                _mock_session_details(mock_sd)

                with patch("unity.events.event_bus.Event") as MockEvent:
                    llm_event = LLMEvent(
                        request={"model": "gpt-4o", "messages": []},
                        billed_cost=0.01,
                    )
                    _llm_event_to_eventbus(llm_event)

                    MockEvent.assert_called_once()
                    payload = MockEvent.call_args.kwargs["payload"]
                    assert isinstance(payload, LLMPayload)
                    assert payload.model_dump()["_attributed_user_id"] == SUPERVISOR_UID
        finally:
            COST_ATTRIBUTION.reset(token)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Concurrent asyncio.create_task isolation
# ═════════════════════════════════════════════════════════════════════════════


class TestConcurrentIsolation:

    @pytest.mark.asyncio
    async def test_create_task_isolates_cost_attribution(self):
        """Two tasks created with create_task get independent COST_ATTRIBUTION."""
        results: dict[str, list[str] | None] = {}
        barrier = asyncio.Barrier(2)

        async def worker(name: str, uid: str):
            COST_ATTRIBUTION.set([uid])
            await barrier.wait()
            await asyncio.sleep(0)
            results[name] = COST_ATTRIBUTION.get()

        t1 = asyncio.create_task(worker("alice", ALICE_UID))
        t2 = asyncio.create_task(worker("bob", BOB_UID))
        await asyncio.gather(t1, t2)

        assert results["alice"] == [ALICE_UID]
        assert results["bob"] == [BOB_UID]

    @pytest.mark.asyncio
    async def test_parent_set_does_not_leak_into_child_task(self):
        """A child task inherits the parent's value at creation, not later mutations."""
        child_saw: list[Any] = []

        COST_ATTRIBUTION.set([ALICE_UID])

        async def child():
            child_saw.append(COST_ATTRIBUTION.get())

        task = asyncio.create_task(child())
        COST_ATTRIBUTION.set([BOB_UID])
        await task

        assert child_saw[0] == [
            ALICE_UID,
        ], "Child should see the value at task creation time, not the later mutation"


# ═════════════════════════════════════════════════════════════════════════════
# 8. request_llm_run stores triggering_contact_id
# ═════════════════════════════════════════════════════════════════════════════


class TestRequestLlmRunMetadata:

    @pytest.mark.asyncio
    async def test_triggering_contact_id_stored_in_meta(self):
        """request_llm_run stores triggering_contact_id in pending meta."""
        from unity.conversation_manager.conversation_manager import (
            ConversationManager,
        )

        cm = MagicMock(spec=ConversationManager)
        cm._llm_request_seq = 0
        cm._current_event_trace = {}
        cm._pending_llm_requests = []
        cm._pending_llm_request_meta = []
        cm._session_logger = MagicMock()

        await ConversationManager.request_llm_run(
            cm,
            delay=0,
            triggering_contact_id=2,
        )

        assert len(cm._pending_llm_request_meta) == 1
        meta = cm._pending_llm_request_meta[0]
        assert meta["triggering_contact_id"] == 2

    @pytest.mark.asyncio
    async def test_triggering_contact_id_defaults_to_none(self):
        """Without explicit triggering_contact_id, meta stores None."""
        from unity.conversation_manager.conversation_manager import (
            ConversationManager,
        )

        cm = MagicMock(spec=ConversationManager)
        cm._llm_request_seq = 0
        cm._current_event_trace = {}
        cm._pending_llm_requests = []
        cm._pending_llm_request_meta = []
        cm._session_logger = MagicMock()

        await ConversationManager.request_llm_run(cm, delay=0)

        meta = cm._pending_llm_request_meta[0]
        assert meta["triggering_contact_id"] is None


# ═════════════════════════════════════════════════════════════════════════════
# 9. Contact provisioning stores user_id
# ═════════════════════════════════════════════════════════════════════════════


class TestContactProvisioningUserId:

    def test_provision_user_contact_stores_user_id(self):
        """provision_user_contact includes user_id from SESSION_DETAILS."""
        from unity.contact_manager.system_contacts import provision_user_contact

        mock_self = MagicMock()
        mock_self._BUILTIN_FIELDS = {
            "first_name",
            "surname",
            "email_address",
            "phone_number",
            "bio",
            "should_respond",
            "response_policy",
            "timezone",
            "is_system",
            "contact_id",
            "rolling_summary",
        }
        mock_self.USER_MANAGER_RESPONSE_POLICY = ""
        mock_self._create_contact = MagicMock()

        with (
            patch(
                "unity.contact_manager.system_contacts._resolve_user_details",
                return_value={
                    "first_name": "Boss",
                    "last_name": "User",
                    "email": "boss@example.com",
                },
            ),
            patch(
                "unity.contact_manager.system_contacts._ensure_columns_exist",
            ),
            patch("unity.session_details.SESSION_DETAILS") as mock_sd,
            patch("unity.settings.SETTINGS") as mock_settings,
        ):
            mock_settings.DEMO_MODE = False
            mock_sd.is_initialized = True
            mock_sd.user.id = "boss_platform_uid"

            provision_user_contact(mock_self, user_log=None)

        mock_self._create_contact.assert_called_once()
        call_kwargs = mock_self._create_contact.call_args
        # _create_contact is called with **kwargs — check both positional and keyword
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        assert all_kwargs.get("user_id") == "boss_platform_uid"

    def test_provision_user_contact_skips_user_id_when_not_initialized(self):
        """user_id is not stored when SESSION_DETAILS is not yet initialized."""
        from unity.contact_manager.system_contacts import provision_user_contact

        mock_self = MagicMock()
        mock_self._BUILTIN_FIELDS = {
            "first_name",
            "surname",
            "email_address",
            "phone_number",
            "bio",
            "should_respond",
            "response_policy",
            "timezone",
            "is_system",
            "contact_id",
            "rolling_summary",
        }
        mock_self.USER_MANAGER_RESPONSE_POLICY = ""
        mock_self._create_contact = MagicMock()

        with (
            patch(
                "unity.contact_manager.system_contacts._resolve_user_details",
                return_value={
                    "first_name": "Boss",
                    "last_name": "User",
                    "email": "boss@example.com",
                },
            ),
            patch(
                "unity.contact_manager.system_contacts._ensure_columns_exist",
            ),
            patch("unity.session_details.SESSION_DETAILS") as mock_sd,
            patch("unity.settings.SETTINGS") as mock_settings,
        ):
            mock_settings.DEMO_MODE = False
            mock_sd.is_initialized = False
            mock_sd.user.id = "default"

            provision_user_contact(mock_self, user_log=None)

        mock_self._create_contact.assert_called_once()
        call_kwargs = mock_self._create_contact.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        assert "user_id" not in all_kwargs
