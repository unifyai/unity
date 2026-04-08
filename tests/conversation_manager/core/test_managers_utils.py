"""
tests/conversation_manager/test_managers_utils.py
======================================================

Tests for the managers_utils module, including the initialization queue
that holds operations until the ConversationManager is fully initialized,
and the exchange_id caching fix for call/meet utterances.
"""

from __future__ import annotations

import asyncio
import time as _time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.contact_manager.simulated import SimulatedContactManager
from unity.contact_manager.types.contact import UNASSIGNED
from unity.conversation_manager.domains import managers_utils
from unity.conversation_manager.domains.event_handlers import EventHandler
from unity.conversation_manager.events import (
    SyncContacts,
    InboundUnifyMeetUtterance,
    OutboundUnifyMeetUtterance,
    InboundPhoneUtterance,
    OutboundPhoneUtterance,
)
from unity.transcript_manager.simulated import SimulatedTranscriptManager


async def _wait_for_condition(
    predicate,
    *,
    timeout: float = 5.0,
    poll: float = 0.02,
) -> bool:
    """Poll predicate() until True or timeout. Returns whether condition was met."""
    start = _time.perf_counter()
    while _time.perf_counter() - start < timeout:
        if predicate():
            return True
        await asyncio.sleep(poll)
    return False


@pytest.mark.asyncio
async def test_queue_operation_waits_for_initialization():
    """
    Verify that operations queued via queue_operation only execute
    after cm.initialized becomes True.

    This tests the real queueing mechanism in managers_utils.
    """
    # Create a mock CM that starts uninitialized
    mock_cm = MagicMock()
    mock_cm._session_logger = MagicMock()
    mock_cm.contact_manager = SimulatedContactManager()
    mock_cm.notifications_bar = MagicMock()
    mock_cm.initialized = False

    # Wrap _sync_required_contacts to track calls
    with patch.object(
        mock_cm.contact_manager,
        "_sync_required_contacts",
        wraps=mock_cm.contact_manager._sync_required_contacts,
    ) as mock_sync:
        # Fire the SyncContacts event (this queues the operation)
        event = SyncContacts(reason="test sync")
        await EventHandler.handle_event(event, mock_cm)

        # Give a moment for any immediate execution (there shouldn't be any)
        await asyncio.sleep(0.05)

        # Sync should NOT have been called yet - still waiting for initialization
        mock_sync.assert_not_called()

        # Start the operations listener in the background
        listener_task = asyncio.create_task(
            managers_utils.listen_to_operations(mock_cm),
        )

        # Still not called - listener is waiting for initialization
        await asyncio.sleep(0.05)
        mock_sync.assert_not_called()

        # Now mark as initialized - this unblocks the listener
        mock_cm.initialized = True

        # Wait for the queued operation to be processed (poll instead of fixed sleep)
        await _wait_for_condition(lambda: mock_sync.called, timeout=2.0)

        # NOW the sync should have been called
        mock_sync.assert_called_once()
        mock_cm.notifications_bar.push_notif.assert_called_once()

        # Cleanup
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Exchange-ID caching: log_message must set exchange_id on the call manager
# synchronously so that the next queued utterance reuses it.
# ---------------------------------------------------------------------------


def _make_cm_for_log_message() -> MagicMock:
    """Build a minimal mock CM that satisfies log_message requirements."""
    cm = MagicMock()
    cm.contact_manager = SimulatedContactManager()
    cm.transcript_manager = SimulatedTranscriptManager()

    # Real call_manager attributes (not MagicMock auto-attrs)
    cm.call_manager.call_exchange_id = UNASSIGNED
    cm.call_manager.unify_meet_exchange_id = UNASSIGNED
    cm.call_manager.google_meet_exchange_id = UNASSIGNED
    cm.call_manager.call_start_timestamp = None
    cm.call_manager.unify_meet_start_timestamp = None
    cm.call_manager.google_meet_start_timestamp = None

    cm.contact_index = MagicMock()
    cm.contact_index.get_contact = MagicMock(
        return_value={"contact_id": 1, "first_name": "Test", "surname": "User"},
    )
    cm._local_to_global_message_ids = {}
    return cm


@pytest.mark.asyncio
async def test_log_message_caches_unify_meet_exchange_id_synchronously():
    """
    Regression test for the exchange_id race condition.

    When two unify_meet utterances are queued back-to-back, the first
    log_message call must set cm.call_manager.unify_meet_exchange_id
    synchronously so that the second call reuses it — rather than waiting
    for the asynchronous LogMessageResponse handler, which may not have
    run yet.
    """
    cm = _make_cm_for_log_message()
    contact = {"contact_id": 1}

    assert cm.call_manager.unify_meet_exchange_id == UNASSIGNED

    utterance1 = OutboundUnifyMeetUtterance(contact=contact, content="Hello")
    utterance2 = InboundUnifyMeetUtterance(contact=contact, content="Hi there")

    with patch.object(managers_utils, "event_broker", new=MagicMock(publish=AsyncMock())):
        await managers_utils.log_message(cm, utterance1)

        # After the first utterance, the exchange_id must be cached
        first_exchange_id = cm.call_manager.unify_meet_exchange_id
        assert first_exchange_id != UNASSIGNED, (
            "unify_meet_exchange_id should be set synchronously after first log_message"
        )

        await managers_utils.log_message(cm, utterance2)

        # The second utterance must reuse the same exchange_id
        assert cm.call_manager.unify_meet_exchange_id == first_exchange_id, (
            "Second utterance should reuse the cached unify_meet_exchange_id"
        )


@pytest.mark.asyncio
async def test_log_message_caches_call_exchange_id_synchronously():
    """Same as above but for phone calls (call_exchange_id)."""
    cm = _make_cm_for_log_message()
    contact = {"contact_id": 1}

    assert cm.call_manager.call_exchange_id == UNASSIGNED

    utterance1 = OutboundPhoneUtterance(contact=contact, content="Hello")
    utterance2 = InboundPhoneUtterance(contact=contact, content="Hi there")

    with patch.object(managers_utils, "event_broker", new=MagicMock(publish=AsyncMock())):
        await managers_utils.log_message(cm, utterance1)

        first_exchange_id = cm.call_manager.call_exchange_id
        assert first_exchange_id != UNASSIGNED, (
            "call_exchange_id should be set synchronously after first log_message"
        )

        await managers_utils.log_message(cm, utterance2)

        assert cm.call_manager.call_exchange_id == first_exchange_id, (
            "Second utterance should reuse the cached call_exchange_id"
        )


@pytest.mark.asyncio
async def test_log_message_does_not_overwrite_existing_exchange_id():
    """
    If the exchange_id is already set (e.g. from a prior utterance),
    log_message must not overwrite it.
    """
    cm = _make_cm_for_log_message()
    contact = {"contact_id": 1}

    # Pre-set an exchange_id as if a previous utterance already cached it
    cm.call_manager.unify_meet_exchange_id = 42

    utterance = OutboundUnifyMeetUtterance(contact=contact, content="Hello")

    with patch.object(managers_utils, "event_broker", new=MagicMock(publish=AsyncMock())):
        await managers_utils.log_message(cm, utterance)

        assert cm.call_manager.unify_meet_exchange_id == 42, (
            "log_message must not overwrite an already-set exchange_id"
        )
