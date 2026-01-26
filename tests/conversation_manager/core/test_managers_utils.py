"""
tests/conversation_manager/test_managers_utils.py
======================================================

Tests for the managers_utils module, including the initialization queue
that holds operations until the ConversationManager is fully initialized.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from unity.conversation_manager.domains import managers_utils
from unity.conversation_manager.domains.event_handlers import EventHandler
from unity.conversation_manager.events import SyncContacts


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
    mock_cm.contact_manager = MagicMock()
    mock_cm.notifications_bar = MagicMock()
    mock_cm.initialized = False

    # Fire the SyncContacts event (this queues the operation)
    event = SyncContacts(reason="test sync")
    await EventHandler.handle_event(event, mock_cm)

    # Give a moment for any immediate execution (there shouldn't be any)
    await asyncio.sleep(0.05)

    # Sync should NOT have been called yet - still waiting for initialization
    mock_cm.contact_manager._sync_required_contacts.assert_not_called()

    # Start the operations listener in the background
    listener_task = asyncio.create_task(managers_utils.listen_to_operations(mock_cm))

    # Still not called - listener is waiting for initialization
    await asyncio.sleep(0.05)
    mock_cm.contact_manager._sync_required_contacts.assert_not_called()

    # Now mark as initialized - this unblocks the listener
    mock_cm.initialized = True

    # Wait for the queued operation to be processed
    await asyncio.sleep(0.2)

    # NOW the sync should have been called
    mock_cm.contact_manager._sync_required_contacts.assert_called_once()
    mock_cm.notifications_bar.push_notif.assert_called_once()

    # Cleanup
    listener_task.cancel()
    try:
        await listener_task
    except asyncio.CancelledError:
        pass
