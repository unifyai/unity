"""
tests/conversation_manager/test_logging.py
================================================

Critical production tests for ConversationManager logging behavior.

Note: Tests for logging format, icons, and label structure have been removed as they
are implementation-locked and don't catch production bugs. Only tests for behavior
that affects production operation are retained.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from unity.conversation_manager.domains.event_handlers import EventHandler
from unity.conversation_manager.events import Ping

# =============================================================================
# Critical Production Tests
# =============================================================================


@pytest.mark.asyncio
async def test_ping_event_prints_keepalive_message(capsys):
    """
    Verify that the Ping event handler prints the exact keepalive message
    to stdout. This message is essential for detecting idle containers
    in the communication adapters layer.

    The exact string "Ping received - keeping conversation manager alive"
    must be printed to stdout for idle job detection to work correctly.
    """
    # Create a minimal mock CM with just the required _session_logger
    mock_cm = MagicMock()
    mock_cm._session_logger = MagicMock()

    # Create and handle the Ping event
    ping_event = Ping(kind="keepalive")
    await EventHandler.handle_event(ping_event, mock_cm)

    # Capture stdout and verify the exact message
    captured = capsys.readouterr()
    expected_message = "Ping received - keeping conversation manager alive"

    assert expected_message in captured.out, (
        f"Expected stdout to contain '{expected_message}', "
        f"but got: {captured.out!r}"
    )

    # Also verify the session logger was called with the message
    mock_cm._session_logger.debug.assert_called_once_with("ping", expected_message)
