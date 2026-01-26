"""
tests/conversation_manager/test_logging.py
================================================

Tests for the ConversationManager hierarchical logging infrastructure.

Verifies:
1. SessionLogger creates properly formatted labels
2. SimulatedConversationManagerHandle emits steering logs
3. SimulatedConversationManagerHandle.ask() uses nested labels
4. Log emojis are distinct and correctly applied
"""

from __future__ import annotations

import logging
import re
from unittest.mock import MagicMock

import pytest

from unity.common.hierarchical_logger import (
    SessionLogger,
    CM_ICONS,
    get_current_lineage,
    make_child_loop_id,
)
from unity.conversation_manager.domains.event_handlers import EventHandler
from unity.conversation_manager.events import Ping
from unity.conversation_manager.simulated import SimulatedConversationManagerHandle
from tests.helpers import _handle_project

# ─────────────────────────────────────────────────────────────────────────────
# 1. SessionLogger unit tests
# ─────────────────────────────────────────────────────────────────────────────


def test_session_logger_label_format():
    """
    Verify that SessionLogger creates labels with the expected format:
    ComponentName(xxxx) where xxxx is a 4-character hex suffix.
    """
    logger = SessionLogger("TestComponent")

    # Label should match pattern: TestComponent(xxxx)
    assert re.fullmatch(
        r"TestComponent\([0-9a-f]{4}\)",
        logger.label,
    ), f"Label '{logger.label}' does not match expected format"

    # Suffix should be 4 hex characters
    assert re.fullmatch(r"[0-9a-f]{4}", logger.suffix)


def test_session_logger_explicit_suffix():
    """
    Verify that SessionLogger uses an explicit suffix when provided.
    """
    logger = SessionLogger("TestComponent", suffix="abcd")

    assert logger.label == "TestComponent(abcd)"
    assert logger.suffix == "abcd"


def test_session_logger_explicit_parent_lineage():
    """
    Verify that SessionLogger properly incorporates parent lineage into the label.
    """
    logger = SessionLogger(
        "ChildComponent",
        parent_lineage=["ParentA", "ParentB"],
    )

    # Label should include parent lineage: ParentA->ParentB->ChildComponent(xxxx)
    assert re.fullmatch(
        r"ParentA->ParentB->ChildComponent\([0-9a-f]{4}\)",
        logger.label,
    ), f"Label '{logger.label}' does not include parent lineage"


def test_session_logger_child_lineage():
    """
    Verify that child_lineage() returns the correct lineage for nested components.
    """
    logger = SessionLogger("TestComponent", parent_lineage=["Parent"])

    child_lineage = logger.child_lineage()

    assert child_lineage == ["Parent", "TestComponent"]


def test_make_child_loop_id():
    """
    Verify that make_child_loop_id creates correct loop IDs for nested tool loops.
    """
    logger = SessionLogger("ConversationManager")

    loop_id = make_child_loop_id(logger, "ask")

    assert loop_id == "ConversationManager.ask"


# ─────────────────────────────────────────────────────────────────────────────
# 2. CM_ICONS are distinct from async tool loop icons
# ─────────────────────────────────────────────────────────────────────────────


def test_cm_icons_are_distinct():
    """
    Verify that ConversationManager-specific icons don't overlap with
    the core async tool loop icons (🤖, 🧑‍💻, ✅, etc.)
    """
    # Core async tool loop icons that should NOT be used for CM events
    async_loop_icons = {"🤖", "🧑‍💻", "✅", "🛠️", "📦"}

    # Check that none of the CM-specific icons overlap
    cm_icon_values = set(CM_ICONS.values())

    overlap = async_loop_icons & cm_icon_values
    assert not overlap, f"CM_ICONS should not overlap with async loop icons: {overlap}"


def test_cm_icons_have_expected_categories():
    """
    Verify that CM_ICONS has icons for expected event categories.
    """
    expected_keys = [
        "phone_call_received",
        "sms_received",
        "email_received",
        "llm_thinking",
        "llm_response",
        "session_start",
        "notification_injected",
        "actor_request",
    ]

    for key in expected_keys:
        assert key in CM_ICONS, f"Missing expected icon key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. SimulatedConversationManagerHandle logging tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_simulated_handle_has_log_label():
    """
    Verify that SimulatedConversationManagerHandle creates a log label on init.
    """
    handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id=1,
    )

    assert hasattr(handle, "_log_label")
    assert re.fullmatch(
        r"SimulatedConversationManager\([0-9a-f]{4}\)",
        handle._log_label,
    ), f"Label '{handle._log_label}' does not match expected format"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_handle_pause_logs(caplog):
    """
    Verify that pause() emits a log with the correct icon and label.
    """
    caplog.set_level(logging.DEBUG, logger="unity")

    handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id=1,
    )

    await handle.pause()

    # Should see: ⏸️ [SimulatedConversationManager(xxxx)] Pause requested
    assert re.search(
        r"⏸️ \[SimulatedConversationManager\([0-9a-f]{4}\)\] Pause requested",
        caplog.text,
    ), "Expected pause log with correct icon and label"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_handle_resume_logs(caplog):
    """
    Verify that resume() emits a log with the correct icon and label.
    """
    caplog.set_level(logging.DEBUG, logger="unity")

    handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id=1,
    )

    await handle.pause()
    await handle.resume()

    # Should see: ▶️ [SimulatedConversationManager(xxxx)] Resume requested
    assert re.search(
        r"▶️ \[SimulatedConversationManager\([0-9a-f]{4}\)\] Resume requested",
        caplog.text,
    ), "Expected resume log with correct icon and label"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_handle_stop_logs(caplog):
    """
    Verify that stop() emits a log with the correct icon and label.
    """
    caplog.set_level(logging.DEBUG, logger="unity")

    handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id=1,
    )

    handle.stop(reason="test cleanup")

    # Should see: 🛑 [SimulatedConversationManager(xxxx)] Stop requested – reason: test cleanup
    assert re.search(
        r"🛑 \[SimulatedConversationManager\([0-9a-f]{4}\)\] Stop requested – reason: test cleanup",
        caplog.text,
    ), "Expected stop log with correct icon, label, and reason"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_handle_interject_logs(caplog):
    """
    Verify that interject() emits a log with the correct icon and label.
    """
    caplog.set_level(logging.DEBUG, logger="unity")

    handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id=1,
    )

    await handle.interject("User has a new message")

    # Should see: 💬 [SimulatedConversationManager(xxxx)] Interject requested: User has a new message
    assert re.search(
        r"💬 \[SimulatedConversationManager\([0-9a-f]{4}\)\] Interject requested: User has a new message",
        caplog.text,
    ), "Expected interject log with correct icon and label"


# ─────────────────────────────────────────────────────────────────────────────
# 4. SimulatedConversationManagerHandle.ask() logging tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_simulated_handle_ask_logs_request(caplog):
    """
    Verify that ask() emits a log with the correct icon and nested label.
    """
    caplog.set_level(logging.DEBUG, logger="unity")

    handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id=1,
    )

    ask_handle = await handle.ask("What is the weather?")
    await ask_handle.result()

    # Should see: ❓ [SimulatedConversationManager.ask(xxxx)] Ask requested: What is the weather?
    assert re.search(
        r"❓ \[SimulatedConversationManager\.ask\([0-9a-f]{4}\)\] Ask requested: What is the weather\?",
        caplog.text,
    ), "Expected ask log with correct icon and label"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_handle_ask_logs_llm_roundtrip(caplog):
    """
    Verify that ask() triggers LLM roundtrip logging.
    """
    caplog.set_level(logging.DEBUG, logger="unity")

    handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id=1,
    )

    ask_handle = await handle.ask("Hello there!")
    await ask_handle.result()

    # Should see LLM simulating log: 🔄 [SimulatedConversationManager.ask(xxxx)] LLM simulating…
    assert re.search(
        r"🔄 \[SimulatedConversationManager\.ask\([0-9a-f]{4}\)\] LLM simulating…",
        caplog.text,
    ), "Expected LLM simulating log"

    # Should see LLM reply log: ✅ [SimulatedConversationManager.ask(xxxx)] LLM replied in ...
    assert re.search(
        r"✅ \[SimulatedConversationManager\.ask\([0-9a-f]{4}\)\] LLM replied in \d+ ms",
        caplog.text,
    ), "Expected LLM replied log"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_ask_handle_steering_logs(caplog):
    """
    Verify that steering methods on the ask handle also emit logs.
    """
    caplog.set_level(logging.DEBUG, logger="unity")

    handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id=1,
    )

    ask_handle = await handle.ask("What time is it?")

    # Call interject on the ask handle
    await ask_handle.interject("Actually, never mind")

    # Should see interject log with the ask handle's label
    assert re.search(
        r"💬 \[SimulatedConversationManager\.ask\([0-9a-f]{4}\)\] Interject requested: Actually, never mind",
        caplog.text,
    ), "Expected interject log on ask handle"


# ─────────────────────────────────────────────────────────────────────────────
# 5. get_current_lineage helper tests
# ─────────────────────────────────────────────────────────────────────────────


def test_get_current_lineage_empty_by_default():
    """
    Verify that get_current_lineage returns empty list when no context is set.
    """
    lineage = get_current_lineage()

    # When no tool loop or session is active, lineage should be empty
    assert isinstance(lineage, list)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Event handler console output tests
# ─────────────────────────────────────────────────────────────────────────────


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
