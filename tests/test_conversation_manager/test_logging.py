"""
tests/test_conversation_manager/test_logging.py
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
import pytest

from unity.common.hierarchical_logger import (
    SessionLogger,
    CM_ICONS,
    get_current_lineage,
    make_child_loop_id,
)
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
        "conductor_request",
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
# 6. LLM IO debug logging tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_llm_log_files_created(tmp_path, monkeypatch):
    """
    Verify that LLM log files are created when using SimulatedConversationManagerHandle.

    The LLM IO hooks capture all LLM requests/responses at the unify client level,
    so ConversationManager LLM calls should produce debug files.
    """
    import unity.common.llm_io_hooks as hooks_mod

    # Set up temp directory for IO debug files
    io_dir = tmp_path / "logs" / "llm" / "test_session"
    io_dir.mkdir(parents=True)
    monkeypatch.setattr(hooks_mod, "_LLM_IO_DIR", str(io_dir))

    handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id=1,
    )

    # Make an LLM call via ask()
    ask_handle = await handle.ask("What is 2 + 2?")
    await ask_handle.result()

    # Check that debug files were created
    files = list(io_dir.glob("*.txt"))
    assert (
        len(files) >= 2
    ), f"Expected at least 2 files (request + response), got {len(files)}"

    # Verify we have both request and response files
    file_contents = [f.read_text() for f in files]
    has_request = any("request" in content.lower() for content in file_contents)
    has_response = any("response" in content.lower() for content in file_contents)

    assert has_request, "Should have at least one request file"
    assert has_response, "Should have at least one response file"


@pytest.mark.asyncio
@_handle_project
async def test_llm_log_writes_to_terminal(tmp_path, monkeypatch, caplog):
    """
    Verify that LLM log writes emit "📝 LLM request/response written to" log messages.

    These terminal logs help developers see where debug files are being written.
    """
    import unity.common.llm_io_hooks as hooks_mod

    # Set up temp directory for IO debug files
    io_dir = tmp_path / "logs" / "llm" / "test_session"
    io_dir.mkdir(parents=True)
    monkeypatch.setattr(hooks_mod, "_LLM_IO_DIR", str(io_dir))

    # Capture logs at INFO level
    caplog.set_level(logging.INFO, logger="unity")

    handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id=1,
    )

    # Make an LLM call via ask()
    ask_handle = await handle.ask("Hello!")
    await ask_handle.result()

    # Check for terminal log messages about file writes
    assert re.search(
        r"📝 LLM request written to",
        caplog.text,
    ), "Expected '📝 LLM request written to' log message"

    assert re.search(
        r"📝 LLM response written to",
        caplog.text,
    ), "Expected '📝 LLM response written to' log message"


@pytest.mark.asyncio
@_handle_project
async def test_llm_log_file_contains_messages(tmp_path, monkeypatch):
    """
    Verify that LLM log files contain the actual message content.
    """
    import unity.common.llm_io_hooks as hooks_mod

    # Set up temp directory for IO debug files
    io_dir = tmp_path / "logs" / "llm" / "test_session"
    io_dir.mkdir(parents=True)
    monkeypatch.setattr(hooks_mod, "_LLM_IO_DIR", str(io_dir))

    handle = SimulatedConversationManagerHandle(
        assistant_id="test_assistant",
        contact_id=1,
    )

    # Use a distinctive message that we can search for
    test_question = "What is the capital of France?"
    ask_handle = await handle.ask(test_question)
    await ask_handle.result()

    # Read all debug files and check for content
    files = list(io_dir.glob("*.txt"))
    all_content = "\n".join(f.read_text() for f in files)

    # The request file should contain our question somewhere in the messages
    assert (
        "France" in all_content or "capital" in all_content
    ), "Debug files should contain the user's question"
