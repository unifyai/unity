"""
tests/conversation_manager/core/test_run_llm_multi_tool.py
=============================================================

Tests for multi-tool handling in ``_run_llm()``.

These tests capture two pre-existing bugs in how ``_run_llm()`` handles
LLM turns where multiple tools are called concurrently:

1. **Lost tool names**: ``_run_llm()`` returns only the first tool name
   via ``result.tool_name``, so callers (including the test driver's
   ``all_tool_calls``) lose visibility of subsequent tool calls.

2. **Missed wait scheduling**: The ``wait(delay=N)`` scheduling logic
   checks only ``result.tool_name`` (first tool). If the LLM calls
   ``wait(delay=N)`` alongside other tools in the same turn and ``wait``
   is not the first tool call, the delayed follow-up turn is never
   scheduled.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.helpers import _handle_project
from unity.common.single_shot import SingleShotResult, ToolExecution


def _make_multi_tool_result(*tool_pairs: tuple[str, dict, object]) -> SingleShotResult:
    """Build a ``SingleShotResult`` with multiple tool executions.

    Each ``tool_pairs`` element is ``(name, args, result)``.
    """
    return SingleShotResult(
        tools=[
            ToolExecution(name=name, args=args, result=result)
            for name, args, result in tool_pairs
        ],
        text_response=None,
        structured_output=None,
    )


# =============================================================================
# Bug 1: _run_llm() drops tool names beyond the first
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_run_llm_returns_all_tool_names(initialized_cm):
    """_run_llm() should surface ALL tool names from a multi-tool turn.

    Currently it returns only the first via ``result.tool_name``, silently
    dropping the rest.  This means any caller relying on the return value
    (including the test driver's ``all_tool_calls``) has an incomplete
    picture of what the LLM decided.
    """
    cm = initialized_cm.cm

    fake_result = _make_multi_tool_result(
        ("desktop_act", {"instruction": "Click Submit"}, {"status": "acting"}),
        (
            "act",
            {"query": "Desktop session active", "persist": True},
            {"status": "acting"},
        ),
        ("send_unify_message", {"content": "On it.", "contact_id": 1}, None),
    )

    with patch(
        "unity.conversation_manager.conversation_manager.single_shot_tool_decision",
        AsyncMock(return_value=fake_result),
    ):
        returned = await cm._run_llm()

    # The correct behavior: _run_llm() should return all tool names
    # so callers can track the full set of actions taken in this turn.
    assert isinstance(returned, list), (
        f"_run_llm() should return a list of tool names when multiple tools "
        f"are called, but got {type(returned).__name__}: {returned!r}"
    )
    assert set(returned) == {
        "desktop_act",
        "act",
        "send_unify_message",
    }, f"Expected all three tool names, got: {returned}"


@pytest.mark.asyncio
@_handle_project
async def test_step_driver_tracks_all_tool_names(initialized_cm):
    """CMStepDriver.all_tool_calls should record EVERY tool called per turn.

    Currently it appends only the single string returned by ``_run_llm()``,
    so when the LLM calls ``[desktop_act, act, send_unify_message]`` in one
    turn, only ``desktop_act`` appears in ``all_tool_calls``.
    """
    cm_driver = initialized_cm

    fake_result = _make_multi_tool_result(
        ("desktop_act", {"instruction": "Click Submit"}, {"status": "acting"}),
        (
            "act",
            {"query": "Desktop session active", "persist": True},
            {"status": "acting"},
        ),
    )

    with patch(
        "unity.conversation_manager.conversation_manager.single_shot_tool_decision",
        AsyncMock(return_value=fake_result),
    ):
        returned = await cm_driver.cm._run_llm()

    # Simulate what the step driver does: append the return value.
    if isinstance(returned, list):
        cm_driver.all_tool_calls.extend(returned)
    elif returned:
        cm_driver.all_tool_calls.append(returned)

    assert "act" in cm_driver.all_tool_calls, (
        f"all_tool_calls should contain 'act' but only has: {cm_driver.all_tool_calls}. "
        f"The second tool call is silently dropped."
    )


# =============================================================================
# Bug 2: wait(delay=N) scheduling missed when wait is not the first tool
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_wait_delay_scheduled_when_not_first_tool(initialized_cm):
    """wait(delay=N) should schedule a delayed follow-up even when called
    alongside other tools in the same turn.

    Currently the scheduling logic checks only ``result.tool_name`` (first
    tool).  If the LLM calls ``[desktop_act, wait(delay=5)]``, the wait is
    the second tool and ``result.tool_name`` is ``"desktop_act"``, so the
    ``delay=5`` scheduling is silently skipped.
    """
    cm = initialized_cm.cm

    fake_result = _make_multi_tool_result(
        ("desktop_act", {"instruction": "Click Submit"}, {"status": "acting"}),
        ("wait", {"delay": 5}, None),
    )

    with (
        patch(
            "unity.conversation_manager.conversation_manager.single_shot_tool_decision",
            AsyncMock(return_value=fake_result),
        ),
        patch.object(cm, "run_llm", new_callable=AsyncMock) as mock_run,
    ):
        await cm._run_llm()

    mock_run.assert_called_once_with(delay=5), (
        f"run_llm(delay=5) should have been called for the wait tool, "
        f"but it was not.  The wait(delay=N) scheduling was missed because "
        f"wait was not the first tool in the multi-tool response."
    )


# =============================================================================
# wait() sets outbound-suppress generation flag
# =============================================================================


@pytest.mark.asyncio
async def test_wait_sets_outbound_suppress_generation():
    """wait() should stamp _outbound_suppress_gen with the current _llm_gen.

    When the LLM calls wait() alongside an outbound tool (send_unify_message,
    send_sms, etc.), the sent-message event handler should NOT trigger a
    redundant follow-up LLM turn.  The suppression uses a generation counter:
    wait() stamps _outbound_suppress_gen = _llm_gen, and the event handler
    skips request_llm_run when the two match.
    """
    from unity.conversation_manager.domains.brain_action_tools import (
        ConversationManagerBrainActionTools,
    )

    cm = MagicMock()
    cm._llm_gen = 7
    cm._outbound_suppress_gen = -1

    with patch(
        "unity.conversation_manager.domains.brain_action_tools.get_event_broker",
    ) as mock_broker:
        mock_broker.return_value = MagicMock()
        mock_broker.return_value.publish = AsyncMock()
        tools = ConversationManagerBrainActionTools(cm)

    result = await tools.wait()
    assert result == {"status": "waiting", "delay": None}
    assert cm._outbound_suppress_gen == 7, (
        "wait() should set _outbound_suppress_gen to the current _llm_gen "
        "so outbound-comms event handlers skip the reflexive follow-up turn"
    )

    result = await tools.wait(delay=30)
    assert result == {"status": "waiting", "delay": 30}
    assert (
        cm._outbound_suppress_gen == 7
    ), "wait(delay=N) should also set the suppression flag"
