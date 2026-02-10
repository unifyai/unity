"""
tests/conversation_manager/actions/test_storage_check.py
=========================================================

Tests that verify the ConversationManager brain correctly handles the
two-phase notification pattern from ``_StorageCheckHandle``-wrapped
actor handles.

When ``storage_check_on_return=True``, the handle lifecycle is:

1. Task completes → notification emitted (action still in-flight during
   storage check) → brain wakes up, should relay result to user.
2. Storage check finishes → ActorResult emitted (action now completed)
   → brain wakes up again, should **no-op** (already relayed).

Neither wake-up should mention internal skill-storage details.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import filter_events_by_type
from tests.conversation_manager.conftest import BOSS
from tests.conversation_manager.actions.integration.helpers import (
    run_cm_until_wait,
)
from unity.conversation_manager.events import (
    SMSReceived,
    SMSSent,
)

pytestmark = pytest.mark.eval


def _make_blocking_handle(result_value: str) -> tuple[MagicMock, asyncio.Event]:
    """Create a mock handle whose ``result()`` blocks until a gate is opened.

    Returns (handle, gate).  Call ``gate.set()`` to unblock ``result()``.
    The ``actor_watch_result`` background task will block on ``result()``
    until the gate opens, keeping the action in-flight for the test.
    """
    gate = asyncio.Event()
    handle = MagicMock()
    handle.done.return_value = False

    async def _blocking_result():
        await gate.wait()
        return result_value

    handle.result = _blocking_result

    # next_notification / next_clarification must also block so the
    # watcher tasks don't spin-loop or error out.
    async def _block_forever():
        await asyncio.Event().wait()
        return {}

    handle.next_notification = _block_forever
    handle.next_clarification = _block_forever
    handle.ask = AsyncMock(return_value=MagicMock(result=AsyncMock(return_value="")))
    handle.interject = AsyncMock()
    handle.stop = AsyncMock()
    handle.pause = AsyncMock(return_value=None)
    handle.resume = AsyncMock(return_value=None)
    handle.answer_clarification = AsyncMock()
    handle.get_history = MagicMock(return_value=[])

    return handle, gate


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_storage_check_two_phase_relay(initialized_cm):
    """The CM brain receives two wake-ups for a storage-check action:

    1. **Notification** (task done, storage check still running, action
       in-flight) — the brain sees the result in the progress event and
       should relay it to the user.
    2. **Completion** (storage check done, action completed) — the brain
       should no-op because the result was already relayed.

    Neither response should mention internal skill-storage details.
    """
    cm = initialized_cm

    simulated_result = "Alice Smith's phone number is +15555552222."
    mock_handle, result_gate = _make_blocking_handle(simulated_result)

    # ── Setup: process the user's message into chat history ───────────
    await cm.step(
        SMSReceived(
            contact=BOSS,
            content="What is Alice Smith's phone number?",
        ),
        run_llm=False,
    )

    from unity.common.prompt_helpers import now as prompt_now

    handle_id = 0

    # This matches the notification message emitted by _StorageCheckHandle
    # at the phase transition. It includes the result so the CM brain can
    # relay it immediately without waiting for the full completion.
    notification_msg = (
        f"Task completed with result:\n\n"
        f"{simulated_result}\n\n"
        f"The agent is now reviewing its execution "
        f"trajectory to store reusable skills. This "
        f"handle will remain active until skill "
        f"consolidation finishes."
    )

    # ── Phase 1: notification arrives (action still IN-FLIGHT) ────────
    #
    # The brain sees an in-flight action with a progress event that
    # carries the task result AND mentions skill consolidation. It
    # should relay the result to the user without mentioning the
    # internal skill-storage details.

    cm.cm.in_flight_actions[handle_id] = {
        "handle": mock_handle,
        "query": "What is Alice Smith's phone number?",
        "persist": False,
        "handle_actions": [
            {
                "action_name": "act_started",
                "query": "What is Alice Smith's phone number?",
                "timestamp": prompt_now(),
            },
            {
                "action_name": "progress",
                "query": notification_msg,
                "timestamp": prompt_now(),
            },
        ],
        "initial_snapshot_state": None,
    }

    phase1_events = await run_cm_until_wait(cm)

    phase1_sms = filter_events_by_type(phase1_events, SMSSent)
    assert phase1_sms, (
        "Phase 1: brain should relay the task result when the "
        "notification arrives (action still in-flight). The progress "
        "event includes the result."
    )
    phase1_text = " ".join(e.content for e in phase1_sms)

    # Must not leak internal skill-storage details.
    skill_terms = [
        "skill",
        "function library",
        "reusable",
        "storing",
        "consolidat",
        "trajectory",
        "handle",
    ]
    for term in skill_terms:
        assert term.lower() not in phase1_text.lower(), (
            f"Phase 1: leaked internal detail '{term}': {phase1_text}"
        )

    # ── Phase 2: action completes (moved to COMPLETED) ────────────────
    #
    # The storage check has finished. The action moves from in-flight to
    # completed. The brain wakes up again and should recognise that it
    # already relayed the result — it should no-op.

    mock_handle.done.return_value = True

    action_data = cm.cm.in_flight_actions.pop(handle_id, None)
    if action_data is None:
        action_data = cm.cm.completed_actions.get(handle_id)
    assert action_data is not None, "Action data lost between phases"

    action_data["handle_actions"].append(
        {
            "action_name": "act_completed",
            "query": simulated_result,
            "timestamp": prompt_now(),
        },
    )
    cm.cm.completed_actions[handle_id] = action_data

    # Release the gate so any background tasks waiting on result() can finish.
    result_gate.set()

    phase2_events = await run_cm_until_wait(cm)

    phase2_sms = filter_events_by_type(phase2_events, SMSSent)
    assert not phase2_sms, (
        f"Phase 2: brain should NOT send a duplicate message after the "
        f"result was already relayed in phase 1. Got: "
        f"{[e.content for e in phase2_sms]}"
    )
