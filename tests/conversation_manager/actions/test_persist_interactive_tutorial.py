"""
tests/conversation_manager/actions/test_persist_interactive_tutorial.py
========================================================================

Tests that the CM slow brain uses ``persist=True`` when the user gives the
first actionable instruction during an interactive tutorial or walkthrough.

The key insight: ``act`` should NOT fire when the user merely *announces*
a tutorial ("I'm going to show you how to …"). It should fire when the
user gives the first concrete instruction ("now click on the Contacts
tab"). At that point, the brain must recognise the instruction is part of
a larger interactive session and:

1. Call ``act(persist=True)`` — not a one-shot action.
2. Include the broader tutorial context in the ``query`` (not just the
   isolated instruction), so the Actor understands the full scope.

These are eval tests: they verify that the LLM correctly identifies the
boundary between conversational context-setting and the first actionable
tutorial step, and that it chooses the right ``persist`` mode.
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.cm_helpers import (
    assert_act_triggered,
    assert_efficient,
    filter_events_by_type,
)
from tests.conversation_manager.conftest import BOSS
from unity.conversation_manager.events import (
    UnifyMessageReceived,
    ActorHandleStarted,
)

pytestmark = pytest.mark.eval


def _assert_no_act_triggered(result, description: str) -> None:
    """Assert that no act() call was made during this step."""
    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) == 0, (
        f"Expected NO ActorHandleStarted events — {description}. "
        f"Got {len(actor_events)} event(s) with queries: "
        f"{[e.query for e in actor_events]}"
    )


def _assert_persist_true(cm, description: str) -> None:
    """Assert that the most recent act() call used persist=True."""
    actions = cm.cm.in_flight_actions
    assert actions, f"No in-flight actions found — {description}"
    last_action = list(actions.values())[-1]
    assert last_action.get("persist") is True, (
        f"Expected persist=True for interactive tutorial action, "
        f"but got persist={last_action.get('persist')}. {description}"
    )


def _get_act_query(result) -> str:
    """Extract the query string from the ActorHandleStarted event."""
    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert actor_events, "No ActorHandleStarted event found"
    return actor_events[0].query


@pytest.mark.asyncio
@_handle_project
async def test_crm_walkthrough_persist_on_first_instruction(initialized_cm):
    """User shares screen, explains CRM context, then gives first instruction.

    Phase 1: "I'm going to show you how to use our CRM system" — pure
    context-setting. The assistant should acknowledge and wait, NOT call act.

    Phase 2: "Now, click on the Contacts tab on the left" — the first
    actionable instruction. The assistant should call act(persist=True)
    with a query that captures the broader CRM tutorial context, not just
    the isolated click instruction.
    """
    cm = initialized_cm
    cm.cm.user_screen_share_active = True

    # Phase 1: Context-setting — should NOT trigger act
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Okay, I'm going to show you how to use our CRM system. "
                "You can see I've got the main dashboard open on my screen."
            ),
        ),
    )
    _assert_no_act_triggered(result1, "Context-setting should not trigger act")
    assert_efficient(result1, 3)

    # Phase 2: First actionable instruction — should trigger act(persist=True)
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content="Now, click on the Contacts tab on the left side.",
        ),
    )
    assert_act_triggered(
        result2,
        ActorHandleStarted,
        "First tutorial instruction should trigger act",
        cm=cm,
    )
    _assert_persist_true(cm, "Tutorial instruction should use persist=True")

    query = _get_act_query(result2)
    query_lower = query.lower()
    assert "crm" in query_lower or "contacts" in query_lower, (
        f"Expected act query to reference the broader CRM tutorial context, "
        f"not just the isolated instruction. Got: {query}"
    )
    assert_efficient(result2, 3)


@pytest.mark.asyncio
@_handle_project
async def test_invoice_workflow_persist_on_first_instruction(initialized_cm):
    """User explains invoice workflow, then asks assistant to perform a step.

    Phase 1: "I'm going to walk you through our invoice processing" — the
    user is narrating what they'll demonstrate. No action needed yet.

    Phase 2: "Go ahead and open the Accounting portal" — now the assistant
    is being told to do something. This should trigger act(persist=True)
    with context about the invoice workflow.
    """
    cm = initialized_cm
    cm.cm.user_screen_share_active = True

    # Phase 1: Narration — no act
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "I'm going to walk you through our invoice processing workflow "
                "step by step. Watch my screen as I go through it."
            ),
        ),
    )
    _assert_no_act_triggered(result1, "Narration should not trigger act")
    assert_efficient(result1, 3)

    # Phase 2: First instruction — act(persist=True) with broader context
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content="Go ahead and open the Accounting portal now.",
        ),
    )
    assert_act_triggered(
        result2,
        ActorHandleStarted,
        "First workflow instruction should trigger act",
        cm=cm,
    )
    _assert_persist_true(cm, "Workflow instruction should use persist=True")

    query = _get_act_query(result2)
    query_lower = query.lower()
    assert "invoice" in query_lower or "accounting" in query_lower, (
        f"Expected act query to reference the broader invoice/accounting "
        f"workflow context, not just 'open the portal'. Got: {query}"
    )
    assert_efficient(result2, 3)


@pytest.mark.asyncio
@_handle_project
async def test_refund_demo_persist_on_first_instruction(initialized_cm):
    """User demonstrates refund process, then tells assistant to navigate.

    Phase 1: "Let me show you how we handle customer refunds" — setting
    the scene. The assistant should listen, not act.

    Phase 2: "Click on the Orders tab and find order #4521" — a concrete
    instruction to perform. Should trigger act(persist=True) with the
    refund process as context.
    """
    cm = initialized_cm
    cm.cm.user_screen_share_active = True

    # Phase 1: Scene-setting — no act
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content=(
                "Let me show you how we handle customer refunds in our system. "
                "I'll walk you through the full process."
            ),
        ),
    )
    _assert_no_act_triggered(result1, "Scene-setting should not trigger act")
    assert_efficient(result1, 3)

    # Phase 2: First concrete instruction — act(persist=True) with refund context
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            contact=BOSS,
            content="Click on the Orders tab and find order #4521.",
        ),
    )
    assert_act_triggered(
        result2,
        ActorHandleStarted,
        "First refund-process instruction should trigger act",
        cm=cm,
    )
    _assert_persist_true(cm, "Refund-process instruction should use persist=True")

    query = _get_act_query(result2)
    query_lower = query.lower()
    assert "refund" in query_lower or "order" in query_lower, (
        f"Expected act query to reference the broader refund process context, "
        f"not just 'click Orders tab'. Got: {query}"
    )
    assert_efficient(result2, 3)
