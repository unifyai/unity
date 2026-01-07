"""
Clarification flow tests for Actor steerability (simulated managers).

These tests validate that:
- Simulated managers can deterministically emit clarification events (forced by a test-only wrapper).
- `SteerableToolPane` indexes pending clarifications.
- `pane.answer_clarification(handle_id, call_id, answer)` routes answers to the correct handle.
- Handles can proceed to completion once clarifications are answered.

We use the same canned-plan + `TEST_GATE` pattern as the rest of this suite to
ensure timing-agnostic behavior regardless of cache speed.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from tests.test_actor.test_state_managers.utils import (
    extract_clarification_details,
    get_pending_clarification_count,
    pick_handle_id_by_origin_tool,
    wait_for_pane_handle_count,
    wait_for_pane_steering_event,
)
from tests.test_async_tool_loop.async_helpers import _wait_for_condition

from .helpers import release_gate_and_finish, start_canned_plan
from .timeouts import (
    CLARIFICATION_TIMEOUT,
    HANDLE_REGISTRATION_TIMEOUT,
    PLAN_COMPLETION_TIMEOUT,
    STEERING_EVENT_TIMEOUT,
)

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_clarification_ambiguous_contact_update_david(
    mock_verification,
    create_primitives_with_clarification_forcing,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Actor attempts to update "David's email", which is ambiguous in the scenario.
    - Simulated contact manager forces a clarification request.
    - Test answers the clarification via `pane.answer_clarification(...)`.

    Validates:
    - A clarification event appears in the pane.
    - The clarification is indexed in `get_pending_clarifications()`.
    - Answering removes the pending clarification and emits `steering_applied(method=answer_clarification,status=ok)`.
    - The plan can complete after the answer is routed.
    """

    primitives = create_primitives_with_clarification_forcing(
        contact_desc=(
            "You have two contacts named David in your CRM: "
            "David Smith (david.smith@example.com) and David Jones (david.jones@example.com)."
        ),
        contact_clarification_triggers=["update david"],
    )
    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor, with_clarification=True)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    update_handle = await primitives.contacts.update(
        "Update David's email address to david.new@example.com"
    )

    # Block deterministically so we can handle clarification before awaiting `.result()`.
    await TEST_GATE.wait()

    result = await update_handle.result()
    return f"Updated: {result}"
"""

        _ = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

        try:
            await wait_for_pane_handle_count(
                h,
                expected=1,
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )

            handles = await h.pane.list_handles()
            contact_hid = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.contacts.update",
            )

            async def _contact_pending() -> bool:
                pending = await h.pane.get_pending_clarifications()
                return any(str(e.get("handle_id")) == str(contact_hid) for e in pending)

            await asyncio.wait_for(
                _wait_for_condition(
                    _contact_pending,
                    poll=0.05,
                    timeout=CLARIFICATION_TIMEOUT,
                ),
                timeout=CLARIFICATION_TIMEOUT + 10.0,
            )
            pending = await h.pane.get_pending_clarifications()
            assert len(pending) == 1
            clar_event = pending[0]

            hid, call_id, question = extract_clarification_details(clar_event)
            assert hid == str(contact_hid)
            assert call_id
            assert question.strip()

            # The Actor's pane supervisor surfaces the question via `clarification_up_q`
            # and routes the user's answer back via `pane.answer_clarification(...)`.
            surfaced_question = await asyncio.wait_for(
                h.clarification_up_q.get(),  # type: ignore[union-attr]
                timeout=CLARIFICATION_TIMEOUT,
            )
            assert isinstance(surfaced_question, str) and surfaced_question.strip()

            await h.clarification_down_q.put("David Smith")  # type: ignore[union-attr]
            await wait_for_pane_steering_event(
                h,
                handle_id=hid,
                method="answer_clarification",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )

            async def _pending_cleared() -> bool:
                return (await get_pending_clarification_count(h)) == 0

            await asyncio.wait_for(
                _wait_for_condition(_pending_cleared, poll=0.05, timeout=30.0),
                timeout=40.0,
            )

            result = await release_gate_and_finish(h, timeout=PLAN_COMPLETION_TIMEOUT)
            assert isinstance(result, str) and result.startswith("Updated:")

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_clarification_missing_timezone_for_meeting_tasks_update(
    mock_verification,
    create_primitives_with_clarification_forcing,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Actor lists Berlin contacts and schedules a meeting.
    - TaskScheduler update is forced to request clarification (timezone ambiguity).
    - Test answers via pane, then the plan completes.

    Validates:
    - Clarification appears + is indexed.
    - Answer routing works and clears pending clarification.
    """

    primitives = create_primitives_with_clarification_forcing(
        contact_desc=(
            "You have 3 contacts in Berlin: Alice (alice@example.com), "
            "Bob (bob@example.com), Carol (carol@example.com)."
        ),
        task_desc=(
            "You manage a task list for a small team. Scheduling a meeting may require timezone clarification."
        ),
        task_clarification_triggers=["schedule a team meeting", "berlin contacts"],
    )
    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor, with_clarification=True)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    contacts_handle = await primitives.contacts.ask(
        "List all contacts in Berlin with their email addresses."
    )
    task_handle = await primitives.tasks.update(
        "Schedule a team meeting with the Berlin contacts for next Tuesday at 2pm."
    )

    await TEST_GATE.wait()

    contacts_result = await contacts_handle.result()
    task_result = await task_handle.result()
    return f"Contacts:\\n{contacts_result}\\n\\nMeeting:\\n{task_result}"
"""

        _ = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

        try:
            await wait_for_pane_handle_count(
                h,
                expected=2,
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )
            handles = await h.pane.list_handles()
            tasks_hid = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.tasks.update",
            )

            # Wait until the tasks handle has a pending clarification.
            async def _tasks_pending() -> bool:
                pending = await h.pane.get_pending_clarifications()
                return any(str(e.get("handle_id")) == str(tasks_hid) for e in pending)

            await asyncio.wait_for(
                _wait_for_condition(
                    _tasks_pending,
                    poll=0.05,
                    timeout=CLARIFICATION_TIMEOUT,
                ),
                timeout=CLARIFICATION_TIMEOUT + 10.0,
            )
            pending = await h.pane.get_pending_clarifications()
            assert len(pending) == 1
            clar_event = pending[0]

            hid, call_id, question = extract_clarification_details(clar_event)
            assert hid == str(tasks_hid)
            assert call_id
            assert question.strip()

            surfaced_question = await asyncio.wait_for(
                h.clarification_up_q.get(),  # type: ignore[union-attr]
                timeout=CLARIFICATION_TIMEOUT,
            )
            assert isinstance(surfaced_question, str) and surfaced_question.strip()

            await h.clarification_down_q.put("CET (UTC+1)")  # type: ignore[union-attr]
            await wait_for_pane_steering_event(
                h,
                handle_id=hid,
                method="answer_clarification",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )

            async def _pending_cleared() -> bool:
                return (await get_pending_clarification_count(h)) == 0

            await asyncio.wait_for(
                _wait_for_condition(_pending_cleared, poll=0.05, timeout=30.0),
                timeout=40.0,
            )

            result = await release_gate_and_finish(h, timeout=PLAN_COMPLETION_TIMEOUT)
            assert isinstance(result, str)
            assert "Contacts:" in result and "Meeting:" in result

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_clarification_concurrent_contacts_and_tasks(
    mock_verification,
    create_primitives_with_clarification_forcing,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Actor asks about an ambiguous contact ("Find David's email") and an ambiguous task ("review task").
    - Both simulated managers are forced to request clarifications concurrently.
    - Test answers both via pane (one-by-one) and verifies pending clarifications track correctly.

    Validates:
    - Pane indexes multiple concurrent pending clarifications.
    - Answering one does not affect the other.
    - Both handles can proceed to completion.
    """

    primitives = create_primitives_with_clarification_forcing(
        contact_desc=(
            "You have David Smith (david.smith@example.com) and David Jones (david.jones@example.com)."
        ),
        contact_clarification_triggers=["find david"],
        task_desc=(
            "You have two tasks: (1) Review Q3 report, (2) Review code changes. Both are marked 'in progress'."
        ),
        task_clarification_triggers=["review task"],
    )
    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor, with_clarification=True)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    contact_handle, task_handle = await asyncio.gather(
        primitives.contacts.ask("Find David's email address."),
        primitives.tasks.ask("What is the status of the review task?"),
    )

    await TEST_GATE.wait()

    contact_result, task_result = await asyncio.gather(
        contact_handle.result(),
        task_handle.result(),
    )
    return f"Contact: {contact_result}\\n\\nTask: {task_result}"
"""

        _ = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

        try:
            await wait_for_pane_handle_count(
                h,
                expected=2,
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )
            handles = await h.pane.list_handles()
            contacts_hid = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.contacts.ask",
            )
            tasks_hid = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.tasks.ask",
            )

            async def _pending_two() -> bool:
                return (await get_pending_clarification_count(h)) == 2

            await asyncio.wait_for(
                _wait_for_condition(
                    _pending_two,
                    poll=0.05,
                    timeout=CLARIFICATION_TIMEOUT,
                ),
                timeout=CLARIFICATION_TIMEOUT + 10.0,
            )
            assert (await get_pending_clarification_count(h)) == 2

            # The pane supervisor will request clarification questions sequentially.
            q1 = await asyncio.wait_for(
                h.clarification_up_q.get(),  # type: ignore[union-attr]
                timeout=CLARIFICATION_TIMEOUT,
            )
            assert isinstance(q1, str) and q1.strip()
            a1 = "David Smith" if "contact" in q1.lower() else "Review code changes"
            await h.clarification_down_q.put(a1)  # type: ignore[union-attr]

            async def _pending_leq_one() -> bool:
                return (await get_pending_clarification_count(h)) <= 1

            await asyncio.wait_for(
                _wait_for_condition(
                    _pending_leq_one,
                    poll=0.05,
                    timeout=CLARIFICATION_TIMEOUT,
                ),
                timeout=40.0,
            )

            q2 = await asyncio.wait_for(
                h.clarification_up_q.get(),  # type: ignore[union-attr]
                timeout=CLARIFICATION_TIMEOUT,
            )
            assert isinstance(q2, str) and q2.strip()
            a2 = "David Smith" if "contact" in q2.lower() else "Review code changes"
            await h.clarification_down_q.put(a2)  # type: ignore[union-attr]

            async def _pending_zero() -> bool:
                return (await get_pending_clarification_count(h)) == 0

            await asyncio.wait_for(
                _wait_for_condition(
                    _pending_zero,
                    poll=0.05,
                    timeout=CLARIFICATION_TIMEOUT,
                ),
                timeout=CLARIFICATION_TIMEOUT + 10.0,
            )

            result = await release_gate_and_finish(h, timeout=PLAN_COMPLETION_TIMEOUT)
            assert isinstance(result, str)
            assert "Contact:" in result and "Task:" in result

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")
