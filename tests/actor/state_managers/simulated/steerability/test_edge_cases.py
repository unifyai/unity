"""
Edge-case steerability tests for `HierarchicalActor` (simulated managers).

These tests exercise boundary conditions that can regress easily:
- Late interjection after completion (should be a graceful no-op / informative message)
- Rapid successive interjections (should be processed without dropping/locking up)
- Interjection while a clarification is pending (both flows should still work)
- Empty / vague interjections (should not crash or force disruptive plan patching)

As with the rest of this suite, we rely on deterministic canned plans and verify
infrastructure behavior via `SteerableToolPane` events where applicable.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from tests.actor.state_managers.utils import (
    extract_clarification_details,
    get_pending_clarification_count,
    get_pane_steering_events,
    pick_handle_id_by_origin_tool,
    wait_for_pane_handle_count,
    wait_for_pane_steering_event,
)
from tests.async_helpers import _wait_for_condition

from .helpers import release_gate_and_finish, start_canned_plan
from .timeouts import (
    CLARIFICATION_TIMEOUT,
    HANDLE_REGISTRATION_TIMEOUT,
    INTERJECT_TIMEOUT,
    PLAN_COMPLETION_TIMEOUT,
    STEERING_EVENT_TIMEOUT,
)

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_late_interject_after_handle_completes(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Plan runs a simple contacts query and completes.
    - User attempts to interject after completion.

    Validates:
    - Interjection returns a stable, graceful message (no crash).
    - No new steering events are required (plan is no longer running).
    """

    primitives = create_primitives(
        contact_desc="Contacts: Alice Schmidt (alice@example.com), Bob Mueller (bob@example.com).",
    )
    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor)

        CANNED_PLAN = """
async def main_plan():
    handle = await primitives.contacts.ask(
        "List all contacts with email addresses."
    )
    result = await handle.result()
    return f"CONTACTS:\\n{result}"
"""

        _ = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

        try:
            result = await asyncio.wait_for(h.result(), timeout=PLAN_COMPLETION_TIMEOUT)
            assert isinstance(result, str) and result.strip()

            # Interject after completion should be a graceful no-op.
            status = await asyncio.wait_for(
                h.interject("One more thing: keep the format concise."),
                timeout=INTERJECT_TIMEOUT,
            )
            assert isinstance(status, str) and status.strip()
            assert (
                "cannot interject" in status.lower() or "not running" in status.lower()
            )

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_rapid_successive_interjections_all_processed(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Plan spawns 3 in-flight handles (contacts/transcripts/knowledge) and blocks.
    - User sends 3 interjections in rapid succession.

    Validates:
    - All interject calls return successfully.
    - Pane records >=3 successful `steering_applied(method=interject,status=ok)` events
      (at least one per interjection).
    - Plan remains routing-only (no patch/restart).
    """

    primitives = create_primitives(
        contact_desc=(
            "Contacts across quarters: Q3: Carol (Aug 2024), David (Sep 2024). "
            "Q4: Alice (Oct 2024), Bob (Nov 2024)."
        ),
        transcript_desc="Messages include Q3 planning and Q4 year-end updates + outreach tone notes.",
        knowledge_desc="Knowledge includes Q3 launch notes and Q4 customer feedback + metrics.",
    )
    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    contacts_handle = await primitives.contacts.ask(
        "Summarize contacts across 2024 (include join month)."
    )
    transcripts_handle = await primitives.transcripts.ask(
        "Summarize key message themes and also provide outreach tone guidance."
    )
    knowledge_handle = await primitives.knowledge.ask(
        "Summarize key project updates and metrics across 2024."
    )

    await TEST_GATE.wait()

    c, t, k = await asyncio.gather(
        contacts_handle.result(),
        transcripts_handle.result(),
        knowledge_handle.result(),
    )
    return f"CONTACTS:\\n{c}\\n\\nTRANSCRIPTS:\\n{t}\\n\\nKNOWLEDGE:\\n{k}"
"""

        original_plan_source = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

        try:
            await wait_for_pane_handle_count(
                h,
                expected=3,
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )

            msgs = [
                "For the contact list you’re producing right now, use full names and include the join month in parentheses.",
                "For the summaries you’re producing right now, focus ONLY on Q4 2024 (Oct/Nov/Dec) and ignore Q3 entirely.",
                "For the outreach tone guidance you’re generating right now, make it casual and friendly (use 'Hi <FirstName>,').",
            ]
            statuses = await asyncio.wait_for(
                asyncio.gather(*(h.interject(m) for m in msgs)),
                timeout=INTERJECT_TIMEOUT,
            )
            assert all(isinstance(s, str) and s.strip() for s in statuses)

            async def _three_ok_interjects() -> bool:
                events = get_pane_steering_events(h, n=800, method="interject")
                ok = [
                    e for e in events if (e.get("payload") or {}).get("status") == "ok"
                ]
                return len(ok) >= 3

            await asyncio.wait_for(
                _wait_for_condition(_three_ok_interjects, poll=0.05, timeout=60.0),
                timeout=70.0,
            )

            assert h.plan_source_code == original_plan_source

            _ = await release_gate_and_finish(h, timeout=PLAN_COMPLETION_TIMEOUT)

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_interject_while_waiting_for_clarification_answer(
    mock_verification,
    create_primitives_with_clarification_forcing,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Contact update triggers a clarification ("Which David?").
    - While clarification is pending, the user interjects with extra guidance.
    - User answers the clarification.

    Validates:
    - Interjection does not break clarification handling (both succeed).
    - Pane records both `interject` and `answer_clarification` steering events.
    - Plan completes.
    """

    primitives = create_primitives_with_clarification_forcing(
        contact_desc=(
            "You have two contacts named David: David Smith (david.smith@example.com) "
            "and David Jones (david.jones@example.com)."
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
    await TEST_GATE.wait()
    result = await update_handle.result()
    return f"Updated: {result}"
"""

        original_plan_source = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

        try:
            await wait_for_pane_handle_count(
                h,
                expected=1,
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )
            handles = await h.pane.list_handles()
            contacts_hid = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.contacts.update",
            )

            # Wait for pending clarification to be indexed.
            async def _pending_ready() -> bool:
                pending = await h.pane.get_pending_clarifications()
                return any(
                    str(e.get("handle_id")) == str(contacts_hid) for e in pending
                )

            await asyncio.wait_for(
                _wait_for_condition(
                    _pending_ready,
                    poll=0.05,
                    timeout=CLARIFICATION_TIMEOUT,
                ),
                timeout=CLARIFICATION_TIMEOUT + 10.0,
            )
            pending = await h.pane.get_pending_clarifications()
            assert len(pending) == 1
            hid, call_id, question = extract_clarification_details(pending[0])
            assert hid == str(contacts_hid)
            assert call_id and question.strip()

            # Supervisor surfaces question to user queue.
            _ = await asyncio.wait_for(h.clarification_up_q.get(), timeout=CLARIFICATION_TIMEOUT)  # type: ignore[union-attr]

            # Interject while we are waiting for the clarification answer.
            interject_status = await asyncio.wait_for(
                h.interject(
                    "When choosing the matching David, use the most recently active record if possible.",
                ),
                timeout=INTERJECT_TIMEOUT,
            )
            assert isinstance(interject_status, str) and interject_status.strip()

            await wait_for_pane_steering_event(
                h,
                handle_id=contacts_hid,
                method="interject",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )

            # Now answer the clarification.
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

            assert h.plan_source_code == original_plan_source

            _ = await release_gate_and_finish(h, timeout=PLAN_COMPLETION_TIMEOUT)

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_empty_and_unclear_interjections_handled_gracefully(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Plan spawns 2 in-flight handles and blocks.
    - User sends an empty interjection and then a vague one.

    Validates:
    - Both interjection calls return without crashing.
    - Plan completes after resume + gate release.
    - Interjections do not trigger plan patching/restart.
    """

    primitives = create_primitives(
        contact_desc="Contacts: Alice (alice@example.com), Bob (bob@example.com).",
        transcript_desc="Messages include a short conversation thread about meeting scheduling.",
    )
    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    c = await primitives.contacts.ask("List contacts with email.")
    t = await primitives.transcripts.ask("Summarize recent messages.")
    await TEST_GATE.wait()
    cr, tr = await asyncio.gather(c.result(), t.result())
    return f"CONTACTS:\\n{cr}\\n\\nTRANSCRIPTS:\\n{tr}"
"""

        original_plan_source = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

        try:
            await wait_for_pane_handle_count(
                h,
                expected=2,
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )

            s1 = await asyncio.wait_for(h.interject(""), timeout=INTERJECT_TIMEOUT)
            s2 = await asyncio.wait_for(
                h.interject("um... maybe change something?"),
                timeout=INTERJECT_TIMEOUT,
            )
            assert isinstance(s1, str)
            assert isinstance(s2, str)

            # Keep this invariant strict: meaningless interjections should not trigger disruptive patching.
            assert h.plan_source_code == original_plan_source

            # Best-effort: ensure plan isn't left paused by an unclear interjection.
            with contextlib.suppress(Exception):
                await h.resume()

            _ = await release_gate_and_finish(h, timeout=PLAN_COMPLETION_TIMEOUT)

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")
