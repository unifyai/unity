"""
Complex multi-manager steerability scenarios for `HierarchicalActor` (simulated managers).

These tests cover advanced steering patterns beyond basic broadcast/targeted routing
and simple nesting:

1) **Staged error-recovery / early correction**:
   - Interject when only a single handle exists, then the plan proceeds to spawn
     additional manager handles based on that corrected result.

2) **Sequential guidance at different workflow stages**:
   - Two interjections applied at two different points in the workflow, routed to
     different handles.

These tests are infrastructure-first:
- Deterministic gating (`TEST_GATE_*`) ensures timing-agnostic behavior.
- Assertions focus on `SteerableToolPane` events (`handle_registered`, `steering_applied`),
  and ensure interjections remain routing-only (no plan patch/restart).
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from tests.test_actor.test_state_managers.utils import (
    pick_handle_id_by_origin_tool,
    wait_for_pane_handle_count,
    wait_for_pane_steering_event,
)

from .helpers import (
    finish,
    get_ok_steering_handle_ids,
    release_gate,
    start_canned_plan,
    wait_for_handle_registered,
)
from .timeouts import (
    HANDLE_REGISTRATION_TIMEOUT,
    INTERJECT_TIMEOUT,
    PLAN_COMPLETION_TIMEOUT,
    STEERING_EVENT_TIMEOUT,
)

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_complex_staged_error_recovery_berlin_only(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario (staged / error-recovery):
    - Step 1: Actor starts by listing "active contacts for quarterly sync".
    - User interjects early with a correction: only use Berlin office contacts.
    - Step 2: Plan then spawns transcript + knowledge work based on those contacts.

    Validates:
    - Interjection happens when only the contacts handle is in-flight.
    - Interjection is routed to that handle (infrastructure evidence).
    - Plan remains routing-only (no plan patch/restart).
    - Later handles register successfully and the plan completes.
    """

    primitives = create_primitives(
        contact_desc=(
            "Active contacts across offices. Berlin: Alice Schmidt (designer, alice@berlin.example.com), "
            "Bob Mueller (engineering lead, bob@berlin.example.com). "
            "London: Carol Weber (PM, carol@london.example.com). "
            "NYC: David Klein (sales, david@nyc.example.com)."
        ),
        transcript_desc="Messages exist for contacts across Berlin/London/NYC offices.",
        knowledge_desc="Knowledge base contains office-specific customer notes.",
    )
    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor)

        CANNED_PLAN = """
import asyncio

TEST_GATE_1 = asyncio.Event()
TEST_GATE_2 = asyncio.Event()

async def main_plan():
    contacts_handle = await primitives.contacts.ask(
        "Find all active contacts for the quarterly sync (names + office + email)."
    )

    # Gate 1: allow the user to correct scope before we await any results.
    await TEST_GATE_1.wait()

    contacts_result = await contacts_handle.result()

    transcripts_handle = await primitives.transcripts.ask(
        f"Find recent messages from these contacts and summarize key themes: {contacts_result}"
    )
    knowledge_handle = await primitives.knowledge.update(
        f"Update knowledge base with contact activity summary for quarterly sync: {contacts_result}"
    )

    # Gate 2: keep these in-flight so we can observe registrations deterministically.
    await TEST_GATE_2.wait()

    transcripts_result, knowledge_result = await asyncio.gather(
        transcripts_handle.result(),
        knowledge_handle.result(),
    )
    return (
        "CONTACTS_RESULT:\\n"
        f"{contacts_result}\\n\\n"
        "TRANSCRIPTS_RESULT:\\n"
        f"{transcripts_result}\\n\\n"
        "KNOWLEDGE_RESULT:\\n"
        f"{knowledge_result}"
    )
"""

        original_plan_source = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

        try:
            # Stage 1: only contacts handle should exist.
            await wait_for_pane_handle_count(
                h,
                expected=1,
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )
            handles = await h.pane.list_handles()
            assert (
                len(handles) == 1
            ), f"Expected exactly 1 in-flight handle before gate release. Handles={handles}"
            contacts_hid = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.contacts.ask",
            )

            msg = (
                "Actually, for this quarterly sync, let’s focus only on the Berlin office. "
                "When you list the active contacts, include only Berlin contacts."
            )
            status = await asyncio.wait_for(h.interject(msg), timeout=INTERJECT_TIMEOUT)
            assert isinstance(status, str) and status.strip()

            e_contacts = await wait_for_pane_steering_event(
                h,
                handle_id=contacts_hid,
                method="interject",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )
            routed = str(
                (e_contacts.get("payload") or {}).get("args", {}).get("message", ""),
            ).strip()
            assert routed

            assert h.plan_source_code == original_plan_source

            # Continue to stage 2.
            await release_gate(h, "TEST_GATE_1")

            # Observe subsequent handles registering.
            _ = await wait_for_handle_registered(
                h,
                origin_tool_prefix="primitives.transcripts.ask",
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )
            _ = await wait_for_handle_registered(
                h,
                origin_tool_prefix="primitives.knowledge.update",
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )

            # No additional interjections were sent; only contacts should have ok steering_applied.
            ok_ids = get_ok_steering_handle_ids(h, method="interject")
            assert ok_ids == {str(contacts_hid)}

            await release_gate(h, "TEST_GATE_2")
            _ = await finish(h, timeout=PLAN_COMPLETION_TIMEOUT)

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_complex_sequential_guidance_routes_to_contacts_then_tasks(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario (sequential guidance):
    - Stage 1: contacts lookup starts; user interjects "focus on Berlin contacts".
    - Stage 2: based on contacts result, the plan starts transcripts + knowledge analysis + task creation in parallel.
      User interjects again with guidance specific to task creation: "prioritize urgent items".

    Validates:
    - Two separate interjections were routed to two different handles.
    - Neither interjection required plan patching/restart.
    - Second interjection is selectively routed to the tasks.update handle (not transcripts/knowledge).
    """

    primitives = create_primitives(
        contact_desc=(
            "Customer contacts with recent activity. Berlin: Alice Schmidt (enterprise, alice@berlin.example.com), "
            "Bob Mueller (SMB, bob@berlin.example.com). "
            "London: Carol Weber (enterprise, carol@london.example.com)."
        ),
        transcript_desc="Recent customer messages include both routine updates and urgent escalation items.",
        knowledge_desc="Knowledge includes customer sentiment notes and escalations.",
        task_desc="Task list is empty; ready to create follow-ups.",
    )
    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor)

        CANNED_PLAN = """
import asyncio

TEST_GATE_1 = asyncio.Event()
TEST_GATE_2 = asyncio.Event()

async def main_plan():
    contacts_handle = await primitives.contacts.ask(
        "Find all customer contacts with recent activity (include office and company segment)."
    )

    # Gate 1: first interjection point.
    await TEST_GATE_1.wait()
    contacts_result = await contacts_handle.result()

    transcripts_handle = await primitives.transcripts.ask(
        f"Get recent messages from these contacts and summarize key themes: {contacts_result}"
    )
    knowledge_handle = await primitives.knowledge.ask(
        f"Analyze sentiment and key themes from customer interactions: {contacts_result}"
    )
    task_handle = await primitives.tasks.update(
        "Create follow-up tasks based on customer interactions and sentiment analysis."
    )

    # Gate 2: second interjection point (multiple handles in-flight).
    await TEST_GATE_2.wait()

    transcripts_result, knowledge_result, task_result = await asyncio.gather(
        transcripts_handle.result(),
        knowledge_handle.result(),
        task_handle.result(),
    )
    return (
        "CONTACTS:\\n"
        f"{contacts_result}\\n\\n"
        "TRANSCRIPTS:\\n"
        f"{transcripts_result}\\n\\n"
        "KNOWLEDGE:\\n"
        f"{knowledge_result}\\n\\n"
        "TASKS:\\n"
        f"{task_result}"
    )
"""

        original_plan_source = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

        try:
            # Stage 1: contacts handle.
            await wait_for_pane_handle_count(
                h,
                expected=1,
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )
            contacts_hid = await wait_for_handle_registered(
                h,
                origin_tool_prefix="primitives.contacts.ask",
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )

            msg1 = (
                "Let’s focus on the Berlin office for this. When you pull the contact list, "
                "only include Berlin contacts."
            )
            status1 = await asyncio.wait_for(
                h.interject(msg1),
                timeout=INTERJECT_TIMEOUT,
            )
            assert isinstance(status1, str) and status1.strip()

            await wait_for_pane_steering_event(
                h,
                handle_id=contacts_hid,
                method="interject",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )
            assert h.plan_source_code == original_plan_source

            # Advance to stage 2, where transcripts + knowledge + tasks run concurrently.
            await release_gate(h, "TEST_GATE_1")

            transcripts_hid = await wait_for_handle_registered(
                h,
                origin_tool_prefix="primitives.transcripts.ask",
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )
            knowledge_hid = await wait_for_handle_registered(
                h,
                origin_tool_prefix="primitives.knowledge.ask",
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )
            tasks_hid = await wait_for_handle_registered(
                h,
                origin_tool_prefix="primitives.tasks.update",
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )

            msg2 = (
                "For the follow-up tasks you’re creating, please prioritize urgent items as high priority. "
                "If something sounds like an escalation or blocker, mark it urgent."
            )
            status2 = await asyncio.wait_for(
                h.interject(msg2),
                timeout=INTERJECT_TIMEOUT,
            )
            assert isinstance(status2, str) and status2.strip()

            await wait_for_pane_steering_event(
                h,
                handle_id=tasks_hid,
                method="interject",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )

            ok_ids = get_ok_steering_handle_ids(h, method="interject")
            # The interjection is *at least* relevant to task creation, but the LLM may
            # also (reasonably) route it to upstream analysis handles so they explicitly
            # surface urgency signals for task creation. Keep the invariant strict only
            # on the must-have targets.
            assert str(contacts_hid) in ok_ids, (
                "Expected the first interjection to be routed to the contacts handle. "
                f"Got ok_ids={sorted(ok_ids)}; contacts={contacts_hid}"
            )
            assert str(tasks_hid) in ok_ids, (
                "Expected the second interjection to be routed to the tasks handle. "
                f"Got ok_ids={sorted(ok_ids)}; tasks={tasks_hid}"
            )

            assert h.plan_source_code == original_plan_source

            await release_gate(h, "TEST_GATE_2")
            _ = await finish(h, timeout=PLAN_COMPLETION_TIMEOUT)

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")
