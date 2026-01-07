"""
Interjection routing tests for Actor steerability (simulated managers).

These tests validate that natural-language `HierarchicalActorHandle.interject(...)` is
routed (by the interjection LLM) to the *relevant* in-flight manager handle(s)
registered in the `SteerableToolPane`, including:
- targeted routing (only one handle receives the interjection)
- broadcast routing (all in-flight handles receive the interjection)
- selective routing by manager type (e.g., "contact-related only")

We use deterministic canned-plan gating (`TEST_GATE`) so handles are registered
before steering, regardless of cache speed.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from tests.test_actor.test_state_managers.test_simulated.test_steerability.helpers import (
    get_ok_steering_handle_ids,
    release_gate_and_finish,
    start_canned_plan,
)
from tests.test_actor.test_state_managers.test_simulated.test_steerability.timeouts import (
    HANDLE_REGISTRATION_TIMEOUT,
    INTERJECT_TIMEOUT,
    STEERING_EVENT_TIMEOUT,
)
from tests.test_actor.test_state_managers.utils import (
    pick_handle_id_by_origin_tool,
    wait_for_pane_handle_count,
    wait_for_pane_steering_event,
)

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_interject_targeted_to_contacts_over_knowledge_update(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Plan spawns two in-flight handles: knowledge.update + contacts.ask.
    - User interjects with a preference only relevant to the contact lookup formatting.

    Validates targeted routing to contacts.ask (not knowledge.update), and routing-only
    behavior (no plan patch/restart).
    """
    primitives = create_primitives(
        contact_desc="You have contact: Alice Schmidt (designer, alice@example.com).",
        knowledge_desc="Empty knowledge base ready for new entries.",
    )

    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    knowledge_handle = await primitives.knowledge.update(
        "Add Alice's contact information to the knowledge base."
    )
    contact_handle = await primitives.contacts.ask(
        "Find Alice's email and role."
    )

    await TEST_GATE.wait()

    contact_result = await contact_handle.result()
    knowledge_result = await knowledge_handle.result()
    return f"Contact: {contact_result}\\nKnowledge: {knowledge_result}"
"""

        original_plan_source = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

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
            _ = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.knowledge.update",
            )

            msg = (
                "For the contact lookup you’re doing right now, please use the full name "
                "(first + last) and include the email domain explicitly."
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

            ok_ids = get_ok_steering_handle_ids(h, method="interject")
            assert ok_ids == {
                str(contacts_hid),
            }, f"Expected targeted routing to contacts.ask. ok_ids={sorted(ok_ids)}"

            assert h.plan_source_code == original_plan_source

            _ = await release_gate_and_finish(h)

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_interject_targeted_to_contacts_over_tasks_update(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Plan spawns tasks.update + contacts.ask.
    - User interjects with guidance that only affects the contact selection criteria.

    Validates targeted routing to contacts.ask and routing-only behavior.
    """
    primitives = create_primitives(
        contact_desc=(
            "Berlin office contacts: Alice Schmidt (designer, alice@example.com), "
            "Bob Mueller (engineering lead, bob@example.com), "
            "Carol Weber (PM, carol@example.com)."
        ),
        task_desc="Task list for Q4 planning.",
    )

    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    task_handle = await primitives.tasks.update(
        "Create a task: Review Q4 budget report, assign to the Berlin office contact."
    )
    contact_handle = await primitives.contacts.ask(
        "Who is the contact in the Berlin office?"
    )

    await TEST_GATE.wait()

    contact_result = await contact_handle.result()
    task_result = await task_handle.result()
    return f"Assignee: {contact_result}\\nTask: {task_result}"
"""

        original_plan_source = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

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
            _ = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.tasks.update",
            )

            msg = (
                "For the Berlin contact lookup, if there are multiple candidates, please prioritize anyone who has "
                "'manager' or 'lead' in their role."
            )
            status = await asyncio.wait_for(h.interject(msg), timeout=INTERJECT_TIMEOUT)
            assert isinstance(status, str) and status.strip()

            await wait_for_pane_steering_event(
                h,
                handle_id=contacts_hid,
                method="interject",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )

            ok_ids = get_ok_steering_handle_ids(h, method="interject")
            assert ok_ids == {
                str(contacts_hid),
            }, f"Expected targeted routing to contacts.ask. ok_ids={sorted(ok_ids)}"

            assert h.plan_source_code == original_plan_source

            _ = await release_gate_and_finish(h)

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_interject_targeted_to_transcripts_handle(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Plan spawns knowledge.update + contacts.ask + transcripts.ask.
    - User interjects with a preference that only applies to the message-history lookup.

    Validates targeted routing to transcripts.ask only.
    """
    primitives = create_primitives(
        contact_desc="Alice Schmidt (designer, Berlin, alice@example.com).",
        transcript_desc=(
            "Messages from Alice: 60 days ago: 'Budget planning discussion', "
            "20 days ago: 'Technical update on new feature', "
            "10 days ago: 'Code review feedback'."
        ),
        knowledge_desc="Knowledge base for team profiles.",
    )

    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    knowledge_handle = await primitives.knowledge.update(
        "Add a comprehensive profile for Alice to the knowledge base."
    )
    contact_handle = await primitives.contacts.ask(
        "Find Alice's contact details (email, role, location)."
    )
    transcript_handle = await primitives.transcripts.ask(
        "Find recent messages from Alice about project updates."
    )

    await TEST_GATE.wait()

    transcript_result = await transcript_handle.result()
    contact_result = await contact_handle.result()
    knowledge_result = await knowledge_handle.result()
    return (
        f"Messages: {transcript_result}\\n"
        f"Contact: {contact_result}\\n"
        f"Knowledge: {knowledge_result}"
    )
"""

        original_plan_source = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

        try:
            await wait_for_pane_handle_count(
                h,
                expected=3,
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )
            handles = await h.pane.list_handles()
            transcripts_hid = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.transcripts.ask",
            )
            _ = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.contacts.ask",
            )
            _ = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.knowledge.update",
            )

            msg = (
                "For the message history you’re pulling right now, please only include items "
                "from the last 30 days and focus on technical updates."
            )
            status = await asyncio.wait_for(h.interject(msg), timeout=INTERJECT_TIMEOUT)
            assert isinstance(status, str) and status.strip()

            await wait_for_pane_steering_event(
                h,
                handle_id=transcripts_hid,
                method="interject",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )

            ok_ids = get_ok_steering_handle_ids(h, method="interject")
            assert ok_ids == {
                str(transcripts_hid),
            }, f"Expected targeted routing to transcripts.ask. ok_ids={sorted(ok_ids)}"

            assert h.plan_source_code == original_plan_source

            _ = await release_gate_and_finish(h)

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_interject_broadcast_to_all_in_flight_handles(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Plan spawns multiple in-flight handles (knowledge + contacts + transcripts).
    - User interjects with a global preference that should apply to all in-flight work.

    Validates broadcast routing: all handles receive the interjection.
    """
    primitives = create_primitives(
        contact_desc="Team members: Alice (designer), Bob (engineer), Carol (PM).",
        transcript_desc="Recent team communications include planning + project updates.",
        knowledge_desc="Knowledge base for Q4 team information.",
    )

    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    knowledge_handle = await primitives.knowledge.update(
        "Update the knowledge base with Q4 team information."
    )
    contact_handle = await primitives.contacts.ask(
        "List all team members with their roles."
    )
    transcript_handle = await primitives.transcripts.ask(
        "Summarize recent team communications."
    )

    await TEST_GATE.wait()

    results = await asyncio.gather(
        knowledge_handle.result(),
        contact_handle.result(),
        transcript_handle.result(),
    )
    return "\\n\\n".join([str(r) for r in results])
"""

        original_plan_source = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

        try:
            await wait_for_pane_handle_count(
                h,
                expected=3,
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )
            handles = await h.pane.list_handles()
            contacts_hid = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.contacts.ask",
            )
            transcripts_hid = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.transcripts.ask",
            )
            knowledge_hid = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.knowledge.update",
            )

            msg = (
                "Across everything you’re working on right now, please keep responses concise "
                "and include only essential details."
            )
            status = await asyncio.wait_for(h.interject(msg), timeout=INTERJECT_TIMEOUT)
            assert isinstance(status, str) and status.strip()

            for hid in (contacts_hid, transcripts_hid, knowledge_hid):
                await wait_for_pane_steering_event(
                    h,
                    handle_id=hid,
                    method="interject",
                    status="ok",
                    timeout=STEERING_EVENT_TIMEOUT,
                )

            ok_ids = get_ok_steering_handle_ids(h, method="interject")
            expected = {str(contacts_hid), str(transcripts_hid), str(knowledge_hid)}
            assert (
                ok_ids == expected
            ), f"Expected broadcast to all handles. ok_ids={sorted(ok_ids)} expected={sorted(expected)}"

            assert h.plan_source_code == original_plan_source

            _ = await release_gate_and_finish(h)

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_interject_selective_contacts_only(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Plan spawns contacts.ask + transcripts.ask + knowledge.ask in parallel.
    - User interjects with a preference explicitly scoped to contact-related work.

    Validates selective routing to contacts.ask only.
    """
    primitives = create_primitives(
        contact_desc="Berlin office: Alice Schmidt (designer), Bob Mueller (engineering lead).",
        transcript_desc="Messages mention Berlin and London office locations.",
        knowledge_desc="Knowledge contains office location facts.",
    )

    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    contact_handle = await primitives.contacts.ask(
        "Find contacts in Berlin and London offices."
    )
    transcript_handle = await primitives.transcripts.ask(
        "Find messages about office locations."
    )
    knowledge_handle = await primitives.knowledge.ask(
        "What do we know about office locations?"
    )

    await TEST_GATE.wait()

    results = await asyncio.gather(
        contact_handle.result(),
        transcript_handle.result(),
        knowledge_handle.result(),
    )
    return "\\n\\n".join([str(r) for r in results])
"""

        original_plan_source = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

        try:
            await wait_for_pane_handle_count(
                h,
                expected=3,
                timeout=HANDLE_REGISTRATION_TIMEOUT,
            )
            handles = await h.pane.list_handles()
            contacts_hid = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.contacts.ask",
            )
            _ = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.transcripts.ask",
            )
            _ = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.knowledge.ask",
            )

            msg = (
                "For any contact-related info you’re listing right now, please use full names "
                "and include the office location in parentheses."
            )
            status = await asyncio.wait_for(h.interject(msg), timeout=INTERJECT_TIMEOUT)
            assert isinstance(status, str) and status.strip()

            await wait_for_pane_steering_event(
                h,
                handle_id=contacts_hid,
                method="interject",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )

            ok_ids = get_ok_steering_handle_ids(h, method="interject")
            assert ok_ids == {
                str(contacts_hid),
            }, f"Expected selective routing to contacts.ask. ok_ids={sorted(ok_ids)}"

            assert h.plan_source_code == original_plan_source

            _ = await release_gate_and_finish(h)

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")
