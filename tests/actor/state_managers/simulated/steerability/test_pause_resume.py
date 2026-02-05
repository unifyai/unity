"""
Pause/resume tests for Actor steerability (simulated managers).

These tests validate that `HierarchicalActorHandle.pause()`, `interject()`, and
`resume()` work together in realistic scenarios where a user needs to interrupt,
correct, and continue execution.

Key properties we validate:
- Deterministic alignment using a plan-local `TEST_GATE = asyncio.Event()` so handles
  are registered (and their `.result()` not yet awaited) before steering.
- Pause propagates to in-flight pane handles (pause events are visible via pane
  `steering_applied` events).
- Interjections sent while paused are routed to the correct in-flight handle(s).
- Resume propagates to paused pane handles and execution completes after releasing the gate.
- Interjections remain routing-only (plan source is not patched/restarted).
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from tests.actor.state_managers.utils import (
    pick_handle_id_by_origin_tool,
    wait_for_pane_handle_count,
    wait_for_pane_steering_event,
)

from .helpers import (
    get_ok_steering_handle_ids,
    release_gate_and_finish,
    start_canned_plan,
)
from .timeouts import (
    HANDLE_REGISTRATION_TIMEOUT,
    INTERJECT_TIMEOUT,
    PAUSE_RESUME_TIMEOUT,
    PLAN_COMPLETION_TIMEOUT,
    STEERING_EVENT_TIMEOUT,
)

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_pause_review_interject_resume_email_drafts(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Actor prepares outreach email guidance for Berlin contacts.
    - User pauses to review.
    - User interjects a tone correction while paused.
    - User resumes and the plan completes.

    Validates pause → interject → resume are applied and observable via pane events.
    """

    primitives = create_primitives(
        contact_desc=(
            "You have 3 contacts in Berlin: Alice Schmidt (designer, alice@example.com), "
            "Bob Mueller (engineer, bob@example.com), Carol Weber (PM, carol@example.com). "
            "All prefer professional communication."
        ),
        transcript_desc=(
            "Recent messages show these contacts respond well to formal, structured emails "
            "with clear subject lines and professional greetings."
        ),
    )
    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    contacts_handle = await primitives.contacts.ask(
        "List all contacts in Berlin with their email addresses and roles. "
        "Do not ask clarifying questions."
    )
    email_handle = await primitives.transcripts.ask(
        "Based on the Berlin contacts, draft a professional outreach email template. "
        "Include: formal greeting, brief introduction, meeting invitation, professional sign-off. "
        "Use formal tone (Dear [Name], Best regards). Do not ask clarifying questions."
    )

    # Deterministic pause point: allow user to pause/review before `.result()` calls.
    await TEST_GATE.wait()

    contacts_result, email_result = await asyncio.gather(
        contacts_handle.result(),
        email_handle.result(),
    )
    return (
        "CONTACTS:\\n"
        f"{contacts_result}\\n\\n"
        "EMAIL TEMPLATE:\\n"
        f"{email_result}"
    )
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
            transcripts_hid = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.transcripts.ask",
            )

            pause_status = await asyncio.wait_for(
                h.pause(),
                timeout=PAUSE_RESUME_TIMEOUT,
            )
            assert isinstance(pause_status, str) and "paused" in pause_status.lower()

            # Pause should propagate to both in-flight handles.
            await wait_for_pane_steering_event(
                h,
                handle_id=contacts_hid,
                method="pause",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )
            await wait_for_pane_steering_event(
                h,
                handle_id=transcripts_hid,
                method="pause",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )

            interject_msg = (
                "Actually, let's make the email tone much more casual and friendly. "
                "Use 'Hi <FirstName>,' (first name only) and avoid 'Dear'. "
                "Sign off with 'Cheers' or 'Best' instead of 'Best regards'."
            )
            await asyncio.wait_for(
                h.interject(interject_msg),
                timeout=INTERJECT_TIMEOUT,
            )

            e_transcripts = await wait_for_pane_steering_event(
                h,
                handle_id=transcripts_hid,
                method="interject",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )
            routed = str(
                (e_transcripts.get("payload") or {}).get("args", {}).get("message", ""),
            ).strip()
            assert routed

            # Tone guidance applies to email drafting, not the contact listing.
            ok_ids = get_ok_steering_handle_ids(h, method="interject")
            assert ok_ids == {str(transcripts_hid)}, (
                "Expected interjection to be routed only to the transcripts handle for this scenario. "
                f"Got ok_ids={sorted(ok_ids)}; contacts_hid={contacts_hid}, transcripts_hid={transcripts_hid}"
            )

            # Interjection should remain routing-only (no patch/restart).
            assert h.plan_source_code == original_plan_source

            resume_status = await asyncio.wait_for(
                h.resume(),
                timeout=PAUSE_RESUME_TIMEOUT,
            )
            assert isinstance(resume_status, str) and "resumed" in resume_status.lower()

            await wait_for_pane_steering_event(
                h,
                handle_id=contacts_hid,
                method="resume",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )
            await wait_for_pane_steering_event(
                h,
                handle_id=transcripts_hid,
                method="resume",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )

            # Release deterministic gate and ensure plan completes.
            _ = await release_gate_and_finish(h, timeout=PLAN_COMPLETION_TIMEOUT)

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_pause_for_data_refresh_then_resume_broadcast(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Actor queries contacts, transcripts, and knowledge for a Q4 report.
    - User pauses while waiting for an external data refresh.
    - User interjects with the refreshed context (broadcast intent).
    - User resumes and the plan completes.

    Validates broadcast interjection during pause + resume propagation.
    """

    primitives = create_primitives(
        contact_desc=(
            "You have Q4 2024 contacts: Alice (joined Oct), Bob (joined Nov). "
            "Data last refreshed: Dec 1, 2024."
        ),
        transcript_desc=(
            "You have Q4 messages about year-end planning and budget reviews. "
            "Data last refreshed: Dec 1, 2024."
        ),
        knowledge_desc=(
            "Knowledge base contains Q4 project updates and metrics. "
            "Data last refreshed: Dec 1, 2024."
        ),
    )
    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    contacts_handle = await primitives.contacts.ask(
        "Summarize Q4 2024 contact activity (new contacts, key interactions). "
        "Do not ask clarifying questions."
    )
    transcripts_handle = await primitives.transcripts.ask(
        "Summarize Q4 2024 message themes (budget, planning, year-end). "
        "Do not ask clarifying questions."
    )
    knowledge_handle = await primitives.knowledge.ask(
        "Summarize Q4 2024 project updates and key metrics. "
        "Do not ask clarifying questions."
    )

    await TEST_GATE.wait()

    contacts_result, transcripts_result, knowledge_result = await asyncio.gather(
        contacts_handle.result(),
        transcripts_handle.result(),
        knowledge_handle.result(),
    )
    return (
        "Q4 2024 REPORT:\\n\\n"
        "CONTACTS:\\n"
        f"{contacts_result}\\n\\n"
        "MESSAGES:\\n"
        f"{transcripts_result}\\n\\n"
        "PROJECTS:\\n"
        f"{knowledge_result}"
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
                origin_tool_prefix="primitives.knowledge.ask",
            )

            pause_status = await asyncio.wait_for(
                h.pause(),
                timeout=PAUSE_RESUME_TIMEOUT,
            )
            assert isinstance(pause_status, str) and "paused" in pause_status.lower()

            for hid in (contacts_hid, transcripts_hid, knowledge_hid):
                await wait_for_pane_steering_event(
                    h,
                    handle_id=hid,
                    method="pause",
                    status="ok",
                    timeout=STEERING_EVENT_TIMEOUT,
                )

            interject_msg = (
                "FYI: the data refresh just completed (Dec 15, 2024). "
                "Please incorporate these updates into the summaries you’re producing right now: "
                "2 new contacts added in December (Carol, David); "
                "5 new messages about holiday schedules and Q1 planning; "
                "updated metrics show a 15% increase in customer engagement."
            )
            await asyncio.wait_for(
                h.interject(interject_msg),
                timeout=INTERJECT_TIMEOUT,
            )

            # Broadcast intent: ensure all three in-flight handles received the interjection.
            for hid in (contacts_hid, transcripts_hid, knowledge_hid):
                e = await wait_for_pane_steering_event(
                    h,
                    handle_id=hid,
                    method="interject",
                    status="ok",
                    timeout=STEERING_EVENT_TIMEOUT,
                )
                routed = str(
                    (e.get("payload") or {}).get("args", {}).get("message", ""),
                ).strip()
                assert routed

            assert h.plan_source_code == original_plan_source

            resume_status = await asyncio.wait_for(
                h.resume(),
                timeout=PAUSE_RESUME_TIMEOUT,
            )
            assert isinstance(resume_status, str) and "resumed" in resume_status.lower()

            for hid in (contacts_hid, transcripts_hid, knowledge_hid):
                await wait_for_pane_steering_event(
                    h,
                    handle_id=hid,
                    method="resume",
                    status="ok",
                    timeout=STEERING_EVENT_TIMEOUT,
                )

            _ = await release_gate_and_finish(h, timeout=PLAN_COMPLETION_TIMEOUT)

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_pause_change_priority_filter_resume(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - Actor performs contact analysis for an outreach campaign.
    - User pauses, then interjects to focus only on high-priority contacts.
    - User resumes and the plan completes.

    Validates pause/resume works and interjection is at least routed to the primary contacts handle.
    """

    primitives = create_primitives(
        contact_desc=(
            "You have 10 contacts: High priority: Alice (enterprise customer, $500K ARR), "
            "Bob (strategic partner). Medium priority: Carol (mid-market, $50K ARR), "
            "David (growing account). Low priority: 6 other contacts with minimal activity."
        ),
        transcript_desc=(
            "Recent messages show high priority contacts need immediate attention for renewals. "
            "Medium/low priority contacts can wait until next quarter."
        ),
    )
    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    contacts_handle = await primitives.contacts.ask(
        "Analyze all contacts and identify who should be included in the Q1 outreach campaign. "
        "Consider activity level, engagement, and potential value. "
        "Do not ask clarifying questions."
    )
    transcripts_handle = await primitives.transcripts.ask(
        "Summarize recent messages to understand which contacts need immediate attention. "
        "Do not ask clarifying questions."
    )

    await TEST_GATE.wait()

    contacts_result, transcripts_result = await asyncio.gather(
        contacts_handle.result(),
        transcripts_handle.result(),
    )
    return (
        "OUTREACH CAMPAIGN ANALYSIS:\\n\\n"
        "CONTACTS:\\n"
        f"{contacts_result}\\n\\n"
        "MESSAGE CONTEXT:\\n"
        f"{transcripts_result}"
    )
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
            transcripts_hid = pick_handle_id_by_origin_tool(
                handles,
                origin_tool_prefix="primitives.transcripts.ask",
            )

            pause_status = await asyncio.wait_for(
                h.pause(),
                timeout=PAUSE_RESUME_TIMEOUT,
            )
            assert isinstance(pause_status, str) and "paused" in pause_status.lower()

            await wait_for_pane_steering_event(
                h,
                handle_id=contacts_hid,
                method="pause",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )
            await wait_for_pane_steering_event(
                h,
                handle_id=transcripts_hid,
                method="pause",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )

            interject_msg = (
                "Change of plans: focus ONLY on high priority contacts for this campaign. "
                "High priority means enterprise customers ($500K+ ARR) or strategic partners. "
                "Exclude medium and low priority contacts."
            )
            await asyncio.wait_for(
                h.interject(interject_msg),
                timeout=INTERJECT_TIMEOUT,
            )

            # Must at least apply to the contacts analysis handle.
            await wait_for_pane_steering_event(
                h,
                handle_id=contacts_hid,
                method="interject",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )

            assert h.plan_source_code == original_plan_source

            resume_status = await asyncio.wait_for(
                h.resume(),
                timeout=PAUSE_RESUME_TIMEOUT,
            )
            assert isinstance(resume_status, str) and "resumed" in resume_status.lower()

            await wait_for_pane_steering_event(
                h,
                handle_id=contacts_hid,
                method="resume",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )
            await wait_for_pane_steering_event(
                h,
                handle_id=transcripts_hid,
                method="resume",
                status="ok",
                timeout=STEERING_EVENT_TIMEOUT,
            )

            _ = await release_gate_and_finish(h, timeout=PLAN_COMPLETION_TIMEOUT)

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")
