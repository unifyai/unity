"""
On-the-fly plan generation + steerability tests (simulated managers).

Unlike the rest of this suite (which uses deterministic canned plans), these tests
exercise the full "front half" of the HierarchicalActor:
  - actor.act(...) → LLM plan generation → plan execution

To keep tests deterministic without injecting canned code, we use the Actor's built-in
`request_clarification(...)` to trigger a natural "user confirmation" gate.
This makes the plan pause awaiting user input, giving tests a stable window to:
  - observe in-flight handles in the `SteerableToolPane`
  - apply steering via `handle.interject()` / `pause()` / `resume()`
  - then answer the clarification to let the plan finish.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from tests.test_actor.test_state_managers.test_simulated.test_steerability.helpers import (
    get_ok_steering_handle_ids,
)
from tests.test_actor.test_state_managers.test_simulated.test_steerability.timeouts import (
    CLARIFICATION_TIMEOUT,
    HANDLE_REGISTRATION_TIMEOUT,
    INTERJECT_TIMEOUT,
    PAUSE_RESUME_TIMEOUT,
    PLAN_COMPLETION_TIMEOUT,
    STEERING_EVENT_TIMEOUT,
)
from tests.test_actor.test_state_managers.utils import (
    pick_handle_id_by_origin_tool,
    wait_for_pane_handle_count,
    wait_for_pane_steering_event,
)
from tests.async_helpers import _wait_for_condition

pytestmark = pytest.mark.eval


async def _wait_for_plan_source(handle, *, timeout: float = 90.0) -> str:
    """Wait until the actor has generated a non-empty `plan_source_code`."""

    async def _has_plan() -> bool:
        src = getattr(handle, "plan_source_code", "") or ""
        return "async def main_plan" in src

    await asyncio.wait_for(
        _wait_for_condition(_has_plan, poll=0.05, timeout=timeout),
        timeout=timeout + 10.0,
    )
    return str(getattr(handle, "plan_source_code", "") or "")


@pytest.mark.asyncio
@pytest.mark.timeout(420)
async def test_otf_plan_interject_targeted_while_waiting_for_user_confirmation(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
):
    """
    Scenario:
    - User asks the Actor to start gathering contacts + draft outreach tone guidance,
      but to ask a question before finalizing (plan uses request_clarification gate).
    - While waiting, user interjects with a tone preference.

    Validates:
    - Plan is generated on-the-fly (non-empty `plan_source_code`).
    - Contacts + transcripts handles are in-flight and registered in the pane.
    - Interjection is routed to the transcripts handle (tone guidance), and is routing-only
      (no disruptive plan patch/restart).
    """

    primitives = create_primitives(
        contact_desc=(
            "You have 3 contacts in Berlin: Alice Schmidt (designer, alice@example.com), "
            "Bob Mueller (engineering lead, bob@example.com), Carol Weber (PM, carol@example.com)."
        ),
        transcript_desc=(
            "Recent messages show these contacts prefer professional but friendly communication."
        ),
        # Provide safe defaults so unexpected manager usage doesn't hit ManagerRegistry.
        knowledge_desc="Knowledge base contains general notes about outreach preferences.",
        task_desc="Task list exists but is not needed for this scenario.",
    )

    async with create_actor_with_primitives(primitives) as actor:
        clar_up: asyncio.Queue[str] = asyncio.Queue()
        clar_down: asyncio.Queue[str] = asyncio.Queue()

        goal = (
            "Pull a list of Berlin contacts (names + emails), and in parallel draft outreach tone guidance "
            "based on prior messages with these contacts. "
            "Kick off BOTH workstreams immediately and in parallel (do NOT wait for my tone choice; do NOT await results before asking). "
            "Let the in-flight work continue while you wait for my answer. "
            "Before you finalize anything, ask me which tone to use (casual vs formal) and wait for my answer."
        )

        h = await actor.act(
            goal,
            persist=False,
            _clarification_up_q=clar_up,
            _clarification_down_q=clar_down,
        )

        try:
            plan_src = await _wait_for_plan_source(h, timeout=90.0)
            assert (
                "request_clarification" in plan_src
            ), "Expected the on-the-fly plan to use request_clarification as a user gate."

            # Wait for the gate question (plan-level clarification).
            question = await asyncio.wait_for(
                clar_up.get(),
                timeout=CLARIFICATION_TIMEOUT,
            )
            assert isinstance(question, str) and question.strip()

            # Ensure at least the two expected handles are in-flight.
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

            original_plan_source = str(h.plan_source_code or "")

            # Interject a tone preference while the plan is waiting for user confirmation.
            msg = (
                "Actually, make the outreach much more casual and friendly. "
                "Use 'Hey <FirstName>,' and avoid 'Dear'. Keep it short."
            )
            status = await asyncio.wait_for(h.interject(msg), timeout=INTERJECT_TIMEOUT)
            assert isinstance(status, str) and status.strip()

            # Wait for *any* interject steering event for the transcripts handle, then
            # assert that it was applied (ok) or safely ignored (no-op) if the handle
            # already completed extremely quickly in the simulated environment.
            event = await wait_for_pane_steering_event(
                h,
                handle_id=transcripts_hid,
                method="interject",
                timeout=STEERING_EVENT_TIMEOUT,
            )
            payload = event.get("payload") or {}
            applied_status = str(payload.get("status") or "")
            assert applied_status in {
                "ok",
                "no-op",
            }, f"Unexpected interject status for transcripts handle: {payload}. Event={event}"
            if applied_status == "ok":
                ok_ids = get_ok_steering_handle_ids(h, method="interject")
                assert str(transcripts_hid) in ok_ids, (
                    "Expected the interjection to reach the transcripts handle.",
                )
            else:
                # If no-op, it must be because the handle is terminal, not because routing
                # failed to resolve the handle ID.
                reason = str(payload.get("reason") or "")
                assert "handle already" in reason, (
                    f"Interject was no-op for an unexpected reason: {reason!r}. Payload={payload}",
                )

            # Routing-only invariant: preference update should not force a plan rewrite/restart.
            assert str(h.plan_source_code or "") == original_plan_source

            # Answer the plan-level clarification to let execution continue.
            await clar_down.put("casual")

            result = await asyncio.wait_for(h.result(), timeout=PLAN_COMPLETION_TIMEOUT)
            # Result type is not part of the routing/steerability contract: plans may return structured objects.
            assert result is not None and str(result).strip()

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")


@pytest.mark.asyncio
@pytest.mark.timeout(480)
async def test_otf_plan_pause_broadcast_interject_resume_then_finish(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
):
    """
    Scenario:
    - Actor starts parallel work across contacts + transcripts + knowledge and asks the user to confirm
      a scope choice before finalizing.
    - User pauses, broadcasts an interjection with a global constraint, then resumes and answers.

    Validates:
    - On-the-fly plan generation produces multiple in-flight handles.
    - Pause/resume propagate to in-flight handles.
    - Broadcast-style interjection reaches multiple handles (infrastructure-level evidence).
    """

    primitives = create_primitives(
        contact_desc=(
            "Contacts across quarters: Q3: Carol (Aug 2024), David (Sep 2024). "
            "Q4: Alice (Oct 2024), Bob (Nov 2024)."
        ),
        transcript_desc=(
            "Messages include Q3 planning and Q4 year-end updates + outreach tone notes."
        ),
        knowledge_desc=(
            "Knowledge includes Q3 launch notes and Q4 customer feedback + metrics."
        ),
        task_desc="Tasks exist but are not needed for this scenario.",
    )

    async with create_actor_with_primitives(primitives) as actor:
        clar_up: asyncio.Queue[str] = asyncio.Queue()
        clar_down: asyncio.Queue[str] = asyncio.Queue()

        goal = (
            "Start building a Q4-only summary package: (1) summarize contacts across 2024, "
            "(2) summarize key message themes across 2024, and (3) summarize knowledge-base updates across 2024. "
            "Kick off all three in parallel immediately. While they are still running, "
            "ask me whether I want to exclude Q3 entirely before you finalize anything. "
            "Once you have the summaries, if I choose to exclude Q3, just trim Q3 sections "
            "from the results instead of re-running the summaries."
        )

        h = await actor.act(
            goal,
            persist=False,
            _clarification_up_q=clar_up,
            _clarification_down_q=clar_down,
        )

        try:
            _ = await _wait_for_plan_source(h, timeout=120.0)

            question = await asyncio.wait_for(
                clar_up.get(),
                timeout=CLARIFICATION_TIMEOUT,
            )
            assert isinstance(question, str) and question.strip()

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

            original_plan_source = str(h.plan_source_code or "")

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

            msg = (
                "Important: for everything you're producing right now, focus ONLY on Q4 (Oct/Nov/Dec). "
                "Ignore Q3 entirely and don't mention it."
            )
            status = await asyncio.wait_for(h.interject(msg), timeout=INTERJECT_TIMEOUT)
            assert isinstance(status, str) and status.strip()

            # We expect this global constraint to be relevant to multiple in-flight summaries.
            for hid in (contacts_hid, transcripts_hid, knowledge_hid):
                await wait_for_pane_steering_event(
                    h,
                    handle_id=hid,
                    method="interject",
                    status="ok",
                    timeout=STEERING_EVENT_TIMEOUT,
                )

            # NOTE: A global constraint like “focus ONLY on Q4” may legitimately cause
            # the Actor to patch the plan to affect *future* (not-yet-started) steps
            # in addition to routing the interjection to in-flight handles.
            # This test is about pause/resume propagation + broadcast routing, so we
            # intentionally do not require a routing-only invariant here.

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

            await clar_down.put("Yes — exclude Q3 entirely.")
            result = await asyncio.wait_for(h.result(), timeout=PLAN_COMPLETION_TIMEOUT)
            # Result type is not part of the routing/steerability contract: plans may return structured objects.
            assert result is not None and str(result).strip()

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")
