"""
Actor handle event surface tests (simulated managers).

These tests validate the *top-level* `HierarchicalActorHandle` bottom-up event APIs:
- `next_clarification()` returns a dict containing `question` + stable `call_id`
- `answer_clarification(call_id, answer)` unblocks the plan and routes the answer
  to the correct in-flight child handle (via `SteerableToolPane`)
- `next_notification()` yields queued notifications surfaced from the pane
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from tests.test_actor.test_state_managers.test_simulated.test_steerability.helpers import (
    release_gate_and_finish,
    start_canned_plan,
)
from tests.test_actor.test_state_managers.test_simulated.test_steerability.timeouts import (
    CLARIFICATION_TIMEOUT,
    HANDLE_REGISTRATION_TIMEOUT,
    PLAN_COMPLETION_TIMEOUT,
    STEERING_EVENT_TIMEOUT,
)
from tests.test_actor.test_state_managers.utils import (
    get_pending_clarification_count,
    pick_handle_id_by_origin_tool,
    wait_for_pane_handle_count,
    wait_for_pane_steering_event,
)
from tests.async_helpers import _wait_for_condition

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_handle_next_clarification_returns_call_id_and_routes_answer(
    mock_verification,
    create_primitives_with_clarification_forcing,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - A simulated state manager handle emits a clarification event via the pane.
    - The top-level HierarchicalActorHandle surfaces it via `next_clarification()`.
    - The test answers via `answer_clarification(call_id, ...)` (call-id-aware surface).
    """

    primitives = create_primitives_with_clarification_forcing(
        contact_desc=(
            "You have two contacts named David in your CRM: "
            "David Smith (david.smith@example.com) and David Jones (david.jones@example.com)."
        ),
        contact_clarification_triggers=["update david"],
    )

    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor, with_clarification=False)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    update_handle = await primitives.contacts.update(
        "Update David's email address to david.new@example.com"
    )

    # Block deterministically so the test can route a clarification answer before awaiting `.result()`.
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

            clar = await asyncio.wait_for(
                h.next_clarification(),
                timeout=CLARIFICATION_TIMEOUT,
            )
            assert isinstance(clar, dict)
            assert isinstance(clar.get("question"), str) and clar["question"].strip()
            assert isinstance(clar.get("call_id"), str) and clar["call_id"].strip()

            # Answer using the handle-surfaced call_id (not the pane's internal call_id).
            await h.answer_clarification(str(clar["call_id"]), "David Smith")

            # The answer should be routed via the pane to the correct in-flight handle.
            await wait_for_pane_steering_event(
                h,
                handle_id=str(contact_hid),
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
async def test_handle_next_notification_surfaces_pane_notifications(
    mock_verification,
    create_primitives,
    create_actor_with_primitives,
    create_canned_handle,
):
    """
    Scenario:
    - A dummy in-flight handle is registered with the plan's SteerableToolPane.
    - The pane emits a notification event from that handle.
    - The top-level HierarchicalActorHandle surfaces it via `next_notification()`.
    """

    primitives = create_primitives(
        contact_desc="Contacts exist but are not used in this test.",
        transcript_desc="Transcripts exist but are not used in this test.",
        knowledge_desc="Knowledge exists but is not used in this test.",
        task_desc="Tasks exist but are not used in this test.",
    )

    async with create_actor_with_primitives(primitives) as actor:
        h = await create_canned_handle(actor=actor, with_clarification=False)

        CANNED_PLAN = """
import asyncio

TEST_GATE = asyncio.Event()

async def main_plan():
    await TEST_GATE.wait()
    return "done"
"""

        _ = start_canned_plan(h, actor=actor, source=CANNED_PLAN)

        try:

            class _DummyNotifHandle:
                def __init__(self) -> None:
                    self._q: asyncio.Queue[dict] = asyncio.Queue()

                async def next_clarification(self) -> dict:  # noqa: D401
                    return {}

                async def next_notification(self) -> dict:  # noqa: D401
                    return await self._q.get()

            dummy = _DummyNotifHandle()

            await h.pane.register_handle(
                handle=dummy,  # type: ignore[arg-type]
                handle_id="dummy_notif_handle",
                parent_handle_id=None,
                origin_tool="dummy.notification",
                origin_step=0,
                environment_namespace="test",
                capabilities=[],
                call_stack=None,
            )

            await dummy._q.put({"message": "hello from dummy"})

            notif = await asyncio.wait_for(h.next_notification(), timeout=30.0)
            assert isinstance(notif, dict)
            assert notif.get("message") == "hello from dummy"
            assert str(notif.get("handle_id") or "") == "dummy_notif_handle"

            result = await release_gate_and_finish(h, timeout=PLAN_COMPLETION_TIMEOUT)
            assert isinstance(result, str) and result.strip()

        finally:
            with contextlib.suppress(Exception):
                await h.stop("test cleanup")
