"""Immediate pause/resume tests for HierarchicalActorHandle."""

import asyncio
import contextlib
import traceback
import textwrap
from unittest.mock import AsyncMock

import pytest

from unity.actor.hierarchical_actor import (
    HierarchicalActor,
    HierarchicalActorHandle,
    VerificationAssessment,
    _HierarchicalHandleState,
)
from unity.function_manager.computer_backends import ComputerAgentError


class _OkVerificationClient:
    """Minimal verification client used by pause/resume tests."""

    def __init__(self):
        self.generate = AsyncMock(
            return_value=VerificationAssessment(
                status="ok",
                reason="Mock OK",
            ).model_dump_json(),
        )

    def set_response_format(self, *_args, **_kwargs):
        pass

    def reset_response_format(self, *_args, **_kwargs):
        pass

    def reset_messages(self, *_args, **_kwargs):
        pass

    def set_system_message(self, *_args, **_kwargs):
        pass


async def _wait_for_state(
    plan: HierarchicalActorHandle,
    expected: _HierarchicalHandleState,
    timeout: float = 60.0,
    poll: float = 0.05,
):
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if plan._state == expected:
            return
        await asyncio.sleep(poll)
    tail = "\n".join(plan.action_log[-15:])
    raise AssertionError(
        f"Timed out waiting for {expected.name}; state={plan._state.name}\n---\n{tail}",
    )


CANNED_PLAN_SIMPLE_IMMEDIATE_PAUSE_RESUME = textwrap.dedent(
    """
    @verify
    async def step():
        await computer_primitives.act("first")
        await computer_primitives.act("second")
        return "done"

    async def main_plan():
        return await step()
    """,
)

CANNED_PLAN_WITH_OBSERVE_IMMEDIATE_PAUSE_RESUME = textwrap.dedent(
    """
    @verify
    async def step_with_observe():
        await computer_primitives.act("open")
        await computer_primitives.observe("what is the title?")
        await computer_primitives.act("click cta")
        return "done/observe"

    async def main_plan():
        return await step_with_observe()
    """,
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_immediate_pause_cancels_action_and_restarts_function_cleanly():
    actor = HierarchicalActor(headless=True, connect_now=False, computer_mode="mock")

    act_entered = asyncio.Event()
    act_proceed = asyncio.Event()
    first_act_done = False

    async def act_side_effect(*args, **kwargs):
        nonlocal first_act_done
        _ = kwargs
        if not first_act_done:
            act_entered.set()
            await act_proceed.wait()
            first_act_done = True
            raise ComputerAgentError("cancelled", "Action was interrupted.")
        return None

    actor.computer_primitives.act = AsyncMock(side_effect=act_side_effect)  # type: ignore[attr-defined]
    actor.computer_primitives.observe = AsyncMock(return_value=None)  # type: ignore[attr-defined]
    actor.computer_primitives.navigate = AsyncMock(return_value=None)  # type: ignore[attr-defined]

    plan = HierarchicalActorHandle(
        actor=actor,
        goal="Immediate pause test",
        persist=False,
    )

    if plan._execution_task:
        plan._execution_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await plan._execution_task

    plan.plan_source_code = actor._sanitize_code(
        CANNED_PLAN_SIMPLE_IMMEDIATE_PAUSE_RESUME,
        plan,
    )
    plan.verification_client = _OkVerificationClient()
    plan._execution_task = asyncio.create_task(plan._initialize_and_run())

    await act_entered.wait()
    await plan.pause(immediate=True)
    act_proceed.set()

    await _wait_for_state(plan, _HierarchicalHandleState.PAUSED, timeout=10)
    await plan.resume()
    result = await plan.result()

    log = "\n".join(plan.action_log)
    assert (
        "Retrying 'step' Reason: Action 'computer_primitives.act((('first',), {}))' interrupted by immediate pause"
        in log
    )
    assert actor.computer_primitives.act.call_count >= 3  # type: ignore[attr-defined]
    assert "ERROR" not in str(result)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_immediate_pause_caches_completed_actions_for_replay_after_resume():
    actor = HierarchicalActor(headless=True, connect_now=False, computer_mode="mock")

    open_called = asyncio.Event()
    observe_called = asyncio.Event()
    cta_entered = asyncio.Event()
    cta_proceed = asyncio.Event()
    cta_cancel_count = 0

    async def act_side_effect(*args, **kwargs):
        nonlocal cta_cancel_count
        _ = kwargs
        verb = args[0] if args else None
        if verb == "open":
            open_called.set()
            return None
        if verb == "click cta":
            cta_entered.set()
            await cta_proceed.wait()
            if cta_cancel_count == 0:
                cta_cancel_count += 1
                raise ComputerAgentError("cancelled", "Action was interrupted.")
            return None
        return None

    async def observe_side_effect(*args, **kwargs):
        _ = (args, kwargs)
        observe_called.set()
        return None

    actor.computer_primitives.act = AsyncMock(side_effect=act_side_effect)  # type: ignore[attr-defined]
    actor.computer_primitives.observe = AsyncMock(side_effect=observe_side_effect)  # type: ignore[attr-defined]
    actor.computer_primitives.navigate = AsyncMock(return_value=None)  # type: ignore[attr-defined]

    plan = HierarchicalActorHandle(
        actor=actor,
        goal="Immediate pause with observe",
        persist=False,
    )
    if plan._execution_task:
        plan._execution_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await plan._execution_task

    plan.plan_source_code = actor._sanitize_code(
        CANNED_PLAN_WITH_OBSERVE_IMMEDIATE_PAUSE_RESUME,
        plan,
    )
    plan.verification_client = _OkVerificationClient()
    plan._execution_task = asyncio.create_task(plan._initialize_and_run())

    await open_called.wait()
    await observe_called.wait()
    await cta_entered.wait()
    await plan.pause(immediate=True)
    cta_proceed.set()

    await _wait_for_state(plan, _HierarchicalHandleState.PAUSED, timeout=10)
    await plan.resume()
    result = await plan.result()

    log = "\n".join(plan.action_log)
    assert (
        "Retrying 'step_with_observe' Reason: Action 'computer_primitives.act((('click cta',), {}))' interrupted by immediate pause."
        in log
    )
    assert log.count("CACHE HIT") >= 2
    assert actor.computer_primitives.act.call_count >= 3  # type: ignore[attr-defined]
    assert actor.computer_primitives.observe.call_count >= 1  # type: ignore[attr-defined]
    assert "ERROR" not in str(result)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_immediate_pause_resume_orchestrator():
    """Deprecated orchestration test retained for compatibility."""
    try:
        await test_immediate_pause_cancels_action_and_restarts_function_cleanly()
        await test_immediate_pause_caches_completed_actions_for_replay_after_resume()
    except Exception as e:
        print(f"\n\n❌❌❌ A TEST FAILED: {e} ❌❌❌")
        traceback.print_exc()
        raise
