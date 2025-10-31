from __future__ import annotations

import asyncio
import functools
import textwrap
import pytest
from unittest.mock import AsyncMock, MagicMock

from unity.conductor.simulated import SimulatedConductor
from unity.actor.hierarchical_actor import (
    HierarchicalActor,
    HierarchicalPlan,
    _HierarchicalPlanState,
    ImplementationDecision,
    InterjectionDecision,
    FunctionPatch,
    VerificationAssessment,
)

# Test helpers
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)

SANDBOX_REQUEST: str = (
    "Open a browser window so we can walk through the setup together."
)


@pytest.mark.asyncio
@_handle_project
async def test_real_conductor_actor_request_routes_to_actor_not_task(monkeypatch):
    """
    Validate Conductor.request routes sandbox-like requests to Actor.act (real actor),
    and does not execute TaskScheduler.execute. The test stops the actor early after
    scheduling to avoid external side-effects, then asserts routing via message log.
    """

    # Wrap HierarchicalActor.act to signal once scheduled so we can stop early
    _orig_act = HierarchicalActor.act

    tool_started_evt = asyncio.Event()

    @functools.wraps(_orig_act)
    async def _wrapped_act(self, *a, **kw):
        handle = await _orig_act(self, *a, **kw)
        tool_started_evt.set()
        return handle

    monkeypatch.setattr(HierarchicalActor, "act", _wrapped_act, raising=True)

    # Use real HierarchicalActor but configure it to avoid eager external connections
    actor = HierarchicalActor(
        browser_mode="legacy",  # legacy avoids running magnitude service
        headless=True,
        connect_now=False,  # lazy-init browser only if used
        timeout=30,
    )

    cond = SimulatedConductor(actor=actor)

    handle = await cond.request(
        SANDBOX_REQUEST,
        _return_reasoning_steps=True,
    )

    # Wait until the Actor tool has been scheduled, then stop to finish quickly
    await asyncio.wait_for(tool_started_evt.wait(), timeout=120)
    handle.stop()

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str)

    # Actor should be invoked at least once
    executed_actor_list = tool_names_from_messages(messages, "Actor")
    assert executed_actor_list, "Expected at least one tool call"
    assert (
        executed_actor_list.count("Actor_act") >= 1
    ), f"Expected Actor_act to run at least once, saw order: {executed_actor_list}"

    # TaskScheduler.execute must NOT be called for sandbox-style requests
    executed_ts_list = tool_names_from_messages(messages, "TaskScheduler")
    assert "TaskScheduler_execute" not in set(
        executed_ts_list,
    ), f"TaskScheduler.execute must not run for sandbox scenarios, saw: {sorted(set(executed_ts_list))}"

    # If assistant explicitly requested tools, it should reference Actor_act for this scenario
    requested_actor = set(assistant_requested_tool_names(messages, "Actor"))
    if requested_actor:
        assert requested_actor <= {
            "Actor_act",
        }, f"Assistant should only request Actor_act here, saw: {sorted(requested_actor)}"


async def wait_for_state(task: HierarchicalPlan, expected_state, timeout=60, poll=0.1):
    """
    Poll the plan's state until it matches expected_state (or timeout).
    Raises AssertionError on timeout with a helpful log tail.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if task._state == expected_state:
            return
        await asyncio.sleep(poll)
    tail = "\n".join(task.action_log[-15:])
    raise AssertionError(
        f"Timed out after {timeout}s waiting for state {expected_state.name}; "
        f"current state={task._state.name}\n--- Action log tail ---\n{tail}",
    )


@pytest.mark.asyncio
@_handle_project
async def test_real_conductor_manages_actor_lifecycle_jit_and_interjection(monkeypatch):
    """
    Validates that the Conductor can manage a real HierarchicalActor plan
    through its full lifecycle: JIT implementation, interjection handling,
    cache replay, and completion.
    """

    real_actor = HierarchicalActor(
        browser_mode="legacy",  # legacy avoids running magnitude service
        headless=True,
        connect_now=False,
    )

    # Mock all external I/O on the ActionProvider instance
    # We are testing the Actor's logic, not the browser.
    real_actor.action_provider.navigate = AsyncMock(return_value=None)
    real_actor.action_provider.act = AsyncMock(return_value=None)
    real_actor.action_provider.observe = AsyncMock(return_value="Mocked Page Heading")

    # Avoid starting Chromium/Keychain access during pre/post-state collection
    class _NoKeychainBrowser:
        def __init__(self):
            self.backend = object()

        async def get_current_url(self) -> str:
            return ""

        async def get_screenshot(self) -> str:
            return ""

    real_actor.action_provider._browser = _NoKeychainBrowser()

    # 2. --- Mock Actor's LLM Dependencies ---

    # Mock initial plan generation
    CANNED_PLAN = textwrap.dedent(
        """
        from pydantic import BaseModel, Field

        async def navigate_to_site():
            '''Navigates to the site.'''
            await action_provider.navigate("https://example.com")

        async def observe_heading():
            '''Finds the main heading. This is a stub.'''
            raise NotImplementedError("I need to see the page layout first.")

        async def main_plan():
            await navigate_to_site()
            result = await observe_heading()
            return result
        """,
    )

    async def _fake_generate_initial_plan(self, plan, goal):
        return self._sanitize_code(CANNED_PLAN, plan)

    monkeypatch.setattr(
        HierarchicalActor,
        "_generate_initial_plan",
        _fake_generate_initial_plan,
        raising=True,
    )

    jit_decision = ImplementationDecision(
        action="implement_function",
        reason="Implementing stub for test.",
        code=textwrap.dedent(
            """
            async def observe_heading():
                '''Finds the main heading. (Implemented by JIT)'''
                print("EXEC: Running JIT-implemented observe_heading")
                return await action_provider.observe("get main heading")
            """,
        ),
    )
    real_actor.implementation_client = MagicMock()
    real_actor.implementation_client.generate = AsyncMock(
        return_value=jit_decision.model_dump_json(),
    )

    ok_assessment = VerificationAssessment(status="ok", reason="Mock OK")
    real_actor.verification_client = MagicMock()
    real_actor.verification_client.generate = AsyncMock(
        return_value=ok_assessment.model_dump_json(),
    )

    interject_decision = InterjectionDecision(
        action="modify_task",
        reason="Adding a final submit step per user request.",
        patches=[
            FunctionPatch(
                function_name="main_plan",
                new_code=textwrap.dedent(
                    """
                    async def main_plan():
                        await navigate_to_site()
                        result = await observe_heading()
                        print("EXEC: Running new interjected step")
                        await action_provider.act("Click Submit Button")
                        return result
                    """,
                ),
            ),
        ],
        cache=None,
    )
    real_actor.modification_client = MagicMock()
    real_actor.modification_client.generate = AsyncMock(
        return_value=interject_decision.model_dump_json(),
    )

    # 3. --- Instantiate Conductor with the Real Actor ---
    cond = SimulatedConductor(actor=real_actor)

    handle = await cond.request(
        "Open a browser window so we can walk through the setup together.",
        _return_reasoning_steps=True,
    )

    async def _wait_for_plan_handle(actor: HierarchicalActor, timeout: float = 60.0):
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if actor._plan_handles:
                return list(actor._plan_handles)[0]
            await asyncio.sleep(0.05)
        raise AssertionError(
            "HierarchicalActor did not create a plan handle within timeout.",
        )

    plan_handle = await _wait_for_plan_handle(real_actor, timeout=120)

    plan_handle.implementation_client = MagicMock()
    plan_handle.implementation_client.set_response_format = MagicMock()
    plan_handle.implementation_client.reset_response_format = MagicMock()
    plan_handle.implementation_client.reset_messages = MagicMock()
    plan_handle.implementation_client.set_system_message = MagicMock()
    plan_handle.implementation_client.generate = AsyncMock(
        return_value=jit_decision.model_dump_json(),
    )

    ok_assessment = VerificationAssessment(status="ok", reason="Mock OK")
    plan_handle.verification_client = MagicMock()
    plan_handle.verification_client.set_response_format = MagicMock()
    plan_handle.verification_client.reset_response_format = MagicMock()
    plan_handle.verification_client.reset_messages = MagicMock()
    plan_handle.verification_client.set_system_message = MagicMock()
    plan_handle.verification_client.generate = AsyncMock(
        return_value=ok_assessment.model_dump_json(),
    )

    plan_handle.modification_client = MagicMock()
    plan_handle.modification_client.set_response_format = MagicMock()
    plan_handle.modification_client.reset_response_format = MagicMock()
    plan_handle.modification_client.reset_messages = MagicMock()
    plan_handle.modification_client.set_system_message = MagicMock()
    plan_handle.modification_client.generate = AsyncMock(
        return_value=interject_decision.model_dump_json(),
    )

    await asyncio.wait_for(
        wait_for_state(plan_handle, _HierarchicalPlanState.PAUSED_FOR_INTERJECTION),
        timeout=60,
    )

    await handle.interject("Great, now click the 'Submit' button.")

    await asyncio.wait_for(
        wait_for_state(plan_handle, _HierarchicalPlanState.RUNNING),
        timeout=180,
    )

    await asyncio.wait_for(
        wait_for_state(plan_handle, _HierarchicalPlanState.PAUSED_FOR_INTERJECTION),
        timeout=60,
    )

    handle.stop("Test complete.")
    final_result, messages = await handle.result()

    actor_log = "\n".join(plan_handle.action_log)

    assert (
        "not implemented. Implementing JIT" in actor_log
    ), "Actor did not trigger JIT implementation."
    assert (
        "Updating implementation of 'observe_heading'" in actor_log
    ), "Actor did not update implementation for the stubbed function."

    assert (
        "Interjection Decision: modify_task" in actor_log
    ), "Actor did not process the interjection."
    assert (
        "Click Submit Button" in actor_log
    ), "Actor did not execute the new code from the interjection."

    assert (
        actor_log.count("CACHE HIT: Using cached result") >= 1
    ), "Actor did not use the cache during replay after interjection."

    assert (
        tool_names_from_messages(messages, "Actor").count("Actor_act") == 1
    ), "Conductor should have logged exactly one call to Actor_act."

    assert any(
        getattr(call, "args", []) and call.args[0] == "https://example.com"
        for call in real_actor.action_provider.navigate.call_args_list
    ), "navigate was not called with the expected URL"
    assert any(
        getattr(call, "args", []) and call.args[0] == "get main heading"
        for call in real_actor.action_provider.observe.call_args_list
    ), "observe was not called with the expected prompt"
    assert any(
        getattr(call, "args", []) and call.args[0] == "Click Submit Button"
        for call in real_actor.action_provider.act.call_args_list
    ), "act was not called with the expected instruction"

    assert (
        "Main plan execution concluded with result: Mocked Page Heading" in actor_log
    ), "Final result did not propagate to plan logs."

    if getattr(plan_handle, "_execution_task", None):
        plan_handle._execution_task.cancel()
        try:
            await plan_handle._execution_task
        except asyncio.CancelledError:
            pass
