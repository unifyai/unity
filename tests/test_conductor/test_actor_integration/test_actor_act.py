from __future__ import annotations

import asyncio
import functools
import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


SANDBOX_REQUESTS: list[str] = [
    "Open a browser window so we can walk through the setup together.",
    "Can you open the Settings app? I want to show you something.",
    (
        "Let's start a quick sandbox session: open the browser and navigate to the "
        "dashboard; I'll guide you live."
    ),
    "Open Notes so we can jot down ideas as we talk.",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text", SANDBOX_REQUESTS)
@_handle_project
async def test_actor_sandbox_requests_use_actor_not_task_execute(request_text: str):
    cond = SimulatedConductor(
        description=(
            "Assistant available to act directly in a sandbox; tasks are not required for these interactions."
        ),
    )

    handle = await cond.request(
        request_text,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Actor should be invoked at least once
    executed_actor_list = tool_names_from_messages(messages, "SimulatedActor")
    executed_actor = set(executed_actor_list)
    assert executed_actor, "Expected at least one tool call"
    assert (
        executed_actor_list.count("SimulatedActor_act") >= 1
    ), f"Expected SimulatedActor_act to run at least once, saw order: {executed_actor_list}"

    # TaskScheduler.execute must NOT be called for sandbox-style requests
    executed_ts_list = tool_names_from_messages(messages, "SimulatedTaskScheduler")
    executed_ts = set(executed_ts_list)
    assert (
        "SimulatedTaskScheduler_execute" not in executed_ts
    ), f"TaskScheduler.execute must not run for sandbox scenarios, saw: {sorted(executed_ts)}"

    # If the assistant explicitly requested tools, they should reference Actor.act for this scenario
    requested_actor = set(assistant_requested_tool_names(messages, "SimulatedActor"))
    if requested_actor:
        assert requested_actor <= {
            "SimulatedActor_act",
        }, f"Assistant should only request SimulatedActor_act here, saw: {sorted(requested_actor)}"


@pytest.mark.asyncio
@_handle_project
async def test_both_executors_hidden_while_actor_running(monkeypatch):
    """
    Verify that `TaskScheduler.execute` is not exposed/selected while `Actor.act` is in-flight.

    We configure the `SimulatedActor` to use `steps=1` so a single interjection can
    deterministically complete the session. We then start a sandbox request, wait for
    the actor to be scheduled, interject once, and assert that prior to completion the
    assistant did not request `SimulatedTaskScheduler_execute`.
    """

    # Trigger once SimulatedActor.act is actually called so we interject at the right time
    from unity.actor import simulated as actor_sim

    tool_started_evt = asyncio.Event()

    _orig_act = actor_sim.SimulatedActor.act

    @functools.wraps(_orig_act)
    async def _wrapped_act(self, *a, **kw):
        h = await _orig_act(self, *a, **kw)
        tool_started_evt.set()
        return h

    monkeypatch.setattr(actor_sim.SimulatedActor, "act", _wrapped_act, raising=True)

    cond = SimulatedConductor(
        description=(
            "Assistant available to act directly in a sandbox; tasks should not execute while sandbox is running."
        ),
    )

    # Capture the set of tools exposed to the LLM on each assistant turn
    import unity.common._async_tool.messages as _tool_msgs

    _orig_generate = _tool_msgs.generate_with_preprocess
    exposed_tools_per_turn: list[list[str]] = []

    @functools.wraps(_orig_generate)
    async def _wrapped_generate(client, preprocess_msgs, **gen_kwargs):  # type: ignore[override]
        tool_schemas = gen_kwargs.get("tools") or []
        names: list[str] = []
        try:
            for s in tool_schemas:
                try:
                    fn = (s or {}).get("function", {})
                    nm = fn.get("name")
                    if isinstance(nm, str):
                        names.append(nm)
                except Exception:
                    pass
        finally:
            exposed_tools_per_turn.append(names)
        return await _orig_generate(client, preprocess_msgs, **gen_kwargs)

    monkeypatch.setattr(
        _tool_msgs,
        "generate_with_preprocess",
        _wrapped_generate,
        raising=True,
    )

    # Configure simulated actor for deterministic completion on single step
    # (and add a short duration as a safety-net in case the model does not forward interjection).
    cond._actor._steps = 1  # type: ignore[attr-defined]
    try:
        cond._actor._duration = None  # type: ignore[attr-defined]
    except Exception:
        pass

    # Start a sandbox-like session
    handle = await cond.request(
        "Open a browser window so we can walk through the setup together.",
        _return_reasoning_steps=True,
    )

    # Wait until the Actor tool is actually scheduled, then interject immediately
    await asyncio.wait_for(tool_started_evt.wait(), timeout=60)
    await handle.interject("Make sure to use Google Chrome.")

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Locate the assistant turn that handled the interjection by calling the dynamic helper
    interject_asst_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function", {}) or {}
                name = fn.get("name") or ""
                if name.startswith("interject_SimulatedActor_act"):
                    interject_asst_idx = i
                    break
        if interject_asst_idx is not None:
            break

    # Require that the interjection was processed via the dynamic helper on the same turn
    assert (
        interject_asst_idx is not None
    ), "Expected the assistant to call interject_SimulatedActor_act when processing the interjection"

    asst = messages[interject_asst_idx]
    requested = set(assistant_requested_tool_names([asst], "SimulatedTaskScheduler"))
    assert (
        "SimulatedTaskScheduler_execute" not in requested
    ), "TaskScheduler.execute should not be available on the same assistant turn that processes the interjection"

    # Helper: map message index → assistant turn number
    def _assistant_turn_no(msg_index: int) -> int:
        return (
            sum(1 for m in messages[: msg_index + 1] if m.get("role") == "assistant")
            - 1
        )

    interject_turn_no = _assistant_turn_no(interject_asst_idx)

    # Assert the tool list exposed to the LLM on the interjection turn does NOT include
    # either the other executor or the in-flight tool itself
    exposed_on_interject = set(
        (
            exposed_tools_per_turn[interject_turn_no]
            if interject_turn_no < len(exposed_tools_per_turn)
            else []
        ),
    )
    assert (
        "SimulatedTaskScheduler_execute" not in exposed_on_interject
    ), f"Tool exposure should hide SimulatedTaskScheduler_execute on interjection turn; exposed: {sorted(exposed_on_interject)}"
    assert (
        "SimulatedActor_act" not in exposed_on_interject
    ), f"Tool exposure should hide SimulatedActor_act itself on interjection turn; exposed: {sorted(exposed_on_interject)}"

    # Additionally ensure that while the Actor session is in-flight, the assistant never
    # requests TaskScheduler.execute on any assistant turn between the initial Actor.act
    # scheduling and the interjection handling (hidden while live session is pending).
    actor_start_asst_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function", {}) or {}
                name = fn.get("name") or ""
                if name == "SimulatedActor_act":
                    actor_start_asst_idx = i
                    break
        if actor_start_asst_idx is not None:
            break

    assert (
        actor_start_asst_idx is not None
    ), "Could not locate the assistant turn that scheduled SimulatedActor_act"

    actor_start_turn_no = _assistant_turn_no(actor_start_asst_idx)

    violations: list[int] = []
    for i in range(actor_start_asst_idx, interject_asst_idx + 1):
        m = messages[i]
        if m.get("role") != "assistant":
            continue
        req = set(assistant_requested_tool_names([m], "SimulatedTaskScheduler"))
        if "SimulatedTaskScheduler_execute" in req:
            violations.append(i)

    assert (
        not violations
    ), f"TaskScheduler.execute must be hidden while Actor.act is in-flight; found requests on assistant turn(s): {violations}"

    # Additionally ensure the tool EXPOSURE hides both executors on every assistant turn during the in-flight window
    exposure_violations_other: list[int] = []
    exposure_violations_self: list[int] = []
    for t in range(actor_start_turn_no, interject_turn_no + 1):
        exposed = set(
            exposed_tools_per_turn[t] if t < len(exposed_tools_per_turn) else [],
        )
        if "SimulatedTaskScheduler_execute" in exposed:
            exposure_violations_other.append(t)
        if "SimulatedActor_act" in exposed:
            exposure_violations_self.append(t)

    assert (
        not exposure_violations_other
    ), f"Tool exposure must hide SimulatedTaskScheduler_execute while Actor.act is in-flight; offending assistant turn(s): {exposure_violations_other}"
    assert (
        not exposure_violations_self
    ), f"Tool exposure must hide SimulatedActor_act itself while in-flight; offending assistant turn(s): {exposure_violations_self}"
