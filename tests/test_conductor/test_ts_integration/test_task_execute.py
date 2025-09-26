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


TASK_LIKE_REQUESTS: list[str] = [
    "Start the task to call Alice about the Q3 budget now.",
    "Execute the task named 'Draft Budget FY26' immediately.",
    "Run the task: Email Contoso about invoices today at 4pm.",
    (
        "Begin execution of the 'Prepare slides for kickoff' task and confirm once started."
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("request_text", TASK_LIKE_REQUESTS)
@_handle_project
async def test_task_like_requests_use_taskscheduler_execute_not_actor(
    request_text: str,
    monkeypatch,
):
    # Ensure the underlying SimulatedActor completes immediately during execute
    import unity.actor.simulated as _actor_sim

    _orig_sim_actor = _actor_sim.SimulatedActor

    @functools.wraps(_orig_sim_actor)
    def _patched_sim_actor(*args, **kwargs):  # type: ignore
        kw = dict(kwargs)
        kw.setdefault("steps", 0)
        kw.setdefault("duration", None)
        return _orig_sim_actor(*args, **kw)

    monkeypatch.setattr(_actor_sim, "SimulatedActor", _patched_sim_actor, raising=True)
    cond = SimulatedConductor(
        description=(
            "Assistant that executes clearly-defined tasks when asked to start or run a task."
        ),
    )

    handle = await cond.request(
        request_text,
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # TaskScheduler.execute should be invoked at least once
    executed_ts_list = tool_names_from_messages(messages, "SimulatedTaskScheduler")
    executed_ts = set(executed_ts_list)
    assert executed_ts, "Expected at least one tool call"
    assert (
        executed_ts_list.count("SimulatedTaskScheduler_execute") >= 1
    ), f"Expected SimulatedTaskScheduler_execute to run at least once, saw order: {executed_ts_list}"

    # Actor.act must NOT be called for task-execution requests
    executed_actor_list = tool_names_from_messages(messages, "SimulatedActor")
    executed_actor = set(executed_actor_list)
    assert (
        "SimulatedActor_act" not in executed_actor
    ), f"Actor.act must not run for execute scenarios, saw: {sorted(executed_actor)}"

    # If the assistant explicitly requested tools, they should reference execute here
    requested_ts = set(
        assistant_requested_tool_names(messages, "SimulatedTaskScheduler"),
    )
    if requested_ts:
        assert requested_ts <= {
            "SimulatedTaskScheduler_execute",
        }, f"Assistant should only request SimulatedTaskScheduler_execute here, saw: {sorted(requested_ts)}"


@pytest.mark.asyncio
@_handle_project
async def test_both_executors_hidden_while_taskscheduler_execute_running(monkeypatch):
    """
    Verify that `Actor.act` is not exposed/selected while `TaskScheduler.execute` is in-flight.

    We configure the `SimulatedActor` (used under the hood by the simulated TaskScheduler)
    to complete deterministically on a single interjection. We then start a task-execution
    request, wait for the scheduler to be scheduled, interject once, and assert that prior to
    completion the assistant did not request `SimulatedActor_act`.
    """

    # Ensure the underlying SimulatedActor used by TaskScheduler completes after a single step
    import unity.actor.simulated as _actor_sim

    _orig_sim_actor = _actor_sim.SimulatedActor

    @functools.wraps(_orig_sim_actor)
    def _patched_sim_actor(*args, **kwargs):  # type: ignore
        kw = dict(kwargs)
        kw.setdefault("steps", 1)
        kw.setdefault("duration", None)
        return _orig_sim_actor(*args, **kw)

    monkeypatch.setattr(_actor_sim, "SimulatedActor", _patched_sim_actor, raising=True)

    # Trigger once SimulatedTaskScheduler.execute is actually called so we interject at the right time
    import unity.task_scheduler.simulated as ts_sim

    tool_started_evt = asyncio.Event()

    _orig_execute = ts_sim.SimulatedTaskScheduler.execute

    @functools.wraps(_orig_execute)
    async def _wrapped_execute(self, *a, **kw):
        h = await _orig_execute(self, *a, **kw)
        tool_started_evt.set()
        return h

    monkeypatch.setattr(
        ts_sim.SimulatedTaskScheduler,
        "execute",
        _wrapped_execute,
        raising=True,
    )

    cond = SimulatedConductor(
        description=(
            "Assistant that executes clearly-defined tasks; Actor should not be exposed while a task is running."
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

    # Start a task-like execution request
    handle = await cond.request(
        "Let's run the 'Prepare slides for kickoff' task.",
        _return_reasoning_steps=True,
    )

    # Wait until the TaskScheduler tool is actually scheduled, then interject immediately
    await asyncio.wait_for(tool_started_evt.wait(), timeout=60)
    await handle.interject("Make sure to use a green colour scheme for the slides.")

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # Locate the assistant turn that handled the interjection by calling the dynamic helper
    interject_asst_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function", {}) or {}
                name = fn.get("name") or ""
                if name.startswith("interject_SimulatedTaskScheduler_execute"):
                    interject_asst_idx = i
                    break
        if interject_asst_idx is not None:
            break

    # Require that the interjection was processed via the dynamic helper on the same turn
    assert (
        interject_asst_idx is not None
    ), "Expected the assistant to call interject_SimulatedTaskScheduler_execute when processing the interjection"

    asst = messages[interject_asst_idx]
    requested_actor = set(assistant_requested_tool_names([asst], "SimulatedActor"))
    assert (
        "SimulatedActor_act" not in requested_actor
    ), "Actor.act should not be available on the same assistant turn that processes the interjection"

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
        "SimulatedActor_act" not in exposed_on_interject
    ), f"Tool exposure should hide SimulatedActor_act on interjection turn; exposed: {sorted(exposed_on_interject)}"
    assert (
        "SimulatedTaskScheduler_execute" not in exposed_on_interject
    ), f"Tool exposure should hide SimulatedTaskScheduler_execute itself on interjection turn; exposed: {sorted(exposed_on_interject)}"

    # Additionally ensure that while the Task execution is in-flight, the assistant never
    # requests Actor.act on any assistant turn between the initial TaskScheduler.execute
    # scheduling and the interjection handling (hidden while live task is pending).
    ts_start_asst_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function", {}) or {}
                name = fn.get("name") or ""
                if name == "SimulatedTaskScheduler_execute":
                    ts_start_asst_idx = i
                    break
        if ts_start_asst_idx is not None:
            break

    assert (
        ts_start_asst_idx is not None
    ), "Could not locate the assistant turn that scheduled SimulatedTaskScheduler_execute"

    ts_start_turn_no = _assistant_turn_no(ts_start_asst_idx)

    violations: list[int] = []
    for i in range(ts_start_asst_idx, interject_asst_idx + 1):
        m = messages[i]
        if m.get("role") != "assistant":
            continue
        req = set(assistant_requested_tool_names([m], "SimulatedActor"))
        if "SimulatedActor_act" in req:
            violations.append(i)

    assert (
        not violations
    ), f"Actor.act must be hidden while TaskScheduler.execute is in-flight; found requests on assistant turn(s): {violations}"

    # Additionally ensure the tool EXPOSURE hides both executors on every assistant turn during the in-flight window
    exposure_violations_other: list[int] = []
    exposure_violations_self: list[int] = []
    for t in range(ts_start_turn_no, interject_turn_no + 1):
        exposed = set(
            exposed_tools_per_turn[t] if t < len(exposed_tools_per_turn) else [],
        )
        if "SimulatedActor_act" in exposed:
            exposure_violations_other.append(t)
        if "SimulatedTaskScheduler_execute" in exposed:
            exposure_violations_self.append(t)

    assert (
        not exposure_violations_other
    ), f"Tool exposure must hide SimulatedActor_act while TaskScheduler.execute is in-flight; offending assistant turn(s): {exposure_violations_other}"
    assert (
        not exposure_violations_self
    ), f"Tool exposure must hide SimulatedTaskScheduler_execute itself while in-flight; offending assistant turn(s): {exposure_violations_self}"
