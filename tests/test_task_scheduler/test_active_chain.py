from __future__ import annotations

import asyncio
import functools
from datetime import datetime, timezone
from typing import Dict

import pytest

from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.actor.simulated import SimulatedActor, SimulatedActorHandle


async def _make_ordered_queue(ts: TaskScheduler, names: list[str]) -> list[int]:
    ids: list[int] = []
    for n in names:
        ids.append(ts._create_task(name=n, description=n)["details"]["task_id"])  # type: ignore[index]
    original = [t.task_id for t in ts._get_task_queue()]
    ts._update_task_queue(original=original, new=ids)
    ts._update_task_start_at(task_id=ids[0], new_start_at=datetime.now(timezone.utc))
    return ids


@pytest.mark.asyncio
@_handle_project
async def test_execute_chain_by_numeric_id_forwards_and_runs_followers(monkeypatch):
    # Steps-based actor: immediate completion to avoid timing races
    class _Short(SimulatedActor):  # type: ignore[misc]
        def __init__(self, *a, **kw):
            kw.pop("duration", None)
            super().__init__(steps=0, duration=None, *a, **kw)

    monkeypatch.setattr("unity.actor.simulated.SimulatedActor", _Short, raising=True)
    monkeypatch.setattr(
        "unity.task_scheduler.task_scheduler.SimulatedActor",
        _Short,
        raising=True,
    )

    ts = TaskScheduler()
    a, b, c = await _make_ordered_queue(ts, ["A", "B", "C"])  # type: ignore[misc]

    # Force chain routing
    async def force_chain(*, request_text: str, parent_chat_context=None):  # type: ignore[override]
        return "chain"

    monkeypatch.setattr(ts, "_decide_execution_scope", force_chain, raising=True)

    # numeric fast path + chain → chain handle adopted
    h = await ts.execute(text=str(a))
    await h.result()

    rows_a = ts._filter_tasks(filter=f"task_id == {a}")
    rows_b = ts._filter_tasks(filter=f"task_id == {b}")
    rows_c = ts._filter_tasks(filter=f"task_id == {c}")
    # After full chain, we expect all instances to be completed or terminal
    assert any(r.get("status") in ("completed", "cancelled", "failed") for r in rows_a)
    assert any(r.get("status") in ("completed", "cancelled", "failed") for r in rows_b)
    assert any(r.get("status") in ("completed", "cancelled", "failed") for r in rows_c)


@pytest.mark.asyncio
@_handle_project
async def test_execute_chain_then_defer_on_second_stops_chain_and_reinstate(
    monkeypatch,
):
    # Steps-based actor: each task completes after a single interject
    class _Short(SimulatedActor):  # type: ignore[misc]
        def __init__(self, *a, **kw):
            kw.pop("duration", None)
            super().__init__(steps=1, duration=None, *a, **kw)

    monkeypatch.setattr("unity.actor.simulated.SimulatedActor", _Short, raising=True)
    monkeypatch.setattr(
        "unity.task_scheduler.task_scheduler.SimulatedActor",
        _Short,
        raising=True,
    )

    ts = TaskScheduler()
    a, b, c = await _make_ordered_queue(ts, ["A", "B", "C"])  # type: ignore[misc]

    # Chain routing
    async def force_chain(*, request_text: str, parent_chat_context=None):  # type: ignore[override]
        return "chain"

    monkeypatch.setattr(ts, "_decide_execution_scope", force_chain, raising=True)

    # Deterministic steering: only defer when the message requests "next week"
    async def force_defer(message: str, parent_chat_context=None):  # type: ignore[override]
        text = (message or "").lower()
        if "next week" in text:
            return ("defer", message)
        return ("continue", message)

    monkeypatch.setattr(ts, "_classify_steering_intent", force_defer, raising=True)

    # Explicit trigger when B becomes active (no timing)
    b_active_evt: asyncio.Event = asyncio.Event()

    orig_update_status = ts._update_task_status_instance

    def spy_update_status(*, task_id: int, instance_id: int, new_status: str, activated_by=None):  # type: ignore[override]
        res = orig_update_status(
            task_id=task_id,
            instance_id=instance_id,
            new_status=new_status,
            activated_by=activated_by,
        )
        try:
            if task_id == b and str(new_status) == "active":
                b_active_evt.set()
        except Exception:
            pass
        return res

    monkeypatch.setattr(
        ts,
        "_update_task_status_instance",
        spy_update_status,
        raising=True,
    )

    # Start chain from A
    h = await ts.execute(text=str(a))

    # Deterministically complete A with one step (non-defer semantic)
    await h.interject("advance A now")

    # Wait explicitly until B is active, then defer via interject
    await asyncio.wait_for(b_active_evt.wait(), timeout=5)
    await h.interject("Let's do the remaining tasks next week as originally scheduled.")
    res = await h.result()
    assert "stopped" in (res or "").lower()

    # B should be reinstated as head with original start_at; C queued after
    row_b = ts._filter_tasks(filter=f"task_id == {b}")[0]
    sched_b = row_b.get("schedule") or {}
    assert sched_b.get("prev_task") is None
    assert row_b["status"] in ("scheduled", "queued", "primed")
    row_c = ts._filter_tasks(filter=f"task_id == {c}")[0]
    sched_c = row_c.get("schedule") or {}
    assert sched_c.get("prev_task") == b


@pytest.mark.asyncio
@_handle_project
async def test_freeform_chain_routed_by_llm(monkeypatch):
    # Steps-based actor: immediate completion per task
    class _Short(SimulatedActor):  # type: ignore[misc]
        def __init__(self, *a, **kw):
            kw.pop("duration", None)
            super().__init__(steps=0, duration=None, *a, **kw)

    monkeypatch.setattr("unity.actor.simulated.SimulatedActor", _Short, raising=True)
    monkeypatch.setattr(
        "unity.task_scheduler.task_scheduler.SimulatedActor",
        _Short,
        raising=True,
    )

    ts = TaskScheduler()
    x, y = await _make_ordered_queue(ts, ["X", "Y"])  # type: ignore[misc]

    # LLM decides chain for freeform request
    async def force_chain(*, request_text: str, parent_chat_context=None):  # type: ignore[override]
        return "chain"

    monkeypatch.setattr(ts, "_decide_execution_scope", force_chain, raising=True)

    # Provide a free-form request that should get routed to the one existing queue
    h = await ts.execute(text="run the whole sequence now")
    await h.result()

    rows_x = ts._filter_tasks(filter=f"task_id == {x}")
    rows_y = ts._filter_tasks(filter=f"task_id == {y}")
    assert any(r.get("status") in ("completed", "cancelled", "failed") for r in rows_x)
    assert any(r.get("status") in ("completed", "cancelled", "failed") for r in rows_y)


@pytest.mark.asyncio
@_handle_project
async def test_chain_pause_resume_and_completion(monkeypatch):
    """
    Make SimulatedActor step-based (no duration) to avoid races. Pause/resume
    while current task is certainly active.
    """
    calls: Dict[str, int] = {"pause": 0, "resume": 0}

    orig_pause = SimulatedActorHandle.pause
    orig_resume = SimulatedActorHandle.resume

    @functools.wraps(orig_pause)
    def spy_pause(self) -> str:  # type: ignore[override]
        calls["pause"] += 1
        return orig_pause(self)

    @functools.wraps(orig_resume)
    def spy_resume(self) -> str:  # type: ignore[override]
        calls["resume"] += 1
        return orig_resume(self)

    monkeypatch.setattr(SimulatedActorHandle, "pause", spy_pause, raising=True)
    monkeypatch.setattr(SimulatedActorHandle, "resume", spy_resume, raising=True)

    class _StepOnly(SimulatedActor):  # type: ignore[misc]
        def __init__(self, *a, **kw):
            # Force step-based completion: exactly 2 steps to finish per task
            kw["steps"] = 2
            kw["duration"] = None
            super().__init__(*a, **kw)

    monkeypatch.setattr("unity.actor.simulated.SimulatedActor", _StepOnly, raising=True)
    monkeypatch.setattr(
        "unity.task_scheduler.task_scheduler.SimulatedActor",
        _StepOnly,
        raising=True,
    )

    ts = TaskScheduler()
    a, b = await _make_ordered_queue(ts, ["A4", "B4"])  # type: ignore[misc]

    async def force_chain(*, request_text: str, parent_chat_context=None):  # type: ignore[override]
        return "chain"

    monkeypatch.setattr(ts, "_decide_execution_scope", force_chain, raising=True)

    # Spy to detect when B becomes active (before starting)
    b_active_evt: asyncio.Event = asyncio.Event()
    orig_update_status = ts._update_task_status_instance

    def spy_update_status(*, task_id: int, instance_id: int, new_status: str, activated_by=None):  # type: ignore[override]
        res = orig_update_status(
            task_id=task_id,
            instance_id=instance_id,
            new_status=new_status,
            activated_by=activated_by,
        )
        try:
            if task_id == b and str(new_status) == "active":
                b_active_evt.set()
        except Exception:
            pass
        return res

    monkeypatch.setattr(
        ts,
        "_update_task_status_instance",
        spy_update_status,
        raising=True,
    )

    h = await ts.execute(text=str(a))

    # Wait deterministically until a task becomes active
    async def _wait_until_active(max_iters: int = 500):
        for _ in range(max_iters):
            try:
                rows = ts._filter_tasks(filter="status == 'active'", limit=1)
            except Exception:
                rows = []
            if rows:
                return
            await asyncio.sleep(0)
        raise AssertionError("No active task detected in time")

    await _wait_until_active()

    # Pause immediately while active (A: step 1), then resume (A: step 2) → A completes
    h.pause()
    h.resume()

    # Wait until B is active, then perform two benign steps for B: interject + ask
    await asyncio.wait_for(b_active_evt.wait(), timeout=5)
    await h.interject("continue")
    ask_handle = await h.ask("status?")
    await ask_handle.result()

    await h.result()

    assert calls == {"pause": 1, "resume": 1}, f"unexpected pause/resume calls: {calls}"


@pytest.mark.asyncio
@_handle_project
async def test_chain_interject_routing_multi_task(monkeypatch):
    """
    Interjections can be routed by an LLM to multiple tasks:
      - current task receives its instructions immediately
      - future tasks receive queued instructions when they become active

    This test fakes the router to return explicit task_ids and spies on
    SimulatedActorHandle.interject to verify delivery order.
    """

    # Make each task require exactly two interjections to complete
    class _Step2(SimulatedActor):  # type: ignore[misc]
        def __init__(self, *a, **kw):
            kw["steps"] = 2
            kw["duration"] = None
            super().__init__(*a, **kw)

    monkeypatch.setattr("unity.actor.simulated.SimulatedActor", _Step2, raising=True)
    monkeypatch.setattr(
        "unity.task_scheduler.task_scheduler.SimulatedActor",
        _Step2,
        raising=True,
    )

    ts = TaskScheduler()
    a_id, b_id, c_id = await _make_ordered_queue(ts, ["A_r", "B_r", "C_r"])  # type: ignore[misc]

    # Force chain routing so numeric execute launches a chain
    async def force_chain(*, request_text: str, parent_chat_context=None):  # type: ignore[override]
        return "chain"

    monkeypatch.setattr(ts, "_decide_execution_scope", force_chain, raising=True)

    # Spy: record interjections delivered to each task; avoid networked LLM
    calls: list[tuple[str, str]] = []

    async def spy_interject(self, instruction: str):  # type: ignore[override]
        try:
            desc = getattr(self, "_description", None) or ""
        except Exception:
            desc = ""
        calls.append((str(desc), str(instruction)))
        try:
            self.simulate_step()
        except Exception:
            pass
        return None

    monkeypatch.setattr(SimulatedActorHandle, "interject", spy_interject, raising=True)

    # Start chain at A and issue one multi-task interjection
    h = await ts.execute(text=str(a_id))
    await h.interject(
        "Please route the following interjections strictly as described. "
        "These are NOT lifecycle controls and must NOT be treated as stop/cancel/defer: "
        "- For ALL tasks, apply instruction: GLOBAL_OK. "
        f"- For the task whose description is '{'B_r'}', apply instruction: SAFE_FOR_B. "
        "- For the LAST task in the chain, apply instruction: SAFE_FOR_LAST. "
        "- For the FIRST task in the chain, apply instruction: SAFE_FOR_FIRST.",
    )
    await h.result()

    # Resolve descriptions for assertion readability
    rows_a = ts._filter_tasks(filter=f"task_id == {a_id}")
    rows_b = ts._filter_tasks(filter=f"task_id == {b_id}")
    rows_c = ts._filter_tasks(filter=f"task_id == {c_id}")
    a_desc = rows_a[0]["description"]
    b_desc = rows_b[0]["description"]
    c_desc = rows_c[0]["description"]

    expected = [
        (a_desc, "GLOBAL_OK"),
        (a_desc, "SAFE_FOR_FIRST"),
        (b_desc, "GLOBAL_OK"),
        (b_desc, "SAFE_FOR_B"),
        (c_desc, "GLOBAL_OK"),
        (c_desc, "SAFE_FOR_LAST"),
    ]
    assert calls == expected, f"unexpected routed interjections: {calls}"


@pytest.mark.asyncio
@_handle_project
async def test_chain_handle_ask_includes_chain_context(monkeypatch):
    """
    Verify that _ChainHandle.ask prepends a chain-wide context preamble so
    questions can be answered about the whole chain, not just the active task.
    """

    # Step-based actor to avoid wall-clock races
    class _StepOnly(SimulatedActor):  # type: ignore[misc]
        def __init__(self, *a, **kw):
            kw["steps"] = 2
            kw["duration"] = None
            super().__init__(*a, **kw)

    monkeypatch.setattr("unity.actor.simulated.SimulatedActor", _StepOnly, raising=True)
    monkeypatch.setattr(
        "unity.task_scheduler.task_scheduler.SimulatedActor",
        _StepOnly,
        raising=True,
    )

    ts = TaskScheduler()
    a, b, c = await _make_ordered_queue(ts, ["A_ctx", "B_ctx", "C_ctx"])  # type: ignore[misc]

    async def force_chain(*, request_text: str, parent_chat_context=None):  # type: ignore[override]
        return "chain"

    monkeypatch.setattr(ts, "_decide_execution_scope", force_chain, raising=True)

    captured_questions: list[str] = []
    orig_actor_ask = SimulatedActorHandle.ask

    async def spy_actor_ask(self, question: str):  # type: ignore[override]
        captured_questions.append(question)
        return "OK"

    monkeypatch.setattr(SimulatedActorHandle, "ask", spy_actor_ask, raising=True)

    h = await ts.execute(text=str(a))

    # Wait deterministically until a task becomes active to ensure the scheduler state is populated
    async def _wait_until_active(max_iters: int = 500):
        for _ in range(max_iters):
            try:
                rows = ts._filter_tasks(filter="status == 'active'", limit=1)
            except Exception:
                rows = []
            if rows:
                return
            await asyncio.sleep(0)
        raise AssertionError("No active task detected in time")

    await _wait_until_active()

    ask_handle = await h.ask("How is the chain going?")
    res = await ask_handle.result()
    assert res == "OK"

    assert captured_questions, "expected SimulatedActorHandle.ask to be called"
    q = captured_questions[-1]

    # Preamble markers and structure
    assert "CHAIN CONTEXT" in q
    assert "Chain status:" in q
    assert "Chain tasks (head→tail):" in q
    # All tasks should be listed with their ids
    assert f"Task {a}" in q
    assert f"Task {b}" in q
    assert f"Task {c}" in q
    # User question should be preserved at the end
    assert "USER QUESTION:" in q
    assert "How is the chain going?" in q


@pytest.mark.asyncio
@_handle_project
async def test_chain_result_summarises_all_completed_tasks(monkeypatch):
    """
    Verify that the chain handle's final result summarises all completed tasks.
    """

    # Immediate completion per task to avoid timing races
    class _Immediate(SimulatedActor):  # type: ignore[misc]
        def __init__(self, *a, **kw):
            kw.pop("duration", None)
            super().__init__(steps=0, duration=None, *a, **kw)

    monkeypatch.setattr(
        "unity.actor.simulated.SimulatedActor",
        _Immediate,
        raising=True,
    )
    monkeypatch.setattr(
        "unity.task_scheduler.task_scheduler.SimulatedActor",
        _Immediate,
        raising=True,
    )

    ts = TaskScheduler()
    a, b, c = await _make_ordered_queue(ts, ["A_sum", "B_sum", "C_sum"])  # type: ignore[misc]

    # Force chain routing for deterministic behaviour
    async def force_chain(*, request_text: str, parent_chat_context=None):  # type: ignore[override]
        return "chain"

    monkeypatch.setattr(ts, "_decide_execution_scope", force_chain, raising=True)

    h = await ts.execute(text=str(a))
    res = await h.result()

    assert isinstance(res, str)
    assert "Completed the following tasks:" in res
    assert f"Task {a}: A_sum" in res
    assert f"Task {b}: B_sum" in res
    assert f"Task {c}: C_sum" in res
