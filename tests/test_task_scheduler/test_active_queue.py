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
    qid = ts._allocate_new_queue_id()
    for n in names:
        ids.append(
            ts._create_task(
                name=n,
                description=n,
                schedule={"queue_id": qid},
            )[
                "details"
            ]["task_id"],
        )  # type: ignore[index]
    ts._set_queue(queue_id=qid, order=ids)
    ts._update_task_start_at(task_id=ids[0], new_start_at=datetime.now(timezone.utc))
    return ids


@pytest.mark.asyncio
@_handle_project
async def test_active_queue_passthrough_then_switch_to_multitask(monkeypatch):
    """Start with a singleton queue (passthrough), then append a follower and
    verify the handle switches to multi-task behaviour (CHAIN preamble in ask).

    Steps
    -----
    1) Create a single queued task and execute it with explicit 'queue' scope so
       the ActiveQueue wrapper is used even for singletons.
    2) Ask a question – inner handle should receive the raw question (no chain context).
    3) Append a new follower behind the active task using `_update_task_queue`.
    4) Ask again – the inner handle should now receive a CHAIN-context preamble.
    """

    # Spy: capture the exact question strings delivered to the inner handle
    captured: list[str] = []

    original_ask = SimulatedActorHandle.ask

    @functools.wraps(original_ask)
    async def spy_ask(self, question: str, *a, **kw):  # type: ignore[override]
        # Record the full question passed to the inner handle
        captured.append(question)
        return await original_ask(self, question, *a, **kw)

    monkeypatch.setattr(SimulatedActorHandle, "ask", spy_ask, raising=True)

    # Use a long-running simulated actor so the task does not auto-complete too soon
    actor = SimulatedActor(steps=50)
    ts = TaskScheduler(actor=actor)

    # Create a single task and start it (queue semantics by default)
    name1 = "Singleton A"
    qid = ts._allocate_new_queue_id()
    tid1 = ts._create_task(name=name1, description=name1, schedule={"queue_id": qid})[
        "details"
    ]["task_id"]
    handle = await ts.execute(text=str(tid1))

    # 1) Passthrough path: queue length == 1 → inner sees raw question
    await handle.ask("Q1: status?")
    # Give the background actor a moment to process
    await asyncio.sleep(0.05)
    assert captured, "Inner ask should have been invoked"
    assert "CHAIN CONTEXT" not in captured[-1]

    # 2) Append a follower behind the active task – this grows the queue to >1
    name2 = "Follower B"
    tid2 = ts._create_task(name=name2, description=name2, schedule={"queue_id": qid})[
        "details"
    ]["task_id"]

    # Establish explicit order: [tid1, tid2]
    ts._set_queue(queue_id=qid, order=[tid1, tid2])

    # 3) Multi-task path: queue length > 1 → passthrough disabled, CHAIN preamble expected
    await handle.ask("Q2: what remains?")
    await asyncio.sleep(0.05)
    assert len(captured) >= 2
    assert (
        "CHAIN CONTEXT" in captured[-1]
    ), "Multi-task ask should include chain context preamble"

    # Cleanup: stop the active task to complete quickly
    handle.stop(cancel=False)
    await handle.result()


@pytest.mark.asyncio
@_handle_project
async def test_execute_queue_by_numeric_id_forwards_and_runs_followers(monkeypatch):
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

    # numeric fast path → queue handle adopted by default
    h = await ts.execute(text=str(a))
    await h.result()

    rows_a = ts._filter_tasks(filter=f"task_id == {a}")
    rows_b = ts._filter_tasks(filter=f"task_id == {b}")
    rows_c = ts._filter_tasks(filter=f"task_id == {c}")
    # After full queue, we expect all instances to be completed or terminal
    assert any(r.get("status") in ("completed", "cancelled", "failed") for r in rows_a)
    assert any(r.get("status") in ("completed", "cancelled", "failed") for r in rows_b)
    assert any(r.get("status") in ("completed", "cancelled", "failed") for r in rows_c)


@pytest.mark.asyncio
@_handle_project
async def test_execute_queue_then_defer_on_second_stops_queue_and_reinstate(
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

    # Start queue from A
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
async def test_execute_queue_by_numeric_id_completes_all(monkeypatch):
    """Numeric id path: starting at head should run through all followers to completion."""

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

    # Ensure the queue order, then start by id
    # queue already materialised by helper
    h = await ts.execute(text=str(x))
    await h.result()

    rows_x = ts._filter_tasks(filter=f"task_id == {x}")
    rows_y = ts._filter_tasks(filter=f"task_id == {y}")
    assert any(r.get("status") in ("completed", "cancelled", "failed") for r in rows_x)
    assert any(r.get("status") in ("completed", "cancelled", "failed") for r in rows_y)


@pytest.mark.asyncio
@_handle_project
async def test_queue_pause_resume_and_completion(monkeypatch):
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
async def test_queue_interject_routing_multi_task(monkeypatch):
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

    # Make routing deterministic by faking the router LLM used inside ActiveQueue
    class _FakeRouterClient:
        def __init__(self, *a, **kw):
            pass

        def set_system_message(self, _sys):
            return None

        async def generate(self, user: str) -> str:
            import json as _json

            try:
                marker = "Chain (head→tail):\n"
                start = user.index(marker) + len(marker)
                end = user.index("\nMetadata:\n", start)
                rows = _json.loads(user[start:end])
            except Exception:
                rows = []
            tid_by_desc = {
                str(r.get("description")): int(r.get("task_id"))
                for r in rows
                if r.get("task_id") is not None
            }
            a_tid = tid_by_desc.get("A_r")
            b_tid = tid_by_desc.get("B_r")
            c_tid = tid_by_desc.get("C_r")
            payload = {
                "routes": [
                    {
                        "task_ids": [t for t in [a_tid, b_tid, c_tid] if t is not None],
                        "instruction": "GLOBAL_OK",
                    },
                    {
                        "task_ids": [b_tid] if b_tid is not None else [],
                        "instruction": "SAFE_FOR_B",
                    },
                    {
                        "task_ids": [c_tid] if c_tid is not None else [],
                        "instruction": "SAFE_FOR_LAST",
                    },
                    {
                        "task_ids": [a_tid] if a_tid is not None else [],
                        "instruction": "SAFE_FOR_FIRST",
                    },
                ],
                "uncovered_directives": [],
            }
            return _json.dumps(payload)

    monkeypatch.setattr("unify.AsyncUnify", _FakeRouterClient, raising=True)

    # Start queue at A and issue one multi-task interjection
    h = await ts.execute(text=str(a_id))
    await h.interject(
        "Please route the following interjections strictly as described. "
        "These are NOT lifecycle controls and must NOT be treated as stop/cancel/defer: "
        "- For ALL tasks, apply instruction: GLOBAL_OK. "
        f"- For the task whose description is '{'B_r'}', apply instruction: SAFE_FOR_B. "
        "- For the LAST task in the queue, apply instruction: SAFE_FOR_LAST. "
        "- For the FIRST task in the queue, apply instruction: SAFE_FOR_FIRST.",
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
async def test_queue_handle_ask_includes_queue_context(monkeypatch):
    """
    Verify that _ChainHandle.ask prepends a queue-wide context preamble so
    questions can be answered about the whole queue, not just the active task.
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

    ask_handle = await h.ask("How is the queue going?")
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
    assert "How is the queue going?" in q


@pytest.mark.asyncio
@_handle_project
async def test_queue_result_summarises_all_completed_tasks(monkeypatch):
    """
    Verify that the queue handle's final result summarises all completed tasks.
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

    h = await ts.execute(text=str(a))
    res = await h.result()

    assert isinstance(res, str)
    assert "Completed the following tasks:" in res
    assert f"Task {a}: A_sum" in res
    assert f"Task {b}: B_sum" in res
    assert f"Task {c}: C_sum" in res


@pytest.mark.asyncio
@_handle_project
async def test_queue_dynamic_queue_edit_add_and_remove_followers(monkeypatch):
    """
    While a queue is running, dynamically remove an existing follower and add a new
    follower behind the current task. The queue should reflect the live queue at the
    next hop: skip the removed task and execute the newly added one.
    """

    # Make each task require exactly two interjections to complete
    class _Step2(SimulatedActor):  # type: ignore[misc]
        def __init__(self, *a, **kw):
            kw["steps"] = 2
            kw["duration"] = None
            super().__init__(*a, **kw)

    # Use step-based simulated actor everywhere
    monkeypatch.setattr("unity.actor.simulated.SimulatedActor", _Step2, raising=True)
    monkeypatch.setattr(
        "unity.task_scheduler.task_scheduler.SimulatedActor",
        _Step2,
        raising=True,
    )

    ts = TaskScheduler()
    a_id, b_id, c_id = await _make_ordered_queue(ts, ["A_dyn", "B_dyn", "C_dyn"])  # type: ignore[misc]

    # Deterministic activation triggers per task-id
    activation_events: Dict[int, asyncio.Event] = {}
    completion_events: Dict[int, asyncio.Event] = {}

    def _evt_for(tid: int) -> asyncio.Event:
        ev = activation_events.get(tid)
        if ev is None:
            ev = asyncio.Event()
            activation_events[tid] = ev
        return ev

    def _completed_evt_for(tid: int) -> asyncio.Event:
        ev = completion_events.get(tid)
        if ev is None:
            ev = asyncio.Event()
            completion_events[tid] = ev
        return ev

    orig_update_status = ts._update_task_status_instance

    def spy_update_status(*, task_id: int, instance_id: int, new_status: str, activated_by=None):  # type: ignore[override]
        res = orig_update_status(
            task_id=task_id,
            instance_id=instance_id,
            new_status=new_status,
            activated_by=activated_by,
        )
        try:
            if str(new_status) == "active":
                _evt_for(task_id).set()
            if str(new_status) == "completed":
                _completed_evt_for(task_id).set()
        except Exception:
            pass
        return res

    monkeypatch.setattr(
        ts,
        "_update_task_status_instance",
        spy_update_status,
        raising=True,
    )

    # Start queue at A
    h = await ts.execute(text=str(a_id))

    # Complete A deterministically with pause/resume (each consumes a step)
    h.pause()
    h.resume()
    await asyncio.wait_for(_completed_evt_for(a_id).wait(), timeout=10)

    # Wait until B is active
    await asyncio.wait_for(_evt_for(b_id).wait(), timeout=10)

    # Dynamically add a new follower D after B and remove C from the queue
    d_id = ts._create_task(
        name="D_dyn",
        description="D_dyn",
        schedule={"prev_task": b_id},
    )["details"][
        "task_id"
    ]  # type: ignore[index]
    ts._delete_task(task_id=c_id)

    # Complete B deterministically
    h.pause()
    h.resume()
    await asyncio.wait_for(_completed_evt_for(b_id).wait(), timeout=10)

    # Wait until D is active before applying steps to the new current handle
    await asyncio.wait_for(_evt_for(d_id).wait(), timeout=10)

    # D should be picked up next and complete after two steps
    h.pause()
    h.resume()
    await asyncio.wait_for(_completed_evt_for(d_id).wait(), timeout=10)

    res = await h.result()
    assert isinstance(res, str)

    # A, B, D should be terminal; C should remain non-terminal
    rows_a = ts._filter_tasks(filter=f"task_id == {a_id}")
    rows_b = ts._filter_tasks(filter=f"task_id == {b_id}")
    rows_c = ts._filter_tasks(filter=f"task_id == {c_id}")
    rows_d = ts._filter_tasks(filter=f"task_id == {d_id}")

    def _is_terminal(row):
        return row.get("status") in ("completed", "cancelled", "failed")

    assert any(_is_terminal(r) for r in rows_a)
    assert any(_is_terminal(r) for r in rows_b)
    assert any(_is_terminal(r) for r in rows_d)
    # C was removed from the queue before activation; ensure it is not terminal/active
    assert all(
        r.get("status") not in ("completed", "cancelled", "failed", "active")
        for r in rows_c
    )


@pytest.mark.asyncio
@_handle_project
async def test_active_task_done_aggregates_all_when_called_late(monkeypatch):
    """
    If called after multiple tasks completed, active_task_done should return
    a JSON mapping containing all completions since never having been called.
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
    a, b, c = await _make_ordered_queue(ts, ["A_done", "B_done", "C_done"])  # type: ignore[misc]

    # Run the entire queue to completion
    h = await ts.execute(text=str(a))
    await h.result()

    # Access inner handle if wrapped
    inner = getattr(h, "_inner", h)

    # Call active_task_done the first time – should aggregate all completions
    import json as _json

    payload_str = await inner.active_task_done()
    data = _json.loads(payload_str or "{}")
    assert isinstance(data, dict)
    assert set(data.keys()) == {"A_done", "B_done", "C_done"}
    assert all(isinstance(v, str) for v in data.values())

    # Second call after everything already consumed should be empty
    payload_str2 = await inner.active_task_done()
    data2 = _json.loads(payload_str2 or "{}")
    assert data2 == {}


@pytest.mark.asyncio
@_handle_project
async def test_active_task_done_incremental(monkeypatch):
    """
    Consecutive calls to active_task_done should return only new completions
    since the previous call.
    """

    # Step-based actor: one step to complete each task
    class _StepOne(SimulatedActor):  # type: ignore[misc]
        def __init__(self, *a, **kw):
            kw["steps"] = 1
            kw["duration"] = None
            super().__init__(*a, **kw)

    monkeypatch.setattr("unity.actor.simulated.SimulatedActor", _StepOne, raising=True)
    monkeypatch.setattr(
        "unity.task_scheduler.task_scheduler.SimulatedActor",
        _StepOne,
        raising=True,
    )

    ts = TaskScheduler()
    a_id, b_id = await _make_ordered_queue(ts, ["A_inc", "B_inc"])  # type: ignore[misc]

    # Detect when B becomes active
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
            if task_id == b_id and str(new_status) == "active":
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

    h = await ts.execute(text=str(a_id))
    inner = getattr(h, "_inner", h)

    # Complete A with a single step (pause triggers a step in simulated actor)
    h.pause()

    # First call should include only A
    import json as _json

    payload1 = await inner.active_task_done()
    data1 = _json.loads(payload1 or "{}")
    assert set(data1.keys()) == {"A_inc"}

    # Ensure B is active, then complete it
    await asyncio.wait_for(b_active_evt.wait(), timeout=5)
    h.pause()

    # Second call should include only B
    payload2 = await inner.active_task_done()
    data2 = _json.loads(payload2 or "{}")
    assert set(data2.keys()) == {"B_inc"}

    # Further calls after consumption should be empty
    payload3 = await inner.active_task_done()
    data3 = _json.loads(payload3 or "{}")
    assert data3 == {}


@pytest.mark.asyncio
@_handle_project
async def test_execute_by_id_returns_active_queue_handle(monkeypatch):
    """Executing by id returns an ActiveQueue handle (composite queue wrapper)."""

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

    # Create a single runnable task
    task_id = ts._create_task(name="ISO", description="ISO")["details"]["task_id"]  # type: ignore[index]

    # Execute by numeric id; we should receive an ActiveQueue handle
    h = await ts.execute(text=str(task_id))

    # Import locally and tolerate logging wrapper by unwrapping the inner handle
    from unity.task_scheduler.active_queue import ActiveQueue

    inner = getattr(h, "_inner", h)
    assert isinstance(inner, ActiveQueue), "execute(isolate) must return ActiveQueue"

    # Complete and verify non-summary final text (singleton queue passthrough)
    res = await h.result()
    assert isinstance(res, str)
    assert "Completed the following tasks:" not in res


@pytest.mark.asyncio
@_handle_project
async def test_singleton_queue_passthrough_to_inner_handle(monkeypatch):
    """
    For a true singleton queue (exactly one task at creation), the queue handle
    should pass through interject, ask, and result directly to the inner task
    handle. Prior to this change, ActiveQueue always applied multi-task
    behaviour, which would have added a CHAIN preamble to ask() and returned a
    multi-task summary from result().
    """

    # Make the actor step-based so ask + interject complete the task
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
    # Build a queue with exactly one task
    (solo_id,) = tuple(await _make_ordered_queue(ts, ["Solo"]))  # type: ignore[misc]

    # Spy: capture question text and interjection count; ensure ask still consumes a step
    captured_questions: list[str] = []
    interject_calls = {"count": 0}

    async def spy_actor_ask(self, question: str):  # type: ignore[override]
        captured_questions.append(question)
        try:
            self.simulate_step()
        except Exception:
            pass
        return "OK"

    async def spy_actor_interject(self, instruction: str):  # type: ignore[override]
        interject_calls["count"] += 1
        try:
            self.simulate_step()
        except Exception:
            pass
        return None

    monkeypatch.setattr(SimulatedActorHandle, "ask", spy_actor_ask, raising=True)
    monkeypatch.setattr(
        SimulatedActorHandle,
        "interject",
        spy_actor_interject,
        raising=True,
    )

    # Start execution of the singleton queue
    h = await ts.execute(text=str(solo_id))

    # Wait until a task is active to avoid races
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

    # Pass-through ask: should be the raw user question (no CHAIN preamble)
    ask_handle = await h.ask("What are you doing?")
    res = await ask_handle.result()
    assert res == "OK"
    assert captured_questions, "expected inner ask to be invoked"
    assert captured_questions[-1] == "What are you doing?"
    assert "CHAIN CONTEXT" not in captured_questions[-1]

    # Pass-through interject increments inner count
    await h.interject("Proceed")
    assert interject_calls["count"] == 1

    # ask + interject complete the two steps; result should NOT be a multi-task summary
    final_res = await h.result()
    assert isinstance(final_res, str)
    assert "Completed the following tasks:" not in final_res
    assert "Solo" in final_res


@pytest.mark.asyncio
@_handle_project
async def test_inner_task_clarification_bubbles_up_to_outer(monkeypatch):
    """
    Verify that an inner task can request clarification and that the question
    is emitted to the provided clarification_up_q, with the answer received on
    clarification_down_q completing the task. Result should pass through for a
    singleton queue.
    """

    class _Clar(SimulatedActor):  # type: ignore[misc]
        def __init__(self, *a, **kw):
            # No step/duration completion; rely solely on clarification to finish
            kw["steps"] = None
            kw["duration"] = None
            kw["_requests_clarification"] = True
            super().__init__(*a, **kw)

    # Use clarification-seeking actor everywhere
    monkeypatch.setattr("unity.actor.simulated.SimulatedActor", _Clar, raising=True)
    monkeypatch.setattr(
        "unity.task_scheduler.task_scheduler.SimulatedActor",
        _Clar,
        raising=True,
    )

    ts = TaskScheduler()
    # Create a single runnable task
    task_id = ts._create_task(name="NeedClar", description="NeedClar")["details"][
        "task_id"
    ]  # type: ignore[index]

    # Clarification channels for the test harness
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    # Execute by id, ensuring queues are wired through
    h = await ts.execute(
        text=str(task_id),
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    # Expect the inner task to ask a clarification question immediately
    question = await asyncio.wait_for(up_q.get(), timeout=5)
    assert isinstance(question, str) and question, "expected a clarification question"

    # Provide an answer and expect completion that reflects the answer (passthrough)
    await down_q.put("YES_PROCEED")
    res = await asyncio.wait_for(h.result(), timeout=5)
    assert "Clarification received: YES_PROCEED" in (res or "")


@pytest.mark.asyncio
@_handle_project
async def test_active_queue_requests_clarification_at_queue_level(monkeypatch):
    """
    Verify that ActiveQueue itself can request clarifications for ambiguous
    multi-task interjections. The question should surface on clarification_up_q.
    """

    # Step-based actor: pause/resume pair completes one task deterministically
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
    a_id, b_id = await _make_ordered_queue(ts, ["QA1", "QA2"])  # type: ignore[misc]

    # Clarification channels for ActiveQueue
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    # Start execution at first task with queues supplied
    h = await ts.execute(
        text=str(a_id),
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    # Wait until a task is active to avoid races
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

    # Send an explicitly ambiguous interjection that should trigger a queue-level clarification
    # The text intentionally avoids concrete task_ids or clear directives
    await h.interject(
        "We should probably adjust things: maybe do the rest later, or whichever seems best.",
    )

    clar_q = await asyncio.wait_for(up_q.get(), timeout=5)
    # Do not assert specific phrasing; just verify a clarification question surfaced
    assert (
        isinstance(clar_q, str) and clar_q.strip()
    ), f"no clarification question received: {clar_q!r}"

    # Provide an answer to unblock routing
    await down_q.put("Apply to last only")

    # Complete A deterministically with pause/resume (each consumes a step)
    h.pause()
    h.resume()

    # Ensure B becomes active using an explicit event; then complete B
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
            if task_id == b_id and str(new_status) == "active":
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

    await asyncio.wait_for(b_active_evt.wait(), timeout=20)
    h.pause()
    h.resume()

    # Expect final completion (summary or inner result depending on chain state)
    res = await asyncio.wait_for(h.result(), timeout=30)
    assert isinstance(res, str)
