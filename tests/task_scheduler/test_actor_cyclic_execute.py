"""
Integration test for Actor → TaskScheduler → Actor cyclic execution via delegate.

This verifies:
- When a `HierarchicalActorHandle` is executing, it sets a run-scoped task execution
  delegate (ContextVar).
- A `TaskScheduler.execute()` call performed from within that execution context routes
  task creation through the delegate (rather than directly calling the scheduler's actor).
- The returned `ActiveQueue` handle remains steerable (pause/resume/stop) end-to-end.
"""

from __future__ import annotations

import asyncio
import textwrap
from unittest.mock import AsyncMock

import pytest

from tests.helpers import _handle_project
from unity.actor.hierarchical_actor import HierarchicalActor, HierarchicalActorHandle
from unity.actor.hierarchical_actor import _HierarchicalActorDelegate
from unity.actor.simulated import SimulatedActor
from unity.common.task_execution_context import current_task_execution_delegate
from unity.function_manager.function_manager import FunctionManager
from unity.manager_registry import ManagerRegistry
from unity.task_scheduler.task_scheduler import TaskScheduler


class _SimpleMockVerificationClient:
    """Mock verification client that always returns success."""

    def __init__(self):
        self.generate = AsyncMock(return_value='{"status":"ok","reason":"mock"}')

    def set_response_format(self, model):
        return None

    def reset_response_format(self):
        return None

    def reset_messages(self):
        return None

    def set_system_message(self, message):
        return None


class _ClarifyingContactHandle:
    """Minimal handle returned by a mocked ContactManager.ask().

    It emits a clarification question via the provided queues and completes once
    an answer is received.
    """

    def __init__(
        self,
        *,
        clarification_up_q: asyncio.Queue[str],
        clarification_down_q: asyncio.Queue[str],
    ) -> None:
        self._up = clarification_up_q
        self._down = clarification_down_q
        self._done = asyncio.Event()
        self._result: str | None = None
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            await self._up.put("Which Alice do you mean?")
            ans = await self._down.get()
            self._result = f"Found contact: {ans}"
        finally:
            self._done.set()

    # --- steerable surface (minimal) ---
    def stop(self, reason: str | None = None, **kwargs):  # type: ignore[override]
        try:
            if not self._task.done():
                self._task.cancel()
        except Exception:
            pass
        self._result = "Stopped."
        self._done.set()
        return "Stopped."

    async def pause(self):  # type: ignore[override]
        return "Paused."

    async def resume(self):  # type: ignore[override]
        return "Resumed."

    def done(self) -> bool:  # type: ignore[override]
        return self._done.is_set()

    async def result(self) -> str:  # type: ignore[override]
        await self._done.wait()
        return self._result or ""

    async def ask(self, question: str, **kwargs):  # type: ignore[override]
        return self

    async def interject(self, message: str, **kwargs):  # type: ignore[override]
        return None

    async def next_clarification(self) -> dict:  # type: ignore[override]
        return {}

    async def next_notification(self) -> dict:  # type: ignore[override]
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:  # type: ignore[override]
        return None


class _MockContactManager:
    async def ask(
        self,
        text: str,
        *,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        **kwargs,
    ):
        if _clarification_up_q is None or _clarification_down_q is None:
            raise RuntimeError(
                "Clarification queues were not provided to ContactManager.ask",
            )
        _ = (text, kwargs)
        return _ClarifyingContactHandle(
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
        )

    async def update(self, text: str, **kwargs):
        # Minimal stub: HierarchicalActor tool resolution expects primitives.contacts.update to exist.
        # This test does not exercise update semantics.
        _ = (text, kwargs)

        class _StaticHandle:
            def stop(self, reason: str | None = None, **kw):  # type: ignore[override]
                return "Stopped."

            async def pause(self):  # type: ignore[override]
                return "Paused."

            async def resume(self):  # type: ignore[override]
                return "Resumed."

            def done(self) -> bool:  # type: ignore[override]
                return True

            async def result(self) -> str:  # type: ignore[override]
                return "ok"

            async def ask(self, question: str, **kw):  # type: ignore[override]
                return self

            async def interject(self, message: str, **kw):  # type: ignore[override]
                return None

            async def next_clarification(self) -> dict:  # type: ignore[override]
                return {}

            async def next_notification(self) -> dict:  # type: ignore[override]
                return {}

            async def answer_clarification(self, call_id: str, answer: str) -> None:  # type: ignore[override]
                return None

        return _StaticHandle()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_delegate_path_used_and_active_queue_steering_works(monkeypatch):
    # Ensure clean registry state for isolation.
    ManagerRegistry.clear()

    # ── Build the scheduler we will route to via primitives.tasks.execute ─────
    # Note: the delegate path should bypass this actor, but it must exist for fallback.
    scheduler = TaskScheduler(actor=SimulatedActor(steps=None, duration=None))

    task_id = scheduler._create_task(name="child task", description="child task")[
        "details"
    ]["task_id"]

    # Force primitives.tasks to use this scheduler instance.
    monkeypatch.setattr(
        ManagerRegistry,
        "get_task_scheduler",
        classmethod(lambda cls, *args, **kwargs: scheduler),
        raising=True,
    )

    # Capture the ActiveQueue returned by TaskScheduler.execute().
    queue_ready = asyncio.Event()
    captured: dict[str, object] = {}
    orig_execute = scheduler.execute

    async def _spy_execute(*args, **kwargs):
        h = await orig_execute(*args, **kwargs)
        captured["queue"] = h
        queue_ready.set()
        return h

    monkeypatch.setattr(scheduler, "execute", _spy_execute, raising=True)

    # Spy on the delegate entry point (proof that ActiveTask.create used delegate mode).
    delegate_called = asyncio.Event()
    delegate_calls: list[dict] = []
    child_handles: list[object] = []
    orig_start = _HierarchicalActorDelegate.start_task_run

    async def _spy_start(self, **kwargs):
        delegate_calls.append(dict(kwargs))
        delegate_called.set()
        h = await orig_start(self, **kwargs)
        child_handles.append(h)
        return h

    monkeypatch.setattr(
        _HierarchicalActorDelegate,
        "start_task_run",
        _spy_start,
        raising=True,
    )

    # ── Build a HierarchicalActor and mock computer immediately ────────────────
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
    )
    # Mock specific computer primitives for test control.
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value="Mock action complete.")
    actor.computer_primitives.observe = AsyncMock(return_value="Mock observation.")

    # When the delegate starts a child run via actor.act(...), inject a deterministic plan
    # so we don't invoke LLM planning. The child plan loops at checkpoints and can be steered.
    CHILD_PLAN = textwrap.dedent(
        """
        async def main_plan():
            while True:
                await _cp("tick")
        """,
    )
    original_actor_act = actor.act

    async def _patched_act(*args, **kwargs):
        h = await original_actor_act(*args, **kwargs)
        # Cancel auto-started task BEFORE injecting code, mirroring the project pattern.
        if getattr(h, "_execution_task", None):
            h._execution_task.cancel()
            try:
                await h._execution_task
            except asyncio.CancelledError:
                pass
        h.verification_client = _SimpleMockVerificationClient()
        h.plan_source_code = actor._sanitize_code(CHILD_PLAN, h)
        h._execution_task = asyncio.create_task(h._initialize_and_run())
        return h

    monkeypatch.setattr(actor, "act", _patched_act, raising=True)

    # ── Outer plan: call TaskScheduler.execute via primitives, then return ────
    OUTER_PLAN = textwrap.dedent(
        f"""
        async def main_plan():
            await primitives.tasks.execute(task_id={int(task_id)})
            return "started"
        """,
    )

    assert current_task_execution_delegate.get() is None

    outer = HierarchicalActorHandle(
        actor=actor,
        goal="trigger cyclic tasks.execute",
        persist=False,
    )
    # Cancel auto-start and inject plan.
    if outer._execution_task:
        outer._execution_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer._execution_task
    outer.verification_client = _SimpleMockVerificationClient()
    outer.plan_source_code = actor._sanitize_code(OUTER_PLAN, outer)
    outer._execution_task = asyncio.create_task(outer._initialize_and_run())

    try:
        # Wait until the delegate was used and the ActiveQueue was returned.
        await asyncio.wait_for(delegate_called.wait(), timeout=10)
        await asyncio.wait_for(queue_ready.wait(), timeout=10)

        assert delegate_calls, "expected delegate.start_task_run to be called"

        queue = captured.get("queue")
        assert queue is not None

        # Basic end-to-end steering on the returned ActiveQueue handle.
        await asyncio.wait_for(queue.pause(), timeout=5)  # type: ignore[union-attr]
        await asyncio.wait_for(queue.resume(), timeout=5)  # type: ignore[union-attr]
        await asyncio.wait_for(queue.stop(cancel=True), timeout=10)  # type: ignore[union-attr]

        result = await asyncio.wait_for(queue.result(), timeout=15)  # type: ignore[union-attr]
        assert isinstance(result, str)

        # Ensure the outer actor context finished and reset the delegate.
        outer_result = await asyncio.wait_for(outer.result(), timeout=10)
        assert "started" in str(outer_result)
        assert current_task_execution_delegate.get() is None
    finally:
        # Ensure any child handles started via the delegate are stopped.
        for h in child_handles:
            try:
                await asyncio.wait_for(h.stop(cancel=True), timeout=10)  # type: ignore[attr-defined]
                # Ensure the child execution task cannot leak past test teardown.
                try:
                    _t = getattr(h, "_execution_task", None)
                    if _t is not None and not _t.done():
                        _t.cancel()
                        try:
                            await asyncio.wait_for(_t, timeout=10)
                        except asyncio.CancelledError:
                            pass
                except Exception:
                    pass
            except Exception:
                pass

        # Stop the outer handle if still running.
        try:
            if not outer.done():
                await asyncio.wait_for(outer.stop(cancel=True), timeout=10)
            # Ensure the outer execution task is cancelled/awaited (prevents pytest-asyncio hang).
            if getattr(outer, "_execution_task", None) is not None and not outer._execution_task.done():  # type: ignore[attr-defined]
                outer._execution_task.cancel()  # type: ignore[attr-defined]
                try:
                    await asyncio.wait_for(outer._execution_task, timeout=10)  # type: ignore[attr-defined]
                except asyncio.CancelledError:
                    pass
        except Exception:
            pass

        # Best-effort cleanup
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_cancel_via_active_task_interject_is_deterministic_in_delegate_mode(
    monkeypatch,
):
    """
    Verify that cancellation via `ActiveTask.interject()` is:
    - deterministic (no LLM dependency; intent classification is patched),
    - correctly stops the underlying actor handle even if stop() is async,
    - correctly mirrors status to the Tasks table and clears the active pointer,
    when running in delegate mode (Actor → TaskScheduler → Actor cyclic path).
    """
    from unity.task_scheduler import active_task as active_task_mod
    from unity.task_scheduler.types.status import Status

    ManagerRegistry.clear()

    scheduler = TaskScheduler(actor=SimulatedActor(steps=None, duration=None))
    task_id = scheduler._create_task(name="cancel me", description="cancel me")[
        "details"
    ]["task_id"]

    monkeypatch.setattr(
        ManagerRegistry,
        "get_task_scheduler",
        classmethod(lambda cls, *args, **kwargs: scheduler),
        raising=True,
    )

    # Capture the ActiveQueue returned by TaskScheduler.execute().
    queue_ready = asyncio.Event()
    captured: dict[str, object] = {}
    orig_execute = scheduler.execute

    async def _spy_execute(*args, **kwargs):
        h = await orig_execute(*args, **kwargs)
        captured["queue"] = h
        queue_ready.set()
        return h

    monkeypatch.setattr(scheduler, "execute", _spy_execute, raising=True)

    # Force steering intent classification to deterministic cancellation (no LLM).
    async def _forced_cancel_intent(*args, **kwargs):
        _ = (args, kwargs)
        return "cancel", "test_cancel"

    monkeypatch.setattr(
        active_task_mod,
        "classify_steering_intent",
        _forced_cancel_intent,
        raising=True,
    )

    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
    )
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value="Mock action complete.")
    actor.computer_primitives.observe = AsyncMock(return_value="Mock observation.")

    # Deterministic child plan (avoid LLM planning).
    CHILD_PLAN = textwrap.dedent(
        """
        async def main_plan():
            while True:
                await _cp("tick")
        """,
    )
    original_actor_act = actor.act

    async def _patched_act(*args, **kwargs):
        h = await original_actor_act(*args, **kwargs)
        if getattr(h, "_execution_task", None):
            h._execution_task.cancel()
            try:
                await h._execution_task
            except asyncio.CancelledError:
                pass
        h.verification_client = _SimpleMockVerificationClient()
        h.plan_source_code = actor._sanitize_code(CHILD_PLAN, h)
        h._execution_task = asyncio.create_task(h._initialize_and_run())
        return h

    monkeypatch.setattr(actor, "act", _patched_act, raising=True)

    OUTER_PLAN = textwrap.dedent(
        f"""
        async def main_plan():
            await primitives.tasks.execute(task_id={int(task_id)})
            return "started"
        """,
    )

    outer = HierarchicalActorHandle(
        actor=actor,
        goal="trigger cancel via interject",
        persist=False,
    )
    if outer._execution_task:
        outer._execution_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer._execution_task
    outer.verification_client = _SimpleMockVerificationClient()
    outer.plan_source_code = actor._sanitize_code(OUTER_PLAN, outer)
    outer._execution_task = asyncio.create_task(outer._initialize_and_run())

    await asyncio.wait_for(queue_ready.wait(), timeout=10)

    queue = captured.get("queue")
    assert queue is not None

    assert scheduler._active_task is not None
    active_handle = scheduler._active_task.handle

    # Cancellation via interject should clear active pointer and mark cancelled.
    await asyncio.wait_for(
        active_handle.interject("please cancel this task"),
        timeout=10,
    )
    _ = await asyncio.wait_for(queue.result(), timeout=20)  # should complete after stop

    assert scheduler._active_task is None
    row = scheduler._filter_tasks(filter=f"task_id == {int(task_id)}", limit=1)[0]
    assert row.status == Status.cancelled

    # Ensure no delegate leakage after outer completes.
    _ = await asyncio.wait_for(outer.result(), timeout=10)
    assert current_task_execution_delegate.get() is None

    try:
        await actor.close()
    except Exception:
        pass


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_nested_manager_clarification_in_delegate_mode(monkeypatch):
    """
    Verify that when a task's entrypoint invokes a nested manager that requests
    clarification, the clarification is observable via next_clarification() and
    can be answered via answer_clarification() end-to-end in delegate mode.
    """
    ManagerRegistry.clear()

    # TaskScheduler instance used by primitives.tasks inside the outer plan.
    scheduler = TaskScheduler(actor=SimulatedActor(steps=None, duration=None))

    # Mock ContactManager and force primitives.contacts to resolve to it.
    monkeypatch.setattr(
        ManagerRegistry,
        "get_contact_manager",
        classmethod(lambda cls, *args, **kwargs: _MockContactManager()),
        raising=True,
    )

    # Use a dedicated FunctionManager for the HierarchicalActor entrypoint.
    fm = FunctionManager()
    try:
        fm.clear()
    except Exception:
        pass

    ENTRYPOINT_WITH_MANAGER_CALL = textwrap.dedent(
        """
        async def ask_contact_question():
            '''Entrypoint that calls ContactManager.ask and awaits its result.'''
            h = await primitives.contacts.ask("Who is Alice?")
            r = await h.result()
            return f"Contact info: {r}"
        """,
    )
    fm.add_functions(
        implementations=ENTRYPOINT_WITH_MANAGER_CALL,
        verify={"ask_contact_question": False},
        overwrite=True,
    )
    fn_rows = fm.filter_functions(filter="name == 'ask_contact_question'", limit=1)
    assert fn_rows and fn_rows[0].get("function_id") is not None
    entrypoint_id = int(fn_rows[0]["function_id"])

    task_id = scheduler._create_task(
        name="clarify_contact",
        description="clarify_contact",
        entrypoint=entrypoint_id,
    )["details"]["task_id"]

    monkeypatch.setattr(
        ManagerRegistry,
        "get_task_scheduler",
        classmethod(lambda cls, *args, **kwargs: scheduler),
        raising=True,
    )

    # Capture the ActiveQueue returned by TaskScheduler.execute().
    queue_ready = asyncio.Event()
    captured: dict[str, object] = {}
    orig_execute = scheduler.execute

    async def _spy_execute(*args, **kwargs):
        h = await orig_execute(*args, **kwargs)
        captured["queue"] = h
        queue_ready.set()
        return h

    monkeypatch.setattr(scheduler, "execute", _spy_execute, raising=True)

    # Prove delegate mode is actually used.
    delegate_called = asyncio.Event()
    orig_start = _HierarchicalActorDelegate.start_task_run

    async def _spy_start(self, **kwargs):
        delegate_called.set()
        return await orig_start(self, **kwargs)

    monkeypatch.setattr(
        _HierarchicalActorDelegate,
        "start_task_run",
        _spy_start,
        raising=True,
    )

    # Outer actor: must have clarification queues enabled so ToolProviderProxy can inject them.
    actor = HierarchicalActor(
        function_manager=fm,
        headless=True,
        computer_mode="mock",
        connect_now=False,
        can_compose=False,
    )
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value="Mock action complete.")
    actor.computer_primitives.observe = AsyncMock(return_value="Mock observation.")

    # Ensure child task handles created via the delegate don't invoke real verification LLMs.
    # (This mirrors existing HierarchicalActor tests that inject a mock verification client.)
    original_actor_act = actor.act

    async def _patched_act(*args, **kwargs):
        h = await original_actor_act(*args, **kwargs)
        try:
            h.verification_client = _SimpleMockVerificationClient()
        except Exception:
            pass
        return h

    monkeypatch.setattr(actor, "act", _patched_act, raising=True)

    OUTER_PLAN = textwrap.dedent(
        f"""
        async def main_plan():
            await primitives.tasks.execute(task_id={int(task_id)})
            return "started"
        """,
    )

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    assert current_task_execution_delegate.get() is None
    outer = HierarchicalActorHandle(
        actor=actor,
        goal="trigger tasks.execute with clarification enabled",
        persist=False,
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    if outer._execution_task:
        outer._execution_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer._execution_task
    outer.verification_client = _SimpleMockVerificationClient()
    outer.plan_source_code = actor._sanitize_code(OUTER_PLAN, outer)
    outer._execution_task = asyncio.create_task(outer._initialize_and_run())

    await asyncio.wait_for(delegate_called.wait(), timeout=10)
    await asyncio.wait_for(queue_ready.wait(), timeout=10)

    queue = captured.get("queue")
    assert queue is not None

    # Clarification should bubble up via ActiveQueue -> ActiveTask -> HierarchicalActorHandle queues.
    clar = await asyncio.wait_for(queue.next_clarification(), timeout=10)  # type: ignore[union-attr]
    assert clar.get("question") == "Which Alice do you mean?"

    await asyncio.wait_for(
        queue.answer_clarification("ignored", "Alice Smith"),  # type: ignore[union-attr]
        timeout=5,
    )

    result = await asyncio.wait_for(queue.result(), timeout=10)  # type: ignore[union-attr]
    assert "Contact info:" in str(result)
    assert "Alice Smith" in str(result)

    # Outer handle completes and delegate resets.
    outer_result = await asyncio.wait_for(outer.result(), timeout=10)
    assert "started" in str(outer_result)
    assert current_task_execution_delegate.get() is None

    try:
        await actor.close()
    except Exception:
        pass


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_delegate_contextvar_reset_no_leakage(monkeypatch):
    """
    Verify that after a HierarchicalActorHandle run completes, the delegate
    ContextVar is reset and does not leak to fresh async contexts.

    This test ensures:
    1. Delegate is None before the run
    2. Delegate is set during the run (verified indirectly by delegate construction)
    3. Delegate is None after the run completes (same task)
    4. Delegate is None in a fresh asyncio task spawned after the run
    """
    ManagerRegistry.clear()

    # Spy on delegate construction (it happens inside HierarchicalActorHandle._initialize_and_run()).
    created = asyncio.Event()
    orig_init = _HierarchicalActorDelegate.__init__

    def _spy_init(self, *args, **kwargs):
        created.set()
        return orig_init(self, *args, **kwargs)

    monkeypatch.setattr(_HierarchicalActorDelegate, "__init__", _spy_init, raising=True)

    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        can_compose=False,
    )
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value="Mock action complete.")
    actor.computer_primitives.observe = AsyncMock(return_value="Mock observation.")

    PLAN = textwrap.dedent(
        """
        async def main_plan():
            return "done"
        """,
    )

    assert current_task_execution_delegate.get() is None

    h = HierarchicalActorHandle(
        actor=actor,
        goal="delegate reset leak test",
        persist=False,
    )
    if h._execution_task:
        h._execution_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await h._execution_task

    h.verification_client = _SimpleMockVerificationClient()
    h.plan_source_code = actor._sanitize_code(PLAN, h)
    h._execution_task = asyncio.create_task(h._initialize_and_run())

    # Delegate should be constructed during the run.
    await asyncio.wait_for(created.wait(), timeout=10)
    res = await asyncio.wait_for(h.result(), timeout=10)
    assert "done" in str(res)

    # Delegate reset in the current task (after completion).
    assert current_task_execution_delegate.get() is None

    # Delegate should not leak into a fresh asyncio task created after the run.
    q: asyncio.Queue[object] = asyncio.Queue(maxsize=1)

    async def _check_fresh_context():
        await q.put(current_task_execution_delegate.get())

    await asyncio.wait_for(asyncio.create_task(_check_fresh_context()), timeout=5)
    v = await asyncio.wait_for(q.get(), timeout=5)
    assert v is None

    try:
        await actor.close()
    except Exception:
        pass
