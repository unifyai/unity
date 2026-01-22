import asyncio

import pytest
from unittest.mock import AsyncMock

from unity.actor.code_act_actor import CodeActActor, PythonExecutionSession


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_code_act_notifications_and_notify_helper():
    """
    Validate that CodeActActor emits execution_started/execution_finished notifications
    and that user code can call notify({...}) from within the sandbox.
    """
    from unity.actor.code_act_actor import _CURRENT_SANDBOX

    notification_q: asyncio.Queue[dict] = asyncio.Queue()

    actor = CodeActActor(headless=True, computer_mode="mock")
    actor._computer_primitives.navigate = AsyncMock(return_value=None)
    actor._computer_primitives.act = AsyncMock(return_value="Action completed")
    actor._computer_primitives.observe = AsyncMock(return_value="Page content observed")

    sandbox = PythonExecutionSession(
        computer_primitives=actor._computer_primitives,
        environments=actor.environments,
        venv_pool=actor._venv_pool,
        shell_pool=actor._shell_pool,
    )
    sandbox.global_state["__notification_up_q__"] = notification_q
    token = _CURRENT_SANDBOX.set(sandbox)

    try:
        tools = actor.get_tools("act")
        execute_code = tools["execute_code"]

        _ = await execute_code(
            "emit notifications",
            "notify({'type': 'custom_progress', 'step': 1})\nprint('hi')",
            language="python",
            state_mode="stateful",
            session_id=0,
            venv_id=None,
        )

        started = await asyncio.wait_for(notification_q.get(), timeout=30)
        assert started.get("type") == "execution_started"
        assert isinstance(started.get("sandbox_id"), str)

        custom = await asyncio.wait_for(notification_q.get(), timeout=30)
        assert custom.get("type") == "custom_progress"
        assert custom.get("step") == 1

        finished = await asyncio.wait_for(notification_q.get(), timeout=30)
        assert finished.get("type") == "execution_finished"
        assert finished.get("status") == "ok"
    finally:
        try:
            _CURRENT_SANDBOX.reset(token)
        except Exception:
            pass
        try:
            await sandbox.close()
        except Exception:
            pass
        if actor:
            try:
                await actor.close()
            except Exception:
                pass


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_actor_handle_next_notification_reads_queue():
    """ActorHandle.next_notification should surface events when a queue is supplied."""
    from unity.actor.handle import ActorHandle

    notification_q: asyncio.Queue[dict] = asyncio.Queue()

    handle: ActorHandle | None = None
    try:
        handle = ActorHandle(
            task_description="No-op task (test notifications only).",
            tools={},
            notification_up_q=notification_q,
            timeout=5,
        )

        await notification_q.put({"type": "notification", "message": "hello"})
        event = await asyncio.wait_for(handle.next_notification(), timeout=10)
        assert event.get("message") == "hello"
    finally:
        if handle and not handle.done():
            try:
                await handle.stop("cleanup")
            except Exception:
                pass
