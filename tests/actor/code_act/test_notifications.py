import asyncio

import pytest
from unittest.mock import AsyncMock

from unity.actor.code_act_actor import CodeActActor, PythonExecutionSession


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_execute_code_notifications_with_notification_queue():
    """
    Validate that execute_code emits execution_started/execution_finished notifications
    and that user code can call notify({...}) from within the sandbox when a
    _notification_up_q is provided (as the async tool loop does automatically).
    """
    from unity.actor.code_act_actor import _CURRENT_SANDBOX

    # Create a notification queue that simulates what the async tool loop provides
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
    token = _CURRENT_SANDBOX.set(sandbox)

    try:
        tools = actor.get_tools("act")
        execute_code = tools["execute_code"]

        # Call execute_code with _notification_up_q (as the tool loop would)
        _ = await execute_code(
            "emit notifications",
            "notify({'type': 'custom_progress', 'step': 1})\nprint('hi')",
            language="python",
            state_mode="stateful",
            session_id=0,
            venv_id=None,
            _notification_up_q=notification_q,  # Tool loop provides this
        )

        # Verify execution_started notification
        started = await asyncio.wait_for(notification_q.get(), timeout=30)
        assert started.get("type") == "execution_started"
        assert isinstance(started.get("sandbox_id"), str)

        # Verify custom notification from notify() call in sandbox
        custom = await asyncio.wait_for(notification_q.get(), timeout=30)
        assert custom.get("type") == "custom_progress"
        assert custom.get("step") == 1

        # Verify execution_finished notification
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
@pytest.mark.timeout(120)
async def test_actor_handle_delegates_next_notification_to_inner_loop():
    """
    ActorHandle.next_notification() should delegate to the inner loop handle
    once it's ready. This tests the standard async tool loop pattern where
    notifications bubble up through the loop.
    """
    from unity.actor.handle import ActorHandle

    # Create a simple tool that emits a notification
    async def notifying_tool(
        message: str,
        *,
        _notification_up_q: asyncio.Queue[dict] | None = None,
    ) -> str:
        """A tool that emits a notification when called."""
        if _notification_up_q is not None:
            await _notification_up_q.put(
                {"type": "test_notification", "message": message},
            )
        return f"Processed: {message}"

    handle: ActorHandle | None = None
    try:
        handle = ActorHandle(
            task_description="Call notifying_tool with message='hello'",
            tools={"notifying_tool": notifying_tool},
            timeout=60,
            custom_system_prompt=(
                "You are a helpful assistant. When asked to call a tool, call it with the "
                "specified arguments. Do not ask for clarification - just call the tool."
            ),
        )

        # Wait for the inner loop handle to be ready
        await asyncio.wait_for(handle._loop_handle_ready.wait(), timeout=30)

        # The inner loop should eventually call the tool and emit a notification.
        # The tool loop wraps notifications with tool_name and call_id, but the original
        # payload (including 'type') is merged in via **event_payload.
        event = await asyncio.wait_for(handle.next_notification(), timeout=60)
        assert event.get("type") == "test_notification"  # From our tool's payload
        assert event.get("tool_name") == "notifying_tool"
        # The message from our tool should be in the event
        assert "hello" in str(event.get("message", ""))

    finally:
        if handle and not handle.done():
            try:
                await handle.stop("cleanup")
            except Exception:
                pass
