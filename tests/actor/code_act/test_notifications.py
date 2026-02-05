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
async def test_tool_loop_handle_next_notification():
    """
    AsyncToolLoopHandle.next_notification() should surface notifications
    emitted by tools running inside the loop.
    """
    from unity.common.async_tool_loop import start_async_tool_loop
    from unity.common.llm_client import new_llm_client

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

    client = new_llm_client()
    client.set_system_message(
        "You are a helpful assistant. When asked to call a tool, call it with the "
        "specified arguments. Do not ask for clarification - just call the tool.",
    )

    handle = start_async_tool_loop(
        client,
        "Call notifying_tool with message='hello'",
        {"notifying_tool": notifying_tool},
        timeout=60,
    )

    try:
        # The inner loop should eventually call the tool and emit a notification.
        # The tool loop wraps notifications with tool_name and call_id, but the original
        # payload (including 'type') is merged in via **event_payload.
        event = await asyncio.wait_for(handle.next_notification(), timeout=60)
        assert event.get("type") == "test_notification"  # From our tool's payload
        assert event.get("tool_name") == "notifying_tool"
        # The message from our tool should be in the event
        assert "hello" in str(event.get("message", ""))

    finally:
        if not handle.done():
            try:
                await handle.stop("cleanup")
            except Exception:
                pass
