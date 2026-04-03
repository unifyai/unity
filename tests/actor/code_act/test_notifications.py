import asyncio

import pytest

from unity.actor.code_act_actor import CodeActActor
from unity.actor.execution import PythonExecutionSession


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_execute_code_notifications_with_notification_queue():
    """
    Validate that user code can call notify({...}) from within the sandbox
    when a _notification_up_q is provided (as the async tool loop does
    automatically).  Lifecycle notifications (execution_started/finished)
    were removed -- only user-emitted notifications should appear.
    """
    from unity.actor.execution import _CURRENT_SANDBOX

    notification_q: asyncio.Queue[dict] = asyncio.Queue()

    actor = CodeActActor()

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

        _ = await execute_code(
            "emit notifications",
            "notify({'type': 'custom_progress', 'step': 1})\nprint('hi')",
            language="python",
            state_mode="stateful",
            session_id=0,
            venv_id=None,
            _notification_up_q=notification_q,
        )

        custom = await asyncio.wait_for(notification_q.get(), timeout=30)
        assert custom.get("type") == "custom_progress"
        assert custom.get("step") == 1

        assert (
            notification_q.empty()
        ), f"Expected no more notifications but queue has {notification_q.qsize()} item(s)"
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
async def test_execute_code_notifications_with_notification_queue_in_named_stateful_session():
    """
    Validate that notify({...}) also surfaces from executor-managed named
    stateful Python sessions, not just the bound sandbox session 0 path.
    """
    from unity.actor.execution import _CURRENT_SANDBOX

    notification_q: asyncio.Queue[dict] = asyncio.Queue()

    actor = CodeActActor()

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

        _ = await execute_code(
            "emit notifications from named session",
            "notify({'type': 'custom_progress', 'step': 1, 'message': 'named session'})\nprint('hi')",
            language="python",
            state_mode="stateful",
            session_name="named_notify_session",
            venv_id=None,
            _notification_up_q=notification_q,
        )

        custom = await asyncio.wait_for(notification_q.get(), timeout=5)
        assert custom.get("type") == "custom_progress"
        assert custom.get("step") == 1
        assert custom.get("message") == "named session"

        assert (
            notification_q.empty()
        ), f"Expected no more notifications but queue has {notification_q.qsize()} item(s)"
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
@pytest.mark.llm_call
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
        await handle.result()

    finally:
        if not handle.done():
            try:
                await handle.stop("cleanup")
            except Exception:
                pass
