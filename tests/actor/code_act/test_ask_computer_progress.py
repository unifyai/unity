import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import unity.actor.code_act_actor as code_act_module
from unity.actor.code_act_actor import CodeActActor
from unity.common import async_tool_loop as atl


class _DummyLoopClient:
    endpoint = "gpt-4o-mini@openai"

    def __init__(self) -> None:
        self.messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "working"},
        ]


def _make_actor_with_mock_computer(
    *,
    timeout: float,
    query_return: str,
) -> CodeActActor:
    """Construct a CodeActActor with a mock _computer_primitives for testing."""
    actor = CodeActActor(timeout=timeout)
    mock_desktop = MagicMock()
    mock_desktop.query = AsyncMock(return_value=query_return)
    mock_desktop.navigate = AsyncMock(return_value="navigated")
    mock_desktop.act = AsyncMock(return_value="acted")
    mock_desktop.observe = AsyncMock(return_value="observed")
    mock_cp = MagicMock()
    mock_cp.desktop = mock_desktop
    mock_cp.pause = AsyncMock()
    mock_cp.resume = AsyncMock()
    actor._computer_primitives = mock_cp
    return actor


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_code_act_ask_includes_computer_progress_tool(monkeypatch):
    """CodeActActor should augment ask() inspection tools with computer progress query.

    This is a symbolic integration test:
    - `CodeActActor.act(...)` is exercised so augmentation happens in real code.
    - The follow-up inspection loop startup is monkeypatched to capture tool exposure.
    - The injected `ask_computer_progress` tool is executed and must delegate to
      `ComputerPrimitives.query(...)`.
    """

    captured: dict = {}

    async def ask_inner_status(question: str) -> str:
        return f"inner:{question}"

    async def _outer_done() -> str:
        return "outer-complete"

    outer_task = asyncio.create_task(_outer_done(), name="CodeActOuterDummyTask")
    setattr(
        outer_task,
        "get_ask_tools",
        lambda: {"ask_inner_status": ask_inner_status},
    )

    outer_handle = atl.AsyncToolLoopHandle(
        task=outer_task,
        interject_queue=asyncio.Queue(),
        cancel_event=asyncio.Event(),
        stop_event=asyncio.Event(),
        client=_DummyLoopClient(),
        loop_id="CodeActActor.act",
    )

    def _fake_actor_start_async_tool_loop(*args, **kwargs):
        extra = kwargs.get("extra_ask_tools") or {}
        _base = outer_task.get_ask_tools
        setattr(outer_task, "get_ask_tools", lambda: {**_base(), **extra})
        return outer_handle

    class _DummyInspectionHandle:
        async def result(self):
            return "inspection-complete"

    def _fake_inspection_start_async_tool_loop(*args, **kwargs):
        captured["tools"] = args[2]
        return _DummyInspectionHandle()

    monkeypatch.setattr(
        code_act_module,
        "start_async_tool_loop",
        _fake_actor_start_async_tool_loop,
    )
    monkeypatch.setattr(
        atl,
        "start_async_tool_loop",
        _fake_inspection_start_async_tool_loop,
    )

    actor = _make_actor_with_mock_computer(
        timeout=30,
        query_return="browser-progress-details",
    )

    handle = None
    try:
        handle = await actor.act(
            "Inspect browser progress.",
            clarification_enabled=False,
            persist=False,
        )

        helper = await handle.ask("What is happening in the browser agent?")
        await helper.result()

        ask_tools = captured.get("tools")
        assert isinstance(
            ask_tools,
            dict,
        ), "Expected inspection loop tools to be captured"
        assert "ask_inner_status" in ask_tools
        assert "ask_computer_progress" in ask_tools

        assert await ask_tools["ask_inner_status"]("ping") == "inner:ping"

        question = "Did the browser finish submitting the form?"
        progress = await ask_tools["ask_computer_progress"](question)
        assert progress == "browser-progress-details"
        actor._computer_primitives.desktop.query.assert_awaited_once_with(question)

        await handle.result()
    finally:
        try:
            if handle is not None and not handle.done():
                await handle.stop("test cleanup")
        except Exception:
            pass
        await actor.close()


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_code_act_ask_uses_computer_progress_for_inflight_action(monkeypatch):
    """Ask inspection should invoke ask_computer_progress for missing act() details.

    This test keeps CodeAct's handle augmentation path real while stubbing only
    the outer act loop bootstrap. The follow-up ask loop is real and must choose
    the injected `ask_computer_progress` tool to answer a progress question that
    cannot be resolved from the transcript alone.
    """

    async def _outer_done() -> str:
        return "outer-complete"

    outer_task = asyncio.create_task(_outer_done(), name="CodeActOuterDummyTask")
    setattr(outer_task, "get_ask_tools", lambda: {})

    outer_client = _DummyLoopClient()
    outer_client.messages = [
        {"role": "user", "content": "Log into the dashboard."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_computer_act_1",
                    "type": "function",
                    "function": {
                        "name": "act",
                        "arguments": '{"instruction":"Log into the dashboard"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "name": "act",
            "tool_call_id": "call_computer_act_1",
            "content": "Command queued.",
        },
        {
            "role": "assistant",
            "content": ("I queued the browser action and I am waiting for completion."),
        },
    ]

    outer_handle = atl.AsyncToolLoopHandle(
        task=outer_task,
        interject_queue=asyncio.Queue(),
        cancel_event=asyncio.Event(),
        stop_event=asyncio.Event(),
        client=outer_client,
        loop_id="CodeActActor.act",
    )

    def _fake_actor_start_async_tool_loop(*args, **kwargs):
        extra = kwargs.get("extra_ask_tools") or {}
        _base = outer_task.get_ask_tools
        setattr(outer_task, "get_ask_tools", lambda: {**_base(), **extra})
        return outer_handle

    monkeypatch.setattr(
        code_act_module,
        "start_async_tool_loop",
        _fake_actor_start_async_tool_loop,
    )

    actor = _make_actor_with_mock_computer(
        timeout=40,
        query_return=(
            "Browser memory says: typed email and password, clicked submit, "
            "waiting for the post-login redirect."
        ),
    )

    handle = None
    try:
        handle = await actor.act(
            "Track login progress.",
            clarification_enabled=False,
            persist=False,
        )

        helper = await handle.ask(
            "What exact steps has the browser agent already completed for login?",
        )
        answer = await helper.result()

        assert actor._computer_primitives.desktop.query.await_count >= 1, (
            "Expected ask inspection to call ask_computer_progress, "
            "which delegates to computer_primitives.desktop.query."
        )

        helper_history = helper.get_history()
        assert any(
            m.get("role") == "assistant"
            and any(
                tc.get("function", {}).get("name") == "ask_computer_progress"
                for tc in (m.get("tool_calls") or [])
            )
            for m in helper_history
        ), "Inspection loop did not issue ask_computer_progress tool call."

        assert (
            isinstance(answer, str) and answer.strip()
        ), "Expected a non-empty answer."
    finally:
        try:
            if handle is not None and not handle.done():
                await handle.stop("test cleanup")
        except Exception:
            pass
        await actor.close()
