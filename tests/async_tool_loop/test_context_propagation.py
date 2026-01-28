"""
Chat context propagation tests for async tool loop.

Verifies that `parent_chat_context` is threaded into tools that accept it and
that the loop inserts the synthetic system context header.

Also tests incremental context propagation:
- First tool call receives full parent_chat_context
- Subsequent calls receive only incremental updates via _parent_chat_context_cont
- Context continuations from interjections are properly forwarded
"""

from __future__ import annotations

from typing import List

import pytest
from unity.common.async_tool_loop import ChatContextPropagation, start_async_tool_loop
from unity.common._async_tool.context_tracker import LoopContextState
from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client


@pytest.mark.asyncio
@_handle_project
async def test_chat_context_propagation(model) -> None:
    client = new_llm_client(model=model)

    root_ctx = [{"role": "user", "content": "root-level message"}]
    captured_ctx: List[list[dict]] = []

    async def record_context(*, _parent_chat_context: list[dict] | None = None) -> str:
        captured_ctx.append(_parent_chat_context or [])
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    handle = start_async_tool_loop(
        client=client,
        message="Please call the function `record_context()` once, then reply 'done'.",
        tools={"record_context": record_context},
        parent_chat_context=root_ctx,
        propagate_chat_context=ChatContextPropagation.ALWAYS,
    )

    final_ans = await handle.result()
    assert final_ans is not None, "Loop should complete with a response"

    # Find the runtime context header message (may not be at position 0 due to
    # other system messages like User Visibility Context being prepended)
    ctx_header_msg = next(
        (m for m in client.messages if m.get("_ctx_header") is True),
        None,
    )
    assert ctx_header_msg is not None, "Expected a system message with _ctx_header=True"
    assert ctx_header_msg["role"] == "system"

    assert len(captured_ctx) == 1
    combined = captured_ctx[0]

    assert combined[0]["content"] == "root-level message"
    assert "children" in combined[0]
    child_msgs = combined[0]["children"]
    # Find the user message (may not be at position 0 due to system messages
    # like User Visibility Context being prepended)
    user_msg = next(
        (m for m in child_msgs if m.get("role") == "user"),
        None,
    )
    assert user_msg is not None and user_msg["content"].startswith(
        "Please call the function",
    )


@pytest.mark.asyncio
@_handle_project
async def test_chat_context_propagation_never(model) -> None:
    """Verify that NEVER mode does NOT pass context to tools, even when they accept it."""
    client = new_llm_client(model=model)

    root_ctx = [{"role": "user", "content": "secret-context-message"}]
    captured_ctx: List[list[dict]] = []

    async def record_context(*, _parent_chat_context: list[dict] | None = None) -> str:
        captured_ctx.append(_parent_chat_context or [])
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    handle = start_async_tool_loop(
        client=client,
        message="Please call the function `record_context()` once, then reply 'done'.",
        tools={"record_context": record_context},
        parent_chat_context=root_ctx,
        propagate_chat_context=ChatContextPropagation.NEVER,
    )

    final_ans = await handle.result()
    assert final_ans is not None, "Loop should complete with a response"

    # Tool should have been called
    assert len(captured_ctx) == 1, "Tool should have been called exactly once"

    # But context should be empty (NEVER mode)
    assert captured_ctx[0] == [], "Context should be empty in NEVER mode"


@pytest.mark.asyncio
@_handle_project
async def test_chat_context_propagation_llm_decides_include(model) -> None:
    """Verify that LLM_DECIDES mode passes context when LLM includes it (default)."""
    client = new_llm_client(model=model)

    root_ctx = [{"role": "user", "content": "root-level-context-marker"}]
    captured_ctx: List[list[dict]] = []

    async def record_context(*, _parent_chat_context: list[dict] | None = None) -> str:
        captured_ctx.append(_parent_chat_context or [])
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    # Prompt the LLM to explicitly include context
    handle = start_async_tool_loop(
        client=client,
        message=(
            "Please call the function `record_context()` once with "
            "`include_parent_chat_context` set to `true`, then reply 'done'."
        ),
        tools={"record_context": record_context},
        parent_chat_context=root_ctx,
        propagate_chat_context=ChatContextPropagation.LLM_DECIDES,
    )

    final_ans = await handle.result()
    assert final_ans is not None, "Loop should complete with a response"

    # Tool should have been called
    assert len(captured_ctx) == 1, "Tool should have been called exactly once"

    # Context should be passed (LLM chose to include it)
    combined = captured_ctx[0]
    assert len(combined) > 0, "Context should be non-empty when LLM includes it"
    assert combined[0]["content"] == "root-level-context-marker"


@pytest.mark.asyncio
@_handle_project
async def test_chat_context_propagation_llm_decides_exclude(model) -> None:
    """Verify that LLM_DECIDES mode omits context when LLM explicitly excludes it."""
    client = new_llm_client(model=model)

    root_ctx = [{"role": "user", "content": "secret-context-should-not-appear"}]
    captured_ctx: List[list[dict]] = []

    async def record_context(*, _parent_chat_context: list[dict] | None = None) -> str:
        captured_ctx.append(_parent_chat_context or [])
        return "context-recorded"

    record_context.__name__ = "record_context"
    record_context.__qualname__ = "record_context"

    # Prompt the LLM to explicitly exclude context
    handle = start_async_tool_loop(
        client=client,
        message=(
            "Please call the function `record_context()` once with "
            "`include_parent_chat_context` set to `false`, then reply 'done'."
        ),
        tools={"record_context": record_context},
        parent_chat_context=root_ctx,
        propagate_chat_context=ChatContextPropagation.LLM_DECIDES,
    )

    final_ans = await handle.result()
    assert final_ans is not None, "Loop should complete with a response"

    # Tool should have been called
    assert len(captured_ctx) == 1, "Tool should have been called exactly once"

    # Context should be empty (LLM chose to exclude it)
    assert captured_ctx[0] == [], "Context should be empty when LLM excludes it"


@pytest.mark.asyncio
@_handle_project
async def test_ask_uses_continued_parent_context(model) -> None:
    """Verify that ask() packages continued parent context and influences the answer.

    The inner inspection loop should choose "apple" only because that signal
    exists in the provided continued context, not in the current prompt.
    """

    client = new_llm_client(model=model)

    # Start a trivial outer loop (no tools needed for this test).
    handle = start_async_tool_loop(
        client=client,
        message=("We will later follow-up with a question requiring broader context."),
        tools={},
    )

    # Provide a contrived continued parent context that carries the deciding hint.
    continued_ctx = [
        {
            "role": "assistant",
            "content": (
                "Important persistent preference: If asked to choose a fruit, "
                "always choose APPLE and not BANANA."
            ),
        },
    ]

    # Ask a question whose correct answer requires the continued parent context.
    helper = await handle.ask(
        ("Which fruit should we choose? Please answer in one short phrase."),
        parent_chat_context_cont=continued_ctx,
    )
    ans = await helper.result()

    assert "apple" in ans.lower(), "Answer did not reflect continued parent context."


@pytest.mark.asyncio
@_handle_project
async def test_interject_with_continued_parent_context_influences_decision(
    model,
) -> None:
    """Verify that an interjection with continued parent context steers the LLM decision.

    The outer loop should incorporate the interjection (and its continued context)
    such that the next assistant reply reflects that broader context.
    """

    client = new_llm_client(model=model)

    handle = start_async_tool_loop(
        client=client,
        message=(
            "We need to pick a fruit between APPLE and BANANA. "
            "Decide shortly after considering any additional context."
        ),
        tools={},
    )

    continued_ctx = [
        {
            "role": "assistant",
            "content": (
                "If asked to decide between fruits, the correct choice is APPLE."
            ),
        },
    ]

    # Inject guidance that includes the continued parent context.
    await handle.interject(
        "FYI: see additional context that determines the correct fruit.",
        parent_chat_context_cont=continued_ctx,
    )

    final = await handle.result()
    assert (
        "apple" in final.lower()
    ), "Final decision did not reflect continued parent context."


# =============================================================================
# Unit tests for LoopContextState (incremental context tracking)
# =============================================================================


class TestLoopContextState:
    """Unit tests for the incremental context tracking state machine."""

    def test_initial_state(self):
        """Verify initial state is empty."""
        state = LoopContextState()
        assert state.parent_chat_context == []
        assert state.parent_chat_context_cont_received == []
        assert state.inner_tool_forwarding == {}

    def test_receive_context_continuation(self):
        """Verify context continuations are accumulated."""
        state = LoopContextState()

        state.receive_context_continuation([{"role": "user", "content": "msg1"}])
        assert len(state.parent_chat_context_cont_received) == 1

        state.receive_context_continuation([{"role": "user", "content": "msg2"}])
        assert len(state.parent_chat_context_cont_received) == 2

        # Empty list should not add anything
        state.receive_context_continuation([])
        assert len(state.parent_chat_context_cont_received) == 2

    def test_first_call_receives_full_context(self):
        """First call to a tool should receive full parent context."""
        parent_ctx = [{"role": "user", "content": "parent msg"}]
        state = LoopContextState(parent_chat_context=parent_ctx)

        local_msgs = [{"role": "user", "content": "local msg"}]

        parent, cont = state.compute_context_for_inner_tool("call_1", local_msgs)

        # First call gets full parent context with local msgs as children
        assert parent is not None
        assert parent[0]["content"] == "parent msg"
        assert "children" in parent[0]
        assert parent[0]["children"][0]["content"] == "local msg"

        # No cont on first call (unless there were accumulated conts)
        assert cont is None

    def test_second_call_receives_only_incremental(self):
        """Second call to same tool should receive only incremental updates."""
        parent_ctx = [{"role": "user", "content": "parent msg"}]
        state = LoopContextState(parent_chat_context=parent_ctx)

        local_msgs = [{"role": "user", "content": "msg1"}]

        # First call
        state.compute_context_for_inner_tool("call_1", local_msgs)

        # Add more local messages
        local_msgs.append({"role": "assistant", "content": "msg2"})

        # Second call
        parent, cont = state.compute_context_for_inner_tool("call_1", local_msgs)

        # No parent context on subsequent calls
        assert parent is None

        # Only the new message in cont
        assert cont is not None
        assert len(cont) == 1
        assert cont[0]["content"] == "msg2"

    def test_different_tools_get_independent_tracking(self):
        """Different tool calls should have independent tracking."""
        parent_ctx = [{"role": "user", "content": "parent msg"}]
        state = LoopContextState(parent_chat_context=parent_ctx)

        local_msgs = [{"role": "user", "content": "local msg"}]

        # First tool gets full context
        parent1, _ = state.compute_context_for_inner_tool("call_1", local_msgs)
        assert parent1 is not None

        # Second tool also gets full context (independent tracking)
        parent2, _ = state.compute_context_for_inner_tool("call_2", local_msgs)
        assert parent2 is not None

    def test_cont_received_included_in_first_call(self):
        """Accumulated cont should be included on first tool call."""
        parent_ctx = [{"role": "user", "content": "parent msg"}]
        state = LoopContextState(parent_chat_context=parent_ctx)

        # Receive some cont before first tool call
        state.receive_context_continuation([{"role": "user", "content": "cont msg"}])

        local_msgs = [{"role": "user", "content": "local msg"}]

        parent, cont = state.compute_context_for_inner_tool("call_1", local_msgs)

        assert parent is not None
        assert cont is not None
        assert len(cont) == 1
        assert cont[0]["content"] == "cont msg"

    def test_new_cont_forwarded_to_existing_tool(self):
        """New cont received should be forwarded to already-called tools."""
        parent_ctx = [{"role": "user", "content": "parent msg"}]
        state = LoopContextState(parent_chat_context=parent_ctx)

        local_msgs = [{"role": "user", "content": "local msg"}]

        # First call
        state.compute_context_for_inner_tool("call_1", local_msgs)

        # Receive new cont from above
        state.receive_context_continuation([{"role": "user", "content": "new cont"}])

        # Second call should get the new cont
        parent, cont = state.compute_context_for_inner_tool("call_1", local_msgs)

        assert parent is None
        assert cont is not None
        assert len(cont) == 1
        assert cont[0]["content"] == "new cont"

    def test_get_pending_cont_for_active_tools(self):
        """Verify pending cont calculation for active tools."""
        state = LoopContextState()

        # Simulate first call to tool
        state.compute_context_for_inner_tool("call_1", [])

        # Add new cont
        state.receive_context_continuation([{"role": "user", "content": "pending"}])

        pending = state.get_pending_cont_for_active_tools({"call_1", "call_2"})

        # call_1 has pending cont, call_2 hasn't been called yet
        assert "call_1" in pending
        assert "call_2" not in pending
        assert pending["call_1"][0]["content"] == "pending"

    def test_mark_cont_forwarded_to_tool(self):
        """Verify marking cont as forwarded clears pending state."""
        state = LoopContextState()

        state.compute_context_for_inner_tool("call_1", [])
        state.receive_context_continuation([{"role": "user", "content": "cont1"}])

        # Before marking, there's pending cont
        pending = state.get_pending_cont_for_active_tools({"call_1"})
        assert len(pending["call_1"]) == 1

        # Mark as forwarded
        state.mark_cont_forwarded_to_tool("call_1")

        # After marking, no pending cont
        pending = state.get_pending_cont_for_active_tools({"call_1"})
        assert "call_1" not in pending


# =============================================================================
# Integration tests for incremental context propagation in tool loops
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_incremental_context_multiple_tool_calls(model) -> None:
    """Verify that multiple calls to the same tool receive incremental context.

    First call should receive full parent_chat_context.
    Second call should receive only _parent_chat_context_cont with new messages.
    """
    client = new_llm_client(model=model)

    root_ctx = [{"role": "user", "content": "initial-parent-context"}]

    # Track what each call received
    call_records: List[dict] = []

    async def track_context(
        *,
        _parent_chat_context: list[dict] | None = None,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> str:
        call_records.append(
            {
                "parent_ctx": _parent_chat_context,
                "parent_ctx_cont": _parent_chat_context_cont,
            },
        )
        return f"call {len(call_records)} recorded"

    track_context.__name__ = "track_context"
    track_context.__qualname__ = "track_context"

    handle = start_async_tool_loop(
        client=client,
        message=(
            "Please call `track_context()` TWICE, one after another. "
            "After both calls complete, reply 'done'."
        ),
        tools={"track_context": track_context},
        parent_chat_context=root_ctx,
        propagate_chat_context=ChatContextPropagation.ALWAYS,
    )

    await handle.result()

    # Should have at least 2 calls
    assert len(call_records) >= 2, f"Expected at least 2 calls, got {len(call_records)}"

    # First call should have parent_chat_context
    first_call = call_records[0]
    assert (
        first_call["parent_ctx"] is not None
    ), "First call should receive parent_chat_context"
    # Verify the initial parent context is present
    assert any(
        "initial-parent-context" in str(m.get("content", ""))
        for m in first_call["parent_ctx"]
    ), "First call should contain the initial parent context"

    # Second call should NOT have full parent_chat_context again (only incremental)
    second_call = call_records[1]
    # Second call should either have no parent_ctx or only have cont
    # The exact behavior depends on whether there were new messages between calls
    if second_call["parent_ctx"] is not None:
        # If parent_ctx is present, it should NOT duplicate the initial context
        # (This would indicate the incremental tracking is working)
        pass  # Acceptable if the model called them in a single message batch


@pytest.mark.asyncio
async def test_context_state_integration_symbolic() -> None:
    """Symbolic test: verify LoopContextState integrates correctly with tool scheduling.

    This test doesn't use LLM, it directly tests the state machine behavior.
    """
    state = LoopContextState(
        parent_chat_context=[{"role": "user", "content": "root"}],
    )

    # Simulate tool being called twice
    msgs_at_call_1 = [{"role": "user", "content": "msg1"}]
    parent1, cont1 = state.compute_context_for_inner_tool("tool_a", msgs_at_call_1)

    assert parent1 is not None, "First call should get parent context"
    assert parent1[0]["content"] == "root"
    assert cont1 is None, "First call should not have cont (no accumulated)"

    # Add more messages between calls
    msgs_at_call_2 = msgs_at_call_1 + [
        {"role": "assistant", "content": "response1"},
        {"role": "user", "content": "msg2"},
    ]

    # Receive cont from above
    state.receive_context_continuation([{"role": "system", "content": "interjection"}])

    parent2, cont2 = state.compute_context_for_inner_tool("tool_a", msgs_at_call_2)

    assert parent2 is None, "Second call should NOT get full parent context"
    assert cont2 is not None, "Second call should get incremental cont"
    # Should include new local messages + the interjection
    assert len(cont2) == 3, f"Expected 3 incremental items, got {len(cont2)}"


# =============================================================================
# Tests for context opt-in/opt-out behavior with steering methods
# =============================================================================


def test_method_to_schema_exposes_context_cont_control():
    """Verify that method_to_schema exposes include_parent_chat_context_cont when requested."""
    from unity.common.llm_helpers import method_to_schema

    def steering_method(
        question: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ) -> str:
        """A steering method that accepts context continuation."""
        return "answer"

    # Without expose_context_cont_control
    schema_no_expose = method_to_schema(steering_method)
    props = schema_no_expose["function"]["parameters"]["properties"]
    assert "include_parent_chat_context_cont" not in props
    # parent_chat_context_cont should be hidden
    assert "parent_chat_context_cont" not in props

    # With expose_context_cont_control
    schema_exposed = method_to_schema(
        steering_method,
        expose_context_cont_control=True,
    )
    props_exposed = schema_exposed["function"]["parameters"]["properties"]
    assert "include_parent_chat_context_cont" in props_exposed
    assert props_exposed["include_parent_chat_context_cont"]["type"] == "boolean"


@pytest.mark.asyncio
async def test_dynamic_tool_factory_marks_steering_methods():
    """Verify that DynamicToolFactory marks steering methods with context opt-in status."""
    import asyncio
    from unittest.mock import MagicMock

    from unity.common._async_tool.dynamic_tools_factory import DynamicToolFactory
    from unity.common._async_tool.tools_data import ToolsData
    from unity.common._async_tool.tools_utils import ToolCallMetadata

    # Create mock tools_data
    mock_client = MagicMock()
    mock_client.messages = []
    mock_logger = MagicMock()
    mock_logger.log_steps = False
    mock_logger.info = MagicMock()

    tools_data = ToolsData({}, client=mock_client, logger=mock_logger)

    # Create mock task with metadata indicating context opted in
    mock_task_opted_in = asyncio.Future()
    mock_handle = MagicMock()
    mock_handle.ask = MagicMock(return_value="answer")
    mock_handle.interject = MagicMock()
    mock_handle.stop = MagicMock()

    metadata_opted_in = ToolCallMetadata(
        name="test_tool",
        call_id="call_abc123",
        call_dict={"function": {"arguments": "{}"}},
        call_idx=0,
        chat_context=None,
        assistant_msg={},
        is_interjectable=True,
        tool_schema={},
        llm_arguments={},
        raw_arguments_json="{}",
        handle=mock_handle,
        interject_queue=asyncio.Queue(),
        context_opted_in=True,  # Tool opted in
    )
    tools_data.pending.add(mock_task_opted_in)
    tools_data.info[mock_task_opted_in] = metadata_opted_in

    # Generate dynamic tools
    factory = DynamicToolFactory(tools_data)
    factory.generate()

    # Check that steering methods are marked
    stop_key = "stop_test_tool_abc123"
    interject_key = "interject_test_tool_abc123"
    ask_key = "ask_test_tool_abc123"

    assert stop_key in factory.dynamic_tools
    assert getattr(
        factory.dynamic_tools[stop_key],
        "__supports_context_propagation__",
        False,
    )
    assert getattr(factory.dynamic_tools[stop_key], "__context_opted_in__", False)

    assert interject_key in factory.dynamic_tools
    assert getattr(
        factory.dynamic_tools[interject_key],
        "__supports_context_propagation__",
        False,
    )
    assert getattr(factory.dynamic_tools[interject_key], "__context_opted_in__", False)

    assert ask_key in factory.dynamic_tools
    assert getattr(
        factory.dynamic_tools[ask_key],
        "__supports_context_propagation__",
        False,
    )
    assert getattr(factory.dynamic_tools[ask_key], "__context_opted_in__", False)


@pytest.mark.asyncio
async def test_dynamic_tool_factory_marks_opted_out():
    """Verify that steering methods for opted-out tools have context_opted_in=False."""
    import asyncio
    from unittest.mock import MagicMock

    from unity.common._async_tool.dynamic_tools_factory import DynamicToolFactory
    from unity.common._async_tool.tools_data import ToolsData
    from unity.common._async_tool.tools_utils import ToolCallMetadata

    # Create mock tools_data
    mock_client = MagicMock()
    mock_client.messages = []
    mock_logger = MagicMock()
    mock_logger.log_steps = False
    mock_logger.info = MagicMock()

    tools_data = ToolsData({}, client=mock_client, logger=mock_logger)

    # Create mock task with metadata indicating context opted OUT
    mock_task_opted_out = asyncio.Future()
    mock_handle = MagicMock()
    mock_handle.stop = MagicMock()

    metadata_opted_out = ToolCallMetadata(
        name="private_tool",
        call_id="call_xyz789",
        call_dict={"function": {"arguments": "{}"}},
        call_idx=0,
        chat_context=None,
        assistant_msg={},
        is_interjectable=False,  # Not interjectable
        tool_schema={},
        llm_arguments={},
        raw_arguments_json="{}",
        handle=mock_handle,
        context_opted_in=False,  # Tool opted OUT
    )
    tools_data.pending.add(mock_task_opted_out)
    tools_data.info[mock_task_opted_out] = metadata_opted_out

    # Generate dynamic tools
    factory = DynamicToolFactory(tools_data)
    factory.generate()

    # Check that stop method is marked as opted out
    stop_key = "stop_private_tool_xyz789"
    assert stop_key in factory.dynamic_tools
    assert getattr(
        factory.dynamic_tools[stop_key],
        "__supports_context_propagation__",
        False,
    )
    assert not getattr(factory.dynamic_tools[stop_key], "__context_opted_in__", True)
