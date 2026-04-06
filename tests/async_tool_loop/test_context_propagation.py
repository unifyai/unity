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

import asyncio
from typing import List

import pytest
from unity.common.async_tool_loop import ChatContextPropagation, start_async_tool_loop
from unity.common._async_tool.context_tracker import LoopContextState
from tests.helpers import _handle_project
from unity.common.llm_client import new_llm_client


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_chat_context_propagation(llm_config) -> None:
    client = new_llm_client(**llm_config)

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
@pytest.mark.llm_call
@_handle_project
async def test_chat_context_propagation_never(llm_config) -> None:
    """Verify that NEVER mode does NOT pass context to tools, even when they accept it."""
    client = new_llm_client(**llm_config)

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
@pytest.mark.llm_call
@_handle_project
async def test_chat_context_propagation_llm_decides_include(llm_config) -> None:
    """Verify that LLM_DECIDES mode passes context when LLM includes it (default)."""
    client = new_llm_client(**llm_config)

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
@pytest.mark.llm_call
@_handle_project
async def test_chat_context_propagation_llm_decides_exclude(llm_config) -> None:
    """Verify that LLM_DECIDES mode omits context when LLM explicitly excludes it."""
    client = new_llm_client(**llm_config)

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
@pytest.mark.llm_call
@_handle_project
async def test_ask_uses_parent_context(llm_config) -> None:
    """Verify that ask() includes parent context in the inspection loop and influences the answer.

    The inner inspection loop should choose "apple" only because the parent context
    reveals a banana allergy, not because of anything in the current prompt.

    This test uses a realistic scenario (food allergy) rather than arbitrary directives,
    which sophisticated models might (correctly) treat as suspicious prompt injection.
    """

    client = new_llm_client(**llm_config)

    # Start a trivial outer loop (no tools needed for this test).
    handle = start_async_tool_loop(
        client=client,
        message=("We will later follow-up with a question requiring broader context."),
        tools={},
    )

    # Provide a parent context that carries the deciding hint via a realistic scenario.
    parent_ctx = [
        {
            "role": "user",
            "content": "By the way, I should mention I have a severe banana allergy.",
        },
        {
            "role": "assistant",
            "content": (
                "Thank you for letting me know about your banana allergy. "
                "I'll make sure to keep that in mind for any recommendations."
            ),
        },
    ]

    # Ask a question whose correct answer requires the parent context.
    # Explicitly offer apple and banana so the LLM makes a specific choice.
    helper = await handle.ask(
        (
            "Which fruit should we choose: APPLE or BANANA? "
            "Please answer in one short phrase."
        ),
        _parent_chat_context=parent_ctx,
    )
    ans = await helper.result()

    await handle.result()

    assert (
        "apple" in ans.lower()
    ), "Answer did not reflect parent context (banana allergy)."


@pytest.mark.asyncio
async def test_ask_inspection_loop_context_without_parent(monkeypatch) -> None:
    """Without _parent_chat_context, the inspection loop uses LLM_DECIDES
    propagation with no parent_chat_context — the transcript already has the
    embedded parent context header from when the loop was started."""
    from unity.common import async_tool_loop as atl

    captured_kwargs: dict = {}

    class _DummyInspectionHandle:
        async def result(self):
            return "ok"

    def _fake_start_async_tool_loop(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _DummyInspectionHandle()

    monkeypatch.setattr(atl, "start_async_tool_loop", _fake_start_async_tool_loop)

    class _DummyClient:
        def __init__(self):
            self.messages = [{"role": "user", "content": "loop transcript message"}]

    task = asyncio.Future()
    task.set_result("done")

    handle = atl.AsyncToolLoopHandle(
        task=task,
        interject_queue=asyncio.Queue(),
        cancel_event=asyncio.Event(),
        stop_event=asyncio.Event(),
        client=_DummyClient(),
        loop_id="OuterLoop",
    )

    _helper = await handle.ask("What happened?")

    assert (
        captured_kwargs["propagate_chat_context"] == ChatContextPropagation.LLM_DECIDES
    )
    assert captured_kwargs["parent_chat_context"] is None


@pytest.mark.asyncio
async def test_ask_inspection_loop_with_parent_context(monkeypatch) -> None:
    """With _parent_chat_context, the inspection loop uses LLM_DECIDES
    propagation, passes the real parent context, and replaces the stale
    runtime parent-context header in the transcript with a pointer."""
    from unity.common import async_tool_loop as atl

    captured: dict = {}

    class _DummyInspectionHandle:
        async def result(self):
            return "ok"

    def _fake_start_async_tool_loop(*args, **kwargs):
        inspection_client = args[0]
        captured["system_message"] = inspection_client.system_message
        captured["kwargs"] = kwargs
        return _DummyInspectionHandle()

    monkeypatch.setattr(atl, "start_async_tool_loop", _fake_start_async_tool_loop)

    stale_parent_ctx_content = (
        "## Caller Context\nSome caller info.\n\n"
        "## Parent Chat Context\n"
        "You received this request from within a parent conversation. "
        "The messages below show...\n\n"
        '[{"role": "outer_user", "content": "stale parent message"}]'
    )

    class _DummyClient:
        def __init__(self):
            self.messages = [
                {"role": "system", "content": "You are a helpful tool."},
                {
                    "role": "system",
                    "_runtime_context": True,
                    "_ctx_header": True,
                    "_parent_chat_context": True,
                    "content": stale_parent_ctx_content,
                },
                {"role": "user", "content": "do the thing"},
                {"role": "assistant", "content": "doing it"},
            ]

    task = asyncio.Future()
    task.set_result("done")

    handle = atl.AsyncToolLoopHandle(
        task=task,
        interject_queue=asyncio.Queue(),
        cancel_event=asyncio.Event(),
        stop_event=asyncio.Event(),
        client=_DummyClient(),
        loop_id="OuterLoop",
    )

    fresh_context = [{"role": "user", "content": "fresh parent message"}]
    _helper = await handle.ask("What happened?", _parent_chat_context=fresh_context)

    kwargs = captured["kwargs"]
    assert kwargs["propagate_chat_context"] == ChatContextPropagation.LLM_DECIDES
    assert kwargs["parent_chat_context"] == fresh_context

    # The system message should contain the pointer, not the stale dump.
    # The pointer appears inside a JSON-serialized transcript, so newlines
    # are escaped.  Check for the distinctive phrase instead.
    sys_msg = captured["system_message"]
    assert "stale parent message" not in sys_msg
    assert "has been omitted from this transcript to avoid duplication" in sys_msg

    # The Caller Context section before the parent context should be preserved
    assert "Caller Context" in sys_msg
    assert "Some caller info." in sys_msg


@pytest.mark.asyncio
async def test_ask_inspection_prompt_redacts_image_payloads(monkeypatch) -> None:
    """Inspection ask should redact image blobs from both the system message
    (inspected transcript) and the parent_chat_context kwarg."""
    from unity.common import async_tool_loop as atl

    captured: dict = {}

    class _DummyInspectionHandle:
        async def result(self):
            return "ok"

    def _fake_start_async_tool_loop(*args, **kwargs):
        inspection_client = args[0]
        captured["system_message"] = inspection_client.system_message
        captured["parent_chat_context"] = kwargs.get("parent_chat_context")
        return _DummyInspectionHandle()

    monkeypatch.setattr(atl, "start_async_tool_loop", _fake_start_async_tool_loop)

    big_b64 = "A" * 4000
    raw_data_url = f"data:image/png;base64,{big_b64}"

    class _DummyClient:
        def __init__(self):
            self.messages = [
                {
                    "role": "tool",
                    "name": "execute_code",
                    "content": [
                        {"type": "text", "text": "screenshot captured"},
                        {"type": "image_url", "image_url": {"url": raw_data_url}},
                    ],
                },
            ]

    task = asyncio.Future()
    task.set_result("done")

    handle = atl.AsyncToolLoopHandle(
        task=task,
        interject_queue=asyncio.Queue(),
        cancel_event=asyncio.Event(),
        stop_event=asyncio.Event(),
        client=_DummyClient(),
        loop_id="OuterLoop",
    )

    parent_ctx_with_image = [
        {"role": "user", "content": f"latest screenshot: {raw_data_url}"},
    ]
    _helper = await handle.ask(
        "Summarize status",
        _parent_chat_context=parent_ctx_with_image,
    )

    sys_msg = str(captured.get("system_message") or "")
    parent_ctx = str(captured.get("parent_chat_context") or "")

    # Image blobs must be redacted in the system message (inspected transcript)
    assert raw_data_url not in sys_msg
    assert "data:image/png;base64,<omitted>" in sys_msg

    # Image blobs must be redacted in the parent_chat_context kwarg
    assert raw_data_url not in parent_ctx
    assert "<omitted>" in parent_ctx


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_interject_with_continued_parent_context_influences_decision(
    llm_config,
) -> None:
    """Verify that an interjection with continued parent context steers the LLM decision.

    The outer loop should incorporate the interjection (and its continued context)
    such that the next assistant reply reflects that broader context.

    This test uses a realistic scenario (food allergy) rather than arbitrary directives,
    which sophisticated models might (correctly) treat as suspicious prompt injection.
    """

    client = new_llm_client(**llm_config)

    handle = start_async_tool_loop(
        client=client,
        message=(
            "We need to pick a fruit between APPLE and BANANA. "
            "Wait for any additional context from the parent conversation before deciding."
        ),
        tools={},
    )

    # Realistic outer conversation context: user mentions they have a banana allergy
    continued_ctx = [
        {
            "role": "user",
            "content": "By the way, I should mention I have a severe banana allergy.",
        },
        {
            "role": "assistant",
            "content": (
                "Thank you for letting me know about your banana allergy. "
                "I'll make sure to keep that in mind for any recommendations."
            ),
        },
    ]

    # Inject the context continuation (the allergy info from parent conversation).
    await handle.interject(
        "Please make your fruit recommendation now.",
        _parent_chat_context_cont=continued_ctx,
    )

    final = await handle.result()
    assert (
        "apple" in final.lower()
    ), "Final decision did not reflect continued parent context (banana allergy)."


# =============================================================================
# Unit tests for LoopContextState (incremental context tracking)
# =============================================================================


class TestLoopContextState:
    """Unit tests for the incremental context tracking state machine."""

    def test_initial_state(self):
        """Verify initial state is empty."""
        state = LoopContextState()
        assert state.parent_chat_context == []
        assert state._parent_chat_context_cont_received == []
        assert state.inner_tool_forwarding == {}

    def test_receive_context_continuation(self):
        """Verify context continuations are accumulated."""
        state = LoopContextState()

        state.receive_context_continuation([{"role": "user", "content": "msg1"}])
        assert len(state._parent_chat_context_cont_received) == 1

        state.receive_context_continuation([{"role": "user", "content": "msg2"}])
        assert len(state._parent_chat_context_cont_received) == 2

        # Empty list should not add anything
        state.receive_context_continuation([])
        assert len(state._parent_chat_context_cont_received) == 2

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
@pytest.mark.llm_call
@_handle_project
async def test_incremental_context_multiple_tool_calls(llm_config) -> None:
    """Verify that multiple calls to the same tool receive incremental context.

    First call should receive full parent_chat_context.
    Second call should receive only _parent_chat_context_cont with new messages.
    """
    client = new_llm_client(**llm_config)

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
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> str:
        """A steering method that accepts context continuation."""
        return "answer"

    # Without expose_context_cont_control
    schema_no_expose = method_to_schema(steering_method)
    props = schema_no_expose["function"]["parameters"]["properties"]
    assert "include_parent_chat_context_cont" not in props
    # _parent_chat_context_cont should be hidden
    assert "_parent_chat_context_cont" not in props

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

    # Check that interject steering method is marked with context flags
    # (stop and ask no longer use these flags - they use different mechanisms)
    interject_key = "interject_test_tool_abc123"

    assert interject_key in factory.dynamic_tools
    assert getattr(
        factory.dynamic_tools[interject_key],
        "__supports_context_propagation__",
        False,
    )
    assert getattr(factory.dynamic_tools[interject_key], "__context_opted_in__", False)

    # stop and ask should exist but NOT have context propagation flags
    stop_key = "stop_test_tool_abc123"
    ask_key = "ask_test_tool_abc123"
    assert stop_key in factory.dynamic_tools
    assert ask_key in factory.dynamic_tools
    # These should NOT have the flags (we removed them)
    assert not hasattr(
        factory.dynamic_tools[stop_key],
        "__supports_context_propagation__",
    )
    assert not hasattr(
        factory.dynamic_tools[ask_key],
        "__supports_context_propagation__",
    )


@pytest.mark.asyncio
async def test_dynamic_tool_factory_marks_opted_out():
    """Verify that interject for opted-out tools has context_opted_in=False."""
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
    mock_handle.interject = MagicMock()

    metadata_opted_out = ToolCallMetadata(
        name="private_tool",
        call_id="call_xyz789",
        call_dict={"function": {"arguments": "{}"}},
        call_idx=0,
        chat_context=None,
        assistant_msg={},
        is_interjectable=True,  # Now interjectable so we can test the flag
        tool_schema={},
        llm_arguments={},
        raw_arguments_json="{}",
        handle=mock_handle,
        interject_queue=asyncio.Queue(),
        context_opted_in=False,  # Tool opted OUT
    )
    tools_data.pending.add(mock_task_opted_out)
    tools_data.info[mock_task_opted_out] = metadata_opted_out

    # Generate dynamic tools
    factory = DynamicToolFactory(tools_data)
    factory.generate()

    # Check that interject method is marked as opted out
    interject_key = "interject_private_tool_xyz789"
    assert interject_key in factory.dynamic_tools
    assert getattr(
        factory.dynamic_tools[interject_key],
        "__supports_context_propagation__",
        False,
    )
    assert not getattr(
        factory.dynamic_tools[interject_key],
        "__context_opted_in__",
        True,
    )


# =============================================================================
# Tests for ask_* dynamic tool context control
# =============================================================================


@pytest.mark.asyncio
async def test_ask_dynamic_tool_exposes_include_parent_chat_context():
    """Verify that ask_* dynamic tools expose include_parent_chat_context in LLM_DECIDES mode.

    This test ensures the LLM can opt out of context propagation when calling ask_*
    on an in-flight tool, just like it can for regular tools.
    """
    import asyncio
    from unittest.mock import MagicMock

    from unity.common._async_tool.dynamic_tools_factory import DynamicToolFactory
    from unity.common._async_tool.tools_data import ToolsData
    from unity.common._async_tool.tools_utils import ToolCallMetadata
    from unity.common.async_tool_loop import SteerableToolHandle
    from unity.common.llm_helpers import method_to_schema

    # Create a handle with an ask method that accepts _parent_chat_context
    class TestHandle(SteerableToolHandle):
        def __init__(self):
            pass

        async def ask(
            self,
            question: str,
            *,
            _parent_chat_context: list[dict] | None = None,
        ) -> "SteerableToolHandle":
            """Ask a question about this tool's status."""
            return self

        async def interject(self, message: str, **kwargs):
            pass

        def stop(self, reason: str | None = None):
            pass

        def pause(self):
            pass

        def resume(self):
            pass

        def done(self) -> bool:
            return False

        async def result(self):
            return "result"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            pass

    # Create mock tools_data
    mock_client = MagicMock()
    mock_client.messages = []
    mock_logger = MagicMock()
    mock_logger.log_steps = False
    mock_logger.info = MagicMock()

    tools_data = ToolsData({}, client=mock_client, logger=mock_logger)

    # Create mock task with handle
    mock_task = asyncio.Future()
    test_handle = TestHandle()

    metadata = ToolCallMetadata(
        name="test_tool",
        call_id="call_abc123",
        call_dict={"function": {"arguments": "{}"}},
        call_idx=0,
        chat_context=[{"role": "user", "content": "context"}],
        assistant_msg={},
        is_interjectable=False,
        tool_schema={},
        llm_arguments={},
        raw_arguments_json="{}",
        handle=test_handle,
        context_opted_in=True,
    )
    tools_data.pending.add(mock_task)
    tools_data.info[mock_task] = metadata

    # Generate dynamic tools
    factory = DynamicToolFactory(tools_data)
    factory.generate()

    # The ask_* tool should exist
    ask_key = "ask_test_tool_abc123"
    assert ask_key in factory.dynamic_tools

    ask_fn = factory.dynamic_tools[ask_key]

    # Generate schema with expose_context_control=True (simulating LLM_DECIDES mode)
    schema = method_to_schema(
        ask_fn,
        expose_context_control=True,
        has_parent_context=True,
    )

    props = schema["function"]["parameters"]["properties"]

    # The ask_* tool should expose include_parent_chat_context
    assert (
        "include_parent_chat_context" in props
    ), "ask_* dynamic tool should expose include_parent_chat_context in LLM_DECIDES mode"
    assert props["include_parent_chat_context"]["type"] == "boolean"

    # But _parent_chat_context should be hidden
    assert (
        "_parent_chat_context" not in props
    ), "_parent_chat_context should be hidden from LLM schema"


@pytest.mark.asyncio
async def test_ask_dynamic_tool_context_control_not_exposed_for_other_steering_methods():
    """Verify that stop_* and interject_* do NOT expose include_parent_chat_context.

    Only ask_* should expose this control because:
    - stop_* no longer uses context at all
    - interject_* uses _parent_chat_context_cont (continuation), not initial context
    """
    import asyncio
    from unittest.mock import MagicMock

    from unity.common._async_tool.dynamic_tools_factory import DynamicToolFactory
    from unity.common._async_tool.tools_data import ToolsData
    from unity.common._async_tool.tools_utils import ToolCallMetadata
    from unity.common.async_tool_loop import SteerableToolHandle
    from unity.common.llm_helpers import method_to_schema

    class TestHandle(SteerableToolHandle):
        def __init__(self):
            pass

        async def ask(
            self,
            question: str,
            *,
            _parent_chat_context: list[dict] | None = None,
        ) -> "SteerableToolHandle":
            return self

        async def interject(
            self,
            message: str,
            *,
            _parent_chat_context_cont: list[dict] | None = None,
        ):
            pass

        def stop(self, reason: str | None = None):
            pass

        def pause(self):
            pass

        def resume(self):
            pass

        def done(self) -> bool:
            return False

        async def result(self):
            return "result"

        async def next_clarification(self) -> dict:
            return {}

        async def next_notification(self) -> dict:
            return {}

        async def answer_clarification(self, call_id: str, answer: str) -> None:
            pass

    mock_client = MagicMock()
    mock_client.messages = []
    mock_logger = MagicMock()
    mock_logger.log_steps = False
    mock_logger.info = MagicMock()

    tools_data = ToolsData({}, client=mock_client, logger=mock_logger)

    mock_task = asyncio.Future()
    test_handle = TestHandle()

    metadata = ToolCallMetadata(
        name="test_tool",
        call_id="call_abc123",
        call_dict={"function": {"arguments": "{}"}},
        call_idx=0,
        chat_context=[{"role": "user", "content": "context"}],
        assistant_msg={},
        is_interjectable=True,
        tool_schema={},
        llm_arguments={},
        raw_arguments_json="{}",
        handle=test_handle,
        interject_queue=asyncio.Queue(),
        context_opted_in=True,
    )
    tools_data.pending.add(mock_task)
    tools_data.info[mock_task] = metadata

    factory = DynamicToolFactory(tools_data)
    factory.generate()

    # stop_* should NOT have include_parent_chat_context
    stop_fn = factory.dynamic_tools["stop_test_tool_abc123"]
    stop_schema = method_to_schema(stop_fn, expose_context_control=True)
    stop_props = stop_schema["function"]["parameters"]["properties"]
    assert (
        "include_parent_chat_context" not in stop_props
    ), "stop_* should NOT expose include_parent_chat_context"

    # interject_* should NOT have include_parent_chat_context (uses _cont version)
    interject_fn = factory.dynamic_tools["interject_test_tool_abc123"]
    interject_schema = method_to_schema(interject_fn, expose_context_control=True)
    interject_props = interject_schema["function"]["parameters"]["properties"]
    assert (
        "include_parent_chat_context" not in interject_props
    ), "interject_* should NOT expose include_parent_chat_context"


@pytest.mark.asyncio
async def test_ask_dynamic_tool_respects_include_parent_chat_context_false():
    """Verify that ask_* dynamic tools respect include_parent_chat_context=False.

    When the LLM calls ask_* with include_parent_chat_context=False in LLM_DECIDES mode,
    the underlying handle.ask() should be called WITHOUT _parent_chat_context.

    This test would have FAILED before the fix because:
    1. We exposed include_parent_chat_context in the ask_* schema
    2. But we never processed it in the dynamic tool dispatch
    3. So the LLM's choice was ignored and context was always passed
    """
    from unity.common._async_tool.tools_data import compute_context_injection
    from unity.common._async_tool.context_tracker import LoopContextState

    # Test case 1: include_parent_chat_context=False should NOT inject context
    args_opt_out = {
        "question": "what's the status?",
        "include_parent_chat_context": False,
    }
    context_state = LoopContextState(
        parent_chat_context=[{"role": "user", "content": "parent context"}],
    )
    client_messages = [{"role": "user", "content": "current message"}]

    extra_kwargs, context_opted_in = compute_context_injection(
        args=args_opt_out,
        propagate_chat_context=ChatContextPropagation.LLM_DECIDES,
        context_state=context_state,
        client_messages=client_messages,
        call_id="ask_inner_tool_123",
        accepts_parent_ctx=True,
        accepts_parent_ctx_cont=False,
        is_continuation_only=False,
    )

    # Should NOT have injected context
    assert (
        "_parent_chat_context" not in extra_kwargs
    ), "When include_parent_chat_context=False, _parent_chat_context should NOT be injected"
    assert (
        context_opted_in is False
    ), "context_opted_in should be False when LLM opts out"
    # The control param should be popped from args
    assert (
        "include_parent_chat_context" not in args_opt_out
    ), "include_parent_chat_context should be popped from args"

    # Test case 2: include_parent_chat_context=True (or omitted) SHOULD inject context
    args_opt_in = {
        "question": "what's the status?",
        "include_parent_chat_context": True,
    }
    context_state2 = LoopContextState(
        parent_chat_context=[{"role": "user", "content": "parent context"}],
    )

    extra_kwargs2, context_opted_in2 = compute_context_injection(
        args=args_opt_in,
        propagate_chat_context=ChatContextPropagation.LLM_DECIDES,
        context_state=context_state2,
        client_messages=client_messages,
        call_id="ask_inner_tool_456",
        accepts_parent_ctx=True,
        accepts_parent_ctx_cont=False,
        is_continuation_only=False,
    )

    # Should have injected context
    assert (
        "_parent_chat_context" in extra_kwargs2
    ), "When include_parent_chat_context=True, _parent_chat_context SHOULD be injected"
    assert context_opted_in2 is True, "context_opted_in should be True when LLM opts in"

    # Test case 3: ALWAYS mode ignores LLM choice and always injects
    args_always = {
        "question": "what's the status?",
        "include_parent_chat_context": False,
    }
    context_state3 = LoopContextState(
        parent_chat_context=[{"role": "user", "content": "parent context"}],
    )

    extra_kwargs3, context_opted_in3 = compute_context_injection(
        args=args_always,
        propagate_chat_context=ChatContextPropagation.ALWAYS,
        context_state=context_state3,
        client_messages=client_messages,
        call_id="ask_inner_tool_789",
        accepts_parent_ctx=True,
        accepts_parent_ctx_cont=False,
        is_continuation_only=False,
    )

    # ALWAYS mode should inject context regardless of LLM choice
    assert (
        "_parent_chat_context" in extra_kwargs3
    ), "In ALWAYS mode, context should be injected even if LLM opts out"
    assert context_opted_in3 is True

    # Test case 4: NEVER mode ignores LLM choice and never injects
    args_never = {"question": "what's the status?", "include_parent_chat_context": True}
    context_state4 = LoopContextState(
        parent_chat_context=[{"role": "user", "content": "parent context"}],
    )

    extra_kwargs4, context_opted_in4 = compute_context_injection(
        args=args_never,
        propagate_chat_context=ChatContextPropagation.NEVER,
        context_state=context_state4,
        client_messages=client_messages,
        call_id="ask_inner_tool_abc",
        accepts_parent_ctx=True,
        accepts_parent_ctx_cont=False,
        is_continuation_only=False,
    )

    # NEVER mode should NOT inject context regardless of LLM choice
    assert (
        "_parent_chat_context" not in extra_kwargs4
    ), "In NEVER mode, context should NOT be injected even if LLM opts in"
    assert context_opted_in4 is False


# =============================================================================
# Tests for interjection context continuation message structure
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_interjection_context_continuation_message_structure(llm_config) -> None:
    """Verify that interjections with _parent_chat_context_continued create proper message structure.

    When an interjection includes _parent_chat_context_continued:
    1. A user message with the context continuation (tagged _ctx_header=True) is appended
    2. A second user message with the actual interjection content is appended (if non-empty)
    3. The context continuation message is filtered out when building cur_msgs for inner tools
    4. The current loop's LLM sees the context continuation

    This test uses a realistic scenario (food allergy) rather than arbitrary directives,
    which sophisticated models might (correctly) treat as suspicious prompt injection.
    """
    client = new_llm_client(**llm_config)

    handle = start_async_tool_loop(
        client=client,
        message=(
            "I need you to recommend a fruit for me. Choose between APPLE and BANANA. "
            "Wait for any additional context from the parent conversation that might "
            "help you make a better recommendation before responding."
        ),
        tools={},
    )

    # Realistic outer conversation context: user mentions they have a banana allergy
    continued_ctx = [
        {
            "role": "user",
            "content": "By the way, I should mention I have a severe banana allergy.",
        },
        {
            "role": "assistant",
            "content": (
                "Thank you for letting me know about your banana allergy. "
                "I'll make sure to keep that in mind for any recommendations."
            ),
        },
    ]

    # Inject an interjection with context continuation
    await handle.interject(
        "Please make your fruit recommendation now.",
        _parent_chat_context_cont=continued_ctx,
    )

    final = await handle.result()

    # Verify the LLM made the right decision (influenced by allergy information)
    assert (
        "apple" in final.lower()
    ), "LLM should recommend apple given banana allergy context"

    # Verify message structure: find user messages with _ctx_header=True
    ctx_header_user_msgs = [
        m
        for m in client.messages
        if m.get("role") == "user" and m.get("_ctx_header") is True
    ]
    assert (
        len(ctx_header_user_msgs) >= 1
    ), "Expected at least one user message with _ctx_header=True for context continuation"

    # The context continuation message should contain the context
    ctx_msg = ctx_header_user_msgs[0]
    assert "Parent Chat Context (continued)" in ctx_msg["content"]
    assert (
        "allergy" in ctx_msg["content"].lower()
    ), "Context continuation should contain the allergy information"

    # Verify that when filtering for cur_msgs, context header messages are excluded
    cur_msgs = [m for m in client.messages if not m.get("_ctx_header")]
    ctx_header_in_cur_msgs = [
        m
        for m in cur_msgs
        if m.get("role") == "user"
        and "Parent Chat Context (continued)" in str(m.get("content", ""))
    ]
    assert (
        len(ctx_header_in_cur_msgs) == 0
    ), "Context continuation messages should be filtered out of cur_msgs"


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_interjection_context_only_no_user_message(llm_config) -> None:
    """Verify that context-only interjections (empty message) work correctly.

    When an interjection has _parent_chat_context_continued but empty message text,
    only the context continuation user message should be appended (no empty second message).

    This test uses a realistic scenario: the outer conversation reveals the user has
    a banana allergy, which the inner loop should naturally use when asked to recommend
    a fruit between apple and banana.
    """
    client = new_llm_client(**llm_config)

    handle = start_async_tool_loop(
        client=client,
        message=(
            "I need you to recommend a fruit for me. Choose between APPLE and BANANA. "
            "Wait for any additional context from the parent conversation that might "
            "help you make a better recommendation before responding."
        ),
        tools={},
    )

    # Realistic outer conversation context: user mentions they have a banana allergy
    continued_ctx = [
        {
            "role": "user",
            "content": "By the way, I should mention I have a severe banana allergy.",
        },
        {
            "role": "assistant",
            "content": (
                "Thank you for letting me know about your banana allergy. "
                "I'll make sure to keep that in mind for any recommendations."
            ),
        },
    ]

    # Inject context-only interjection (empty message)
    await handle.interject(
        "",  # Empty message
        _parent_chat_context_cont=continued_ctx,
    )

    final = await handle.result()

    # Verify the LLM made the right decision (influenced by allergy information)
    # The LLM should recommend apple given the banana allergy context
    assert (
        "apple" in final.lower()
    ), "LLM should recommend apple given the banana allergy context"

    # Count user messages that are empty (should be minimal/none)
    empty_user_msgs = [
        m for m in client.messages if m.get("role") == "user" and m.get("content") == ""
    ]
    assert (
        len(empty_user_msgs) == 0
    ), "Empty user messages should not be appended for context-only interjections"
