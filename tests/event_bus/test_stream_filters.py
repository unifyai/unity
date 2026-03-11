"""Unit tests for unity.events.stream_filters and ToolLoopKind classification.

All tests are synchronous, zero-IO, and test the predicate/classifier
functions directly.  No EventBus, Pub/Sub, or project setup is needed.
"""

import json

from unity.events.stream_filters import (
    is_streaming_noise,
    is_suppressed_manager_tree,
)
from unity.events.types.tool_loop import (
    ToolLoopKind,
    classify_tool_loop_message,
)

# ============================================================================
#  classify_tool_loop_message — assistant messages
# ============================================================================


def test_classify_thinking_sentinel():
    msg = {"role": "assistant", "_thinking_in_flight": True}
    assert classify_tool_loop_message(msg) == ToolLoopKind.THINKING_SENTINEL


def test_classify_thought_with_thinking_blocks():
    msg = {"role": "assistant", "thinking_blocks": [{"text": "hmm"}], "content": ""}
    assert classify_tool_loop_message(msg) == ToolLoopKind.THOUGHT


def test_classify_thought_with_reasoning_content():
    msg = {"role": "assistant", "reasoning_content": "Let me think...", "content": ""}
    assert classify_tool_loop_message(msg) == ToolLoopKind.THOUGHT


def test_classify_thought_with_provider_specific_thinking():
    msg = {
        "role": "assistant",
        "provider_specific_fields": {"thinking_blocks": [{"text": "..."}]},
        "content": "",
    }
    assert classify_tool_loop_message(msg) == ToolLoopKind.THOUGHT


def test_classify_tool_call():
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "toolu_xyz",
                "type": "function",
                "function": {"name": "web_search", "arguments": "{}"},
            },
        ],
    }
    assert classify_tool_loop_message(msg) == ToolLoopKind.TOOL_CALL


def test_classify_response():
    msg = {"role": "assistant", "content": "Here are the results..."}
    assert classify_tool_loop_message(msg) == ToolLoopKind.RESPONSE


def test_classify_empty_assistant_as_response():
    msg = {"role": "assistant", "content": ""}
    assert classify_tool_loop_message(msg) == ToolLoopKind.RESPONSE


# ============================================================================
#  classify_tool_loop_message — user messages
# ============================================================================


def test_classify_request():
    msg = {"role": "user", "content": "What is the price?"}
    assert classify_tool_loop_message(msg) == ToolLoopKind.REQUEST


def test_classify_interjection():
    msg = {"role": "user", "_interjection": True, "content": "Actually, also check..."}
    assert classify_tool_loop_message(msg) == ToolLoopKind.INTERJECTION


def test_classify_context_continuation():
    msg = {"role": "user", "_ctx_header": True, "content": "## Parent Chat Context..."}
    assert classify_tool_loop_message(msg) == ToolLoopKind.CONTEXT_CONTINUATION


# ============================================================================
#  classify_tool_loop_message — tool messages
# ============================================================================


def test_classify_tool_result():
    msg = {"role": "tool", "name": "web_search", "content": '{"answer": "42"}'}
    assert classify_tool_loop_message(msg) == ToolLoopKind.TOOL_RESULT


def test_classify_status_check_tool():
    msg = {
        "role": "tool",
        "name": "check_status_toolu_abc123",
        "content": '{"answer": "some result"}',
    }
    assert classify_tool_loop_message(msg) == ToolLoopKind.STATUS_CHECK


def test_classify_status_check_assistant():
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "toolu_abc123_completed",
                "type": "function",
                "function": {"name": "check_status_toolu_abc123", "arguments": "{}"},
            },
        ],
    }
    # check_status assistant stubs are classified as regular tool_calls;
    # the STATUS_CHECK kind only applies to the tool-role reply.
    assert classify_tool_loop_message(msg) == ToolLoopKind.TOOL_CALL


def test_classify_wait_noop():
    msg = {"role": "tool", "name": "wait", "content": ""}
    assert classify_tool_loop_message(msg) == ToolLoopKind.WAIT_NOOP


def test_classify_placeholder_pending():
    msg = {
        "role": "tool",
        "name": "some_tool",
        "content": json.dumps({"_placeholder": "pending"}),
    }
    assert classify_tool_loop_message(msg) == ToolLoopKind.PLACEHOLDER


def test_classify_placeholder_progress():
    msg = {
        "role": "tool",
        "name": "web_search",
        "content": json.dumps({"_placeholder": "progress", "tool": "web_search"}),
    }
    assert classify_tool_loop_message(msg) == ToolLoopKind.PLACEHOLDER


def test_classify_placeholder_completed():
    msg = {
        "role": "tool",
        "name": "some_tool",
        "content": json.dumps(
            {
                "_placeholder": "completed",
                "status": "Tool completed.",
                "result_call_id": "toolu_abc_completed",
            },
        ),
    }
    assert classify_tool_loop_message(msg) == ToolLoopKind.PLACEHOLDER


def test_classify_placeholder_nested_start():
    msg = {
        "role": "tool",
        "name": "some_tool",
        "content": json.dumps({"_placeholder": "nested_start"}),
    }
    assert classify_tool_loop_message(msg) == ToolLoopKind.PLACEHOLDER


def test_classify_tool_non_json_content():
    msg = {"role": "tool", "name": "web_search", "content": "Here are the results..."}
    assert classify_tool_loop_message(msg) == ToolLoopKind.TOOL_RESULT


def test_classify_tool_json_without_placeholder():
    msg = {
        "role": "tool",
        "name": "web_search",
        "content": json.dumps({"answer": "The RTX 5070 costs $549"}),
    }
    assert classify_tool_loop_message(msg) == ToolLoopKind.TOOL_RESULT


# ============================================================================
#  classify_tool_loop_message — system messages
# ============================================================================


def test_classify_steering_pause():
    msg = {"role": "system", "_steering": True, "_steering_action": "pause"}
    assert classify_tool_loop_message(msg) == ToolLoopKind.STEERING_PAUSE


def test_classify_steering_resume():
    msg = {"role": "system", "_steering": True, "_steering_action": "resume"}
    assert classify_tool_loop_message(msg) == ToolLoopKind.STEERING_RESUME


def test_classify_steering_stop():
    msg = {
        "role": "system",
        "_steering": True,
        "_steering_action": "stop",
        "content": "reason",
    }
    assert classify_tool_loop_message(msg) == ToolLoopKind.STEERING_STOP


def test_classify_runtime_context():
    msg = {
        "role": "system",
        "_runtime_context": True,
        "content": "## Parent Chat Context...",
    }
    assert classify_tool_loop_message(msg) == ToolLoopKind.RUNTIME_CONTEXT


def test_classify_time_explanation():
    msg = {
        "role": "system",
        "_time_explanation": True,
        "_ctx_header": True,
        "_runtime_context": True,
        "content": "## Time Context...",
    }
    assert classify_tool_loop_message(msg) == ToolLoopKind.TIME_EXPLANATION


def test_classify_visibility_guidance():
    msg = {
        "role": "system",
        "_visibility_guidance": True,
        "content": "## User Visibility Context...",
    }
    assert classify_tool_loop_message(msg) == ToolLoopKind.VISIBILITY_GUIDANCE


def test_classify_generic_system_message():
    msg = {"role": "system", "content": "You are a helpful assistant."}
    assert classify_tool_loop_message(msg) == ToolLoopKind.SYSTEM_NOTICE


# ============================================================================
#  is_streaming_noise — kind-based filtering
# ============================================================================


def test_placeholder_noise():
    assert is_streaming_noise("ToolLoop", {"kind": ToolLoopKind.PLACEHOLDER}) is True


def test_runtime_context_noise():
    assert (
        is_streaming_noise("ToolLoop", {"kind": ToolLoopKind.RUNTIME_CONTEXT}) is True
    )


def test_time_explanation_noise():
    assert (
        is_streaming_noise("ToolLoop", {"kind": ToolLoopKind.TIME_EXPLANATION}) is True
    )


def test_visibility_guidance_noise():
    assert (
        is_streaming_noise("ToolLoop", {"kind": ToolLoopKind.VISIBILITY_GUIDANCE})
        is True
    )


def test_status_check_not_noise():
    """STATUS_CHECK is intentionally NOT stream noise — frontend needs them
    to resolve pending parallel tool calls via resolvedToolCallIds."""
    assert is_streaming_noise("ToolLoop", {"kind": ToolLoopKind.STATUS_CHECK}) is False


def test_wait_noop_not_noise():
    assert is_streaming_noise("ToolLoop", {"kind": ToolLoopKind.WAIT_NOOP}) is False


# ============================================================================
#  is_streaming_noise — user-facing kinds pass through
# ============================================================================


def test_tool_call_passes():
    assert is_streaming_noise("ToolLoop", {"kind": ToolLoopKind.TOOL_CALL}) is False


def test_tool_result_passes():
    assert is_streaming_noise("ToolLoop", {"kind": ToolLoopKind.TOOL_RESULT}) is False


def test_response_passes():
    assert is_streaming_noise("ToolLoop", {"kind": ToolLoopKind.RESPONSE}) is False


def test_request_passes():
    assert is_streaming_noise("ToolLoop", {"kind": ToolLoopKind.REQUEST}) is False


def test_interjection_passes():
    assert is_streaming_noise("ToolLoop", {"kind": ToolLoopKind.INTERJECTION}) is False


def test_thinking_sentinel_passes():
    assert (
        is_streaming_noise("ToolLoop", {"kind": ToolLoopKind.THINKING_SENTINEL})
        is False
    )


def test_thought_passes():
    assert is_streaming_noise("ToolLoop", {"kind": ToolLoopKind.THOUGHT}) is False


def test_steering_passes():
    for kind in (
        ToolLoopKind.STEERING_PAUSE,
        ToolLoopKind.STEERING_RESUME,
        ToolLoopKind.STEERING_STOP,
        ToolLoopKind.STEERING_HELPER,
    ):
        assert is_streaming_noise("ToolLoop", {"kind": kind}) is False


# ============================================================================
#  ManagerMethod safety invariant
# ============================================================================


def test_non_suppressed_manager_method_never_filtered():
    """ManagerMethod events for non-suppressed managers must NEVER be filtered."""
    payloads = [
        {"manager": "CodeActActor", "kind": ToolLoopKind.TOOL_CALL},
        {"manager": "ContactManager", "kind": ToolLoopKind.STATUS_CHECK},
        {"manager": "KnowledgeManager", "kind": ToolLoopKind.RUNTIME_CONTEXT},
        {"manager": "WebSearcher", "kind": ToolLoopKind.VISIBILITY_GUIDANCE},
        {"manager": "TaskScheduler", "kind": ToolLoopKind.PLACEHOLDER},
        {},
    ]
    for payload in payloads:
        assert (
            is_streaming_noise("ManagerMethod", payload) is False
        ), f"Non-suppressed ManagerMethod must not be filtered: {payload}"


def test_toolloop_noise_only_applies_to_toolloop_type():
    """Kind-based noise rules only apply to ToolLoop events."""
    for event_type in ("Message", "Comms", "LLM", "DesktopPrimitive"):
        assert (
            is_streaming_noise(event_type, {"kind": ToolLoopKind.PLACEHOLDER}) is False
        ), f"Kind-based noise rules should not apply to {event_type}"


# ============================================================================
#  Edge cases
# ============================================================================


def test_missing_kind_passes():
    """Payload dict without a 'kind' key should not be filtered."""
    assert is_streaming_noise("ToolLoop", {}) is False
    assert is_streaming_noise("ToolLoop", {"method": "foo"}) is False


def test_unknown_kind_passes():
    assert is_streaming_noise("ToolLoop", {"kind": "something_new"}) is False


# ============================================================================
#  Disjoint tree suppression (MemoryManager)
# ============================================================================


def test_memory_manager_method_incoming_filtered():
    """ManagerMethod with manager=MemoryManager is suppressed."""
    payload = {
        "manager": "MemoryManager",
        "method": "process_chunk",
        "phase": "incoming",
        "hierarchy": ["MemoryManager.process_chunk(a1b2)"],
    }
    assert is_suppressed_manager_tree("ManagerMethod", payload) is True
    assert is_streaming_noise("ManagerMethod", payload) is True


def test_memory_manager_method_outgoing_filtered():
    payload = {
        "manager": "MemoryManager",
        "method": "update_contacts",
        "phase": "outgoing",
        "answer": "Updated 3 contacts.",
        "hierarchy": ["MemoryManager.update_contacts(c3d4)"],
    }
    assert is_suppressed_manager_tree("ManagerMethod", payload) is True
    assert is_streaming_noise("ManagerMethod", payload) is True


def test_memory_manager_toolloop_root_filtered():
    """ToolLoop event whose hierarchy root is MemoryManager is suppressed."""
    payload = {
        "kind": ToolLoopKind.RESPONSE,
        "message": {"role": "assistant", "content": "Analyzing transcript..."},
        "method": "MemoryManager.process_chunk",
        "hierarchy": ["MemoryManager.process_chunk(a1b2)"],
    }
    assert is_suppressed_manager_tree("ToolLoop", payload) is True
    assert is_streaming_noise("ToolLoop", payload) is True


def test_memory_manager_nested_toolloop_filtered():
    """ToolLoop from an inner manager called BY MemoryManager is also suppressed."""
    payload = {
        "kind": ToolLoopKind.TOOL_RESULT,
        "message": {"role": "tool", "name": "filter_contacts", "content": "[]"},
        "method": "ContactManager.ask",
        "hierarchy": [
            "MemoryManager.process_chunk(a1b2)",
            "ContactManager.ask(e5f6)",
        ],
    }
    assert is_suppressed_manager_tree("ToolLoop", payload) is True
    assert is_streaming_noise("ToolLoop", payload) is True


def test_memory_manager_nested_manager_method_filtered():
    """ManagerMethod from an inner manager called BY MemoryManager is also
    suppressed — the hierarchy root belongs to a suppressed manager."""
    payload = {
        "manager": "ContactManager",
        "method": "ask",
        "phase": "incoming",
        "hierarchy": [
            "MemoryManager.process_chunk(a1b2)",
            "ContactManager.ask(e5f6)",
        ],
    }
    assert is_suppressed_manager_tree("ManagerMethod", payload) is True
    assert is_streaming_noise("ManagerMethod", payload) is True


def test_non_memory_manager_not_suppressed():
    """Other managers' ManagerMethod events are NOT suppressed."""
    for manager in (
        "CodeActActor",
        "ContactManager",
        "KnowledgeManager",
        "TaskScheduler",
        "WebSearcher",
        "SecretManager",
        "TranscriptManager",
        "FileManager",
    ):
        payload = {
            "manager": manager,
            "method": "ask",
            "phase": "incoming",
            "hierarchy": [f"{manager}.ask(a1b2)"],
        }
        assert is_suppressed_manager_tree("ManagerMethod", payload) is False
        assert is_streaming_noise("ManagerMethod", payload) is False


def test_non_memory_manager_toolloop_not_suppressed():
    """ToolLoop events rooted in non-suppressed managers pass through."""
    payload = {
        "kind": ToolLoopKind.RESPONSE,
        "message": {"role": "assistant", "content": "Searching..."},
        "method": "WebSearcher.ask",
        "hierarchy": [
            "CodeActActor.act(abdf)",
            "execute_code(d67e)",
            "WebSearcher.ask(a0ce)",
        ],
    }
    assert is_suppressed_manager_tree("ToolLoop", payload) is False


def test_suppressed_tree_toolloop_empty_hierarchy_passes():
    """ToolLoop with empty or missing hierarchy is not suppressed."""
    assert is_suppressed_manager_tree("ToolLoop", {"hierarchy": []}) is False
    assert is_suppressed_manager_tree("ToolLoop", {}) is False
    assert is_suppressed_manager_tree("ToolLoop", {"hierarchy": None}) is False
