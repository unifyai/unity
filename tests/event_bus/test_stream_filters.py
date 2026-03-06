"""Unit tests for unity.events.stream_filters.

All tests are synchronous, zero-IO, and test the predicate functions directly.
No EventBus, Pub/Sub, or project setup is needed.
"""

import json

from unity.events.stream_filters import (
    is_streaming_noise,
    is_suppressed_manager_tree,
    is_synthetic_status_check,
    is_placeholder_message,
    is_runtime_context_header,
    is_visibility_guidance,
)

# ============================================================================
#  Synthetic check_status_* filtering
# ============================================================================


def test_check_status_assistant_stub_filtered():
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "toolu_abc123_completed",
                "type": "function",
                "function": {
                    "name": "check_status_toolu_abc123",
                    "arguments": "{}",
                },
            },
        ],
    }
    assert is_synthetic_status_check(msg) is True
    assert is_streaming_noise("ToolLoop", {"message": msg}) is True


def test_check_status_tool_reply_filtered():
    msg = {
        "role": "tool",
        "toolCallId": "toolu_abc123_completed",
        "name": "check_status_toolu_abc123",
        "content": '{"answer": "some result"}',
    }
    assert is_synthetic_status_check(msg) is True
    assert is_streaming_noise("ToolLoop", {"message": msg}) is True


# ============================================================================
#  Placeholder message filtering
# ============================================================================


def test_placeholder_pending_filtered():
    msg = {
        "role": "tool",
        "name": "some_tool",
        "content": json.dumps({"_placeholder": "pending"}),
    }
    assert is_placeholder_message(msg) is True
    assert is_streaming_noise("ToolLoop", {"message": msg}) is True


def test_placeholder_progress_filtered():
    msg = {
        "role": "tool",
        "name": "web_search",
        "content": json.dumps(
            {"_placeholder": "progress", "tool": "web_search", "partial": "..."},
        ),
    }
    assert is_placeholder_message(msg) is True
    assert is_streaming_noise("ToolLoop", {"message": msg}) is True


def test_placeholder_completed_filtered():
    msg = {
        "role": "tool",
        "name": "some_tool",
        "content": json.dumps(
            {
                "_placeholder": "completed",
                "status": "Tool completed. See check_status result below.",
                "result_call_id": "toolu_abc_completed",
            },
        ),
    }
    assert is_placeholder_message(msg) is True


def test_placeholder_nested_start_filtered():
    msg = {
        "role": "tool",
        "name": "some_tool",
        "content": json.dumps({"_placeholder": "nested_start"}),
    }
    assert is_placeholder_message(msg) is True


def test_placeholder_non_json_content_passes():
    """Tool reply with plain text content is NOT a placeholder."""
    msg = {
        "role": "tool",
        "name": "web_search",
        "content": "Here are the search results...",
    }
    assert is_placeholder_message(msg) is False


def test_placeholder_json_without_key_passes():
    """Tool reply with valid JSON but no _placeholder key passes through."""
    msg = {
        "role": "tool",
        "name": "web_search",
        "content": json.dumps({"answer": "The RTX 5070 costs $549"}),
    }
    assert is_placeholder_message(msg) is False


# ============================================================================
#  Runtime context header filtering
# ============================================================================


def test_runtime_context_header_filtered():
    msg = {
        "role": "system",
        "_runtimeContext": True,
        "_ctxHeader": True,
        "content": "## Parent Chat Context\nYou received this request...",
    }
    assert is_runtime_context_header(msg) is True
    assert is_streaming_noise("ToolLoop", {"message": msg}) is True


# ============================================================================
#  Visibility guidance filtering
# ============================================================================


def test_visibility_guidance_filtered():
    msg = {
        "role": "system",
        "_visibility_guidance": True,
        "content": "## User Visibility Context\nIMPORTANT: The end-user...",
    }
    assert is_visibility_guidance(msg) is True
    assert is_streaming_noise("ToolLoop", {"message": msg}) is True


# ============================================================================
#  Pass-through cases (should NOT be filtered)
# ============================================================================


def test_normal_assistant_tool_call_passes():
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "toolu_xyz",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "RTX 5070 price"}',
                },
            },
        ],
    }
    assert is_streaming_noise("ToolLoop", {"message": msg}) is False


def test_normal_tool_reply_passes():
    msg = {
        "role": "tool",
        "toolCallId": "toolu_xyz",
        "name": "web_search",
        "content": '{"answer": "The RTX 5070 Super costs $549 at Best Buy"}',
    }
    assert is_streaming_noise("ToolLoop", {"message": msg}) is False


def test_normal_system_message_passes():
    msg = {
        "role": "system",
        "content": "You are a helpful assistant.",
    }
    assert is_streaming_noise("ToolLoop", {"message": msg}) is False


def test_user_message_passes():
    msg = {
        "role": "user",
        "content": "What is the price of the RTX 5070?",
    }
    assert is_streaming_noise("ToolLoop", {"message": msg}) is False


def test_assistant_text_response_passes():
    msg = {
        "role": "assistant",
        "content": "Based on my research, the RTX 5070 Super costs...",
    }
    assert is_streaming_noise("ToolLoop", {"message": msg}) is False


# ============================================================================
#  ManagerMethod safety invariant
# ============================================================================


def test_non_suppressed_manager_method_never_filtered():
    """ManagerMethod events for non-suppressed managers must NEVER be filtered."""
    payloads = [
        {
            "manager": "CodeActActor",
            "message": {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "check_status_fake"}}],
            },
        },
        {
            "manager": "ContactManager",
            "message": {"role": "tool", "name": "check_status_fake", "content": "{}"},
        },
        {
            "manager": "KnowledgeManager",
            "message": {"role": "system", "_runtimeContext": True},
        },
        {
            "manager": "WebSearcher",
            "message": {"role": "system", "_visibility_guidance": True},
        },
        {
            "manager": "TaskScheduler",
            "message": {
                "role": "tool",
                "content": json.dumps({"_placeholder": "pending"}),
            },
        },
        {},
    ]
    for payload in payloads:
        assert (
            is_streaming_noise("ManagerMethod", payload) is False
        ), f"Non-suppressed ManagerMethod must not be filtered: {payload}"


def test_toolloop_noise_only_applies_to_toolloop_type():
    """Per-message noise rules (check_status, placeholder, etc.) only apply
    to ToolLoop events.  Non-ToolLoop/non-suppressed events always pass."""
    noise_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": "check_status_x"}}],
    }
    assert is_streaming_noise("ToolLoop", {"message": noise_msg}) is True
    for event_type in ("Message", "Comms", "LLM", "DesktopPrimitive"):
        assert (
            is_streaming_noise(event_type, {"message": noise_msg}) is False
        ), f"Per-message noise rules should not apply to {event_type}"


# ============================================================================
#  Edge cases
# ============================================================================


def test_missing_message_key_passes():
    """Payload dict without a 'message' key should not be filtered."""
    assert is_streaming_noise("ToolLoop", {}) is False
    assert is_streaming_noise("ToolLoop", {"method": "foo"}) is False


def test_non_dict_message_passes():
    """If message is not a dict, it should not be filtered."""
    assert is_streaming_noise("ToolLoop", {"message": "just a string"}) is False
    assert is_streaming_noise("ToolLoop", {"message": None}) is False
    assert is_streaming_noise("ToolLoop", {"message": 42}) is False


def test_empty_tool_calls_passes():
    """Assistant message with empty tool_calls list passes through."""
    msg = {"role": "assistant", "content": "thinking...", "tool_calls": []}
    assert is_streaming_noise("ToolLoop", {"message": msg}) is False


def test_check_status_mixed_tool_calls():
    """Assistant message with one normal and one check_status tool call
    should still be filtered (the check_status presence is sufficient)."""
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"function": {"name": "web_search"}},
            {"function": {"name": "check_status_toolu_abc"}},
        ],
    }
    assert is_synthetic_status_check(msg) is True
    assert is_streaming_noise("ToolLoop", {"message": msg}) is True


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
        "message": {"role": "assistant", "content": "Analyzing transcript..."},
        "method": "MemoryManager.process_chunk",
        "hierarchy": ["MemoryManager.process_chunk(a1b2)"],
    }
    assert is_suppressed_manager_tree("ToolLoop", payload) is True
    assert is_streaming_noise("ToolLoop", payload) is True


def test_memory_manager_nested_toolloop_filtered():
    """ToolLoop from an inner manager called BY MemoryManager is also suppressed."""
    payload = {
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
