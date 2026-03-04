"""Unit tests for unity.events.stream_filters.

All tests are synchronous, zero-IO, and test the predicate functions directly.
No EventBus, Pub/Sub, or project setup is needed.
"""

import json

from unity.events.stream_filters import (
    is_streaming_noise,
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


def test_manager_method_never_filtered():
    """ManagerMethod events must NEVER be filtered regardless of payload."""
    payloads = [
        {
            "message": {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "check_status_fake"}}],
            },
        },
        {"message": {"role": "tool", "name": "check_status_fake", "content": "{}"}},
        {"message": {"role": "system", "_runtimeContext": True}},
        {"message": {"role": "system", "_visibility_guidance": True}},
        {
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
        ), f"ManagerMethod must never be filtered, but was for payload={payload}"


def test_hierarchy_safe_all_filtered_types_are_toolloop():
    """Every individual rule only matches dict messages, and the top-level
    predicate only applies to ToolLoop events.  This test verifies the
    invariant programmatically for all registered rules."""
    noise_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": "check_status_x"}}],
    }
    assert is_streaming_noise("ToolLoop", {"message": noise_msg}) is True
    for event_type in ("ManagerMethod", "Message", "Comms", "LLM", "DesktopPrimitive"):
        assert (
            is_streaming_noise(event_type, {"message": noise_msg}) is False
        ), f"Only ToolLoop events should be filtered, but {event_type} was filtered"


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
