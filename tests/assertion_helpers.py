"""
Shared assertion helpers for tests
==================================

Contains common functions for formatting assertion error messages
with detailed context including reasoning steps from LLM tool usage,
as well as reusable prompt formatting assertions for system messages.
"""

import json
import re
from typing import Any, List, Dict, Optional, Callable, Type, Tuple
from itertools import zip_longest


def format_reasoning_steps(reasoning: List[Dict[str, Any]]) -> str:
    """Format reasoning steps from LLM tool use loops for better readability."""
    if not reasoning:
        return "No reasoning steps available"

    # Pretty print the reasoning steps, handling nested content fields
    def format_json_content(msg):
        if "content" in msg and msg["content"]:
            try:
                msg["content"] = json.loads(msg["content"])
            except (json.JSONDecodeError, TypeError):
                pass
        return msg

    formatted_reasoning = [format_json_content(msg) for msg in reasoning]
    formatted_reasoning = json.dumps(formatted_reasoning, indent=4)
    formatted_reasoning = formatted_reasoning.replace("\\n", "\n")

    return formatted_reasoning


def find_tool_calls_and_results(
    messages: List[Dict[str, Any]],
    tool_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Find all tool calls with the given name and their corresponding results.

    Handles both sync (direct) and async (check_status_*) tool patterns:
    - Sync: tool result has tool_call_id == original call id
    - Async: tool result has tool_call_id == original call id + "_completed"

    Parameters
    ----------
    messages : List[Dict[str, Any]]
        The messages list from an LLM tool loop (e.g., from handle.result()).
    tool_name : str
        The name of the tool to find (e.g., "ask_image", "filter_contacts").

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        A tuple of (tool_calls, tool_results) where tool_calls are the assistant's
        invocations and tool_results are the corresponding results.
    """
    # Find all tool_calls with the given name
    tool_calls = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if (tc.get("function") or {}).get("name") == tool_name:
                tool_calls.append(tc)

    # Find corresponding tool results by matching tool_call_id
    tool_results = []
    for tc in tool_calls:
        call_id = tc.get("id")
        if not call_id:
            continue

        # Look for tool result with matching id (sync) or id_completed (async)
        # Prefer the _completed version since async tools emit both:
        # 1. A placeholder with tool_call_id == call_id (contains _placeholder status)
        # 2. The actual result with tool_call_id == call_id + "_completed"
        completed_id = f"{call_id}_completed"
        sync_result = None
        async_result = None

        for m in messages:
            if m.get("role") != "tool":
                continue
            result_id = m.get("tool_call_id", "")
            if result_id == completed_id:
                async_result = m
                break  # _completed is definitive, stop searching
            if result_id == call_id and sync_result is None:
                sync_result = m
                # Don't break - keep looking for _completed version

        # Prefer the async (_completed) result over the sync placeholder
        result = async_result or sync_result
        if result is not None:
            tool_results.append(result)

    return tool_calls, tool_results


def assertion_failed(
    expected: Any,
    actual: Any,
    reasoning: List[Dict[str, Any]],
    description: str = "",
    context_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a detailed error message for assertion failures with LLM reasoning.

    Args:
        expected: The expected value
        actual: The actual value received
        reasoning: List of reasoning steps from LLM tool use
        description: Optional description of the assertion
        context_data: Optional additional context data to include (e.g., tasks, messages)

    Returns:
        Formatted error message string
    """
    context_str = ""
    if context_data:
        for label, data in context_data.items():
            context_str += f"\n{label}:\n{json.dumps(data, indent=4)}\n"

    formatted_reasoning = format_reasoning_steps(reasoning)

    return (
        f"\n{description}\n"
        f"Expected:\n{expected}\n"
        f"Got:\n{actual}\n"
        f"{context_str}"
        f"Reasoning:\n{formatted_reasoning}\n"
    )


# ---------------------------------------------------------------------------
# System prompt assertion helpers (shared across system message tests)
# ---------------------------------------------------------------------------


def extract_tools_dict(prompt: str) -> dict:
    """Parse the tools JSON block from a standardized system prompt.

    Looks for the "Tools (name → argspec):" header, then captures the first
    well-formed JSON object that follows it by matching braces depth.
    """
    idx = prompt.find("Tools (name")
    assert idx != -1, "Missing tools block header"
    json_start = prompt.find("{", idx)
    assert json_start != -1, "Missing tools JSON start"
    depth = 0
    end = None
    for i in range(json_start, len(prompt)):
        c = prompt[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assert end is not None, "Unclosed tools JSON block"
    return json.loads(prompt[json_start:end])


def assert_in_order(prompt: str, markers: List[str]) -> None:
    """Assert each marker appears in `prompt` in the given sequence.

    Raises an AssertionError if any marker is missing or out of order.
    """
    pos = -1
    missing: list[str] = []
    for m in markers:
        i = prompt.find(m)
        if i == -1:
            missing.append(m)
            continue
        assert i > pos, f"Marker out of order: {m!r}\n\nFull system prompt:\n\n{prompt}"
        pos = i
    assert (
        not missing
    ), f"Missing markers (order check): {missing}\n\nFull system prompt:\n\n{prompt}"


def assert_section_spacing(prompt: str) -> None:
    """Assert that dashed underline headers are preceded by a blank line."""
    lines = prompt.splitlines()
    errors: list[str] = []
    for idx in range(len(lines) - 1):
        line = lines[idx]
        next_line = lines[idx + 1]
        if re.fullmatch(r"-+", next_line.strip()):
            if idx == 0 or lines[idx - 1].strip() != "":
                errors.append(f"Missing blank line before section header: '{line}'")
    assert not errors, "\n".join(errors) + f"\n\nFull system prompt:\n\n{prompt}"


def assert_selected_headers_have_blank_line(prompt: str, titles: List[str]) -> None:
    """Assert that provided section titles are preceded by a blank line."""
    lines = prompt.splitlines()
    missing: list[str] = []
    for i, line in enumerate(lines):
        title = line.strip()
        if title in titles:
            if i == 0 or lines[i - 1].strip() != "":
                missing.append(title)
    assert (
        not missing
    ), f"Missing blank line before: {missing}\n\nFull system prompt:\n\n{prompt}"


def assert_time_footer(prompt: str, prefix: str) -> None:
    """Assert the final non-empty line matches the standardized time footer.

    `prefix` is the literal prefix used by the builder (e.g.,
    "Current UTC time is ").

    Expected format: "Friday, June 13, 2025 at 12:00 PM UTC."
    """
    non_empty_lines = [ln for ln in prompt.splitlines() if ln.strip()]
    assert non_empty_lines, (
        "Prompt should not be empty\n\nFull system prompt:\n\n" + prompt
    )
    last = non_empty_lines[-1]
    # Human-readable format: "Friday, June 13, 2025 at 12:00 PM UTC."
    pattern = re.compile(
        rf"{re.escape(prefix)}[A-Z][a-z]+, [A-Z][a-z]+ \d{{1,2}}, \d{{4}} at \d{{1,2}}:\d{{2}} [AP]M [A-Za-z/_]+\.",
    )
    assert pattern.fullmatch(
        last,
    ), f"Unexpected last line: {last!r}\n\nFull system prompt:\n\n{prompt}"


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------


def first_diff_block(
    a: str,
    b: str,
    context: int = 3,
    label_a: str = "First",
    label_b: str = "Second",
) -> str:
    """
    Return a compact snippet showing the first differing line between two strings,
    with a few lines of context above and below the differing line.
    Only the first differing location is shown (further diffs are not included).
    """
    a_lines = a.splitlines()
    b_lines = b.splitlines()

    for idx, (la, lb) in enumerate(zip_longest(a_lines, b_lines, fillvalue="<EOF>")):
        if la != lb:
            start = max(0, idx - context)
            end = idx + context + 1
            a_block = "\n".join(
                a_lines[start : end if end <= len(a_lines) else len(a_lines)],
            )
            b_block = "\n".join(
                b_lines[start : end if end <= len(b_lines) else len(b_lines)],
            )
            return (
                f"First differing line at index {idx}:\n"
                f"--- {label_a} ---\n{a_block}\n"
                f"--- {label_b} ---\n{b_block}"
            )

    return "No differing line found (contents are identical)."


# ---------------------------------------------------------------------------
# Dynamic mock tool and column generation for system message tests
# ---------------------------------------------------------------------------


def get_tools_from_manager(
    manager_class: Type,
    method: str,
) -> Dict[str, Callable]:
    """Get real tools from a manager instance.

    This creates a manager instance and extracts the actual tools dictionary.
    This ensures test tools have the exact same signatures as production code.

    Note: This may hit the database. If database state causes issues, the test
    should use subprocess-based extraction instead.

    Parameters
    ----------
    manager_class : Type
        The manager class (e.g., ContactManager, TaskScheduler).
    method : str
        The method name ("ask" or "update").

    Returns
    -------
    Dict[str, Callable]
        Dictionary of real tools from the manager.
    """
    manager = manager_class()
    return dict(manager.get_tools(method))


def mock_columns_from_model(model_class: Type) -> Dict[str, str]:
    """Generate mock columns dictionary from a Pydantic model.

    This extracts field names and types from the model, ensuring
    test columns match the actual schema.

    Parameters
    ----------
    model_class : Type
        A Pydantic model class.

    Returns
    -------
    Dict[str, str]
        Dictionary of {column_name: type_string}.
    """
    columns = {}
    for field_name, field_info in model_class.model_fields.items():
        # Get a simplified type string
        annotation = field_info.annotation
        if annotation is None:
            type_str = "Any"
        elif hasattr(annotation, "__name__"):
            type_str = annotation.__name__
        elif hasattr(annotation, "__origin__"):
            # Handle generic types like Optional[str], List[int]
            origin = annotation.__origin__
            type_str = getattr(origin, "__name__", str(origin))
        else:
            type_str = str(annotation)
        columns[field_name] = type_str.lower()
    return columns


def mock_columns_with_custom(
    model_class: Type,
    custom_columns: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Generate mock columns from a model plus optional custom columns.

    Parameters
    ----------
    model_class : Type
        A Pydantic model class.
    custom_columns : Optional[Dict[str, str]]
        Additional custom columns to add.

    Returns
    -------
    Dict[str, str]
        Combined columns dictionary.
    """
    columns = mock_columns_from_model(model_class)
    if custom_columns:
        columns.update(custom_columns)
    return columns
