from __future__ import annotations


from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.common.llm_helpers import method_to_schema
import json


def _unwrap_callable(tool):
    """Return the underlying callable from either a ToolSpec or a function."""
    return getattr(tool, "fn", tool)


def test_all_ask_tools_have_sufficient_docstrings():
    tm = TranscriptManager()
    tools = tm.get_tools("ask")

    assert tools, "TranscriptManager.ask should expose at least one tool"

    for name, value in tools.items():
        fn = _unwrap_callable(value)
        doc = (getattr(fn, "__doc__", None) or "").strip()
        assert doc, f"Tool '{name}' is missing a docstring"
        assert (
            len(doc) >= 100
        ), f"Docstring for tool '{name}' is too short (len={len(doc)})"


def test_ask_tool_schemas_are_stable_across_serial_calls():
    """
    The fully unpacked tool schemas (as seen by the LLM in the async tool loop)
    should be identical across serial calls to the representation function.
    """
    tm = TranscriptManager()
    tools = tm.get_tools("ask")
    assert tools, "TranscriptManager.ask should expose at least one tool"

    # First pass
    first = {
        name: method_to_schema(_unwrap_callable(value), name)
        for name, value in tools.items()
    }
    # Second pass
    second = {
        name: method_to_schema(_unwrap_callable(value), name)
        for name, value in tools.items()
    }

    # Direct dict equality (order-insensitive)
    assert first == second, "Tool schemas for ask-tools changed between serial calls"

    # Also ensure JSON-rendered form is stable (order-sensitive, sorted keys)
    f_dump = json.dumps(first, sort_keys=True)
    s_dump = json.dumps(second, sort_keys=True)
    assert (
        f_dump == s_dump
    ), "JSON rendering of ask-tool schemas changed between serial calls"
