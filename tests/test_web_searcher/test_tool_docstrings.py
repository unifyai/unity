from __future__ import annotations


from unity.web_searcher.web_searcher import WebSearcher
from tests.assertion_helpers import first_diff_block
import sys
import subprocess
import textwrap


def _unwrap_callable(tool):
    """Return the underlying callable from either a ToolSpec or a function."""
    return getattr(tool, "fn", tool)


def test_all_ask_tools_have_sufficient_docstrings():
    ws = WebSearcher()
    tools = ws.get_tools("ask")

    assert tools, "WebSearcher.ask should expose at least one tool"

    for name, value in tools.items():
        fn = _unwrap_callable(value)
        doc = (getattr(fn, "__doc__", None) or "").strip()
        assert doc, f"Tool '{name}' is missing a docstring"
        assert (
            len(doc) >= 100
        ), f"Docstring for tool '{name}' is too short (len={len(doc)})"


def _build_tools_schema_in_subprocess(method: str) -> str:
    """
    Build tools→schema JSON in a fresh Python process to catch cross-session drift.
    """
    assert method in {"ask", "update"}
    code = textwrap.dedent(
        f"""
		import os, sys, json
		sys.path.insert(0, os.getcwd())
		from unity.common.llm_helpers import method_to_schema
		def _unwrap_callable(tool):
			return getattr(tool, "fn", tool)
		from unity.web_searcher.web_searcher import WebSearcher
		ws = WebSearcher()
		tools = ws.get_tools("{method}")
		if not tools:
			raise AssertionError("WebSearcher.{method} should expose at least one tool")
		mapping = {{
			name: method_to_schema(_unwrap_callable(value), name)
			for name, value in tools.items()
		}}
		sys.stdout.write(json.dumps(mapping, sort_keys=True, indent=2))
		""",
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return proc.stdout


def test_all_update_tools_have_sufficient_docstrings():
    ws = WebSearcher()
    tools = ws.get_tools("update")

    assert tools, "WebSearcher.update should expose at least one tool"

    for name, value in tools.items():
        fn = _unwrap_callable(value)
        doc = (getattr(fn, "__doc__", None) or "").strip()
        assert doc, f"Tool '{name}' is missing a docstring"
        assert (
            len(doc) >= 100
        ), f"Docstring for tool '{name}' is too short (len={len(doc)})"


def test_ask_tool_schemas_are_stable_across_python_sessions():
    p1 = _build_tools_schema_in_subprocess("ask")
    p2 = _build_tools_schema_in_subprocess("ask")
    if p1 != p2:
        snippet = first_diff_block(
            p1,
            p2,
            context=3,
            label_a="First JSON",
            label_b="Second JSON",
        )
        raise AssertionError(
            "Tool schemas for ask-tools changed between separate Python sessions.\n\n"
            + snippet,
        )


def test_update_tool_schemas_are_stable_across_python_sessions():
    p1 = _build_tools_schema_in_subprocess("update")
    p2 = _build_tools_schema_in_subprocess("update")
    if p1 != p2:
        snippet = first_diff_block(
            p1,
            p2,
            context=3,
            label_a="First JSON",
            label_b="Second JSON",
        )
        raise AssertionError(
            "Tool schemas for update-tools changed between separate Python sessions.\n\n"
            + snippet,
        )
