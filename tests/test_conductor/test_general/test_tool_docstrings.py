from __future__ import annotations

from unity.conductor.conductor import Conductor
from tests.assertion_helpers import first_diff_block
import sys
import subprocess
import textwrap


def _unwrap_callable(tool):
    """Return the underlying callable from either a ToolSpec or a function."""
    return getattr(tool, "fn", tool)


def _build_tools_schema_in_subprocess(method: str) -> str:
    """
    Build tools→schema JSON in a fresh Python process to catch cross-session drift.
    """
    assert method in {"request"}
    code = textwrap.dedent(
        f"""
		import os, sys, json
		sys.path.insert(0, os.getcwd())
		from unity.common.llm_helpers import method_to_schema
		def _unwrap_callable(tool):
			return getattr(tool, "fn", tool)
		from unity.conductor.conductor import Conductor
		c = Conductor()
		tools = c.get_tools("{method}")
		if not tools:
			raise AssertionError("Conductor.{method} should expose at least one tool")
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


def test_request_tools_have_docstrings():
    c = Conductor()
    tools = c.get_tools("request")

    assert tools, "Conductor.request should expose at least one tool"

    for name, value in tools.items():
        fn = _unwrap_callable(value)
        doc = (getattr(fn, "__doc__", None) or "").strip()
        assert doc, f"Tool '{name}' is missing a docstring"
        assert (
            len(doc) >= 100
        ), f"Docstring for tool '{name}' is too short (len={len(doc)})"


def test_request_schemas_stable():
    p1 = _build_tools_schema_in_subprocess("request")
    p2 = _build_tools_schema_in_subprocess("request")
    if p1 != p2:
        snippet = first_diff_block(
            p1,
            p2,
            context=3,
            label_a="First JSON",
            label_b="Second JSON",
        )
        raise AssertionError(
            "Tool schemas for request-tools changed between separate Python sessions.\n\n"
            + snippet,
        )
