from __future__ import annotations

import os
import sys
import subprocess
import textwrap

from unity.session_details import UNASSIGNED_USER_CONTEXT, UNASSIGNED_ASSISTANT_CONTEXT
from tests.assertion_helpers import first_diff_block
from tests.helpers import _handle_project


def _unwrap_callable(tool):
    """Return the underlying callable from either a ToolSpec or a function."""
    return getattr(tool, "fn", tool)


def test_all_ask_tools_have_sufficient_docstrings(file_manager):
    tools = file_manager.get_tools("ask_about_file")

    assert tools, "FileManager.ask_about_file should expose at least one tool"

    for name, value in tools.items():
        fn = _unwrap_callable(value)
        doc = (getattr(fn, "__doc__", None) or "").strip()
        assert doc, f"Tool '{name}' is missing a docstring"
        assert (
            len(doc) >= 100
        ), f"Docstring for tool '{name}' is too short (len={len(doc)})"


def _build_tools_schema_in_subprocess(method: str, test_context: str) -> str:
    """
    Build tools→schema JSON in a fresh Python process to catch cross-session drift.

    The test_context is passed via environment variable to ensure the subprocess
    uses an isolated context rather than the shared default context.
    """
    assert method == "ask_about_file"
    code = textwrap.dedent(
        f"""
		import os, sys, json
		sys.path.insert(0, os.getcwd())
		import unify
		# Activate the test project before setting context
		project_name = os.environ.get("UNITY_TEST_PROJECT_NAME", "UnityTests")
		unify.activate(project_name, overwrite=False)
		# Set test-specific context before creating FileManager to avoid races
		test_ctx = os.environ.get("_TEST_CONTEXT")
		if test_ctx:
			unify.set_context(test_ctx, relative=False)
		from unity.common.llm_helpers import method_to_schema
		def _unwrap_callable(tool):
			return getattr(tool, "fn", tool)
		from unity.file_manager.managers.local import LocalFileManager as FileManager
		fm = FileManager()
		tools = fm.get_tools("{method}")
		if not tools:
			raise AssertionError("FileManager.{method} should expose at least one tool")
		mapping = {{
			name: method_to_schema(_unwrap_callable(value), name)
			for name, value in tools.items()
		}}
		sys.stdout.write(json.dumps(mapping, sort_keys=True, indent=2))
		""",
    )
    env = os.environ.copy()
    env["_TEST_CONTEXT"] = test_context
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )
    return proc.stdout


@_handle_project
def test_ask_tool_schemas_are_stable_across_python_sessions():
    # Build a test-specific context path matching _handle_project pattern
    test_ctx = f"tests/file_manager/test_tool_docstrings/test_ask_tool_schemas_are_stable_across_python_sessions/{UNASSIGNED_USER_CONTEXT}/{UNASSIGNED_ASSISTANT_CONTEXT}"
    p1 = _build_tools_schema_in_subprocess("ask_about_file", test_ctx)
    p2 = _build_tools_schema_in_subprocess("ask_about_file", test_ctx)
    if p1 != p2:
        snippet = first_diff_block(
            p1,
            p2,
            context=3,
            label_a="First JSON",
            label_b="Second JSON",
        )
        raise AssertionError(
            "Tool schemas for ask_about_file tools changed between separate Python sessions.\n\n"
            + snippet,
        )
