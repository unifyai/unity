from __future__ import annotations

import os
import sys
import subprocess
import textwrap

from unity.contact_manager.contact_manager import ContactManager
from unity.session_details import DEFAULT_USER_CONTEXT, DEFAULT_ASSISTANT_CONTEXT
from tests.assertion_helpers import first_diff_block
from tests.helpers import _handle_project


def _unwrap_callable(tool):
    """Return the underlying callable from either a ToolSpec or a function."""
    return getattr(tool, "fn", tool)


@_handle_project
def test_ask_tools_docstrings():
    cm = ContactManager()
    tools = cm.get_tools("ask")

    assert tools, "ContactManager.ask should expose at least one tool"

    for name, value in tools.items():
        fn = _unwrap_callable(value)
        doc = (getattr(fn, "__doc__", None) or "").strip()
        assert doc, f"Tool '{name}' is missing a docstring"
        assert (
            len(doc) >= 100
        ), f"Docstring for tool '{name}' is too short (len={len(doc)})"


@_handle_project
def test_update_tools_docstrings():
    cm = ContactManager()
    tools = cm.get_tools("update")

    assert tools, "ContactManager.update should expose at least one tool"

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
    assert method in {"ask", "update"}
    code = textwrap.dedent(
        f"""
		import os, sys, json
		sys.path.insert(0, os.getcwd())
		import unify
		# Activate the test project before setting context
		project_name = os.environ.get("UNITY_TEST_PROJECT_NAME", "UnityTests")
		unify.activate(project_name, overwrite=False)
		# Set test-specific context before creating ContactManager to avoid races
		test_ctx = os.environ.get("_TEST_CONTEXT")
		if test_ctx:
			unify.set_context(test_ctx, relative=False)
		from unity.common.llm_helpers import method_to_schema
		def _unwrap_callable(tool):
			return getattr(tool, "fn", tool)
		from unity.contact_manager.contact_manager import ContactManager
		cm = ContactManager()
		tools = cm.get_tools("{method}")
		if not tools:
			raise AssertionError("ContactManager.{method} should expose at least one tool")
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
def test_ask_schemas_stable():
    # Build a test-specific context path matching _handle_project pattern
    test_ctx = f"tests/test_contact_manager/test_tool_docstrings/test_ask_schemas_stable/{DEFAULT_USER_CONTEXT}/{DEFAULT_ASSISTANT_CONTEXT}"
    p1 = _build_tools_schema_in_subprocess("ask", test_ctx)
    p2 = _build_tools_schema_in_subprocess("ask", test_ctx)
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


@_handle_project
def test_update_schemas_stable():
    # Build a test-specific context path matching _handle_project pattern
    test_ctx = f"tests/test_contact_manager/test_tool_docstrings/test_update_schemas_stable/{DEFAULT_USER_CONTEXT}/{DEFAULT_ASSISTANT_CONTEXT}"
    p1 = _build_tools_schema_in_subprocess("update", test_ctx)
    p2 = _build_tools_schema_in_subprocess("update", test_ctx)
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
