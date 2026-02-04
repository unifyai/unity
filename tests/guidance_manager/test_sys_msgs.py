"""
System message tests for GuidanceManager.

Tests dynamically extract tools from the actual manager to ensure
tests stay in sync with API changes.
"""

import os
import re
import sys
import subprocess
import textwrap

from tests.assertion_helpers import (
    extract_tools_dict,
    assert_in_order,
    assert_section_spacing,
    assert_selected_headers_have_blank_line,
    assert_time_footer,
    first_diff_block,
)
from tests.helpers import _handle_project

from unity.guidance_manager.guidance_manager import GuidanceManager
from unity.guidance_manager.prompt_builders import build_ask_prompt, build_update_prompt
from unity.session_details import DEFAULT_USER_CONTEXT, DEFAULT_ASSISTANT_CONTEXT


@_handle_project
def test_ask_system_prompt_formatting():
    """Test ask prompt structure with dynamically extracted tools."""
    gm = GuidanceManager()
    tools = dict(gm.get_tools("ask"))
    num_items = gm._num_items()

    prompt = build_ask_prompt(
        tools=tools,
        num_items=num_items,
        columns=gm._list_columns(),
    ).flatten()

    # Verify tools match what's in the prompt
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt

    # Schema-based table info
    m = re.search(r"There are currently\s+(\d+)\s+guidance entries\.", prompt)
    assert m, "Missing counts line"
    assert int(m.group(1)) == num_items
    assert "Columns are defined in the Guidance schema above." in prompt
    assert "Schemas" in prompt
    assert "Guidance = " in prompt

    # Standard blocks
    assert "Images policy (when images are present)" in prompt
    assert "Images forwarding to nested tools" in prompt
    assert "Parallelism and single" in prompt

    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks
    counts_line = f"There are currently {num_items} guidance entries."
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "Schemas",
            "Guidance = ",
            counts_line,
            "Columns are defined in the Guidance schema above.",
            "Tools (name",
            "Examples",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
            "Parallelism and single",
            "Current UTC time is ",
        ],
    )

    assert_selected_headers_have_blank_line(
        prompt,
        [
            "Examples",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
        ],
    )
    assert_section_spacing(prompt)
    assert_time_footer(prompt, "Current UTC time is ")


@_handle_project
def test_update_system_prompt_formatting():
    """Test update prompt structure with dynamically extracted tools."""
    gm = GuidanceManager()
    tools = dict(gm.get_tools("update"))
    num_items = gm._num_items()

    prompt = build_update_prompt(
        tools=tools,
        num_items=num_items,
        columns=gm._list_columns(),
    ).flatten()

    # Verify tools match what's in the prompt
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())

    # Schema-based table info
    m = re.search(r"There are currently\s+(\d+)\s+guidance entries\.", prompt)
    assert m, "Missing counts line"
    assert int(m.group(1)) == num_items
    assert "Columns are defined in the Guidance schema above." in prompt

    assert "Schemas" in prompt
    assert "Guidance = " in prompt
    assert "Images policy (when images are present)" in prompt
    assert "Images forwarding to nested tools" in prompt
    assert "Parallelism and single" in prompt

    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks
    counts_line = f"There are currently {num_items} guidance entries."
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "Schemas",
            "Guidance = ",
            counts_line,
            "Columns are defined in the Guidance schema above.",
            "Tools (name",
            "Tool selection",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
            "Parallelism and single",
            "Current UTC time is ",
        ],
    )

    assert_selected_headers_have_blank_line(
        prompt,
        [
            "Tool selection",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
        ],
    )
    assert_section_spacing(prompt)
    assert_time_footer(prompt, "Current UTC time is ")


# ─────────────────────────────────────────────────────────────────────────────
# Stability tests (subprocess for cross-session comparison)
# ─────────────────────────────────────────────────────────────────────────────


def _build_prompt_in_subprocess(method: str, test_context: str) -> str:
    """Build prompt in subprocess for stability comparison.

    The test_context is passed via environment variable to ensure the subprocess
    uses an isolated context rather than the shared default context.
    """
    code = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, os.getcwd())
        import unify
        # Activate the test project before setting context
        project_name = os.environ.get("UNITY_TEST_PROJECT_NAME", "UnityTests")
        unify.activate(project_name, overwrite=False)
        # Set test-specific context before creating GuidanceManager to avoid races
        test_ctx = os.environ.get("_TEST_CONTEXT")
        if test_ctx:
            unify.set_context(test_ctx, relative=False)
        import unity.common.prompt_helpers as _ph
        from datetime import datetime, timezone
        def _static_now(time_only: bool = False):
            dt = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
            label = "UTC"
            if time_only:
                return dt.strftime("%I:%M %p ") + label
            return dt.strftime("%A, %B %d, %Y at %I:%M %p ") + label
        _ph.now = _static_now
        from unity.guidance_manager.guidance_manager import GuidanceManager
        from unity.guidance_manager.prompt_builders import build_ask_prompt, build_update_prompt
        gm = GuidanceManager()
        tools = dict(gm.get_tools("{method}"))
        prompt = build_{method}_prompt(tools=tools, num_items=gm._num_items(), columns=gm._list_columns()).flatten()
        sys.stdout.write(prompt)
        """,
    )
    env = os.environ.copy()
    env["_TEST_CONTEXT"] = test_context
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return proc.stdout


@_handle_project
def test_ask_prompt_stable():
    """Verify ask prompt is identical across Python sessions."""
    # Build a test-specific context path matching _handle_project pattern
    test_ctx = f"tests/guidance_manager/test_sys_msgs/test_ask_prompt_stable/{DEFAULT_USER_CONTEXT}/{DEFAULT_ASSISTANT_CONTEXT}"
    p1 = _build_prompt_in_subprocess("ask", test_ctx)
    p2 = _build_prompt_in_subprocess("ask", test_ctx)
    if p1 != p2:
        raise AssertionError(
            "Ask prompt changed between sessions.\n\n" + first_diff_block(p1, p2),
        )


@_handle_project
def test_update_prompt_stable():
    """Verify update prompt is identical across Python sessions."""
    # Build a test-specific context path matching _handle_project pattern
    test_ctx = f"tests/guidance_manager/test_sys_msgs/test_update_prompt_stable/{DEFAULT_USER_CONTEXT}/{DEFAULT_ASSISTANT_CONTEXT}"
    p1 = _build_prompt_in_subprocess("update", test_ctx)
    p2 = _build_prompt_in_subprocess("update", test_ctx)
    if p1 != p2:
        raise AssertionError(
            "Update prompt changed between sessions.\n\n" + first_diff_block(p1, p2),
        )
