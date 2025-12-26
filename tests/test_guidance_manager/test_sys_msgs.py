"""
System message tests for GuidanceManager.

Tests dynamically extract tools from the actual manager to ensure
tests stay in sync with API changes.
"""

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

from unity.guidance_manager.guidance_manager import GuidanceManager
from unity.guidance_manager.prompt_builders import build_ask_prompt, build_update_prompt


def test_ask_system_prompt_formatting():
    """Test ask prompt structure with dynamically extracted tools."""
    gm = GuidanceManager()
    tools = dict(gm.get_tools("ask"))
    num_items = gm._num_items()

    prompt = build_ask_prompt(
        tools=tools,
        num_items=num_items,
        columns=gm._list_columns(),
    )

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


def test_update_system_prompt_formatting():
    """Test update prompt structure with dynamically extracted tools."""
    gm = GuidanceManager()
    tools = dict(gm.get_tools("update"))
    num_items = gm._num_items()

    prompt = build_update_prompt(
        tools=tools,
        num_items=num_items,
        columns=gm._list_columns(),
    )

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


def _build_prompt_in_subprocess(method: str) -> str:
    """Build prompt in subprocess for stability comparison."""
    code = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, os.getcwd())
        import unity.common.prompt_helpers as _ph
        from datetime import datetime, timezone
        def _static_now(time_only: bool = False):
            dt = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
            return dt.strftime("%H:%M:%S UTC" if time_only else "%Y-%m-%d %H:%M:%S UTC")
        _ph.now = _static_now
        from unity.guidance_manager.guidance_manager import GuidanceManager
        from unity.guidance_manager.prompt_builders import build_ask_prompt, build_update_prompt
        gm = GuidanceManager()
        tools = dict(gm.get_tools("{method}"))
        prompt = build_{method}_prompt(tools=tools, num_items=gm._num_items(), columns=gm._list_columns())
        sys.stdout.write(prompt)
        """,
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def test_ask_prompt_stable():
    """Verify ask prompt is identical across Python sessions."""
    p1 = _build_prompt_in_subprocess("ask")
    p2 = _build_prompt_in_subprocess("ask")
    if p1 != p2:
        raise AssertionError(
            "Ask prompt changed between sessions.\n\n" + first_diff_block(p1, p2),
        )


def test_update_prompt_stable():
    """Verify update prompt is identical across Python sessions."""
    p1 = _build_prompt_in_subprocess("update")
    p2 = _build_prompt_in_subprocess("update")
    if p1 != p2:
        raise AssertionError(
            "Update prompt changed between sessions.\n\n" + first_diff_block(p1, p2),
        )
