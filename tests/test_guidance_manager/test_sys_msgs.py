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


from unity.guidance_manager.prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
)


def _make_mock_ask_tools():
    """Create mock tools for ask prompt testing (avoids GuidanceManager instantiation)."""

    def filter(**kwargs):
        return []

    def search(**kwargs):
        return []

    def list_columns(**kwargs):
        return {}

    def reduce(**kwargs):
        return {}

    return {
        "filter": filter,
        "search": search,
        "list_columns": list_columns,
        "reduce": reduce,
    }


def _make_mock_update_tools():
    """Create mock tools for update prompt testing (avoids GuidanceManager instantiation)."""

    def ask(**kwargs):
        return ""

    def add_guidance(**kwargs):
        return {}

    def update_guidance(**kwargs):
        return {}

    def remove_guidance(**kwargs):
        return {}

    def delete_guidance(**kwargs):
        return {}

    def create_custom_column(**kwargs):
        return {}

    def delete_custom_column(**kwargs):
        return {}

    return {
        "ask": ask,
        "add_guidance": add_guidance,
        "update_guidance": update_guidance,
        "remove_guidance": remove_guidance,
        "delete_guidance": delete_guidance,
        "create_custom_column": create_custom_column,
        "delete_custom_column": delete_custom_column,
    }


def _mock_columns():
    """Return mock columns that include both built-in and custom columns."""
    return {
        "guidance_id": "int",
        "topic": "str",
        "instruction": "str",
        "category": "str",
        "priority": "str",
        "active": "bool",
        # Custom column
        "custom_field": "str",
    }


def _build_prompt_in_subprocess(method: str) -> str:
    """
    Build the GuidanceManager system prompt in a fresh Python process and return it.
    This ensures we catch differences that only manifest across Python sessions.
    """
    assert method in {"ask", "update"}
    code = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, os.getcwd())
        # Install the same static timestamp override used by pytest's autouse fixture,
        # but inside this fresh process so the time footer is deterministic.
        import unity.common.prompt_helpers as _ph
        from datetime import datetime, timezone
        def _static_now(time_only: bool = False):
            dt = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
            label = "UTC"
            return (
                dt.strftime("%H:%M:%S ") + label
                if time_only
                else dt.strftime("%Y-%m-%d %H:%M:%S ") + label
            )
        _ph.now = _static_now
        from unity.guidance_manager.guidance_manager import GuidanceManager
        from unity.guidance_manager.prompt_builders import build_ask_prompt, build_update_prompt

        gm = GuidanceManager()
        if "{method}" == "ask":
            tools = dict(gm.get_tools("ask"))
            prompt = build_ask_prompt(
                tools=tools,
                num_items=gm._num_items(),
                columns=gm._list_columns(),
            )
        else:
            tools = dict(gm.get_tools("update"))
            prompt = build_update_prompt(
                tools=tools,
                num_items=gm._num_items(),
                columns=gm._list_columns(),
            )
        sys.stdout.write(prompt)
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


def test_ask_system_prompt_formatting():
    """Test ask prompt structure using mock tools (avoids database state issues)."""
    tools = _make_mock_ask_tools()
    num_items = 10
    columns = _mock_columns()

    prompt = build_ask_prompt(
        tools=tools,
        num_items=num_items,
        columns=columns,
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt

    # Schema-based table info: count line + schema reference
    m = re.search(r"There are currently\s+(\d+)\s+guidance entries\.", prompt)
    assert m, "Missing counts line"
    assert int(m.group(1)) == num_items
    assert "Columns are defined in the Guidance schema above." in prompt
    assert "Schemas" in prompt
    assert "Guidance = " in prompt  # Schema rendered early

    # Custom columns should appear separately
    assert "Additional custom columns:" in prompt
    assert '"custom_field"' in prompt

    assert "Images policy (when images are present)" in prompt
    assert "Images forwarding to nested tools" in prompt
    assert "Parallelism and single" in prompt  # header starts with this substring
    # Clarification top sentence (no clarification tool provided → else-policy)
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks - schemas now appear EARLY (before table info)
    counts_line = f"There are currently {num_items} guidance entries."
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "Schemas",  # Schemas appear early now
            "Guidance = ",  # Schema definition
            counts_line,  # Table info references schema
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
    print(
        "GuidanceManager ask system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n" + prompt,
    )


def test_update_system_prompt_formatting():
    """Test update prompt structure using mock tools (avoids database state issues)."""
    tools = _make_mock_update_tools()
    num_items = 10
    columns = _mock_columns()

    prompt = build_update_prompt(
        tools=tools,
        num_items=num_items,
        columns=columns,
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())

    # Schema-based table info: count line + schema reference
    m = re.search(r"There are currently\s+(\d+)\s+guidance entries\.", prompt)
    assert m, "Missing counts line"
    assert int(m.group(1)) == num_items
    assert "Columns are defined in the Guidance schema above." in prompt

    # Custom columns should appear separately
    assert "Additional custom columns:" in prompt
    assert '"custom_field"' in prompt

    assert "Schemas" in prompt
    assert "Guidance = " in prompt  # Schema
    assert "Images policy (when images are present)" in prompt
    assert "Images forwarding to nested tools" in prompt
    assert "Parallelism and single" in prompt
    # Clarification top sentence (no clarification tool provided → else-policy)
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks - schemas now appear EARLY (before table info)
    counts_line = f"There are currently {num_items} guidance entries."
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "Schemas",  # Schemas appear early now
            "Guidance = ",  # Schema definition
            counts_line,  # Table info references schema
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
    print(
        "GuidanceManager update system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n" + prompt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stability: prompts should be identical across serial builder calls
# ─────────────────────────────────────────────────────────────────────────────


def test_ask_prompt_stable():
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess("ask")
    p2 = _build_prompt_in_subprocess("ask")
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "Ask system prompt changed between separate Python sessions.\n\n" + snippet,
        )


def test_update_prompt_stable():
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess("update")
    p2 = _build_prompt_in_subprocess("update")
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "Update system prompt changed between separate Python sessions.\n\n"
            + snippet,
        )
