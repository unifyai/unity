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


from unity.web_searcher.prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
)
from unity.web_searcher.web_searcher import WebSearcher


def _build_prompt_in_subprocess(method: str) -> str:
    """
    Build the WebSearcher system prompt in a fresh Python process and return it.
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
        from unity.web_searcher.web_searcher import WebSearcher
        from unity.web_searcher.prompt_builders import build_ask_prompt, build_update_prompt

        ws = WebSearcher()
        if "{method}" == "ask":
            tools = dict(ws.get_tools("ask"))
            prompt = build_ask_prompt(tools=tools)
        else:
            tools = dict(ws.get_tools("update"))
            prompt = build_update_prompt(tools=tools)
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
    ws = WebSearcher()
    tools = dict(ws.get_tools("ask"))
    prompt = build_ask_prompt(tools=tools)

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt
    # WebSearcher has its own Tools Available section (domain-specific)
    assert "Tools Available" in prompt
    assert "Parallelism and single" in prompt  # header starts with this substring
    # WebSearcher-specific sections
    assert "General Rules and Guidance" in prompt
    assert "Website-aware Routing" in prompt
    assert "Decision Policy and When to Stop" in prompt
    assert "Answer Requirements" in prompt
    # Clarification top sentence (no clarification tool provided → else-policy)
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks - use "Examples\n--------" to match the header, not examples in tool docs
    assert_in_order(
        prompt,
        [
            "web research assistant",
            "Do not ask the user questions in your final response",
            "Tools Available",
            "General Rules and Guidance",
            "Website-aware Routing",
            "Decision Policy and When to Stop",
            "Tools (name",
            "Examples\n--------",
            "Parallelism and single",
            "Current UTC time is ",
        ],
    )

    assert_selected_headers_have_blank_line(
        prompt,
        [
            "Tools Available",
            "General Rules and Guidance",
            "Website-aware Routing",
            "Decision Policy and When to Stop",
        ],
    )
    assert_section_spacing(prompt)
    assert_time_footer(prompt, "Current UTC time is ")
    print(
        "WebSearcher ask system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n" + prompt,
    )


def test_update_system_prompt_formatting():
    ws = WebSearcher()
    tools = dict(ws.get_tools("update"))
    prompt = build_update_prompt(tools=tools)

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Parallelism and single" in prompt
    # WebSearcher-specific sections
    assert "Tools Available" in prompt
    assert "Tool selection" in prompt
    assert "General Rules" in prompt
    assert "Security & Data Hygiene" in prompt
    # Clarification top sentence (no clarification tool provided → else-policy)
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks - note that some sections might appear in different positions
    # due to the usage_examples including Tool selection, General Rules, etc.
    assert_in_order(
        prompt,
        [
            "manages the WebSearcher configuration",
            "Tools Available",
            "Tools (name",
            "Tool selection",
            "General Rules",
            "Security & Data Hygiene",
            "Parallelism and single",
            "Current UTC time is ",
        ],
    )

    assert_section_spacing(prompt)
    assert_time_footer(prompt, "Current UTC time is ")
    print(
        "WebSearcher update system message passed formatting checks;\n"
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
