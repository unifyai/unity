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


from unity.secret_manager.prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
)
from unity.secret_manager.secret_manager import SecretManager
from unity.session_details import DEFAULT_USER_CONTEXT, DEFAULT_ASSISTANT_CONTEXT


def _build_prompt_in_subprocess(method: str, test_context: str) -> str:
    """
    Build the SecretManager system prompt in a fresh Python process and return it.
    This ensures we catch differences that only manifest across Python sessions.

    The test_context is passed via environment variable to ensure the subprocess
    uses an isolated context rather than the shared default context.
    """
    assert method in {"ask", "update"}
    code = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, os.getcwd())
        import unify
        # Activate the test project before setting context
        project_name = os.environ.get("UNITY_TEST_PROJECT_NAME", "UnityTests")
        unify.activate(project_name, overwrite=False)
        # Set test-specific context before creating SecretManager to avoid races
        test_ctx = os.environ.get("_TEST_CONTEXT")
        if test_ctx:
            unify.set_context(test_ctx, relative=False)
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
        from unity.secret_manager.secret_manager import SecretManager
        from unity.secret_manager.prompt_builders import build_ask_prompt, build_update_prompt

        sm = SecretManager()
        if "{method}" == "ask":
            tools = dict(sm.get_tools("ask"))
            prompt = build_ask_prompt(tools=tools)
        else:
            tools = dict(sm.get_tools("update"))
            prompt = build_update_prompt(tools=tools)
        sys.stdout.write(prompt)
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
def test_ask_system_prompt_formatting():
    sm = SecretManager()
    tools = dict(sm.get_tools("ask"))
    prompt = build_ask_prompt(tools=tools)

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt
    # SecretManager doesn't have counts/columns block (fixed schema)
    assert "Parallelism and single" in prompt  # header starts with this substring
    assert "Security (CRITICAL)" in prompt
    # Clarification top sentence (no clarification tool provided → else-policy)
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks (no counts block for SecretManager)
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "Tools (name",
            "Examples",
            "Parallelism and single",
            "Security (CRITICAL)",
            "Current UTC time is ",
        ],
    )

    assert_selected_headers_have_blank_line(
        prompt,
        [
            "Examples",
            "Security (CRITICAL)",
        ],
    )
    assert_section_spacing(prompt)
    assert_time_footer(prompt, "Current UTC time is ")
    print(
        "SecretManager ask system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n" + prompt,
    )


@_handle_project
def test_update_system_prompt_formatting():
    sm = SecretManager()
    tools = dict(sm.get_tools("update"))
    prompt = build_update_prompt(tools=tools)

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Parallelism and single" in prompt
    assert "Security (CRITICAL)" in prompt
    # Clarification top sentence (no clarification tool provided → else-policy)
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks (no counts block for SecretManager)
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "Tools (name",
            "Tool selection",
            "Parallelism and single",
            "Security (CRITICAL)",
            "Current UTC time is ",
        ],
    )

    assert_selected_headers_have_blank_line(
        prompt,
        [
            "Tool selection",
            "Security (CRITICAL)",
        ],
    )
    assert_section_spacing(prompt)
    assert_time_footer(prompt, "Current UTC time is ")
    print(
        "SecretManager update system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n" + prompt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stability: prompts should be identical across serial builder calls
# ─────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_ask_prompt_stable():
    # Build a test-specific context path matching _handle_project pattern
    test_ctx = f"tests/test_secret_manager/test_sys_msgs/test_ask_prompt_stable/{DEFAULT_USER_CONTEXT}/{DEFAULT_ASSISTANT_CONTEXT}"
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess("ask", test_ctx)
    p2 = _build_prompt_in_subprocess("ask", test_ctx)
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "Ask system prompt changed between separate Python sessions.\n\n" + snippet,
        )


@_handle_project
def test_update_prompt_stable():
    # Build a test-specific context path matching _handle_project pattern
    test_ctx = f"tests/test_secret_manager/test_sys_msgs/test_update_prompt_stable/{DEFAULT_USER_CONTEXT}/{DEFAULT_ASSISTANT_CONTEXT}"
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess("update", test_ctx)
    p2 = _build_prompt_in_subprocess("update", test_ctx)
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "Update system prompt changed between separate Python sessions.\n\n"
            + snippet,
        )
