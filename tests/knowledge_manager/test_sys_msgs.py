import os
import re
import sys
import subprocess
import textwrap

from tests.assertion_helpers import (
    extract_tools_dict,
    assert_in_order,
    assert_section_spacing,
    assert_time_footer,
    first_diff_block,
)
from tests.helpers import _handle_project


from unity.knowledge_manager.prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
    build_refactor_prompt,
)
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.session_details import DEFAULT_USER_CONTEXT, DEFAULT_ASSISTANT_CONTEXT


def _build_prompt_in_subprocess(
    method: str,
    test_context: str,
    table_schemas_json: str = "{}",
) -> str:
    """
    Build the KnowledgeManager system prompt in a fresh Python process and return it.
    This ensures we catch differences that only manifest across Python sessions.

    The test_context is passed via environment variable to ensure the subprocess
    uses an isolated context rather than the shared default context.
    """
    assert method in {"ask", "update", "refactor"}
    code = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, os.getcwd())
        import unify
        # Activate the test project before setting context
        project_name = os.environ.get("UNITY_TEST_PROJECT_NAME", "UnityTests")
        unify.activate(project_name, overwrite=False)
        # Set test-specific context before creating KnowledgeManager to avoid races
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
            if time_only:
                return dt.strftime("%I:%M %p ") + label
            return dt.strftime("%A, %B %d, %Y at %I:%M %p ") + label
        _ph.now = _static_now
        from unity.knowledge_manager.knowledge_manager import KnowledgeManager
        from unity.knowledge_manager.prompt_builders import (
            build_ask_prompt,
            build_update_prompt,
            build_refactor_prompt,
        )

        km = KnowledgeManager()
        table_schemas_json = '''{table_schemas_json}'''
        if "{method}" == "ask":
            tools = dict(km.get_tools("ask"))
            prompt = build_ask_prompt(tools=tools, table_schemas_json=table_schemas_json)
        elif "{method}" == "update":
            tools = dict(km.get_tools("update"))
            prompt = build_update_prompt(tools=tools, table_schemas_json=table_schemas_json)
        else:
            tools = dict(km.get_tools("refactor"))
            prompt = build_refactor_prompt(tools=tools, table_schemas_json=table_schemas_json)
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
    km = KnowledgeManager()
    tools = dict(km.get_tools("ask"))
    table_schemas_json = "{}"
    prompt = build_ask_prompt(tools=tools, table_schemas_json=table_schemas_json)

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt
    assert "Parallelism and single" in prompt  # parallelism guidance
    # KnowledgeManager-specific sections
    assert "Mandatory steps" in prompt
    assert "ColumnType Schema" in prompt
    assert "Current table schemas" in prompt
    assert "Examples" in prompt
    # Clarification top sentence (no clarification tool provided → else-policy)
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks
    assert_in_order(
        prompt,
        [
            "retrieve",
            "Mandatory steps",
            "Tools (name",
            "Examples",
            "Parallelism and single",
            "ColumnType Schema",
            "Current table schemas",
            "Current UTC time",
        ],
    )

    assert_section_spacing(prompt)
    assert_time_footer(prompt, "Current UTC time is ")
    print(
        "KnowledgeManager ask system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n" + prompt,
    )


@_handle_project
def test_update_system_prompt_formatting():
    km = KnowledgeManager()
    tools = dict(km.get_tools("update"))
    table_schemas_json = "{}"
    prompt = build_update_prompt(tools=tools, table_schemas_json=table_schemas_json)

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Parallelism and single" in prompt
    # KnowledgeManager-specific sections
    assert "Tool selection" in prompt
    assert "Ask vs Clarification" in prompt
    assert "Schema evolution" in prompt
    assert "ColumnType Schema" in prompt
    assert "Current table schemas" in prompt
    # Clarification top sentence (no clarification tool provided → else-policy)
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks
    assert_in_order(
        prompt,
        [
            "store",
            "Follow this workflow",
            "Tools (name",
            "Tool selection",
            "Parallelism and single",
            "ColumnType Schema",
            "Current table schemas",
            "Current UTC time",
        ],
    )

    assert_section_spacing(prompt)
    assert_time_footer(prompt, "Current UTC time is ")
    print(
        "KnowledgeManager update system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n" + prompt,
    )


@_handle_project
def test_refactor_system_prompt_formatting():
    km = KnowledgeManager()
    tools = dict(km.get_tools("refactor"))
    table_schemas_json = "{}"
    prompt = build_refactor_prompt(tools=tools, table_schemas_json=table_schemas_json)

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Parallelism and single" in prompt
    # KnowledgeManager refactor-specific sections
    assert "Current schema (JSON)" in prompt
    assert "How to work" in prompt
    assert "Tool availability groups" in prompt
    assert "Tool selection" in prompt
    # Clarification top sentence (no clarification tool provided → else-policy)
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks
    assert_in_order(
        prompt,
        [
            "Schema Refactor Assistant",
            "Current schema (JSON)",
            "How to work",
            "Tool availability groups",
            "Tools (name",
            "Tool selection",
            "Parallelism and single",
            "Current UTC time",
        ],
    )

    assert_section_spacing(prompt)
    assert_time_footer(prompt, "Current UTC time is ")
    print(
        "KnowledgeManager refactor system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n" + prompt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stability: prompts should be identical across serial builder calls
# ─────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_ask_prompt_stable():
    # Build a test-specific context path matching _handle_project pattern
    test_ctx = f"tests/knowledge_manager/test_sys_msgs/test_ask_prompt_stable/{DEFAULT_USER_CONTEXT}/{DEFAULT_ASSISTANT_CONTEXT}"
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
    test_ctx = f"tests/knowledge_manager/test_sys_msgs/test_update_prompt_stable/{DEFAULT_USER_CONTEXT}/{DEFAULT_ASSISTANT_CONTEXT}"
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess("update", test_ctx)
    p2 = _build_prompt_in_subprocess("update", test_ctx)
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "Update system prompt changed between separate Python sessions.\n\n"
            + snippet,
        )


@_handle_project
def test_refactor_prompt_stable():
    # Build a test-specific context path matching _handle_project pattern
    test_ctx = f"tests/knowledge_manager/test_sys_msgs/test_refactor_prompt_stable/{DEFAULT_USER_CONTEXT}/{DEFAULT_ASSISTANT_CONTEXT}"
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess("refactor", test_ctx)
    p2 = _build_prompt_in_subprocess("refactor", test_ctx)
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "Refactor system prompt changed between separate Python sessions.\n\n"
            + snippet,
        )
