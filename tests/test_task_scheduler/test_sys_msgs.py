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


from unity.task_scheduler.prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
)
from unity.task_scheduler.task_scheduler import TaskScheduler


def _build_prompt_in_subprocess(method: str) -> str:
    """
    Build the TaskScheduler system prompt in a fresh Python process and return it.
    Installs the same static time override used in tests so time is deterministic,
    and catches any cross-session instability in prompt composition.
    """
    assert method in {"ask", "update"}
    code = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, os.getcwd())
        # Install a deterministic timestamp inside this fresh process
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

        from unity.task_scheduler.task_scheduler import TaskScheduler
        from unity.task_scheduler.prompt_builders import build_ask_prompt, build_update_prompt

        ts = TaskScheduler()
        if "{method}" == "ask":
            tools = dict(ts.get_tools("ask"))
            prompt = build_ask_prompt(
                tools=tools,
                num_tasks=ts._num_tasks(),
                columns=ts._list_columns(),
            )
        else:
            tools = dict(ts.get_tools("update"))
            prompt = build_update_prompt(
                tools=tools,
                num_tasks=ts._num_tasks(),
                columns=ts._list_columns(),
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
    ts = TaskScheduler()
    tools = dict(ts.get_tools("ask"))
    prompt = build_ask_prompt(
        tools=tools,
        num_tasks=ts._num_tasks(),
        columns=ts._list_columns(),
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt
    m = re.search(
        r"There are currently\s+(\d+)\s+tasks\s+stored in a table with the following columns:",
        prompt,
    )
    assert m, "Missing counts/columns line"
    assert int(m.group(1)) == ts._num_tasks()
    assert "Images-first workflow for ask()" in prompt
    assert "Images policy (when images are present)" in prompt
    assert "Images forwarding to nested tools" in prompt
    assert "Parallelism and single" in prompt
    assert "Schemas" in prompt and "Task schema = " in prompt
    # Clarification top sentence (no clarification tool provided → else-policy)
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks
    counts_line = f"There are currently {ts._num_tasks()} tasks stored in a table with the following columns:"
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            counts_line,
            "Tools (name",
            "Examples",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
            "Images-first workflow for ask()",
            "Parallelism and single",
            "Schemas",
            "Current UTC time is ",
        ],
    )

    assert_section_spacing(prompt)
    assert_time_footer(prompt, "Current UTC time is ")
    print(
        "TaskScheduler ask system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n" + prompt,
    )


def test_update_system_prompt_formatting():
    ts = TaskScheduler()
    tools = dict(ts.get_tools("update"))
    prompt = build_update_prompt(
        tools=tools,
        num_tasks=ts._num_tasks(),
        columns=ts._list_columns(),
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    m = re.search(
        r"There are currently\s+(\d+)\s+tasks\s+stored in a table with the following columns:",
        prompt,
    )
    assert m, "Missing counts/columns line"
    assert int(m.group(1)) == ts._num_tasks()
    assert "Images policy (when images are present)" in prompt
    assert "Images forwarding to nested tools" in prompt
    assert "Parallelism and single" in prompt
    assert "Schemas" in prompt and "Task schema = " in prompt
    # Clarification top sentence (no clarification tool provided → else-policy)
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks
    counts_line = f"There are currently {ts._num_tasks()} tasks stored in a table with the following columns:"
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            counts_line,
            "Tools (name",
            "Tool selection",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
            "Parallelism and single",
            "Schemas",
            "Current UTC time is ",
        ],
    )

    assert_section_spacing(prompt)
    assert_time_footer(prompt, "Current UTC time is ")
    print(
        "TaskScheduler update system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n" + prompt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stability: prompts should be identical across serial builder calls
# ─────────────────────────────────────────────────────────────────────────────


def test_ask_prompt_is_stable_across_serial_builds():
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess("ask")
    p2 = _build_prompt_in_subprocess("ask")
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "TaskScheduler.ask system prompt changed between separate Python sessions.\n\n"
            + snippet,
        )


def test_update_prompt_is_stable_across_serial_builds():
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess("update")
    p2 = _build_prompt_in_subprocess("update")
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "TaskScheduler.update system prompt changed between separate Python sessions.\n\n"
            + snippet,
        )
