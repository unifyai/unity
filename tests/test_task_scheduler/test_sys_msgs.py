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


def _make_mock_ask_tools():
    """Create mock tools for ask prompt testing (avoids TaskScheduler instantiation)."""

    def filter_tasks(**kwargs):
        return []

    def search_tasks(**kwargs):
        return []

    def reduce(**kwargs):
        return {}

    def list_queues():
        return []

    def get_queue(**kwargs):
        return {}

    def get_queue_for_task(**kwargs):
        return {}

    def ContactManager_ask(**kwargs):
        return ""

    return {
        "filter_tasks": filter_tasks,
        "search_tasks": search_tasks,
        "reduce": reduce,
        "list_queues": list_queues,
        "get_queue": get_queue,
        "get_queue_for_task": get_queue_for_task,
        "ContactManager_ask": ContactManager_ask,
    }


def _make_mock_update_tools():
    """Create mock tools for update prompt testing (avoids TaskScheduler instantiation)."""

    def ask(**kwargs):
        return ""

    def create_task(**kwargs):
        return {}

    def create_tasks(**kwargs):
        return []

    def delete_task(**kwargs):
        return {}

    def cancel_tasks(**kwargs):
        return []

    def update_task(**kwargs):
        return {}

    def list_queues():
        return []

    def get_queue(**kwargs):
        return {}

    def get_queue_for_task(**kwargs):
        return {}

    def set_queue(**kwargs):
        return {}

    def reorder_queue(**kwargs):
        return {}

    def move_tasks_to_queue(**kwargs):
        return {}

    def partition_queue(**kwargs):
        return {}

    def set_schedules_atomic(**kwargs):
        return {}

    def reinstate_task_to_previous_queue(**kwargs):
        return {}

    def ContactManager_ask(**kwargs):
        return ""

    return {
        "ask": ask,
        "create_task": create_task,
        "create_tasks": create_tasks,
        "delete_task": delete_task,
        "cancel_tasks": cancel_tasks,
        "update_task": update_task,
        "list_queues": list_queues,
        "get_queue": get_queue,
        "get_queue_for_task": get_queue_for_task,
        "set_queue": set_queue,
        "reorder_queue": reorder_queue,
        "move_tasks_to_queue": move_tasks_to_queue,
        "partition_queue": partition_queue,
        "set_schedules_atomic": set_schedules_atomic,
        "reinstate_task_to_previous_queue": reinstate_task_to_previous_queue,
        "ContactManager_ask": ContactManager_ask,
    }


def _mock_columns():
    """Return mock columns that include both built-in and custom columns."""
    return {
        "task_id": "int",
        "instance_id": "int",
        "queue_id": "int",
        "name": "str",
        "description": "str",
        "status": "str",
        "schedule": "dict",
        "trigger": "dict",
        "deadline": "datetime",
        "repeat": "list",
        "priority": "str",
        "response_policy": "str",
        "entrypoint": "int",
        "activated_by": "dict",
        "info": "str",
        # Custom column
        "custom_field": "str",
    }


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
    """Test ask prompt structure using mock tools (avoids database state issues)."""
    tools = _make_mock_ask_tools()
    num_tasks = 10
    columns = _mock_columns()

    prompt = build_ask_prompt(
        tools=tools,
        num_tasks=num_tasks,
        columns=columns,
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt

    # Schema-based table info: count line + schema reference
    m = re.search(r"There are currently\s+(\d+)\s+tasks\.", prompt)
    assert m, "Missing counts line"
    assert int(m.group(1)) == num_tasks
    assert "Columns are defined in the Task schema above." in prompt
    assert "Schemas" in prompt
    assert "Task = " in prompt  # Schema rendered early

    # Custom columns should appear separately
    assert "Additional custom columns:" in prompt
    assert '"custom_field"' in prompt

    assert "Images-first workflow for ask()" in prompt
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
    counts_line = f"There are currently {num_tasks} tasks."
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "Schemas",  # Schemas appear early now
            "Task = ",  # Schema definition
            counts_line,  # Table info references schema
            "Columns are defined in the Task schema above.",
            "Tools (name",
            "Examples",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
            "Images-first workflow for ask()",
            "Parallelism and single",
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
    """Test update prompt structure using mock tools (avoids database state issues)."""
    tools = _make_mock_update_tools()
    num_tasks = 10
    columns = _mock_columns()

    prompt = build_update_prompt(
        tools=tools,
        num_tasks=num_tasks,
        columns=columns,
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())

    # Schema-based table info: count line + schema reference
    m = re.search(r"There are currently\s+(\d+)\s+tasks\.", prompt)
    assert m, "Missing counts line"
    assert int(m.group(1)) == num_tasks
    assert "Columns are defined in the Task schema above." in prompt

    # Custom columns should appear separately
    assert "Additional custom columns:" in prompt
    assert '"custom_field"' in prompt

    assert "Schemas" in prompt
    assert "Task = " in prompt  # Schema
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
    counts_line = f"There are currently {num_tasks} tasks."
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "Schemas",  # Schemas appear early now
            "Task = ",  # Schema definition
            counts_line,  # Table info references schema
            "Columns are defined in the Task schema above.",
            "Tools (name",
            "Tool selection",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
            "Parallelism and single",
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
