import re

from tests.assertion_helpers import (
    extract_tools_dict,
    assert_in_order,
    assert_section_spacing,
    assert_time_footer,
)


from unity.task_scheduler.prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
    build_execute_prompt,
)
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.common.llm_helpers import methods_to_tool_dict


def test_task_scheduler_ask_system_prompt_formatting():
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


def test_task_scheduler_update_system_prompt_formatting():
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


def test_task_scheduler_execute_system_prompt_formatting():
    ts = TaskScheduler()
    # Build a tools dict that mirrors TaskScheduler.execute()
    tools = methods_to_tool_dict(
        # Read-only helpers
        ts.ask,
        ts._list_queues,
        ts._get_queue,
        # Start execution (public by-id helper)
        ts.execute_by_id,
        # Creation (name + description only)
        ts._create_task,
        include_class_name=False,
    )
    prompt = build_execute_prompt(tools=tools)

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Decision policy (isolation vs chain)" in prompt
    assert "CLARIFICATION POLICY (always prefer tool over prose)" in prompt
    assert "Reporting" in prompt
    assert "Images forwarding to nested tools" in prompt
    # Not included for execute
    assert "Parallelism and single" not in prompt
    assert "Images policy (when images are present)" not in prompt

    # Ordering checks
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "Decision policy (isolation vs chain)",
            "CLARIFICATION POLICY (always prefer tool over prose)",
            "Tools (name",
            "A. If the request contains a *numeric task_id*:",
            "Images forwarding to nested tools",
            "Current UTC time is ",
        ],
    )

    assert_section_spacing(prompt)
    assert_time_footer(prompt, "Current UTC time is ")
    print(
        "TaskScheduler execute system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n" + prompt,
    )
