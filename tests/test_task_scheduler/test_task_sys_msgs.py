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


def _dummy(*args, **kwargs):
    pass


def _tools_for_ask():
    # Omit clarification tool on purpose so the time line is the final non-empty line
    return {
        "filter_tasks": _dummy,
        "search_tasks": _dummy,
        "list_queues": _dummy,
        "get_queue": _dummy,
        "get_queue_for_task": _dummy,
        "ContactManager_ask": _dummy,
    }


def _tools_for_update():
    # Omit clarification tool on purpose so the time line is the final non-empty line
    return {
        "ask": _dummy,
        "create_task": _dummy,
        "create_tasks": _dummy,
        "delete_task": _dummy,
        "cancel_tasks": _dummy,
        "update_task": _dummy,
        # Queue helpers
        "list_queues": _dummy,
        "get_queue": _dummy,
        "get_queue_for_task": _dummy,
        "set_queue": _dummy,
        "reorder_queue": _dummy,
        "move_tasks_to_queue": _dummy,
        "partition_queue": _dummy,
        "reinstate_task_to_previous_queue": _dummy,
        "set_schedules_atomic": _dummy,
        # Contact manager lookup
        "ContactManager_ask": _dummy,
    }


def _tools_for_execute():
    # Omit clarification tool on purpose so the time line is the final non-empty line
    return {
        "ask": _dummy,
        "execute_by_id": _dummy,
        "execute_isolated_by_id": _dummy,
        "create_task": _dummy,
        # Read-only queue helpers
        "list_queues": _dummy,
        "get_queue": _dummy,
    }


def test_task_scheduler_ask_system_prompt_formatting():
    prompt = build_ask_prompt(
        tools=_tools_for_ask(),
        num_tasks=3,
        columns={"task_id": "int", "name": "str"},
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(_tools_for_ask().keys())
    assert "Tools (name" in prompt
    assert re.search(
        r"There are currently\s+3\s+tasks\s+stored in a table with the following columns:",
        prompt,
    )
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
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "There are currently 3 tasks stored in a table with the following columns:",
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
        "The following system message resulted in no assertion errors:\n" + prompt,
    )


def test_task_scheduler_update_system_prompt_formatting():
    prompt = build_update_prompt(
        tools=_tools_for_update(),
        num_tasks=3,
        columns={"task_id": {"type": "int"}, "name": {"type": "str"}},
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(_tools_for_update().keys())
    assert re.search(
        r"There are currently\s+3\s+tasks\s+stored in a table with the following columns:",
        prompt,
    )
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
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "There are currently 3 tasks stored in a table with the following columns:",
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
        "The following system message resulted in no assertion errors:\n" + prompt,
    )


def test_task_scheduler_execute_system_prompt_formatting():
    prompt = build_execute_prompt(tools=_tools_for_execute())

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(_tools_for_execute().keys())
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
        "The following system message resulted in no assertion errors:\n" + prompt,
    )
