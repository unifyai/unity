import re


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


def _assert_section_spacing(prompt: str) -> None:
    lines = prompt.splitlines()
    errors: list[str] = []
    for idx in range(len(lines) - 1):
        line = lines[idx]
        next_line = lines[idx + 1]
        if re.fullmatch(r"-+", next_line.strip()):
            # Expect a blank line before the section title line
            if idx == 0 or lines[idx - 1].strip() != "":
                errors.append(f"Missing blank line before section header: '{line}'")
    assert not errors, "\n".join(errors) + f"\n\nFull system prompt:\n{prompt}"


def _assert_time_footer(prompt: str) -> None:
    non_empty_lines = [ln for ln in prompt.splitlines() if ln.strip()]
    assert non_empty_lines, (
        "Prompt should not be empty\n\nFull system prompt:\n" + prompt
    )
    last = non_empty_lines[-1]
    assert re.fullmatch(
        r"Current UTC time is \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC\.",
        last,
    ), f"Unexpected last line: {last!r}\n\nFull system prompt:\n{prompt}"


def test_task_scheduler_ask_system_prompt_formatting():
    prompt = build_ask_prompt(
        tools=_tools_for_ask(),
        num_tasks=3,
        columns={"task_id": "int", "name": "str"},
    )

    _assert_section_spacing(prompt)
    _assert_time_footer(prompt)
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

    _assert_section_spacing(prompt)
    _assert_time_footer(prompt)
    print(
        "TaskScheduler update system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n" + prompt,
    )


def test_task_scheduler_execute_system_prompt_formatting():
    prompt = build_execute_prompt(tools=_tools_for_execute())

    _assert_section_spacing(prompt)
    _assert_time_footer(prompt)
    print(
        "TaskScheduler execute system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n" + prompt,
    )
