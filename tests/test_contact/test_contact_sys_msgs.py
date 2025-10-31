import re

import pytest

from unity.contact_manager.prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
)


def _dummy(*args, **kwargs):
    pass


def _tools_for_ask():
    # Intentionally omit clarification tool so the time footer is last
    return {
        "filter_contacts": _dummy,
        "search_contacts": _dummy,
        "list_columns": _dummy,
    }


def _tools_for_update():
    # Intentionally omit clarification tool so the time footer is last
    return {
        "create_contact": _dummy,
        "update_contact": _dummy,
        "delete_contact": _dummy,
        "merge_contacts": _dummy,
        "create_custom_column": _dummy,
        "delete_custom_column": _dummy,
        "ask": _dummy,
    }


def _assert_section_spacing(prompt: str) -> None:
    lines = prompt.splitlines()
    errors: list[str] = []
    for idx in range(len(lines) - 1):
        line = lines[idx]
        next_line = lines[idx + 1]
        if re.fullmatch(r"-+", next_line.strip()):
            if idx == 0 or lines[idx - 1].strip() != "":
                errors.append(f"Missing blank line before section header: '{line}'")
    assert not errors, "\n".join(errors) + f"\n\nFull system prompt:\n{prompt}"


def _assert_selected_headers_have_blank_line(prompt: str, titles: list[str]) -> None:
    lines = prompt.splitlines()
    missing: list[str] = []
    for i, line in enumerate(lines):
        title = line.strip()
        if title in titles:
            if i == 0 or lines[i - 1].strip() != "":
                missing.append(title)
    assert (
        not missing
    ), f"Missing blank line before: {missing}\n\nFull system prompt:\n{prompt}"


def _assert_time_footer_is(prompt: str) -> None:
    non_empty_lines = [ln for ln in prompt.splitlines() if ln.strip()]
    assert non_empty_lines, (
        "Prompt should not be empty\n\nFull system prompt:\n" + prompt
    )
    last = non_empty_lines[-1]
    assert re.fullmatch(
        r"Current UTC time is \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC\.",
        last,
    ), f"Unexpected last line: {last!r}\n\nFull system prompt:\n{prompt}"


@pytest.mark.parametrize(
    "builder,tools,kwargs,headers",
    [
        (
            build_ask_prompt,
            _tools_for_ask,
            {
                "num_contacts": 3,
                "columns": [{"name": "first_name", "type": "str"}],
            },
            [
                "Examples",
                "Special contacts",
                "Images policy (when images are present)",
                "Images forwarding to nested tools",
            ],
        ),
        (
            build_update_prompt,
            _tools_for_update,
            {
                "num_contacts": 3,
                "columns": [{"name": "first_name", "type": "str"}],
            },
            [
                "Tool selection",
                "Images policy (when images are present)",
                "Images forwarding to nested tools",
            ],
        ),
    ],
)
def test_contact_manager_system_prompts_formatting(builder, tools, kwargs, headers):
    prompt = builder(tools=tools(), **kwargs)

    # Selected top-level sections should have a blank line above
    _assert_selected_headers_have_blank_line(prompt, headers)

    # Footer: last non-empty line is always the UTC time sentence
    _assert_time_footer_is(prompt)


def test_contact_manager_system_prompts_global_spacing():
    # ask()
    ask_prompt = build_ask_prompt(
        tools=_tools_for_ask(),
        num_contacts=3,
        columns=[{"name": "first_name", "type": "str"}],
    )
    _assert_section_spacing(ask_prompt)
    _assert_time_footer_is(ask_prompt)

    # update()
    upd_prompt = build_update_prompt(
        tools=_tools_for_update(),
        num_contacts=3,
        columns=[{"name": "first_name", "type": "str"}],
    )
    _assert_section_spacing(upd_prompt)
    _assert_time_footer_is(upd_prompt)
