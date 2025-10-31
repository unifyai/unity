import re

from tests.assertion_helpers import (
    extract_tools_dict,
    assert_in_order,
    assert_section_spacing,
    assert_selected_headers_have_blank_line,
    assert_time_footer,
)


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


def test_contact_manager_ask_system_prompt_formatting():
    prompt = build_ask_prompt(
        tools=_tools_for_ask(),
        num_contacts=3,
        columns=[{"name": "first_name", "type": "str"}],
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(_tools_for_ask().keys())
    assert "Tools (name" in prompt
    assert re.search(
        r"There are currently\s+3\s+contacts\s+stored in a table with the following columns:",
        prompt,
    )
    assert "Special contacts" in prompt
    assert "contact_id==0 is the assistant" in prompt
    assert "contact_id==1 is the central user" in prompt
    assert "Images policy (when images are present)" in prompt
    assert "Images forwarding to nested tools" in prompt
    assert "Parallelism and single" in prompt  # header starts with this substring
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
            "There are currently 3 contacts stored in a table with the following columns:",
            "Tools (name",
            "Examples",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
            "Parallelism and single",
            "Special contacts",
            "Current UTC time is ",
        ],
    )

    assert_selected_headers_have_blank_line(
        prompt,
        [
            "Examples",
            "Special contacts",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
        ],
    )
    assert_section_spacing(prompt)
    assert_time_footer(prompt, "Current UTC time is ")
    print(
        "ContactManager ask system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n" + prompt,
    )


def test_contact_manager_update_system_prompt_formatting():
    prompt = build_update_prompt(
        tools=_tools_for_update(),
        num_contacts=3,
        columns=[{"name": "first_name", "type": "str"}],
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(_tools_for_update().keys())
    assert re.search(
        r"There are currently\s+3\s+contacts\s+stored in a table with the following columns:",
        prompt,
    )
    assert "Schemas" in prompt
    assert "Contact schema = " in prompt
    assert "ColumnType schema (for custom columns) = " in prompt
    assert "Do not create new columns if an alias already exists." in prompt
    assert "Images policy (when images are present)" in prompt
    assert "Images forwarding to nested tools" in prompt
    assert "Parallelism and single" in prompt
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
            "There are currently 3 contacts stored in a table with the following columns:",
            "Tools (name",
            "Tool selection",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
            "Parallelism and single",
            "Schemas",
            "Special contacts",
            "Do not create new columns if an alias already exists.",
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
        "ContactManager update system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n" + prompt,
    )
