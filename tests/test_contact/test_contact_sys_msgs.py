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
from unity.contact_manager.contact_manager import ContactManager


def test_contact_manager_ask_system_prompt_formatting():
    cm = ContactManager()
    tools = dict(cm.get_tools("ask"))
    prompt = build_ask_prompt(
        tools=tools,
        num_contacts=cm._num_contacts(),
        columns=cm._list_columns(),
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt
    m = re.search(
        r"There are currently\s+(\d+)\s+contacts\s+stored in a table with the following columns:",
        prompt,
    )
    assert m, "Missing counts/columns line"
    assert int(m.group(1)) == cm._num_contacts()
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
    counts_line = f"There are currently {cm._num_contacts()} contacts stored in a table with the following columns:"
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            counts_line,
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
        "The following system message resulted in no assertion errors:\n\n\n" + prompt,
    )


def test_contact_manager_update_system_prompt_formatting():
    cm = ContactManager()
    tools = dict(cm.get_tools("update"))
    prompt = build_update_prompt(
        tools=tools,
        num_contacts=cm._num_contacts(),
        columns=cm._list_columns(),
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    m = re.search(
        r"There are currently\s+(\d+)\s+contacts\s+stored in a table with the following columns:",
        prompt,
    )
    assert m, "Missing counts/columns line"
    assert int(m.group(1)) == cm._num_contacts()
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
    counts_line = f"There are currently {cm._num_contacts()} contacts stored in a table with the following columns:"
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
        "The following system message resulted in no assertion errors:\n\n\n" + prompt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stability: prompts should be identical across serial builder calls
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_prompt_for_stability(prompt: str) -> str:
    """
    Remove/normalize inherently dynamic parts from a system prompt to enable
    stable equality checks across serial builds.
    """
    # 1) Normalize the time footer line (e.g., "Current UTC time is 2025-01-01 12:34:56 UTC.")
    prompt = re.sub(
        r"Current UTC time is \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC\.\Z",
        "Current UTC time is <TIMESTAMP>.",
        prompt.strip(),
        flags=re.M,
    )
    # 2) Normalize any memory addresses that may appear in repr() of default values
    prompt = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", prompt)
    return prompt


def test_contact_manager_ask_prompt_is_stable_across_serial_builds():
    cm = ContactManager()
    tools = dict(cm.get_tools("ask"))

    p1 = build_ask_prompt(
        tools=tools,
        num_contacts=cm._num_contacts(),
        columns=cm._list_columns(),
    )
    p2 = build_ask_prompt(
        tools=tools,
        num_contacts=cm._num_contacts(),
        columns=cm._list_columns(),
    )

    n1 = _normalize_prompt_for_stability(p1)
    n2 = _normalize_prompt_for_stability(p2)
    assert (
        n1 == n2
    ), f"Ask system prompt changed between serial builds.\n\nFirst:\n\n{p1}\n\nSecond:\n\n{p2}"


def test_contact_manager_update_prompt_is_stable_across_serial_builds():
    cm = ContactManager()
    tools = dict(cm.get_tools("update"))

    p1 = build_update_prompt(
        tools=tools,
        num_contacts=cm._num_contacts(),
        columns=cm._list_columns(),
    )
    p2 = build_update_prompt(
        tools=tools,
        num_contacts=cm._num_contacts(),
        columns=cm._list_columns(),
    )

    n1 = _normalize_prompt_for_stability(p1)
    n2 = _normalize_prompt_for_stability(p2)
    assert (
        n1 == n2
    ), f"Update system prompt changed between serial builds.\n\nFirst:\n\n{p1}\n\nSecond:\n\n{p2}"
