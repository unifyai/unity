import re

from tests.assertion_helpers import (
    extract_tools_dict,
    assert_in_order,
    assert_section_spacing,
    assert_selected_headers_have_blank_line,
    assert_time_footer,
)

from unity.transcript_manager.prompt_builders import build_ask_prompt


def _dummy(*args, **kwargs):
    pass


def _tools_for_ask():
    # Intentionally omit clarification and image tools to keep the footer last
    return {
        "filter_messages": _dummy,
        "search_messages": _dummy,
    }


def test_transcript_manager_ask_system_prompt_formatting():
    prompt = build_ask_prompt(
        tools=_tools_for_ask(),
        num_messages=2,
        transcript_columns={"message_id": "int", "content": "str"},
        contact_columns={"contact_id": "int", "first_name": "str"},
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(_tools_for_ask().keys())
    assert "Tools (name" in prompt
    assert re.search(
        r"There are currently\s+2\s+messages\s+stored in a table with the following sections:",
        prompt,
    )
    assert "Transcript columns" in prompt
    assert (
        "Sender contact columns (fields available on the Contacts table for the message sender)"
        in prompt
    )
    assert "Two-table reasoning:" in prompt
    assert "`search_messages`" in prompt and "`filter_messages`" in prompt
    assert "Images policy (when images are present)" in prompt
    assert "Images forwarding to nested tools" in prompt
    assert "Parallelism and single" in prompt
    # Clarification top sentence (no clarification tool provided → else-policy)
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )
    # Schemas and shorthand sections
    assert "Schemas" in prompt
    assert "Message field shorthand (full → shorthand)" in prompt
    assert "Message field shorthand (shorthand → full)" in prompt

    # Ordering checks
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "Two-table reasoning:",
            "There are currently 2 messages stored in a table with the following sections:",
            "Tools (name",
            "Examples",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
            "Parallelism and single",
            "Schemas",
            "Current UTC time is ",
        ],
    )

    assert_selected_headers_have_blank_line(
        prompt,
        [
            "Examples",
            "Guidance on when to use which image tool",
            "Schemas",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
        ],
    )

    assert_section_spacing(prompt)
    assert_time_footer(prompt, "Current UTC time is ")

    print(
        "TranscriptManager ask system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n" + prompt,
    )
