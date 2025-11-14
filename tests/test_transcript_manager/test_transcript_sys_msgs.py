import re

from tests.assertion_helpers import (
    extract_tools_dict,
    assert_in_order,
    assert_section_spacing,
    assert_selected_headers_have_blank_line,
    assert_time_footer,
)

from unity.transcript_manager.prompt_builders import build_ask_prompt
from unity.transcript_manager.transcript_manager import TranscriptManager


def test_transcript_manager_ask_system_prompt_formatting():
    tm = TranscriptManager()
    tools = dict(tm.get_tools("ask"))

    prompt = build_ask_prompt(
        tools=tools,
        num_messages=tm._num_messages(),
        transcript_columns=tm._list_columns(),
        contact_columns=tm._contact_manager._list_columns(),
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt

    # Counts line should reflect the real number of messages
    m = re.search(
        r"There are currently\s+(\d+)\s+messages\s+stored in a table with the following sections:",
        prompt,
    )
    assert m, "Missing counts/sections line"
    assert int(m.group(1)) == tm._num_messages()

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

    # Ordering checks (build the dynamic counts line fragment)
    counts_line = f"There are currently {tm._num_messages()} messages stored in a table with the following sections:"
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "Two-table reasoning:",
            counts_line,
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
        "The following system message resulted in no assertion errors:\n\n\n" + prompt,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stability: prompts should be identical across serial builder calls
# ─────────────────────────────────────────────────────────────────────────────


def test_transcript_manager_ask_prompt_is_stable_across_serial_builds():
    tm = TranscriptManager()
    tools = dict(tm.get_tools("ask"))

    p1 = build_ask_prompt(
        tools=tools,
        num_messages=tm._num_messages(),
        transcript_columns=tm._list_columns(),
        contact_columns=tm._contact_manager._list_columns(),
    )
    p2 = build_ask_prompt(
        tools=tools,
        num_messages=tm._num_messages(),
        transcript_columns=tm._list_columns(),
        contact_columns=tm._contact_manager._list_columns(),
    )

    assert (
        p1 == p2
    ), f"TranscriptManager.ask system prompt changed between serial builds.\n\nFirst:\n\n{p1}\n\nSecond:\n\n{p2}"
