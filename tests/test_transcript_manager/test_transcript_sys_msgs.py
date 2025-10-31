import re

from unity.transcript_manager.prompt_builders import build_ask_prompt


def _dummy(*args, **kwargs):
    pass


def _tools_for_ask():
    # Intentionally omit clarification and image tools to keep the footer last
    return {
        "filter_messages": _dummy,
        "search_messages": _dummy,
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


def _assert_time_footer_colon(prompt: str) -> None:
    non_empty_lines = [ln for ln in prompt.splitlines() if ln.strip()]
    assert non_empty_lines, (
        "Prompt should not be empty\n\nFull system prompt:\n" + prompt
    )
    last = non_empty_lines[-1]
    assert re.fullmatch(
        r"Current UTC time: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC\.",
        last,
    ), f"Unexpected last line: {last!r}\n\nFull system prompt:\n{prompt}"


def test_transcript_manager_system_prompt_formatting():
    prompt = build_ask_prompt(
        tools=_tools_for_ask(),
        num_messages=2,
        transcript_columns={"message_id": "int", "content": "str"},
        contact_columns={"contact_id": "int", "first_name": "str"},
    )

    _assert_selected_headers_have_blank_line(
        prompt,
        [
            "Examples",
            "Guidance on when to use which image tool",
            "Schemas",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
        ],
    )

    _assert_time_footer_colon(prompt)

    # Also print full prompt on success for quick inspection when running with -s
    print(
        "TranscriptManager ask system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n" + prompt,
    )


def test_transcript_manager_section_spacing_and_footer():
    prompt = build_ask_prompt(
        tools=_tools_for_ask(),
        num_messages=2,
        transcript_columns={"message_id": "int", "content": "str"},
        contact_columns={"contact_id": "int", "first_name": "str"},
    )

    _assert_section_spacing(prompt)
    _assert_time_footer_colon(prompt)
    print(
        "TranscriptManager ask system message passed spacing/footer checks;\n"
        "The following system message resulted in no assertion errors:\n" + prompt,
    )
