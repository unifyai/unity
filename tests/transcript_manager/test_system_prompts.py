import os
import re
import sys
import subprocess
import textwrap

from tests.assertion_helpers import (
    extract_tools_dict,
    assert_in_order,
    assert_section_spacing,
    assert_selected_headers_have_blank_line,
    assert_time_footer,
    first_diff_block,
)
from tests.helpers import _handle_project

from unity.transcript_manager.prompt_builders import build_ask_prompt
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.session_details import UNASSIGNED_USER_CONTEXT, UNASSIGNED_ASSISTANT_CONTEXT


def _build_prompt_in_subprocess(test_context: str) -> str:
    """
    Build the TranscriptManager.ask system prompt in a fresh Python process and return it.
    This catches cross-session instabilities (e.g., object-id defaults) while keeping
    the time footer deterministic by installing the same static_now override used in tests.

    The test_context is passed via environment variable to ensure the subprocess
    uses an isolated context rather than the shared default context.
    """
    code = textwrap.dedent(
        """
        import os, sys
        sys.path.insert(0, os.getcwd())
        import unify
        # Activate the test project before setting context
        project_name = os.environ.get("UNITY_TEST_PROJECT_NAME", "UnityTests")
        unify.activate(project_name, overwrite=False)
        # Set test-specific context before creating TranscriptManager to avoid races
        test_ctx = os.environ.get("_TEST_CONTEXT")
        if test_ctx:
            unify.set_context(test_ctx, relative=False)
        # Install a deterministic timestamp inside this fresh process
        import unity.common.prompt_helpers as _ph
        from datetime import datetime, timezone
        def _static_now(time_only: bool = False):
            dt = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
            label = "UTC"
            if time_only:
                return dt.strftime("%I:%M %p ") + label
            return dt.strftime("%A, %B %d, %Y at %I:%M %p ") + label
        _ph.now = _static_now

        from unity.transcript_manager.transcript_manager import TranscriptManager
        from unity.transcript_manager.prompt_builders import build_ask_prompt

        tm = TranscriptManager()
        tools = dict(tm.get_tools("ask"))
        prompt = build_ask_prompt(
            tools=tools,
            num_messages=tm._num_messages(),
            transcript_columns=tm._list_columns(),
            contact_columns=tm._contact_manager._list_columns(),
        ).flatten()
        sys.stdout.write(prompt)
        """,
    )
    env = os.environ.copy()
    env["_TEST_CONTEXT"] = test_context
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        env=env,
    )
    return proc.stdout


@_handle_project
def test_ask_system_prompt_formatting():
    tm = TranscriptManager()
    tools = dict(tm.get_tools("ask"))

    prompt = build_ask_prompt(
        tools=tools,
        num_messages=tm._num_messages(),
        transcript_columns=tm._list_columns(),
        contact_columns=tm._contact_manager._list_columns(),
    ).flatten()

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt

    # Schema-based two-table info: count line + data architecture
    m = re.search(r"There are currently\s+(\d+)\s+messages\.", prompt)
    assert m, "Missing counts line"
    assert int(m.group(1)) == tm._num_messages()

    # Two-table architecture explanation (references schemas)
    assert "Data architecture:" in prompt
    assert "**Transcripts table**" in prompt
    assert "**Contacts table**" in prompt
    assert "Columns defined in the Message schema above" in prompt
    assert "Columns defined in the Contact schema above" in prompt

    # Schemas rendered early
    assert "Schemas" in prompt
    assert "Contact = " in prompt
    assert "Message = " in prompt

    # Two-table reasoning guidance
    assert "Two-table reasoning:" in prompt
    assert "`search_messages`" in prompt and "`filter_messages`" in prompt

    # Standard blocks
    assert "Images policy (when images are present)" in prompt
    assert "Images forwarding to nested tools" in prompt
    assert "Parallelism and single" in prompt

    # Clarification top sentence (no clarification tool provided → else-policy)
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Shorthand sections
    assert "Message field shorthand (full → shorthand)" in prompt
    assert "Message field shorthand (shorthand → full)" in prompt

    # Ordering checks - schemas appear early, special blocks (two-table info) appear late
    counts_line = f"There are currently {tm._num_messages()} messages."
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "Two-table reasoning:",
            "Schemas",
            "Contact = ",
            "Message = ",
            "Tools (name",
            "Examples",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
            "Parallelism and single",
            counts_line,  # special_blocks come after parallelism
            "Data architecture:",
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


# ─────────────────────────────────────────────────────────────────────────────
# Stability: prompts should be identical across serial builder calls
# ─────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_ask_prompt_stability():
    # Build a test-specific context path matching _handle_project pattern
    test_ctx = f"tests/transcript_manager/test_system_prompts/test_ask_prompt_stability/{UNASSIGNED_USER_CONTEXT}/{UNASSIGNED_ASSISTANT_CONTEXT}"
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess(test_ctx)
    p2 = _build_prompt_in_subprocess(test_ctx)
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "TranscriptManager.ask system prompt changed between separate Python sessions.\n\n"
            + snippet,
        )
