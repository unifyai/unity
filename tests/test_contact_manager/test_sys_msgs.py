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


from unity.contact_manager.prompt_builders import (
    build_ask_prompt,
    build_update_prompt,
)


def _make_mock_ask_tools():
    """Create mock tools for ask prompt testing (avoids ContactManager instantiation)."""

    def filter_contacts(filter: str = None, limit: int = 100, offset: int = 0):
        return []

    def search_contacts(references: dict = None, k: int = 10):
        return []

    def list_columns(include_types: bool = True):
        return {}

    def reduce(metric: str = "count", keys: str = None, group_by: str = None):
        return {}

    return {
        "filter_contacts": filter_contacts,
        "search_contacts": search_contacts,
        "list_columns": list_columns,
        "reduce": reduce,
    }


def _make_mock_update_tools():
    """Create mock tools for update prompt testing (avoids ContactManager instantiation)."""

    def ask(text: str):
        return ""

    def create_contact(**kwargs):
        return {}

    def update_contact(contact_id: int, **kwargs):
        return {}

    def delete_contact(contact_id: int):
        return {}

    def merge_contacts(contact_id_1: int, contact_id_2: int, overrides: dict = None):
        return {}

    def create_custom_column(column_name: str, column_type: str = "str"):
        return {}

    def delete_custom_column(column_name: str):
        return {}

    return {
        "ask": ask,
        "create_contact": create_contact,
        "update_contact": update_contact,
        "delete_contact": delete_contact,
        "merge_contacts": merge_contacts,
        "create_custom_column": create_custom_column,
        "delete_custom_column": delete_custom_column,
    }


def _mock_columns():
    """Return mock columns that include both built-in and custom columns."""
    return {
        "contact_id": "int",
        "first_name": "str",
        "surname": "str",
        "email_address": "str",
        "phone_number": "str",
        "bio": "str",
        "rolling_summary": "str",
        "respond_to": "bool",
        "response_policy": "str",
        "timezone": "str",
        # Custom column
        "occupation": "str",
    }


def _build_prompt_in_subprocess(method: str) -> str:
    """
    Build the ContactManager system prompt in a fresh Python process and return it.
    This ensures we catch differences that only manifest across Python sessions.
    """
    assert method in {"ask", "update"}
    code = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, os.getcwd())
        # Install the same static timestamp override used by pytest's autouse fixture,
        # but inside this fresh process so the time footer is deterministic.
        import unity.common.prompt_helpers as _ph
        from datetime import datetime, timezone
        def _static_now(time_only: bool = False):
            dt = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
            label = "UTC"
            return (
                dt.strftime("%H:%M:%S ") + label
                if time_only
                else dt.strftime("%Y-%m-%d %H:%M:%S ") + label
            )
        _ph.now = _static_now
        from unity.contact_manager.contact_manager import ContactManager
        from unity.contact_manager.prompt_builders import build_ask_prompt, build_update_prompt

        cm = ContactManager()
        if "{method}" == "ask":
            tools = dict(cm.get_tools("ask"))
            prompt = build_ask_prompt(
                tools=tools,
                num_contacts=cm._num_contacts(),
                columns=cm._list_columns(),
            )
        else:
            tools = dict(cm.get_tools("update"))
            prompt = build_update_prompt(
                tools=tools,
                num_contacts=cm._num_contacts(),
                columns=cm._list_columns(),
            )
        sys.stdout.write(prompt)
        """,
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return proc.stdout


def test_ask_system_prompt_formatting():
    """Test ask prompt structure using mock tools (avoids database state issues)."""
    tools = _make_mock_ask_tools()
    num_contacts = 10
    columns = _mock_columns()

    prompt = build_ask_prompt(
        tools=tools,
        num_contacts=num_contacts,
        columns=columns,
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt

    # Schema-based table info: count line + schema reference
    m = re.search(r"There are currently\s+(\d+)\s+contacts\.", prompt)
    assert m, "Missing counts line"
    assert int(m.group(1)) == num_contacts
    assert "Columns are defined in the Contact schema above." in prompt
    assert "Schemas" in prompt
    assert "Contact = " in prompt  # Schema rendered early

    # Custom columns should appear separately
    assert "Additional custom columns:" in prompt
    assert '"occupation"' in prompt

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

    # Ordering checks - schemas now appear EARLY (before table info)
    counts_line = f"There are currently {num_contacts} contacts."
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "Schemas",  # Schemas appear early now
            "Contact = ",  # Schema definition
            counts_line,  # Table info references schema
            "Columns are defined in the Contact schema above.",
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


def test_update_system_prompt_formatting():
    """Test update prompt structure using mock tools (avoids database state issues)."""
    tools = _make_mock_update_tools()
    num_contacts = 10
    columns = _mock_columns()

    prompt = build_update_prompt(
        tools=tools,
        num_contacts=num_contacts,
        columns=columns,
    )

    # Standardized blocks
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())

    # Schema-based table info: count line + schema reference
    m = re.search(r"There are currently\s+(\d+)\s+contacts\.", prompt)
    assert m, "Missing counts line"
    assert int(m.group(1)) == num_contacts
    assert "Columns are defined in the Contact schema above." in prompt

    # Custom columns should appear separately
    assert "Additional custom columns:" in prompt
    assert '"occupation"' in prompt

    assert "Schemas" in prompt
    assert "Contact = " in prompt  # Primary schema
    assert "ColumnType (for custom columns) = " in prompt  # Secondary schema
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

    # Ordering checks - schemas now appear EARLY (before table info)
    counts_line = f"There are currently {num_contacts} contacts."
    assert_in_order(
        prompt,
        [
            "Do not ask the user questions in your final response",
            "Schemas",  # Schemas appear early now
            "Contact = ",  # Schema definition
            counts_line,  # Table info references schema
            "Columns are defined in the Contact schema above.",
            "Tools (name",
            "Tool selection",
            "Images policy (when images are present)",
            "Images forwarding to nested tools",
            "Parallelism and single",
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


def test_ask_prompt_stable():
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess("ask")
    p2 = _build_prompt_in_subprocess("ask")
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "Ask system prompt changed between separate Python sessions.\n\n" + snippet,
        )


def test_update_prompt_stable():
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess("update")
    p2 = _build_prompt_in_subprocess("update")
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "Update system prompt changed between separate Python sessions.\n\n"
            + snippet,
        )
