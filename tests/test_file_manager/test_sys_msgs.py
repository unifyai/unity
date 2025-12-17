"""System message tests for FileManager prompt builders.

FileManager uses a "slot-filling" pattern rather than PromptSpec, so these tests
validate the current structure while ensuring key elements are present.
"""

import re
import sys
import subprocess
import textwrap

from tests.assertion_helpers import (
    extract_tools_dict,
    assert_in_order,
    assert_time_footer,
    first_diff_block,
)


from unity.file_manager.prompt_builders import (
    build_file_manager_ask_prompt,
    build_file_manager_ask_about_file_prompt,
    build_file_manager_organize_prompt,
)


def _make_mock_tools() -> dict:
    """Create mock tool functions for testing."""

    def exists(path: str) -> bool:
        """Check if file exists."""
        return True

    def list_dir(path: str = "/") -> list:
        """List directory contents."""
        return []

    def list_columns(table: str, *, include_types: bool = True) -> dict:
        """List columns in a table."""
        return {}

    def tables_overview(*, file: str = None) -> dict:
        """Get tables overview."""
        return {}

    def schema_explain(*, table: str) -> str:
        """Explain a table schema."""
        return ""

    def filter_files(
        *,
        filter: str = None,
        tables: list = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list:
        """Filter files."""
        return []

    def search_files(
        *,
        references: dict,
        table: str = None,
        filter: str = None,
        k: int = 10,
    ) -> list:
        """Search files semantically."""
        return []

    def ask_about_file(file_path: str, question: str) -> str:
        """Ask a question about a specific file."""
        return ""

    def reduce(
        *,
        table: str,
        metric: str,
        keys: str,
        filter: str = None,
        group_by: str = None,
    ) -> dict:
        """Reduce/aggregate data."""
        return {}

    def filter_join(
        *,
        tables: list,
        join_expr: str,
        select: dict,
        result_where: str = None,
    ) -> list:
        """Filter with join."""
        return []

    def search_join(
        *,
        tables: list,
        join_expr: str,
        select: dict,
        references: dict,
        k: int = 10,
    ) -> list:
        """Search with join."""
        return []

    def filter_multi_join(*, joins: list, result_where: str = None) -> list:
        """Multi-step filter join."""
        return []

    def search_multi_join(*, joins: list, references: dict, k: int = 10) -> list:
        """Multi-step search join."""
        return []

    def stat(path_or_uri: str) -> dict:
        """Get file stat."""
        return {}

    def ask(text: str) -> str:
        """Ask for information."""
        return ""

    def rename_file(file_id_or_path: str, new_name: str) -> dict:
        """Rename file."""
        return {}

    def move_file(file_id_or_path: str, new_parent_path: str) -> dict:
        """Move file."""
        return {}

    def delete_file(file_id_or_path: str) -> dict:
        """Delete file."""
        return {}

    def sync(file_path: str) -> dict:
        """Sync file."""
        return {}

    return {
        "exists": exists,
        "list": list_dir,
        "list_columns": list_columns,
        "tables_overview": tables_overview,
        "schema_explain": schema_explain,
        "filter_files": filter_files,
        "search_files": search_files,
        "reduce": reduce,
        "filter_join": filter_join,
        "search_join": search_join,
        "filter_multi_join": filter_multi_join,
        "search_multi_join": search_multi_join,
        "stat": stat,
        "ask": ask,
        "ask_about_file": ask_about_file,
        "rename_file": rename_file,
        "move_file": move_file,
        "delete_file": delete_file,
        "sync": sync,
    }


def test_file_manager_ask_system_prompt_formatting():
    tools = _make_mock_tools()
    prompt = build_file_manager_ask_prompt(
        tools=tools,
        num_files=10,
    )

    # Check key structural elements
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt

    # FileManager-specific sections
    assert "Context map" in prompt
    assert "Discover schema" in prompt

    # Clarification sentence
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks (slot-filling pattern)
    assert_in_order(
        prompt,
        [
            "retrieving file information",  # Role
            "Do not ask the user questions",  # Clarification
            "Context map",  # Generic block
            "Discover schema",  # Retrieval usage
            "Tools (name",  # Tools
            "Current UTC time",  # Timestamp
        ],
    )

    assert_time_footer(prompt, "Current UTC time is ")
    print(
        "FileManager ask system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n"
        + prompt[:2000],
    )


def test_file_manager_ask_about_file_system_prompt_formatting():
    tools = _make_mock_tools()
    prompt = build_file_manager_ask_about_file_prompt(
        tools=tools,
    )

    # Check key structural elements
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt

    # FileManager-specific sections
    assert "analyzing the content of a specific file" in prompt
    assert "Context map" in prompt
    assert "Structured extraction" in prompt

    # Clarification sentence
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    assert_time_footer(prompt, "Current UTC time is ")
    print(
        "FileManager ask_about_file system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n"
        + prompt[:2000],
    )


def test_file_manager_organize_system_prompt_formatting():
    tools = _make_mock_tools()
    prompt = build_file_manager_organize_prompt(
        tools=tools,
        num_files=10,
    )

    # Check key structural elements
    tools_json = extract_tools_dict(prompt)
    assert set(tools_json.keys()) == set(tools.keys())
    assert "Tools (name" in prompt

    # FileManager organize-specific sections
    assert "organizing files" in prompt
    assert "Context map" in prompt
    assert "rename" in prompt.lower()
    assert "move" in prompt.lower()
    assert "delete" in prompt.lower()

    # Clarification sentence
    assert re.search(
        r"Do not ask the user questions in your final response\..*sensible defaults",
        prompt,
        re.S,
    )

    # Ordering checks
    assert_in_order(
        prompt,
        [
            "organizing files",
            "Do not ask the user questions",
            "Context map",
            "Tools (name",
            "Current UTC time",
        ],
    )

    assert_time_footer(prompt, "Current UTC time is ")
    print(
        "FileManager organize system message passed formatting checks;\n"
        "The following system message resulted in no assertion errors:\n\n\n"
        + prompt[:2000],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stability: prompts should be identical across serial builder calls
# ─────────────────────────────────────────────────────────────────────────────


def _build_prompt_in_subprocess(method: str) -> str:
    """
    Build the FileManager system prompt in a fresh Python process and return it.
    This ensures we catch differences that only manifest across Python sessions.
    """
    assert method in {"ask", "ask_about_file", "organize"}
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
        from unity.file_manager.prompt_builders import (
            build_file_manager_ask_prompt,
            build_file_manager_ask_about_file_prompt,
            build_file_manager_organize_prompt,
        )

        # Create mock tools
        def exists(path: str) -> bool: return True
        def list_dir(path: str = "/") -> list: return []
        def list_columns(table: str, *, include_types: bool = True) -> dict: return {{}}
        def tables_overview(*, file: str = None) -> dict: return {{}}
        def schema_explain(*, table: str) -> str: return ""
        def filter_files(*, filter: str = None, tables: list = None, offset: int = 0, limit: int = 100) -> list: return []
        def search_files(*, references: dict, table: str = None, filter: str = None, k: int = 10) -> list: return []
        def ask_about_file(file_path: str, question: str) -> str: return ""
        def reduce(*, table: str, metric: str, keys: str, filter: str = None, group_by: str = None) -> dict: return {{}}
        def filter_join(*, tables: list, join_expr: str, select: dict, result_where: str = None) -> list: return []
        def search_join(*, tables: list, join_expr: str, select: dict, references: dict, k: int = 10) -> list: return []
        def filter_multi_join(*, joins: list, result_where: str = None) -> list: return []
        def search_multi_join(*, joins: list, references: dict, k: int = 10) -> list: return []
        def stat(path_or_uri: str) -> dict: return {{}}
        def ask(text: str) -> str: return ""
        def rename_file(file_id_or_path: str, new_name: str) -> dict: return {{}}
        def move_file(file_id_or_path: str, new_parent_path: str) -> dict: return {{}}
        def delete_file(file_id_or_path: str) -> dict: return {{}}
        def sync(file_path: str) -> dict: return {{}}

        tools = {{
            "exists": exists, "list": list_dir, "list_columns": list_columns,
            "tables_overview": tables_overview, "filter_files": filter_files,
            "schema_explain": schema_explain, "search_files": search_files,
            "ask_about_file": ask_about_file, "reduce": reduce, "filter_join": filter_join,
            "search_join": search_join, "filter_multi_join": filter_multi_join,
            "search_multi_join": search_multi_join, "stat": stat, "ask": ask,
            "rename_file": rename_file, "move_file": move_file,
            "delete_file": delete_file, "sync": sync,
        }}

        if "{method}" == "ask":
            prompt = build_file_manager_ask_prompt(
                tools=tools, num_files=10,
            )
        elif "{method}" == "ask_about_file":
            prompt = build_file_manager_ask_about_file_prompt(
                tools=tools,
            )
        else:
            prompt = build_file_manager_organize_prompt(
                tools=tools, num_files=10,
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


def test_ask_prompt_stable():
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess("ask")
    p2 = _build_prompt_in_subprocess("ask")
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "Ask system prompt changed between separate Python sessions.\n\n" + snippet,
        )


def test_ask_about_file_prompt_stable():
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess("ask_about_file")
    p2 = _build_prompt_in_subprocess("ask_about_file")
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "Ask about file system prompt changed between separate Python sessions.\n\n"
            + snippet,
        )


def test_organize_prompt_stable():
    # Build prompts in two separate Python processes to catch cross-session drift
    p1 = _build_prompt_in_subprocess("organize")
    p2 = _build_prompt_in_subprocess("organize")
    if p1 != p2:
        snippet = first_diff_block(p1, p2, context=3, label_a="First", label_b="Second")
        raise AssertionError(
            "Organize system prompt changed between separate Python sessions.\n\n"
            + snippet,
        )
