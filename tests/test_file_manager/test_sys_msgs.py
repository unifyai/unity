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
    assert_time_footer,
    first_diff_block,
)


from unity.file_manager.prompt_builders import (
    build_file_manager_ask_about_file_prompt,
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

    def describe(*, file_path: str = None, file_id: int = None) -> dict:
        """Get file storage layout with contexts and schemas."""
        return {}

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

    def visualize(
        *,
        tables: str,
        plot_type: str,
        x_axis: str,
        y_axis: str = None,
    ) -> dict:
        """Visualize data."""
        return {}

    return {
        "exists": exists,
        "list": list_dir,
        "list_columns": list_columns,
        "describe": describe,
        "filter_files": filter_files,
        "search_files": search_files,
        "reduce": reduce,
        "filter_join": filter_join,
        "search_join": search_join,
        "filter_multi_join": filter_multi_join,
        "search_multi_join": search_multi_join,
        "stat": stat,
        "visualize": visualize,
    }


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


# ─────────────────────────────────────────────────────────────────────────────
# Stability: prompts should be identical across serial builder calls
# ─────────────────────────────────────────────────────────────────────────────


def _build_prompt_in_subprocess(method: str) -> str:
    """
    Build the FileManager system prompt in a fresh Python process and return it.
    This ensures we catch differences that only manifest across Python sessions.
    """
    assert method in {"ask_about_file"}
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
            build_file_manager_ask_about_file_prompt,
        )

        # Create mock tools
        def exists(path: str) -> bool: return True
        def list_dir(path: str = "/") -> list: return []
        def list_columns(table: str, *, include_types: bool = True) -> dict: return {{}}
        def describe(*, file_path: str = None, file_id: int = None) -> dict: return {{}}
        def filter_files(*, filter: str = None, tables: list = None, offset: int = 0, limit: int = 100) -> list: return []
        def search_files(*, references: dict, table: str = None, filter: str = None, k: int = 10) -> list: return []
        def reduce(*, table: str, metric: str, keys: str, filter: str = None, group_by: str = None) -> dict: return {{}}
        def filter_join(*, tables: list, join_expr: str, select: dict, result_where: str = None) -> list: return []
        def search_join(*, tables: list, join_expr: str, select: dict, references: dict, k: int = 10) -> list: return []
        def filter_multi_join(*, joins: list, result_where: str = None) -> list: return []
        def search_multi_join(*, joins: list, references: dict, k: int = 10) -> list: return []
        def stat(path_or_uri: str) -> dict: return {{}}
        def visualize(*, tables: str, plot_type: str, x_axis: str, y_axis: str = None) -> dict: return {{}}

        tools = {{
            "exists": exists, "list": list_dir, "list_columns": list_columns,
            "describe": describe, "filter_files": filter_files,
            "search_files": search_files,
            "reduce": reduce, "filter_join": filter_join,
            "search_join": search_join, "filter_multi_join": filter_multi_join,
            "search_multi_join": search_multi_join, "stat": stat,
            "visualize": visualize,
        }}

        prompt = build_file_manager_ask_about_file_prompt(
            tools=tools,
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
