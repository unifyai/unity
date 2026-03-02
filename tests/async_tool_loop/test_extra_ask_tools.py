"""
tests/async_tool_loop/test_extra_ask_tools.py
==============================================

Regression tests for the ``extra_ask_tools`` parameter on
``start_async_tool_loop`` / ``ToolsData``.

``extra_ask_tools`` lets callers inject domain-specific read-only tools into
the ``handle.ask()`` inspection loop without monkeypatching
``task.get_ask_tools``.  These tests verify the plumbing at the ``ToolsData``
level (pure unit tests, no LLM calls).
"""

from unittest.mock import MagicMock

import pytest

from unity.common._async_tool.tools_data import ToolsData


def _make_tools_data(**extra_kw) -> ToolsData:
    """Create a minimal ToolsData with a no-op logger/client."""
    logger = MagicMock()
    logger.log_steps = False

    async def _noop():
        return "ok"

    return ToolsData(
        {"noop": _noop},
        client=MagicMock(),
        logger=logger,
        **extra_kw,
    )


# ---------------------------------------------------------------------------
# 1. extra_ask_tools are merged into get_ask_tools()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_extra_ask_tools_appear_in_get_ask_tools():
    """extra_ask_tools passed at construction are returned by get_ask_tools()."""

    async def ask_custom_status() -> str:
        return "custom status ok"

    td = _make_tools_data(
        extra_ask_tools={"ask_custom_status": ask_custom_status},
    )

    snapshot = td.get_ask_tools()
    assert (
        "ask_custom_status" in snapshot
    ), f"Extra ask tool missing from snapshot: {list(snapshot.keys())}"
    assert snapshot["ask_custom_status"] is ask_custom_status


# ---------------------------------------------------------------------------
# 2. extra_ask_tools coexist with completed + dynamic ask tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_extra_ask_tools_coexist_with_dynamic_and_completed():
    """extra_ask_tools, completed ask handles, and dynamic ask tools all
    appear in the merged snapshot without overwriting each other."""

    async def ask_extra() -> str:
        return "extra"

    async def ask_completed() -> str:
        return "completed"

    async def ask_dynamic() -> str:
        return "dynamic"

    td = _make_tools_data(
        extra_ask_tools={"ask_extra": ask_extra},
    )

    # Simulate a completed ask handle
    td._completed_ask_handles["ask_completed"] = ask_completed

    # Simulate a live dynamic ask tool
    td._dynamic_tools_ref = {"ask_dynamic": ask_dynamic, "other_tool": lambda: None}

    snapshot = td.get_ask_tools()

    assert "ask_extra" in snapshot, "Extra ask tool missing"
    assert "ask_completed" in snapshot, "Completed ask tool missing"
    assert "ask_dynamic" in snapshot, "Dynamic ask tool missing"
    assert "other_tool" not in snapshot, "Non-ask dynamic tools should be filtered"

    assert snapshot["ask_extra"] is ask_extra
    assert snapshot["ask_completed"] is ask_completed
    assert snapshot["ask_dynamic"] is ask_dynamic


# ---------------------------------------------------------------------------
# 3. Precedence: dynamic > extra > completed (most-live wins)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_extra_ask_tools_precedence():
    """When the same key exists in multiple sources, live dynamic tools win
    over extra_ask_tools, and extra_ask_tools win over completed handles."""

    sentinel_completed = lambda: "completed"
    sentinel_extra = lambda: "extra"
    sentinel_dynamic = lambda: "dynamic"

    td = _make_tools_data(
        extra_ask_tools={
            "ask_shared": sentinel_extra,
            "ask_extra_only": sentinel_extra,
        },
    )
    td._completed_ask_handles["ask_shared"] = sentinel_completed
    td._completed_ask_handles["ask_completed_only"] = sentinel_completed
    td._dynamic_tools_ref = {
        "ask_shared": sentinel_dynamic,
        "ask_dynamic_only": sentinel_dynamic,
    }

    snapshot = td.get_ask_tools()

    # Dynamic should win for the shared key
    assert snapshot["ask_shared"] is sentinel_dynamic
    # Each unique key should be present
    assert snapshot["ask_extra_only"] is sentinel_extra
    assert snapshot["ask_completed_only"] is sentinel_completed
    assert snapshot["ask_dynamic_only"] is sentinel_dynamic


# ---------------------------------------------------------------------------
# 4. ask_about_completed_tool meta-dispatcher excluded from get_ask_tools()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_get_ask_tools_excludes_ask_about_completed_tool():
    """The ``ask_about_completed_tool`` meta-dispatcher created by
    DynamicToolFactory must not leak into get_ask_tools().

    It routes by call_id into _completed_askable_tools, so including it
    alongside genuine per-tool ask functions (keyed by function name) causes
    signature mismatches when downstream code calls fn(question=...).
    """

    async def ask_real_tool() -> str:
        return "real"

    async def ask_about_completed_tool(tool_id: str, question: str) -> str:
        return "meta-dispatcher"

    td = _make_tools_data()
    td._dynamic_tools_ref = {
        "ask_real_tool": ask_real_tool,
        "ask_about_completed_tool": ask_about_completed_tool,
    }

    snapshot = td.get_ask_tools()

    assert "ask_real_tool" in snapshot, "Genuine ask tool should be included"
    assert "ask_about_completed_tool" not in snapshot, (
        "Meta-dispatcher ask_about_completed_tool must be excluded from "
        "get_ask_tools() to prevent signature mismatches in downstream consumers"
    )
