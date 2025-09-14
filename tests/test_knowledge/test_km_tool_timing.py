from __future__ import annotations

import os
import time
import pytest

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project


def _enable_timing():
    os.environ["KNOWLEDGE_MANAGER_TOOL_TIMING"] = "1"
    # Keep prints off by default to keep CI logs clean
    # os.environ["KNOWLEDGE_MANAGER_TOOL_TIMING_PRINT"] = "1"


@pytest.mark.unit
@_handle_project
def test_tool_tables_overview_timing():
    _enable_timing()
    km = KnowledgeManager()
    t0 = time.perf_counter()
    tabs = km._tables_overview()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(tabs, dict)
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_create_table_and_add_rows_timing():
    _enable_timing()
    km = KnowledgeManager()
    km._create_table(name="PerfTable")
    t0 = time.perf_counter()
    resp = km._add_rows(
        table="PerfTable",
        rows=[{"item": "A", "qty": 1}, {"item": "B", "qty": 2}],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # Verify via a read instead of assuming a specific return type
    data = km._filter(tables=["PerfTable"])["PerfTable"]
    assert isinstance(data, list) and len(data) == 2
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_filter_timing():
    _enable_timing()
    km = KnowledgeManager()
    km._create_table(name="PerfFilter")
    km._add_rows(
        table="PerfFilter",
        rows=[{"x": 0}, {"x": 1}, {"x": 2}],
    )
    t0 = time.perf_counter()
    data = km._filter(filter="x > 1", tables=["PerfFilter"])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert (
        isinstance(data, dict)
        and data["PerfFilter"]
        and data["PerfFilter"][0]["x"] == 2
    )
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_create_empty_column_timing():
    _enable_timing()
    km = KnowledgeManager()
    km._create_table(name="PerfCols")
    t0 = time.perf_counter()
    out = km._create_empty_column(
        table="PerfCols",
        column_name="score",
        column_type="int",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(out, dict)
    cols = km._tables_overview(include_column_info=True)["PerfCols"]["columns"]
    assert "score" in cols
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_rename_column_timing():
    _enable_timing()
    km = KnowledgeManager()
    km._create_table(name="PerfRename")
    km._add_rows(table="PerfRename", rows=[{"x": 1}])
    t0 = time.perf_counter()
    out = km._rename_column(table="PerfRename", old_name="x", new_name="X")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(out, dict)
    data = km._filter(tables=["PerfRename"])["PerfRename"]
    assert "X" in data[0] and "x" not in data[0]
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_delete_column_timing():
    _enable_timing()
    km = KnowledgeManager()
    km._create_table(name="PerfDeleteCol")
    km._add_rows(table="PerfDeleteCol", rows=[{"x": 1, "y": 2}])
    t0 = time.perf_counter()
    out = km._delete_column(table="PerfDeleteCol", column_name="x")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(out, dict)
    data = km._filter(tables=["PerfDeleteCol"])["PerfDeleteCol"]
    assert "x" not in data[0]
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_delete_rows_timing():
    _enable_timing()
    km = KnowledgeManager()
    km._create_table(name="PerfDelRows")
    km._add_rows(table="PerfDelRows", rows=[{"x": i} for i in range(5)])
    t0 = time.perf_counter()
    res = km._delete_rows(filter="x >= 3", tables=["PerfDelRows"], limit=100)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(res, dict) and "PerfDelRows" in res
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@_handle_project
def test_tool_delete_tables_timing():
    _enable_timing()
    km = KnowledgeManager()
    km._create_table(name="TmpDelTable")
    t0 = time.perf_counter()
    km._delete_tables(tables="TmpDelTable")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    tabs = km._tables_overview()
    assert "TmpDelTable" not in tabs
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_tool_search_timing():
    _enable_timing()
    km = KnowledgeManager()
    tbl = "PerfSearch"
    km._create_table(name=tbl)
    km._add_rows(
        table=tbl,
        rows=[
            {"content": "banking and budgeting"},
            {"content": "random unrelated"},
        ],
    )
    t0 = time.perf_counter()
    nearest = km._search(table=tbl, references={"content": "banking"}, k=1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(nearest, list) and nearest
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_tool_search_join_timing():
    _enable_timing()
    km = KnowledgeManager()
    left = "PerfLeft"
    right = "PerfRight"
    km._create_table(name=left, columns={"lk": "int", "content": "str"})
    km._create_table(name=right, columns={"rk": "int", "tag": "str"})
    km._add_rows(
        table=left,
        rows=[
            {"lk": 1, "content": "short cheatsheet terminal"},
            {"lk": 2, "content": "long article"},
        ],
    )
    km._add_rows(
        table=right,
        rows=[
            {"rk": 1, "tag": "linux"},
            {"rk": 2, "tag": "misc"},
        ],
    )
    t0 = time.perf_counter()
    results = km._search_join(
        tables=[left, right],
        join_expr=f"{left}.lk == {right}.rk",
        select={
            f"{left}.content": "content",
            f"{right}.tag": "tag",
        },
        references={"content": "cheatsheet"},
        k=1,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(results, list) and results
    # assert elapsed_ms < X
    print(f"elapsed: {elapsed_ms} < X")
