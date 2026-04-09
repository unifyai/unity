"""
Unit-style test for `_filter_multi_join`
=======================================

Verifies that KnowledgeManager delegates multi-join work to
DataManager.filter_multi_join and returns the correct rows.
"""

from __future__ import annotations

import functools

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project


@_handle_project
def test_filter_multi_join(monkeypatch):
    """
    Scenario: *Authors → Books → Reviews*  (expect three Rowling reviews).
    """

    km = KnowledgeManager()

    # ---------- seed -------------------------------------------------------
    km._create_table(
        name="Authors",
        columns={"author_id": "int", "author_name": "str"},
    )
    km._create_table(
        name="Books",
        columns={"book_id": "int", "author_id": "int", "title": "str"},
    )
    km._create_table(
        name="Reviews",
        columns={"review_id": "int", "book_id": "int", "rating": "int"},
    )

    km._add_rows(
        table="Authors",
        rows=[
            {"author_id": 1, "author_name": "J.K. Rowling"},
            {"author_id": 2, "author_name": "George R.R. Martin"},
        ],
    )
    km._add_rows(
        table="Books",
        rows=[
            {"book_id": 100, "author_id": 1, "title": "HP1"},
            {"book_id": 101, "author_id": 1, "title": "HP2"},
            {"book_id": 102, "author_id": 2, "title": "GoT"},
        ],
    )
    km._add_rows(
        table="Reviews",
        rows=[
            {"review_id": 1000, "book_id": 100, "rating": 5},
            {"review_id": 1001, "book_id": 100, "rating": 4},
            {"review_id": 1002, "book_id": 101, "rating": 3},
            {"review_id": 1003, "book_id": 102, "rating": 5},
        ],
    )

    # ---------- spy on DataManager.filter_multi_join -----------------------
    dm_calls: list[dict] = []
    dm = km._data_manager
    original_fm = dm.filter_multi_join

    @functools.wraps(original_fm)
    def _dm_spy(**kwargs):
        dm_calls.append(kwargs)
        return original_fm(**kwargs)

    monkeypatch.setattr(dm, "filter_multi_join", _dm_spy)

    # ---------- exercise ---------------------------------------------------
    pipeline = [
        {
            "tables": ["Authors", "Books"],
            "join_expr": "Authors.author_id == Books.author_id",
            "select": {"Books.book_id": "book_id"},
            "mode": "inner",
            "left_where": "author_name == 'J.K. Rowling'",
        },
        {
            "tables": ["$prev", "Reviews"],
            "join_expr": "$prev.book_id == Reviews.book_id",
            "select": {"Reviews.review_id": "review_id"},
        },
    ]

    res = km._filter_multi_join(joins=pipeline)

    # ---------- assertions -------------------------------------------------
    assert (
        len(res) == 3
    ), f"Should return exactly three Rowling reviews, got {len(res)}."

    assert (
        len(dm_calls) == 1
    ), "KM should delegate to DataManager.filter_multi_join exactly once."

    passed_joins = dm_calls[0]["joins"]
    assert len(passed_joins) == 2, "Both join steps should be forwarded to DM."

    for step in passed_joins:
        for t in step["tables"]:
            if t != "$prev":
                assert "Knowledge/" in t or t.startswith(
                    "$",
                ), f"Table names should be resolved to Knowledge/… paths, got {t!r}"
