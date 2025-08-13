"""
Unit-style test for `_search_multi_join`
=======================================

We avoid the full LLM retrieval loop – instead we *directly* invoke the new
helper and spy on:

  • internal calls to `_search_join` (there must be one per join step);
  • automatic clean-up of every temporary context.
"""

from __future__ import annotations

import re
import functools

import pytest

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def _tmp_ctx_survivors(km: KnowledgeManager) -> list[str]:
    """Return *any* context names that look like a leftover temp join."""
    return [
        t
        for t in km._tables_overview(include_column_info=False).keys()
        if re.match(r"_tmp_mjoin_[0-9a-f]{6}", t)
    ]


# --------------------------------------------------------------------------- #
# actual test                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@_handle_project
def test_search_multi_join(monkeypatch):
    """
    Scenario: *Authors → Books → Reviews*  (expect three reviews).
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

    # ---------- spies ------------------------------------------------------
    join_calls = []

    original_join = KnowledgeManager._create_join

    @functools.wraps(original_join)
    def _join_spy(self, *a, **k):
        join_calls.append(k.copy())
        return original_join(self, *a, **k)

    monkeypatch.setattr(KnowledgeManager, "_create_join", _join_spy, raising=True)

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
    # ➊ correct row-count
    assert len(res) == 3, "Should return exactly three Rowling reviews."

    # ➋ internal two-table join used twice
    assert len(join_calls) == 2, "_search_join should be called once per step."

    # ➌ temp contexts cleaned up
    survivors = _tmp_ctx_survivors(km)
    assert not survivors, f"Temporary join contexts not deleted: {survivors}"
