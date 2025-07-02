"""
Integration tests for multi-step joins
======================================

A single ↔︎ join is not enough for many real-world questions.  These tests
ensure the LLM understands how to:

  1. join **A ↔︎ B** into a _temporary_ table (name must start with “_”);
  2. join that temporary table with **C**;
  3. remove the temporary artefact via `_delete_tables`.

Two independent scenarios are covered:

  •  *Authors → Books → Reviews* – counting the reviews for J.K. Rowling.
  •  *Customers → Orders → OrderItems* – summing the quantity Alice bought.
"""

from __future__ import annotations

import re
import json
import pytest
import functools

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project
from tests.assertion_helpers import assertion_failed


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _contains(text: str, *needles: str) -> bool:
    """Return True iff every *needle* regex is present (case-insensitive)."""
    return all(re.search(n, text, re.I) for n in needles)


# --------------------------------------------------------------------------- #
# 1️⃣  AUTHORS – BOOKS – REVIEWS                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.eval
@pytest.mark.asyncio
# @pytest.mark.timeout(300)
@_handle_project
async def test_multi_join_author_reviews(monkeypatch):
    """
    Q: *“How many reviews have been written for books by J.K. Rowling?”*

    Expected answer … **3**.
    Must use **two** joins and delete the temporary table.
    """

    # 🔧 Spies ----------------------------------------------------------------
    join_calls: list[dict] = []
    deleted_tables: list[str] = []

    orig_join = KnowledgeManager._search_join
    orig_delete = KnowledgeManager._delete_tables

    @functools.wraps(orig_join)
    def _join_spy(self, *args, **kwargs):
        join_calls.append(kwargs.copy())
        return orig_join(self, *args, **kwargs)

    @functools.wraps(orig_delete)
    def _delete_spy(self, *args, **kwargs):
        # record explicit table-names …
        if "tables" in kwargs and kwargs["tables"] is not None:
            tbls = kwargs["tables"]
            deleted_tables.extend(tbls if isinstance(tbls, list) else [tbls])

        # … and prefix deletions done via `startswith`
        if "startswith" in kwargs and kwargs["startswith"]:
            deleted_tables.append(kwargs["startswith"])

        return orig_delete(self, *args, **kwargs)

    monkeypatch.setattr(KnowledgeManager, "_search_join", _join_spy, raising=True)
    monkeypatch.setattr(KnowledgeManager, "_delete_tables", _delete_spy, raising=True)

    # 🗄️  Seed data -----------------------------------------------------------
    km = KnowledgeManager()

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

    # ❓  Ask the multi-hop question ------------------------------------------
    handle = await km.ask(
        "How many reviews have been written for books by J.K. Rowling?  "
        "First join Authors with Books in a private table (name MUST start "
        "with an underscore), then join that temp table with Reviews, and "
        "delete the temp table when done.  Please use `_search_join` as "
        "described.",
        _return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()

    # ✅  Assertions -----------------------------------------------------------
    assert _contains(answer, r"\b3\b") or _contains(answer, "three"), assertion_failed(
        "Answer containing review count 3",
        answer,
        reasoning,
        "Incorrect number of reviews returned",
        {"All Knowledge": json.dumps(km._search())},
    )

    # exactly TWO joins
    assert len(join_calls) >= 2, "Expected at least two _search_join calls."

    # first join should create a *_temp* table
    first_new_table = join_calls[0].get("new_table")
    assert first_new_table and first_new_table.startswith(
        "_",
    ), f"First join should create a private temp table – got {first_new_table!r}"

    # subsequent join should read from that table (new_table None is OK)
    involved_tables = join_calls[1].get("tables")
    assert (
        first_new_table in involved_tables
    ), "Second join did not use the temp table produced by the first join."

    # temp table must be explicitly deleted
    assert any(
        first_new_table == d or first_new_table.startswith(d) for d in deleted_tables
    ), "Temporary table was not deleted"


# --------------------------------------------------------------------------- #
# 2️⃣  CUSTOMERS – ORDERS – ORDERITEMS                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_multi_join_customer_item_sum(monkeypatch):
    """
    Q: *“How many items has Alice bought in total?”*   (Expected: **7**)

    Reinforces that:

      • two joins are performed;
      • a private temp table is cleaned up afterwards.
    """

    join_calls: list[dict] = []
    deleted_tables: list[str] = []

    # patching …
    monkeypatch.setattr(
        KnowledgeManager,
        "_search_join",
        lambda self, *a, **k: (
            join_calls.append(k.copy())
            or KnowledgeManager._search_join.__wrapped__(self, *a, **k)
        ),
        raising=True,
    )
    monkeypatch.setattr(
        KnowledgeManager,
        "_delete_tables",
        lambda self, *a, **k: (
            deleted_tables.extend(
                (
                    k.get("tables", [])
                    if isinstance(k.get("tables", []), list)
                    else [k.get("tables")]
                ),
            )
            or KnowledgeManager._delete_tables.__wrapped__(self, *a, **k)
        ),
        raising=True,
    )

    # seed -------------------------------------------------------------------
    km = KnowledgeManager()

    km._create_table(
        name="Customers",
        columns={"customer_id": "int", "customer_name": "str"},
    )
    km._create_table(
        name="Orders",
        columns={"order_id": "int", "customer_id": "int"},
    )
    km._create_table(
        name="OrderItems",
        columns={"order_item_id": "int", "order_id": "int", "quantity": "int"},
    )

    km._add_rows(
        table="Customers",
        rows=[
            {"customer_id": 1, "customer_name": "Alice"},
            {"customer_id": 2, "customer_name": "Bob"},
        ],
    )
    km._add_rows(
        table="Orders",
        rows=[
            {"order_id": 10, "customer_id": 1},
            {"order_id": 11, "customer_id": 1},
            {"order_id": 12, "customer_id": 2},
        ],
    )
    km._add_rows(
        table="OrderItems",
        rows=[
            {"order_item_id": 100, "order_id": 10, "quantity": 2},
            {"order_item_id": 101, "order_id": 10, "quantity": 1},
            {"order_item_id": 102, "order_id": 11, "quantity": 4},
            {"order_item_id": 103, "order_id": 12, "quantity": 3},
        ],
    )

    # ask --------------------------------------------------------------------
    handle = await km.ask(
        "Alice would like to know how many items she has purchased in total. "
        "Please: (a) join Customers with Orders into a private temporary table; "
        "(b) join that result with OrderItems; (c) sum the `quantity` column; "
        "(d) delete the temp table.  Use `_search_join` twice.",
        _return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()

    # check ------------------------------------------------------------------
    assert _contains(answer, r"\b7\b") or _contains(answer, "seven"), assertion_failed(
        "Answer containing quantity 7",
        answer,
        reasoning,
        "Incorrect total quantity",
        {"All Knowledge": json.dumps(km._search())},
    )

    assert len(join_calls) >= 2, "Should perform at least two joins."

    temp_table = join_calls[0].get("new_table")
    assert temp_table and temp_table.startswith(
        "_",
    ), "First join must write into a private temp table."

    assert temp_table in (
        join_calls[1].get("tables") or []
    ), "Second join didn't consume the first join's result."

    assert temp_table in deleted_tables, "Temp table not cleaned up."
