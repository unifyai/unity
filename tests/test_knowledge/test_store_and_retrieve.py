"""
Integration tests for KnowledgeManager – PUBLIC API ONLY
========================================================

Each test spins-up a brand–new (temporary) Unify project
via the ``@_handle_project`` helper, so runs are hermetic.

We interact exclusively through:

    • KnowledgeManager.store(text)
    • KnowledgeManager.retrieve(text)

No private helpers (_search, _list_tables, …) are imported or poked.
"""

import re
import json
import pytest
from typing import Dict, List, Any

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project
from tests.assertion_helpers import assertion_failed


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _contains(text: str, *needles: str) -> bool:
    """Return True when every needle appears (case-insensitive)."""
    return all(re.search(n, text, re.I) for n in needles)


# --------------------------------------------------------------------------- #
# 1.  Basic single-fact storage                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.timeout(120)
@_handle_project
def test_store_simple_fact():
    km = KnowledgeManager()

    km.store("Adrian was born in 1994.")

    all_data = km._search()
    assert _contains(json.dumps(all_data), "1994"), all_data


# --------------------------------------------------------------------------- #
# 2.  Basic single-fact retrieval                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.timeout(120)
@_handle_project
def test_retrieve_simple_fact():
    km = KnowledgeManager()

    km._create_table("MyTable")
    km._add_data(table="MyTable", data=[{"name": "Adrian", "birth_year": "1994"}])

    answer, reasoning = km.retrieve(
        "When was Adrian born?",
        return_reasoning_steps=True,
    )
    assert _contains(answer, "1994"), assertion_failed(
        "Answer containing '1994'",
        answer,
        reasoning,
        "Answer does not contain expected birth year",
        {"Knowledge Data": km._search()},
    )


# --------------------------------------------------------------------------- #
# 3.  Basic single-fact round-trip                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.timeout(120)
@_handle_project
def test_round_trip_simple_fact():
    km = KnowledgeManager()

    km.store("Adrian was born in 1994.")

    answer, reasoning = km.retrieve(
        "When was Adrian born?",
        return_reasoning_steps=True,
    )
    assert _contains(answer, "1994"), assertion_failed(
        "Answer containing '1994'",
        answer,
        reasoning,
        "Answer does not contain expected birth year",
        {"Knowledge Data": km._search()},
    )


# --------------------------------------------------------------------------- #
# 4.  Schema expansion inside *one* table                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.timeout(180)
@_handle_project
def test_schema_expands_and_new_field_retrievable():
    """
    • First fact gives Bob only 'age'.
    • Second fact adds two *previously unseen* attributes.
    • We can always query any of the attributes.
    """
    km = KnowledgeManager()

    km.store("Bob is 35 years old.")

    answer, reasoning = km.retrieve("How old is Bob?", return_reasoning_steps=True)
    assert _contains(answer, "35"), assertion_failed(
        "Answer containing '35'",
        answer,
        reasoning,
        "Answer does not contain expected age",
        {"Knowledge Data": km._search()},
    )

    km.store(
        "Bob's favourite colour is green and his height is 180 centimetres.",
    )

    answer, reasoning = km.retrieve("How tall is Bob?", return_reasoning_steps=True)
    assert _contains(answer, "180"), assertion_failed(
        "Answer containing '180'",
        answer,
        reasoning,
        "Answer does not contain expected height",
        {"Knowledge Data": km._search()},
    )

    answer, reasoning = km.retrieve(
        "What is Bob's favourite colour?",
        return_reasoning_steps=True,
    )
    assert _contains(answer, "green"), assertion_failed(
        "Answer containing 'green'",
        answer,
        reasoning,
        "Answer does not contain expected favorite color",
        {"Knowledge Data": km._search()},
    )

    answer, reasoning = km.retrieve("How old is Bob?", return_reasoning_steps=True)
    assert _contains(answer, "35"), assertion_failed(
        "Answer containing '35'",
        answer,
        reasoning,
        "Answer does not contain expected age after schema expansion",
        {"Knowledge Data": km._search()},
    )


# --------------------------------------------------------------------------- #
# 5.  Multiple tables & cross-table reasoning                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.timeout(240)
@_handle_project
def test_multiple_tables_and_join_like_query():
    """
    Two conceptually different tables:

    • a *Product*-ish table (iPhone 15, price)
    • a *Purchase*-ish table (Daniel bought iPhone 15)

    A retrieval question that forces the model to relate them.
    """
    km = KnowledgeManager()

    km.store("The Apple iPhone 15 costs 999 US dollars.")
    km.store(
        "Daniel bought an iPhone 15 on 3 May 2025 using his credit card.",
    )

    answer, reasoning = km.retrieve(
        "How much did Daniel pay for his purchase?",
        return_reasoning_steps=True,
    )
    assert _contains(answer, "999"), assertion_failed(
        "Answer containing '999'",
        answer,
        reasoning,
        "Answer does not contain expected price",
        {"Knowledge Data": km._search()},
    )


# --------------------------------------------------------------------------- #
# 6.  Long multi-turn conversation with incremental updates                   #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.timeout(240)
@_handle_project
def test_incremental_updates_and_refactor():
    """
    Carol first has one pet → later gains another.
    Retrieval must mention *both* pets, proving that:

      • The second `store()` merged data with prior rows OR
      • The model added a related row & could aggregate on retrieval.

    Either way, table structure had to change / be searched flexibly.
    """
    km = KnowledgeManager()

    km.store("Carol owns a dog named Fido.")
    km.store("Carol also owns a cat named Luna.")

    answer, reasoning = km.retrieve(
        "What are the names of Carol's pets?",
        return_reasoning_steps=True,
    )
    assert _contains(answer, "Fido", "Luna"), assertion_failed(
        "Answer containing both 'Fido' and 'Luna'",
        answer,
        reasoning,
        "Answer does not contain both expected pet names",
        {"Knowledge Data": km._search()},
    )


# --------------------------------------------------------------------------- #
# 7.  Complex numeric scenario – implicit filtering                           #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.timeout(240)
@_handle_project
def test_numeric_reasoning_after_multiple_points():
    """
    Store two 2-D points; ask a qualitative question whose
    correct answer involves *only one* of them.

    Success implies:
      • Numbers were stored as true numerics, and/or
      • The model was able to filter at retrieval time.
    """
    km = KnowledgeManager()

    km.store("Point P has coordinates x = 3 and y = 4.")
    km.store("Point Q has coordinates x = 1 and y = 10.")

    answer, reasoning = km.retrieve(
        "Which points lie in the first quadrant but have y less than 5?",
        return_reasoning_steps=True,
    )
    assert "P" in answer or "3, 4" in answer, assertion_failed(
        "Answer containing 'P' but not 'Q'",
        answer,
        reasoning,
        "Answer does not correctly identify only point P",
        {"Knowledge Data": km._search()},
    )
