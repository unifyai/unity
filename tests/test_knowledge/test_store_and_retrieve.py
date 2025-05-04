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

from knowledge.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _contains(text: str, *needles: str) -> bool:
    """Return True when every needle appears (case-insensitive)."""
    return all(re.search(n, text, re.I) for n in needles)


# --------------------------------------------------------------------------- #
# 1.  Basic single-fact storage                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.timeout(120)
@_handle_project
def test_store_simple_fact():
    km = KnowledgeManager()
    km.start()

    km.store("Please remember that Adrian was born in 1994.")

    all_data = km._search()
    assert _contains(json.dumps(all_data), "1994"), all_data


# --------------------------------------------------------------------------- #
# 2.  Basic single-fact round-trip                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.timeout(120)
@_handle_project
def test_round_trip_simple_fact():
    km = KnowledgeManager()
    km.start()

    km.store("Please remember that Adrian was born in 1994.")

    answer = km.retrieve("When was Adrian born?")
    assert _contains(answer, "1994"), answer


# --------------------------------------------------------------------------- #
# 3.  Schema expansion inside *one* table                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.timeout(180)
@_handle_project
def test_schema_expands_and_new_field_retrievable():
    """
    • First fact gives Bob only ‘age’.
    • Second fact adds two *previously unseen* attributes.
    • We can immediately query one of the new attributes.
    """
    km = KnowledgeManager()
    km.start()

    km.store("Remember that Bob is 35 years old.")
    km.store(
        "Also remember that Bob's favourite colour is green "
        "and his height is 180 centimetres.",
    )

    answer = km.retrieve("How tall is Bob?")
    assert _contains(answer, "180"), answer


# --------------------------------------------------------------------------- #
# 4.  Multiple tables & cross-table reasoning                                 #
# --------------------------------------------------------------------------- #


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
    km.start()

    km.store("Store that the Apple iPhone 15 costs 999 US dollars.")
    km.store(
        "Store that Daniel bought an iPhone 15 on 3 May 2025 " "using his credit card.",
    )

    answer = km.retrieve("How much did Daniel pay for his purchase?")
    assert _contains(answer, "999"), answer


# --------------------------------------------------------------------------- #
# 5.  Long multi-turn conversation with incremental updates                   #
# --------------------------------------------------------------------------- #


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
    km.start()

    km.store("Remember that Carol owns a dog named Fido.")
    km.store("Update: Carol also owns a cat named Luna.")

    answer = km.retrieve("List all of Carol's pets by name.")
    assert _contains(answer, "Fido", "Luna"), answer


# --------------------------------------------------------------------------- #
# 6.  Complex numeric scenario – implicit filtering                           #
# --------------------------------------------------------------------------- #


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
    km.start()

    km.store("Record that point P has coordinates x = 3 and y = 4.")
    km.store("Record that point Q has coordinates x = 1 and y = 10.")

    answer = km.retrieve(
        "Which points lie in the first quadrant but have y less than 5?",
    )
    assert "P" in answer and "Q" not in answer, answer
