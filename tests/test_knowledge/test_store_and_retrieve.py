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
import asyncio
import pytest

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
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_store_simple_fact():
    km = KnowledgeManager()

    handle = km.store("Adrian was born in 1994.")
    await handle.result()

    all_data = km._search()
    assert _contains(json.dumps(all_data), "1994"), all_data


# --------------------------------------------------------------------------- #
# 2.  Basic single-fact retrieval                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_retrieve_simple_fact():
    km = KnowledgeManager()

    km._create_table("MyTable")
    km._add_data(table="MyTable", data=[{"name": "Adrian", "birth_year": "1994"}])

    handle = km.retrieve(
        "When was Adrian born?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
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
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_round_trip_simple_fact():
    km = KnowledgeManager()

    handle = km.store("Adrian was born in 1994.")
    await handle.result()

    handle = km.retrieve(
        "When was Adrian born?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
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
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_schema_expands_and_new_field_retrievable():
    """
    • First fact gives Bob only 'age'.
    • Second fact adds two *previously unseen* attributes.
    • We can always query any of the attributes.
    """
    km = KnowledgeManager()

    handle = km.store("Bob is 35 years old.")
    await handle.result()

    handle = km.retrieve(
        "How old is Bob?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "35"), assertion_failed(
        "Answer containing '35'",
        answer,
        reasoning,
        "Answer does not contain expected age",
        {"Knowledge Data": km._search()},
    )

    handle = km.store(
        "Bob's favourite colour is green and his height is 180 centimetres.",
    )
    await handle.result()

    handle = km.retrieve(
        "How tall is Bob?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "180"), assertion_failed(
        "Answer containing '180'",
        answer,
        reasoning,
        "Answer does not contain expected height",
        {"Knowledge Data": km._search()},
    )

    handle = km.retrieve(
        "What is Bob's favourite colour?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "green"), assertion_failed(
        "Answer containing 'green'",
        answer,
        reasoning,
        "Answer does not contain expected favorite color",
        {"Knowledge Data": km._search()},
    )

    handle = km.retrieve(
        "How old is Bob?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
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
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_multiple_tables_and_join_like_query():
    """
    Two conceptually different tables:

    • a *Product*-ish table (iPhone 15, price)
    • a *Purchase*-ish table (Daniel bought iPhone 15)

    A retrieval question that forces the model to relate them.
    """
    km = KnowledgeManager()

    handle = km.store("The Apple iPhone 15 costs 999 US dollars.")
    await handle.result()

    handle = km.store(
        "Daniel bought an iPhone 15 on 3 May 2025 using his credit card.",
    )
    await handle.result()

    handle = km.retrieve(
        "How much did Daniel pay for his purchase?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
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
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_incremental_updates_and_refactor():
    """
    Carol first has one pet → later gains another.
    Retrieval must mention *both* pets, proving that:

      • The second `store()` merged data with prior rows OR
      • The model added a related row & could aggregate on retrieval.

    Either way, table structure had to change / be searched flexibly.
    """
    km = KnowledgeManager()

    handle = km.store("Carol owns a dog named Fido.")
    await handle.result()

    handle = km.store("Carol also owns a cat named Luna.")
    await handle.result()

    handle = km.retrieve(
        "What are the names of Carol's pets?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
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
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_numeric_reasoning_after_multiple_points():
    """
    Store two 2-D points; ask a qualitative question whose
    correct answer involves *only one* of them.

    Success implies:
      • Numbers were stored as true numerics, and/or
      • The model was able to filter at retrieval time.
    """
    km = KnowledgeManager()

    handle = km.store("Point P has coordinates x = 3 and y = 4.")
    await handle.result()

    handle = km.store("Point Q has coordinates x = 1 and y = 10.")
    await handle.result()

    handle = km.retrieve(
        "Which points lie in the first quadrant but have y less than 5?",
        return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert "P" in answer or "3, 4" in answer, assertion_failed(
        "Answer containing 'P' but not 'Q'",
        answer,
        reasoning,
        "Answer does not correctly identify only point P",
        {"Knowledge Data": km._search()},
    )


# --------------------------------------------------------------------------- #
# 8.  Store with interjection                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_store_interjection():
    """
    Test that we can interject during a store operation and
    the interjection is incorporated into the final result.
    """
    km = KnowledgeManager()

    # store some informatiion
    handle = km.store("Bob lives in Bangkok, Thailand.")
    # Mid-operation, add another detail that should also get stored.
    await handle.interject("He was born in 1990.")

    handle = km.retrieve("Which city does Bob live in and what is his age?")
    out = await handle.result()

    # The confirmation text returned by `store()` should include both pieces of information.
    assert _contains(out, "Bangkok", "1990"), assertion_failed(
        "Output containing both 'Bangkok' and '1990'",
        out,
        "Output does not contain both expected details about Bob",
        {"Knowledge Data": km._search()},
    )


# --------------------------------------------------------------------------- #
# 9.  Store with stop                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_store_stop():
    km = KnowledgeManager()

    # Provide multiple facts in one go so that cancelling halfway through still yields a partial, meaningful result.
    handle = km.store(
        "Bob lives in Bangkok. Alice is 30 years old. Carl is 25 years old.",
    )
    await asyncio.sleep(0.05)
    handle.stop()
    with pytest.raises(asyncio.CancelledError):
        await handle.result()
    assert handle.done()


# --------------------------------------------------------------------------- #
# 10. Retrieve with interjection                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_retrieve_interjection():
    """
    Test that we can interject during a retrieve operation and
    the interjection is incorporated into the final result.
    """
    km = KnowledgeManager()

    # Store some data first
    handle = km.store("Alice is 30 years old.")
    handle = km.store("Alice lives in New York.")
    await handle.result()

    # Now retrieve with interjection
    handle = km.retrieve("How old is Alice?")
    await handle.interject("Also, where does she live?")
    out = await handle.result()

    assert _contains(out, "30", "New York"), assertion_failed(
        "Output containing both '30' and 'New York'",
        out,
        "Output does not contain both expected details about Alice",
    )


# --------------------------------------------------------------------------- #
# 11. Retrieve with stop                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_retrieve_stop():
    """
    Test that we can stop a retrieve operation mid-execution
    """
    km = KnowledgeManager()

    # Store some data first
    handle = km.store(
        "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome.",
    )
    await handle.result()

    # Now retrieve with stop
    handle = km.retrieve("List the capitals of European countries.")
    await asyncio.sleep(0.05)
    handle.stop()
    with pytest.raises(asyncio.CancelledError):
        await handle.result()
    assert handle.done()
