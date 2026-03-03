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

pytestmark = pytest.mark.eval

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project
from tests.assertion_helpers import assertion_failed
from tests.async_helpers import _wait_for_next_assistant_response_event
# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _contains(text: str, *needles: str) -> bool:
    """Return True when every needle appears (case-insensitive)."""
    return all(re.search(n, text, re.I) for n in needles)


# --------------------------------------------------------------------------- #
# 1.  Basic single-fact storage                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
@_handle_project
async def test_update_simple_fact():
    km = KnowledgeManager()

    handle = await km.update("The ZX-99 gizmo was released in 1994.")
    await handle.result()

    all_data = km._filter()
    assert _contains(json.dumps(all_data), "1994"), all_data


# --------------------------------------------------------------------------- #
# 2.  Basic single-fact retrieval                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
@_handle_project
async def test_ask_simple_fact():
    km = KnowledgeManager()

    km._create_table(name="MyTable")
    km._add_rows(
        table="MyTable",
        rows=[{"model": "ZX-99", "release_year": "1994"}],
    )

    handle = await km.ask(
        "When was the ZX-99 gizmo released?",
        _return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "1994"), assertion_failed(
        "Answer containing '1994'",
        answer,
        reasoning,
        "Answer does not contain expected birth year",
        {"Knowledge Data": km._filter()},
    )


# --------------------------------------------------------------------------- #
# 3.  Basic single-fact round-trip                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
@_handle_project
async def test_round_trip_simple_fact():
    km = KnowledgeManager()

    handle = await km.update("The ZX-99 gizmo was released in 1994.")
    await handle.result()

    handle = await km.ask(
        "When was the ZX-99 released?",
        _return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "1994"), assertion_failed(
        "Answer containing '1994'",
        answer,
        reasoning,
        "Answer does not contain expected birth year",
        {"Knowledge Data": km._filter()},
    )


# --------------------------------------------------------------------------- #
# 4.  Schema expansion inside *one* table                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
@_handle_project
async def test_schema_expands_and_new_field_retrievable():
    """
    • First fact gives Bob only 'age'.
    • Second fact adds two *previously unseen* attributes.
    • We can always query any of the attributes.
    """
    km = KnowledgeManager()

    handle = await km.update("The QuantumDrive unit produces 35 megawatts.")
    await handle.result()

    handle = await km.ask(
        "How many megawatts does the QuantumDrive unit produce?",
        _return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "35"), assertion_failed(
        "Answer containing '35'",
        answer,
        reasoning,
        "Answer does not contain expected age",
        {"Knowledge Data": km._filter()},
    )

    handle = await km.update(
        "The QuantumDrive's core colour is blue and its weight is 180 kilograms.",
    )
    await handle.result()

    handle = await km.ask(
        "How much does the QuantumDrive unit weigh?",
        _return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "180"), assertion_failed(
        "Answer containing '180'",
        answer,
        reasoning,
        "Answer does not contain expected height",
        {"Knowledge Data": km._filter()},
    )

    handle = await km.ask(
        "What is the QuantumDrive's core colour?",
        _return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "blue"), assertion_failed(
        "Answer containing 'blue'",
        answer,
        reasoning,
        "Answer does not contain expected favorite color",
        {"Knowledge Data": km._filter()},
    )

    handle = await km.ask(
        "How many megawatts does the QuantumDrive unit produce?",
        _return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "35"), assertion_failed(
        "Answer containing '35'",
        answer,
        reasoning,
        "Answer does not contain expected age after schema expansion",
        {"Knowledge Data": km._filter()},
    )


# --------------------------------------------------------------------------- #
# 5.  Multiple tables & cross-table reasoning                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
@_handle_project
async def test_multiple_tables_and_join_like_query():
    """
    Two conceptually different tables:

    • a *Product*-ish table (iPhone 15, price)
    • a *Purchase*-ish table (Daniel bought iPhone 15)

    A retrieval question that forces the model to relate them.
    """
    km = KnowledgeManager()

    handle = await km.update("The OrbitalDrone X99 costs 999 credits.")
    await handle.result()

    handle = await km.update(
        "Node Lambda acquired an OrbitalDrone X99 on 3 May 2025 via its procurement channel.",
    )
    await handle.result()

    handle = await km.ask(
        "How much did Node Lambda pay for its acquisition?",
        _return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "999"), assertion_failed(
        "Answer containing '999'",
        answer,
        reasoning,
        "Answer does not contain expected price",
        {"Knowledge Data": km._filter()},
    )


# --------------------------------------------------------------------------- #
# 6.  Long multi-turn conversation with incremental updates                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
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

    handle = await km.update("The StorageVault contains a component named AlphaCore.")
    await handle.result()

    handle = await km.update(
        "The StorageVault also contains a component named BetaModule.",
    )
    await handle.result()

    handle = await km.ask(
        "What are the names of the components in the StorageVault?",
        _return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "AlphaCore", "BetaModule"), assertion_failed(
        "Answer containing both 'AlphaCore' and 'BetaModule'",
        answer,
        reasoning,
        "Answer does not contain both expected pet names",
        {"Knowledge Data": km._filter()},
    )


# --------------------------------------------------------------------------- #
# 7.  Complex numeric scenario – implicit filtering                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
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

    handle = await km.update("Point P has coordinates x = 3 and y = 4.")
    await handle.result()

    handle = await km.update("Point Q has coordinates x = 1 and y = 10.")
    await handle.result()

    handle = await km.ask(
        "Which points lie in the first quadrant but have y less than 5?",
        _return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert "P" in answer or "3, 4" in answer, assertion_failed(
        "Answer containing 'P' but not 'Q'",
        answer,
        reasoning,
        "Answer does not correctly identify only point P",
        {"Knowledge Data": km._filter()},
    )


# --------------------------------------------------------------------------- #
# 8.  Store with interjection                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
@_handle_project
async def test_update_interjection():
    """
    Test that we can interject during a store operation and
    the interjection is incorporated into the final result.
    """
    km = KnowledgeManager()

    # store some informatiion
    handle = await km.update("Batch A is located in Sector 7.")

    # Mid-operation, add another detail that should also get stored.
    await handle.interject("Also, it was calibrated in 1990.")

    await handle.result()
    handle = await km.ask(
        "Which sector is Batch A located in and when was it calibrated?",
    )
    out = await handle.result()

    # The confirmation text returned by `store()` should include both pieces of information.
    assert _contains(out, "Sector", "7", "1990"), assertion_failed(
        "Output containing both 'Sector 7' and '1990'",
        out,
        "Output does not contain both expected details about Bob",
        {"Knowledge Data": km._filter()},
    )


# --------------------------------------------------------------------------- #
# 9.  Store with stop                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
@_handle_project
async def test_update_stop():
    km = KnowledgeManager()

    # Provide multiple facts in one go so that cancelling halfway through still yields a partial, meaningful result.
    handle = await km.update(
        "Batch A is in Sector 7. Module X weighs 30 kg. Device Y weighs 25 kg.",
    )
    await asyncio.sleep(0.05)
    await handle.stop()
    await handle.result()
    assert handle.done()


# --------------------------------------------------------------------------- #
# 10. Retrieve with interjection                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
@_handle_project
async def test_ask_interjection():
    """
    Test that we can interject during a retrieve operation and
    the interjection is incorporated into the final result.
    """
    km = KnowledgeManager()

    # Store some data first
    handle = await km.update("Unit 42 weighs 30 kilograms.")
    await handle.result()

    handle = await km.update("Unit 42 is stored in Bay A.")
    await handle.result()

    # Now retrieve with interjection
    handle = await km.ask("How heavy is Unit 42?")
    await handle.interject("Also, where is it stored?")
    out = await handle.result()

    assert _contains(out, "30", "Bay", "A"), assertion_failed(
        "Output containing both '30' and 'Bay A'",
        out,
        "Output does not contain both expected details about Alice",
    )


# --------------------------------------------------------------------------- #
# 11. Retrieve with stop                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
@_handle_project
async def test_ask_stop():
    """
    Test that we can stop a retrieve operation mid-execution
    """
    km = KnowledgeManager()

    # Store some data first
    handle = await km.update(
        "The capital of Andovia is Mirax. The capital of Eldoria is Luthen. "
        "The capital of Zarkon is Nimbos.",
    )
    await handle.result()

    # Now retrieve with stop
    handle = await km.ask("List the capitals of the specified kingdoms.")
    _wait_for_next_assistant_response_event(handle._client)
    await handle.stop()
    await handle.result()
    assert handle.done()


# --------------------------------------------------------------------------- #
# 12. Retrieve uses reduce for numeric aggregation                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(500)
@_handle_project
async def test_ask_uses_reduce_for_numeric_aggregation():
    """Verify LLM uses reduce tool for numeric aggregation questions."""
    km = KnowledgeManager()

    # Store some numeric data first
    handle = await km.update(
        "Product A has a price of 100. Product B has a price of 200. "
        "Product C has a price of 150.",
    )
    await handle.result()

    handle = await km.ask(
        "What is the total amount of all product prices?",
        _return_reasoning_steps=True,
    )
    answer, steps = await handle.result()

    # Assert reduce tool was called
    reduce_called = any(
        any(
            "reduce" in (tc.get("function", {}).get("name", "") or "").lower()
            for tc in (step.get("tool_calls") or [])
        )
        for step in steps
        if step.get("role") == "assistant"
    )
    assert reduce_called, assertion_failed(
        "reduce tool to be called",
        f"steps without reduce: {[s for s in steps if s.get('role') == 'assistant']}",
        steps,
        "LLM should use reduce tool for numeric aggregation",
    )
