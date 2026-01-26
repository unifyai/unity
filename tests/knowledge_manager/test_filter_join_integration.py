"""
Integration test for KnowledgeManager's *join* helper
=====================================================

Scenario
--------
* Two tables
    • **Employees** (employee_id, full_name, department_id)
    • **Departments** (department_id, department_name)
* Query: *"How many people are in John Smith's department?"*

Correct answering demands a join on ``department_id``.  We patch
``KnowledgeManager._filter_join`` with a spy so the test can assert that the
LLM chose the dedicated join tool instead of piecing the answer together from
two independent filters.
"""

import re
import json
import pytest
import functools

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project
from tests.assertion_helpers import assertion_failed

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _contains(text: str, *needles: str) -> bool:
    """Return True when every *needle* appears in *text* (case-insensitive)."""
    return all(re.search(n, text, re.I) for n in needles)


# --------------------------------------------------------------------------- #
#  Join-aware retrieval                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_join_used_for_department_query(monkeypatch):
    """
    The answer should be *3* and – crucially – come from a call to
    ``_filter_join``.
    """

    # 1️⃣  Patch _filter_join with a spy so we can detect invocation --------
    called = {"flag": False}
    original_join = KnowledgeManager._filter_join

    @functools.wraps(original_join)
    def _filter_join_spy(self, *args, **kwargs):
        called["flag"] = True
        return original_join(self, *args, **kwargs)

    monkeypatch.setattr(
        KnowledgeManager,
        "_filter_join",
        _filter_join_spy,
        raising=True,
    )

    # 2️⃣  Prepare data -------------------------------------------------------
    km = KnowledgeManager()

    km._create_table(
        name="Employees",
        columns={
            "employee_id": "int",
            "full_name": "str",
            "department_id": "int",
        },
    )
    km._create_table(
        name="Departments",
        columns={
            "department_id": "int",
            "department_name": "str",
        },
    )

    km._add_rows(
        table="Employees",
        rows=[
            {"employee_id": 1, "full_name": "John Smith", "department_id": 10},
            {"employee_id": 2, "full_name": "Alice Jones", "department_id": 10},
            {"employee_id": 3, "full_name": "Bob Brown", "department_id": 10},
            {"employee_id": 4, "full_name": "Charlie Zed", "department_id": 20},
        ],
    )
    km._add_rows(
        table="Departments",
        rows=[
            {"department_id": 10, "department_name": "Engineering"},
            {"department_id": 20, "department_name": "Marketing"},
        ],
    )

    # 3️⃣  Ask the question ---------------------------------------------------
    handle = await km.ask(
        "How many people are in John Smith's department? "
        "Please use the tool 'filter_join' to answer the question.",
        _return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()

    # 4️⃣  Assertions ---------------------------------------------------------
    assert _contains(answer, r"\b3\b") or _contains(answer, "three"), assertion_failed(
        "Answer containing head-count 3",
        answer,
        reasoning,
        "Answer does not reflect the correct number of employees",
        {"All Knowledge": json.dumps(km._filter())},
    )

    assert called["flag"], (
        "KnowledgeManager._filter_join was NOT invoked – the LLM did not use "
        "the dedicated join tool."
    )
