"""
Integration test for the new **KnowledgeManager.refactor** public API.

Scenario
--------
• A `Companies` table and the Contact-Manager's `Contacts` table both carry an
  **opening_hours** field – a clear duplication.
• The contact rows also store a free-text *employer_name* which we would like
  to replace by a foreign-key style reference.

The test triggers a `refactor()` call with a natural-language request to
deduplicate the schema.  We then verify that the call:
  1. completes successfully, and
  2. returns a non-empty summary mentioning the duplicated column name
     ("opening_hours"), proving the LLM analysed the duplication.

We do **not** assert the *exact* column layout post-migration (LLM decisions
may evolve) – only that the refactor ran and acknowledged the issue.
"""

from __future__ import annotations

import pytest

from unity.contact_manager.contact_manager import ContactManager
from unity.knowledge_manager.knowledge_manager import KnowledgeManager

# helper – spins up an isolated temporary Unify project
from tests.helpers import _handle_project

# --------------------------------------------------------------------------- #
#  refactor()                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.requires_real_unify
@_handle_project
async def test_refactor_removes_duplicate_opening_hours():
    """
    End-to-end check that .refactor() can detect and act upon a duplicated
    **opening_hours** column present in both the Companies table and the
    Contacts table.
    """

    # ➊  Seed duplicated schema & data --------------------------------------
    cm = ContactManager()
    km = KnowledgeManager()

    # Contacts table: add duplicate columns + one sample row
    cm._create_custom_column(column_name="opening_hours", column_type="str")
    cm._create_custom_column(column_name="employer_name", column_type="str")
    cm._create_contact(
        first_name="Alice",
        surname="Smith",
        opening_hours="Mon-Fri 09:00-17:00",
        employer_name="Acme",
    )

    # Companies table with the same opening_hours column
    km._create_table(
        name="Companies",
        columns={
            "name": "str",
            "revenue": "int",
            "opening_hours": "str",
        },
    )
    km._add_rows(
        table="Companies",
        rows=[
            {
                "name": "Acme",
                "revenue": 1_000_000,
                "opening_hours": "Mon-Fri 09:00-17:00",
            },
        ],
    )

    # ➋  Run the refactor command ------------------------------------------
    handle = await km.refactor(
        "Please normalise the schema – remove the duplicate opening_hours "
        "field and replace employer_name with a company_id foreign key.",
        _return_reasoning_steps=True,
    )
    summary, reasoning = await handle.result()

    # ➌  Basic, deterministic assertions ------------------------------------
    assert isinstance(summary, str) and summary.strip(), (
        "Refactor summary should be a non-empty string.",
    )
    assert "opening" in summary.lower(), (
        "Summary should mention the duplicated 'opening_hours' column.",
        summary,
    )
