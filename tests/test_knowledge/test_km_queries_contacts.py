"""
Integration tests: the LLM running inside *KnowledgeManager* must be able to
call the embedded *ContactManager.ask* and *ContactManager.update* tools.

The tests are `@pytest.mark.eval` and `@pytest.mark.requires_real_unify`
because they execute a real tool-use reasoning loop.
"""

from __future__ import annotations

import pytest

from unity.contact_manager.contact_manager import ContactManager
from unity.knowledge_manager.knowledge_manager import KnowledgeManager

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# 1.  KM.retrieve → CM.ask → KM internal knowledge                           #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.requires_real_unify
@_handle_project
async def test_km_ask_joins_contact_and_company():
    """
    The KM should discover the employer via CM.ask and then look up the
    company's employee count in its own tables.
    """
    cm = ContactManager()
    km = KnowledgeManager()

    # ➊ Set up contact – add custom "employer" field then create Steve Taylor
    cm._create_custom_column(column_name="employer", column_type="str")
    cm._create_contact(
        first_name="Steve",
        surname="Taylor",
        employer="BigCorp",
    )

    # ➋ Set up companies knowledge
    km._create_table(name="Companies", columns={"name": "str", "employees": "int"})
    km._add_rows(table="Companies", rows=[{"name": "BigCorp", "employees": 1200}])

    # ➌ Ask a question that *requires* calling CM.ask internally
    q = "How many employees are at the company Steve Taylor works at?"
    h = await km.ask(q, _return_reasoning_steps=True)
    answer, reasoning = await h.result()

    # Basic semantic check – the answer should quote 1200 employees somewhere
    assert "1200" in answer.replace(",", ""), (
        f"Expected headcount '1200' in answer, got: {answer}",
    )


# ────────────────────────────────────────────────────────────────────────────
# 2.  KM.store → CM.update                                                   #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.requires_real_unify
@_handle_project
async def test_km_update_updates_contact():
    """
    A natural-language *store* instruction routed through KM should trigger
    ContactManager.update to add WhatsApp info.
    """
    cm = ContactManager()
    km = KnowledgeManager()

    # ‣ Pre-existing contact *without* WhatsApp
    cm._create_contact(
        first_name="Jane",
        surname="Doe",
        email_address="jane@example.com",
    )

    # ‣ Instruction to store extra info
    cmd = "Add Jane Doe's WhatsApp number +15559998877."
    h = await km.update(cmd, _return_reasoning_steps=True)
    _, steps = await h.result()  # we don't need the textual confirmation here

    # ‣ Verify that the ContactManager state really changed
    updated = cm._filter_contacts(filter="email_address == 'jane@example.com'")
    assert updated, "Jane Doe should exist after KM.store call"
    assert updated[0].whatsapp_number == "+15559998877"
