"""Tests for ContactManager response_format parameter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from typing import List, Optional

from unity.contact_manager.contact_manager import ContactManager
from unity.contact_manager.simulated import SimulatedContactManager
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Response format schemas
# ────────────────────────────────────────────────────────────────────────────


class ContactSummary(BaseModel):
    """Structured summary of contacts matching a query."""

    total_count: int = Field(..., description="Total number of contacts found")
    contact_names: List[str] = Field(
        ...,
        description="List of full names of matching contacts",
    )
    summary: str = Field(..., description="Brief natural language summary")


class ContactUpdateResult(BaseModel):
    """Structured result after a contact update operation."""

    success: bool = Field(..., description="Whether the update was successful")
    contact_name: Optional[str] = Field(
        None,
        description="Name of the contact that was updated",
    )
    action_taken: str = Field(..., description="Description of what was done")


# ────────────────────────────────────────────────────────────────────────────
# Simulated ContactManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_simulated_ask_response_format():
    """Simulated ContactManager.ask should return structured output when response_format is provided."""
    cm = SimulatedContactManager("Demo CRM with several contacts for testing.")

    handle = await cm.ask(
        "How many contacts are stored and what are their names?",
        response_format=ContactSummary,
    )
    result = await handle.result()

    # Should be valid JSON conforming to the schema
    parsed = ContactSummary.model_validate_json(result)

    assert isinstance(parsed.total_count, int)
    assert parsed.total_count >= 0
    assert isinstance(parsed.contact_names, list)
    assert parsed.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_update_response_format():
    """Simulated ContactManager.update should return structured output when response_format is provided."""
    cm = SimulatedContactManager("Demo CRM for testing updates.")

    handle = await cm.update(
        "Create a new contact named John Doe with email john@example.com",
        response_format=ContactUpdateResult,
    )
    result = await handle.result()

    parsed = ContactUpdateResult.model_validate_json(result)

    assert isinstance(parsed.success, bool)
    assert parsed.action_taken.strip(), "Action description should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# Real ContactManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_real_ask_response_format(
    contact_manager_scenario: tuple[ContactManager, dict],
):
    """Real ContactManager.ask should return structured output when response_format is provided."""
    cm, _ = contact_manager_scenario

    handle = await cm.ask(
        "How many contacts are in the database and list their names?",
        response_format=ContactSummary,
    )
    result = await handle.result()

    parsed = ContactSummary.model_validate_json(result)

    # We know from the fixture there are multiple contacts
    assert parsed.total_count > 0, "Should find at least one contact"
    assert len(parsed.contact_names) > 0, "Should have at least one contact name"
    assert parsed.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_real_update_response_format(
    contact_manager_mutation_scenario: tuple[ContactManager, dict],
):
    """Real ContactManager.update should return structured output when response_format is provided."""
    cm, _ = contact_manager_mutation_scenario

    handle = await cm.update(
        "Update Alice Smith's notes to mention she prefers email contact",
        response_format=ContactUpdateResult,
    )
    result = await handle.result()

    parsed = ContactUpdateResult.model_validate_json(result)

    assert isinstance(parsed.success, bool)
    assert parsed.action_taken.strip(), "Action description should be non-empty"
