"""Tests for GuidanceManager response_format parameter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from typing import List, Optional

from unity.guidance_manager.guidance_manager import GuidanceManager
from unity.guidance_manager.simulated import SimulatedGuidanceManager
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Response format schemas
# ────────────────────────────────────────────────────────────────────────────


class GuidanceQueryResult(BaseModel):
    """Structured result from a guidance query."""

    guidance_count: int = Field(..., description="Number of guidance items found")
    titles: List[str] = Field(..., description="Titles of matching guidance items")
    categories: List[str] = Field(
        default_factory=list,
        description="Categories of guidance covered",
    )
    summary: str = Field(..., description="Brief natural language summary")


class GuidanceUpdateResult(BaseModel):
    """Structured result after a guidance update operation."""

    success: bool = Field(..., description="Whether the update was successful")
    guidance_title: Optional[str] = Field(
        None,
        description="Title of the guidance that was modified",
    )
    action_taken: str = Field(..., description="Description of what was done")


# ────────────────────────────────────────────────────────────────────────────
# Simulated GuidanceManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_simulated_ask_response_format():
    """Simulated GuidanceManager.ask should return structured output when response_format is provided."""
    gm = SimulatedGuidanceManager(
        "Demo guidance DB with onboarding and deployment docs.",
    )

    handle = await gm.ask(
        "What guidance do we have about onboarding?",
        response_format=GuidanceQueryResult,
    )
    result = await handle.result()

    # Should be valid JSON conforming to the schema
    parsed = GuidanceQueryResult.model_validate_json(result)

    assert isinstance(parsed.guidance_count, int)
    assert parsed.guidance_count >= 0
    assert isinstance(parsed.titles, list)
    assert isinstance(parsed.categories, list)
    assert parsed.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_update_response_format():
    """Simulated GuidanceManager.update should return structured output when response_format is provided."""
    gm = SimulatedGuidanceManager("Demo guidance for testing updates.")

    handle = await gm.update(
        "Create new guidance titled 'Security Best Practices' about password policies",
        response_format=GuidanceUpdateResult,
    )
    result = await handle.result()

    parsed = GuidanceUpdateResult.model_validate_json(result)

    assert isinstance(parsed.success, bool)
    assert parsed.action_taken.strip(), "Action description should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# Real GuidanceManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_real_ask_response_format():
    """Real GuidanceManager.ask should return structured output when response_format is provided."""
    gm = GuidanceManager()
    # Seed some guidance entries
    gm._add_guidance(title="Onboarding", content="How to onboard a user step by step")
    gm._add_guidance(title="Billing", content="Explains invoices and payments")

    handle = await gm.ask(
        "What guidance items do we have and what topics do they cover?",
        response_format=GuidanceQueryResult,
    )
    result = await handle.result()

    parsed = GuidanceQueryResult.model_validate_json(result)

    assert isinstance(parsed.guidance_count, int)
    assert isinstance(parsed.titles, list)
    assert parsed.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_real_update_response_format():
    """Real GuidanceManager.update should return structured output when response_format is provided."""
    gm = GuidanceManager()
    # First create a guidance entry
    gm._add_guidance(title="API Guidelines", content="How to use our REST API")

    handle = await gm.update(
        "Update the API Guidelines to mention rate limiting of 100 requests per minute",
        response_format=GuidanceUpdateResult,
    )
    result = await handle.result()

    parsed = GuidanceUpdateResult.model_validate_json(result)

    assert isinstance(parsed.success, bool)
    assert parsed.action_taken.strip(), "Action description should be non-empty"
