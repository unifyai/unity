"""Tests for Actor response_format parameter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from typing import List

from unity.actor.simulated import SimulatedActor
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# Response format schemas
# ────────────────────────────────────────────────────────────────────────────


class ActionResult(BaseModel):
    """Structured result from an actor action."""

    completed: bool = Field(..., description="Whether the action completed")
    steps_taken: List[str] = Field(
        default_factory=list,
        description="List of steps taken",
    )
    outcome: str = Field(..., description="Description of the outcome")


# ────────────────────────────────────────────────────────────────────────────
# Simulated Actor tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_simulated_act_response_format():
    """Simulated Actor.act should return structured output when response_format is provided."""
    actor = SimulatedActor(steps=2, duration=1)

    handle = await actor.act(
        "Perform a quick demo task and report results",
        response_format=ActionResult,
    )
    result = await handle.result()

    # Should be valid JSON conforming to the schema
    parsed = ActionResult.model_validate_json(result)

    assert isinstance(parsed.completed, bool)
    assert isinstance(parsed.steps_taken, list)
    assert parsed.outcome.strip(), "Outcome should be non-empty"
