"""Tests for Actor response_format parameter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from typing import List, Optional

from unity.actor.simulated import SimulatedActor
from unity.actor.single_function_actor import SingleFunctionActor
from unity.function_manager.function_manager import FunctionManager
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


# ────────────────────────────────────────────────────────────────────────────
# SingleFunctionActor tests
# ────────────────────────────────────────────────────────────────────────────


def _create_test_function(fm: FunctionManager) -> dict:
    """Add a test function to the FunctionManager."""
    implementation = '''
def calculate_sum(a: int, b: int) -> dict:
    """Calculate the sum of two numbers."""
    return {"result": a + b, "message": f"Sum of {a} and {b} is {a + b}"}
'''
    result = fm.add_functions(implementations=[implementation])
    assert result.get("calculate_sum") in ("added", "skipped: already exists")
    functions = fm.list_functions(include_implementations=True)
    return functions["calculate_sum"]


class FunctionExecutionResult(BaseModel):
    """Structured result from function execution."""

    success: bool = Field(..., description="Whether execution succeeded")
    result_value: Optional[str] = Field(
        None,
        description="The computed result or value",
    )
    summary: str = Field(..., description="Summary of what was done")


@pytest.mark.asyncio
@_handle_project
async def test_single_function_actor_act_response_format():
    """SingleFunctionActor.act should return structured output when response_format is provided."""
    fm = FunctionManager()
    func_data = _create_test_function(fm)

    actor = SingleFunctionActor(
        computer_primitives=None,
        function_manager=fm,
    )

    handle = await actor.act(
        function_id=func_data["function_id"],
        call_kwargs={"a": 5, "b": 3},
        response_format=FunctionExecutionResult,
    )

    result = await handle.result()

    # The response should be valid JSON conforming to the schema
    parsed = FunctionExecutionResult.model_validate_json(result)

    assert isinstance(parsed.success, bool)
    assert parsed.summary.strip(), "Summary should be non-empty"
