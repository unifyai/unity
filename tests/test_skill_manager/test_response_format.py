"""Tests for SkillManager response_format parameter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from typing import List

from unity.skill_manager.skill_manager import SkillManager
from unity.skill_manager.simulated import SimulatedSkillManager
from unity.function_manager.function_manager import FunctionManager
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# Response format schemas
# ────────────────────────────────────────────────────────────────────────────


class SkillQueryResult(BaseModel):
    """Structured result from a skill query."""

    skills_count: int = Field(..., description="Number of skills found")
    skill_names: List[str] = Field(..., description="Names of available skills")
    categories: List[str] = Field(
        default_factory=list,
        description="Categories of skills",
    )
    summary: str = Field(..., description="Brief natural language summary")


# ────────────────────────────────────────────────────────────────────────────
# Simulated SkillManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_simulated_ask_response_format():
    """Simulated SkillManager.ask should return structured output when response_format is provided."""
    sm = SimulatedSkillManager("Demo skills catalogue with various capabilities.")

    handle = await sm.ask(
        "What skills do you have and how many are there?",
        response_format=SkillQueryResult,
    )
    result = await handle.result()

    # Should be valid JSON conforming to the schema
    parsed = SkillQueryResult.model_validate_json(result)

    assert isinstance(parsed.skills_count, int)
    assert parsed.skills_count >= 0
    assert isinstance(parsed.skill_names, list)
    assert isinstance(parsed.categories, list)
    assert parsed.summary.strip(), "Summary should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# Real SkillManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_real_ask_response_format():
    """Real SkillManager.ask should return structured output when response_format is provided."""
    # Seed some functions
    fm = FunctionManager()
    src1 = (
        "def calculate_sum(a: int, b: int) -> int:\n"
        '    """Calculate sum of two integers"""\n'
        "    return a + b\n"
    )
    src2 = (
        "def format_currency(amount: float) -> str:\n"
        '    """Format amount as currency string"""\n'
        "    return f'${amount:.2f}'\n"
    )
    fm.add_functions(implementations=[src1, src2])

    sk = SkillManager()
    handle = await sk.ask(
        "List your skills and categorize them",
        response_format=SkillQueryResult,
    )
    result = await handle.result()

    parsed = SkillQueryResult.model_validate_json(result)

    assert isinstance(parsed.skills_count, int)
    assert isinstance(parsed.skill_names, list)
    assert parsed.summary.strip(), "Summary should be non-empty"
