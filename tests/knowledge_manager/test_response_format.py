"""Tests for KnowledgeManager response_format parameter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Response format schemas
# ────────────────────────────────────────────────────────────────────────────


class KnowledgeQueryResult(BaseModel):
    """Structured result from a knowledge query."""

    facts_found: int = Field(..., description="Number of relevant facts found")
    facts: List[str] = Field(..., description="List of relevant facts")
    categories: List[str] = Field(
        default_factory=list,
        description="Categories of knowledge covered",
    )
    summary: str = Field(..., description="Brief natural language summary")


class KnowledgeUpdateResult(BaseModel):
    """Structured result after a knowledge update operation."""

    success: bool = Field(..., description="Whether the update was successful")
    table_affected: Optional[str] = Field(
        None,
        description="Name of the table that was modified",
    )
    rows_added: int = Field(0, description="Number of rows added")
    action_taken: str = Field(..., description="Description of what was done")


class SchemaRefactorResult(BaseModel):
    """Structured result from a schema refactor operation."""

    changes_proposed: int = Field(..., description="Number of schema changes proposed")
    tables_affected: List[str] = Field(
        default_factory=list,
        description="Tables that would be affected",
    )
    migration_plan: str = Field(..., description="Description of the migration plan")


# ────────────────────────────────────────────────────────────────────────────
# Simulated KnowledgeManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_simulated_ask_response_format():
    """Simulated KnowledgeManager.ask should return structured output when response_format is provided."""
    km = SimulatedKnowledgeManager("Demo KB with product specifications and facts.")

    handle = await km.ask(
        "What facts do we have about products?",
        response_format=KnowledgeQueryResult,
    )
    result = await handle.result()

    assert isinstance(result, KnowledgeQueryResult)
    assert isinstance(result.facts_found, int)
    assert result.facts_found >= 0
    assert isinstance(result.facts, list)
    assert isinstance(result.categories, list)
    assert result.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_update_response_format():
    """Simulated KnowledgeManager.update should return structured output when response_format is provided."""
    km = SimulatedKnowledgeManager("Demo KB for testing updates.")

    handle = await km.update(
        "Add a new fact: The QuantumX processor runs at 5GHz and costs $500",
        response_format=KnowledgeUpdateResult,
    )
    result = await handle.result()

    assert isinstance(result, KnowledgeUpdateResult)
    assert isinstance(result.success, bool)
    assert isinstance(result.rows_added, int)
    assert result.action_taken.strip(), "Action description should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_refactor_response_format():
    """Simulated KnowledgeManager.refactor should return structured output when response_format is provided."""
    km = SimulatedKnowledgeManager(
        "Demo KB with redundant columns and denormalized tables.",
    )

    handle = await km.refactor(
        "Normalize the schema and remove duplicate columns",
        response_format=SchemaRefactorResult,
    )
    result = await handle.result()

    assert isinstance(result, SchemaRefactorResult)
    assert isinstance(result.changes_proposed, int)
    assert isinstance(result.tables_affected, list)
    assert result.migration_plan.strip(), "Migration plan should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# Real KnowledgeManager tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_real_ask_response_format(
    knowledge_manager_scenario: tuple[KnowledgeManager, Dict[str, Any]],
):
    """Real KnowledgeManager.ask should return structured output when response_format is provided."""
    km, _ = knowledge_manager_scenario

    handle = await km.ask(
        "What facts do we have about products and their specifications?",
        response_format=KnowledgeQueryResult,
    )
    result = await handle.result()

    assert isinstance(result, KnowledgeQueryResult)
    assert isinstance(result.facts_found, int)
    assert isinstance(result.facts, list)
    assert result.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_real_update_response_format(
    knowledge_manager_scenario: tuple[KnowledgeManager, Dict[str, Any]],
):
    """Real KnowledgeManager.update should return structured output when response_format is provided."""
    km, _ = knowledge_manager_scenario

    handle = await km.update(
        "Add a note that the ZX-99 gizmo was discontinued in 2020",
        response_format=KnowledgeUpdateResult,
    )
    result = await handle.result()

    assert isinstance(result, KnowledgeUpdateResult)
    assert isinstance(result.success, bool)
    assert result.action_taken.strip(), "Action description should be non-empty"
