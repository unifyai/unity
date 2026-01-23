"""Tests for TaskScheduler response_format parameter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field
from typing import List, Optional

from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Response format schemas
# ────────────────────────────────────────────────────────────────────────────


class TaskListSummary(BaseModel):
    """Structured summary of task query results."""

    total_tasks: int = Field(..., description="Total number of tasks found")
    task_names: List[str] = Field(..., description="List of task names")
    has_primed_task: bool = Field(
        ...,
        description="Whether there is a primed (ready to execute) task",
    )
    summary: str = Field(..., description="Brief natural language summary")


class TaskUpdateResult(BaseModel):
    """Structured result after a task update operation."""

    success: bool = Field(..., description="Whether the update was successful")
    task_name: Optional[str] = Field(
        None,
        description="Name of the task that was modified",
    )
    action_taken: str = Field(..., description="Description of what was done")


class TaskExecutionResult(BaseModel):
    """Structured result of task execution."""

    completed: bool = Field(..., description="Whether execution completed")
    outcome: str = Field(..., description="Description of the execution outcome")
    steps_taken: List[str] = Field(
        default_factory=list,
        description="List of steps taken during execution",
    )


# ────────────────────────────────────────────────────────────────────────────
# Simulated TaskScheduler tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_simulated_ask_response_format():
    """Simulated TaskScheduler.ask should return structured output when response_format is provided."""
    ts = SimulatedTaskScheduler("Demo task list with several pending tasks.")

    handle = await ts.ask(
        "How many tasks are there and what are their names?",
        response_format=TaskListSummary,
    )
    result = await handle.result()

    # Should be valid JSON conforming to the schema
    parsed = TaskListSummary.model_validate_json(result)

    assert isinstance(parsed.total_tasks, int)
    assert parsed.total_tasks >= 0
    assert isinstance(parsed.task_names, list)
    assert isinstance(parsed.has_primed_task, bool)
    assert parsed.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_update_response_format():
    """Simulated TaskScheduler.update should return structured output when response_format is provided."""
    ts = SimulatedTaskScheduler("Demo task list for testing updates.")

    handle = await ts.update(
        "Create a new task called 'Review budget proposal' with high priority",
        response_format=TaskUpdateResult,
    )
    result = await handle.result()

    parsed = TaskUpdateResult.model_validate_json(result)

    assert isinstance(parsed.success, bool)
    assert parsed.action_taken.strip(), "Action description should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_simulated_execute_response_format():
    """Simulated TaskScheduler.execute should return structured output when response_format is provided."""
    ts = SimulatedTaskScheduler(actor_steps=2, actor_duration=1)

    handle = await ts.execute(
        "Draft a simple email reminder",
        response_format=TaskExecutionResult,
    )
    result = await handle.result()

    parsed = TaskExecutionResult.model_validate_json(result)

    assert isinstance(parsed.completed, bool)
    assert parsed.outcome.strip(), "Outcome should be non-empty"
    assert isinstance(parsed.steps_taken, list)


# ────────────────────────────────────────────────────────────────────────────
# Real TaskScheduler tests
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_real_ask_response_format(
    task_scheduler_read_scenario: tuple[TaskScheduler, list],
):
    """Real TaskScheduler.ask should return structured output when response_format is provided."""
    ts, _ = task_scheduler_read_scenario

    handle = await ts.ask(
        "How many tasks are in the system and list their names?",
        response_format=TaskListSummary,
    )
    result = await handle.result()

    parsed = TaskListSummary.model_validate_json(result)

    # We know from the fixture there are seeded tasks
    assert parsed.total_tasks >= 0
    assert isinstance(parsed.task_names, list)
    assert parsed.summary.strip(), "Summary should be non-empty"


@pytest.mark.asyncio
@_handle_project
async def test_real_update_response_format(
    task_scheduler_mutation_scenario: tuple[TaskScheduler, list],
):
    """Real TaskScheduler.update should return structured output when response_format is provided."""
    ts, _ = task_scheduler_mutation_scenario

    handle = await ts.update(
        "Add a note to the 'Write quarterly report' task indicating it needs review",
        response_format=TaskUpdateResult,
    )
    result = await handle.result()

    parsed = TaskUpdateResult.model_validate_json(result)

    assert isinstance(parsed.success, bool)
    assert parsed.action_taken.strip(), "Action description should be non-empty"
