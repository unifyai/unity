# tests/test_real/conductor/test_real_conductor.py
from __future__ import annotations

import pytest
from typing import Dict, Any

from unity.conductor.conductor import Conductor
from tests.assertion_helpers import assertion_failed
from tests.helpers import _handle_project


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
async def test_ask_across_managers(
    real_conductor_scenario: tuple[Conductor, Dict[str, Any]],
):
    """
    Tests a read-only `ask` query that requires information from multiple managers.
    "What is the status of the task related to the last phone call with Julia?"
    This requires finding the call (TranscriptManager) and then the task (TaskScheduler).
    """
    conductor, id_maps = real_conductor_scenario

    question = "What is the status of the task named 'Prepare slide deck' which we discussed in our last call with Julia?"
    handle = await conductor.ask(question, _return_reasoning_steps=True)
    answer, reasoning = await handle.result()

    # The seeded "Prepare slide deck" task has a default status of 'queued'.
    assert "queued" in answer.lower(), assertion_failed(
        "Answer containing 'queued'",
        answer,
        reasoning,
        "Conductor failed to join info from transcripts and tasks",
    )


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
async def test_request_multi_step_workflow(
    real_conductor_scenario: tuple[Conductor, Dict[str, Any]],
):
    """
    Tests a conversational workflow:
    1. Update a contact's info.
    2. Create a task based on that new info.
    """
    conductor, id_maps = real_conductor_scenario
    cm = conductor._contact_manager
    ts = conductor._task_scheduler
    carlos_id = id_maps["contacts"]["carlos"]

    # Step 1: Update contact
    update_req = f"Add a description 'Lead from GlobalCorp' to contact Carlos Diaz."
    update_handle = await conductor.request(update_req, _return_reasoning_steps=True)
    await update_handle.result()

    # Programmatic check
    carlos_contact = cm._filter_contacts(filter=f"contact_id == {carlos_id}")[0]
    assert carlos_contact.description == "Lead from GlobalCorp"

    # Step 2: Create a task based on the update
    task_req = "Now, create a task to 'Follow up with Carlos from GlobalCorp'."
    task_handle = await conductor.request(task_req, _return_reasoning_steps=True)
    await task_handle.result()

    # Programmatic check
    tasks = ts._search_tasks(filter="'Carlos' in name")
    assert len(tasks) > 0, "Task for Carlos was not created."
    assert "GlobalCorp" in tasks[0]["name"], "Task name did not include updated info."


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
async def test_request_complex_single_shot(
    real_conductor_scenario: tuple[Conductor, Dict[str, Any]],
):
    """
    Tests a single, complex request that should trigger tools from multiple managers.
    We verify this by inspecting the reasoning steps.
    """
    conductor, id_maps = real_conductor_scenario

    request_text = (
        "Create a task to 'Review Q2 Report'. In the description, "
        "include the location of 'GlobalCorp' from our knowledge base "
        "and mention that it's for 'Dan Turner', whose contact info we have."
    )

    handle = await conductor.request(request_text, _return_reasoning_steps=True)
    _, reasoning = await handle.result()

    # Inspect the tool calls in the reasoning steps
    tool_calls_json = [
        call["function"]["name"]
        for step in reasoning
        if "tool_calls" in step and step["tool_calls"]
        for call in step["tool_calls"]
    ]

    assert (
        "KnowledgeManager_ask" in tool_calls_json
    ), "KnowledgeManager.ask was not called."
    assert "ContactManager_ask" in tool_calls_json, "ContactManager.ask was not called."
    assert (
        "TaskScheduler_update" in tool_calls_json
    ), "TaskScheduler.update was not called."

    # Final state check
    created_task = conductor._task_scheduler._search_tasks(
        filter="'Review Q2 Report' in name",
    )
    assert len(created_task) == 1
    description = created_task[0]["description"].lower()
    assert "london" in description  # From KnowledgeBase
    assert "dan turner" in description  # From ContactManager


@_handle_project
@pytest.mark.eval
@pytest.mark.asyncio
async def test_request_executes_task(
    real_conductor_scenario: tuple[Conductor, Dict[str, Any]],
):
    """
    Tests that a request to execute a task correctly calls `task_scheduler.execute_task`.
    """
    conductor, id_maps = real_conductor_scenario
    ts = conductor._task_scheduler
    task_to_run_id = id_maps["tasks"]["write_quarterly_report"]

    # The task starts as 'primed' from the scenario
    initial_task = ts._search_tasks(filter=f"task_id == {task_to_run_id}")[0]
    assert initial_task["status"] == "primed"

    # Execute the task via a natural language request
    request_text = (
        f"Please start work on the task with ID {task_to_run_id} immediately."
    )
    handle = await conductor.request(request_text)
    await handle.result()

    # Check the task's final status. Since we injected a SimulatedPlanner,
    # it should complete quickly.
    final_task = ts._search_tasks(filter=f"task_id == {task_to_run_id}")[0]
    assert final_task["status"] == "completed", assertion_failed(
        "Task status 'completed'",
        final_task["status"],
        f"Task did not complete after execution request.",
    )
