from datetime import datetime, timezone

from unity.task_scheduler.prompt_builders import (
    build_task_execution_request,
    build_task_run_guidelines,
)
from unity.task_scheduler.types.activated_by import ActivatedBy
from unity.task_scheduler.types.priority import Priority
from unity.task_scheduler.types.repetition import Frequency, RepeatPattern
from unity.task_scheduler.types.schedule import Schedule
from unity.task_scheduler.types.status import Status
from unity.task_scheduler.types.task import Task


def test_build_task_execution_request_includes_run_metadata():
    task = Task(
        task_id=7,
        instance_id=2,
        name="Weekly AI report",
        description="Summarize the previous week's AI research.",
        status=Status.scheduled,
        priority=Priority.normal,
        response_policy="Email the user a concise document.",
        schedule=Schedule(start_at=datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)),
        repeat=[RepeatPattern(frequency=Frequency.WEEKLY)],
    )

    request = build_task_execution_request(task)

    assert "Execute this TaskScheduler task as a contained task run." in request
    assert "Task id: 7" in request
    assert "Instance id: 2" in request
    assert "Weekly AI report" in request
    assert "Summarize the previous week's AI research." in request
    assert "Task response policy:" in request
    assert "Schedule metadata:" in request
    assert "Repeat metadata:" in request


def test_build_task_run_guidelines_keep_child_actor_focused_on_one_task():
    task = Task(
        task_id=3,
        instance_id=1,
        name="Invoice follow-up",
        description="Draft an invoice reply.",
        status=Status.triggerable,
        priority=Priority.normal,
    )

    guidelines = build_task_run_guidelines(task, ActivatedBy.trigger)

    assert "executing exactly one TaskScheduler task" in guidelines
    assert "do not create another task" in guidelines
    assert "interpret the natural-language description" in guidelines
    assert "Activation reason: trigger" in guidelines
    assert "Task id: 3" in guidelines
    assert "Instance id: 1" in guidelines
