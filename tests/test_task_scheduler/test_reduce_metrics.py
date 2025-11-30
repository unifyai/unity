from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.task_scheduler.task_scheduler import TaskScheduler


@pytest.mark.requires_real_unify
@_handle_project
def test_task_scheduler_reduce_param_shapes():
    ts = TaskScheduler()

    # Seed tasks so metrics have real data to aggregate
    ts._create_task(
        name="Draft project update email",
        description="Write a detailed project update email for stakeholders",
    )
    ts._create_task(
        name="Send quick status text",
        description="Send a short status update via text message",
    )
    ts._create_task(
        name="Prepare weekly notes",
        description="Compile notes from this week's readings for the team",
    )

    # Single key, no grouping
    scalar = ts._reduce(metric="sum", keys="task_id")
    assert isinstance(scalar, (int, float))

    # Multiple keys, no grouping
    multi = ts._reduce(metric="max", keys=["task_id"])
    assert isinstance(multi, dict)
    assert set(multi.keys()) == {"task_id"}

    # Single key, group_by string
    grouped_str = ts._reduce(
        metric="sum",
        keys="task_id",
        group_by="status",
    )
    assert isinstance(grouped_str, dict)

    # Multiple keys, group_by string
    grouped_str_multi = ts._reduce(
        metric="min",
        keys=["task_id"],
        group_by="status",
    )
    assert isinstance(grouped_str_multi, dict)

    # Single key, group_by list
    grouped_list = ts._reduce(
        metric="sum",
        keys="task_id",
        group_by=["status", "queue_id"],
    )
    assert isinstance(grouped_list, dict)

    # Multiple keys, group_by list
    grouped_list_multi = ts._reduce(
        metric="mean",
        keys=["task_id"],
        group_by=["status", "queue_id"],
    )
    assert isinstance(grouped_list_multi, dict)

    # Filter as string
    filtered_scalar = ts._reduce(
        metric="sum",
        keys="task_id",
        filter="task_id >= 0",
    )
    assert isinstance(filtered_scalar, (int, float))

    # Filter as per-key dict
    filtered_multi = ts._reduce(
        metric="sum",
        keys=["task_id"],
        filter={"task_id": "task_id >= 0"},
    )
    assert isinstance(filtered_multi, dict)
