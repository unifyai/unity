from __future__ import annotations

import pytest
from datetime import datetime, UTC

from tests.helpers import _handle_project
from unity.task_scheduler.types.task import Task


@pytest.mark.unit
@_handle_project
def test_task_json_shorthand_aliases_keys_no_prune():
    task = Task(
        task_id=1,
        instance_id=0,
        name="Email client",
        description="Write an email",
        status="queued",
        deadline=datetime.now(UTC),
        priority="normal",
    )

    dumped = task.model_dump(mode="json", context={"shorthand": True})

    # Aliased keys should exist
    for k in ("tid", "iid", "nm", "desc", "st", "prio"):
        assert k in dumped, f"expected shorthand key {k} in dump"

    # Original keys should not be present
    for k in ("task_id", "instance_id", "name", "description", "status", "priority"):
        assert k not in dumped, f"did not expect original key {k} in dump"


@pytest.mark.unit
@_handle_project
def test_task_json_shorthand_with_prune_omits_empties():
    task = Task(
        task_id=2,
        instance_id=0,
        name="No extras",
        description="Minimal",
        status="queued",
        priority="normal",
        # schedule=None, trigger=None, response_policy=None (implicit empties)
    )

    dumped = task.model_dump(
        mode="json",
        context={"shorthand": True, "prune_empty": True},
    )

    # Aliased keys present
    for k in ("tid", "iid", "nm", "desc", "st", "prio"):
        assert k in dumped

    # Optional empty fields should be pruned
    for k in ("sched", "trig", "policy", "entry", "ab"):
        assert k not in dumped
