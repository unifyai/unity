# --------------------------------------------------------------------------- #
#  Helper to seed a deterministic task set                                   #
# --------------------------------------------------------------------------- #
import os
import pytest
import unify
from unity.task_scheduler.task_scheduler import TaskScheduler
from typing import List


def _seed_basic_tasks(ts: TaskScheduler) -> List[int]:
    """Return list of task-ids in creation order."""

    ids = []
    ids.append(
        ts._create_task(
            name="Write quarterly report",
            description="Draft the Q2 report (send email to finance).",
            status="primed",
        ),
    )
    ids.append(
        ts._create_task(
            name="Prepare slide deck",
            description="Create slides for the board meeting. Email once done.",
            status="queued",
        ),
    )
    ids.append(
        ts._create_task(
            name="Client follow-up email",
            description="Send email to prospective client about proposal.",
            status="queued",
        ),
    )
    return ids


@pytest.fixture(scope="session", autouse=True)
def setup_session_context():
    """Set up a session-wide context for all tests in this module."""
    file_path = __file__
    ctx = "/".join(file_path.split("/tests/")[1].split("/"))[:-3]
    if unify.get_contexts(prefix=ctx):
        unify.delete_context(ctx)
    with unify.Context(ctx):
        unify.set_trace_context("Traces")
        yield

    if os.environ.get("UNIFY_DELETE_CONTEXT_ON_EXIT", "false").lower() == "true":
        unify.delete_context(ctx)


@pytest.fixture(scope="session")
def basic_task_scenario(setup_session_context):
    """
    Snapshot task state before each test that uses basic_task_scenario and restore after.
    """
    ts = TaskScheduler()
    ids = _seed_basic_tasks(ts)
    snapshot = ts._search_tasks()
    yield ts, ids
    new_snapshot = ts._search_tasks()
    for t_original, t_new in zip(snapshot, new_snapshot):
        if t_original["name"] != t_new["name"]:
            ts._update_task_name(
                task_id=t_original["task_id"],
                new_name=t_original["name"],
            )
        if t_original["description"] != t_new["description"]:
            ts._update_task_description(
                task_id=t_original["task_id"],
                new_description=t_original["description"],
            )
        if t_original["status"] != t_new["status"]:
            ts._update_task_status(
                task_ids=[t_original["task_id"]],
                new_status=t_original["status"],
            )
        if t_original["priority"] != t_new["priority"]:
            ts._update_task_priority(
                task_id=t_original["task_id"],
                new_priority=t_original["priority"],
            )
        if t_original["deadline"] != t_new["deadline"]:
            ts._update_task_deadline(
                task_id=t_original["task_id"],
                new_deadline=t_original["deadline"],
            )
        if t_original["repeat"] != t_new["repeat"]:
            ts._update_task_repetition(
                task_id=t_original["task_id"],
                new_repeat=t_original["repeat"],
            )
