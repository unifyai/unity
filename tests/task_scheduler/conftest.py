# --------------------------------------------------------------------------- #
#  Helper to seed a deterministic task set for TaskScheduler testing         #
# --------------------------------------------------------------------------- #
from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple

import pytest
import pytest_asyncio
import unify

from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.task_scheduler.types.status import Status
from unity.manager_registry import ManagerRegistry
from unity.common.context_registry import ContextRegistry
from tests.helpers import (
    is_task_scenario_seeded,
    rebuild_task_id_mapping,
    scenario_file_lock,
    mutation_test_lock,
)

# Separate commit hash storage for read vs mutation contexts
_READ_SCENARIO_COMMIT_HASHES: Dict[str, Any] = {}
_MUTATION_SCENARIO_COMMIT_HASHES: Dict[str, Any] = {}


# Task data for seeding (shared by both scenarios)
_TASKS_DATA: List[Dict[str, str]] = [
    {
        "name": "Write quarterly report",
        "description": "Draft the Q2 report (send email to finance).",
        "status": "primed",
    },
    {
        "name": "Prepare slide deck",
        "description": "Create slides for the board meeting. Email once done.",
        "status": "queued",
    },
    {
        "name": "Client follow-up email",
        "description": "Send email to prospective client about proposal.",
        "status": "queued",
    },
]


def _seed_tasks(ts: TaskScheduler) -> List[int]:
    """Create tasks if they don't exist. Returns list of task IDs in order."""
    task_ids: List[int] = []
    for task_data in _TASKS_DATA:
        name = task_data["name"]
        # Check if task already exists
        try:
            existing = ts._filter_tasks(filter=f"name == {name!r}", limit=1)
            if existing:
                task_id = existing[0].task_id
                if existing[0].status == Status.primed:
                    ts._primed_task = existing[0]
            else:
                result = ts._create_task(**task_data)
                task_id = result["details"]["task_id"]
            task_ids.append(task_id)
        except Exception as e:
            print(f"Warning: Could not create/find task '{name}': {e}")
    return task_ids


def _rebuild_commit_hashes(
    ctx_prefix: str,
    commit_hashes: Dict[str, Any],
) -> None:
    """Rebuild commit hashes from existing context commits."""
    existing_contexts = unify.get_contexts(prefix=ctx_prefix)
    for ctx_name in existing_contexts.keys():
        history = unify.get_context_commits(ctx_name)
        if history:
            commit_hashes[ctx_name] = history[0]["commit_hash"]


def _commit_contexts_for_rollback(
    ctx_prefix: str,
    commit_hashes: Dict[str, Any],
) -> None:
    """Commit all contexts under prefix and store hashes for rollback."""
    existing_contexts = unify.get_contexts(prefix=ctx_prefix)
    for ctx_name in existing_contexts.keys():
        commit_info = unify.commit_context(
            name=ctx_name,
            commit_message="Initial seed data for task scheduler tests",
        )
        commit_hashes[ctx_name] = commit_info["commit_hash"]


def _setup_scenario(
    request: pytest.FixtureRequest,
    ctx: str,
    lock_name: str,
    commit_hashes: Dict[str, Any],
) -> Tuple[TaskScheduler, List[int]]:
    """
    Common setup logic for seeding a task scheduler scenario.

    Creates/reuses a versioned context, seeds tasks if needed,
    and returns the scheduler + task ID list.
    """
    ManagerRegistry.clear()
    ContextRegistry.clear()
    os.environ["TQDM_DISABLE"] = "1"

    overwrite_scenarios = request.config.getoption("--overwrite-scenarios")

    # If --overwrite-scenarios is set, delete existing contexts first
    if overwrite_scenarios:
        existing_contexts = unify.get_contexts(prefix=ctx)
        for ctx_name in existing_contexts.keys():
            unify.delete_context(ctx_name)

    # Set context before any operations
    unify.create_context(ctx)  # exist_ok=True by default
    unify.set_context(ctx, relative=False)

    # Create scheduler
    ts = TaskScheduler()
    task_ids: List[int] = []

    # Use file lock to coordinate seeding across parallel processes
    with scenario_file_lock(lock_name):
        if is_task_scenario_seeded(ts, _TASKS_DATA):
            # Scenario exists - just rebuild local state
            print(f"Scenario already seeded ({ctx}), rebuilding local state...")
            task_ids = rebuild_task_id_mapping(ts, _TASKS_DATA)
            _rebuild_commit_hashes(ctx, commit_hashes)
        else:
            # Scenario not seeded - seed it
            print(f"Seeding task scheduler scenario ({ctx})...")
            task_ids = _seed_tasks(ts)
            _commit_contexts_for_rollback(ctx, commit_hashes)

    unify.unset_context()
    return ts, task_ids


# ---------------------------------------------------------------------------
# READ-ONLY SCENARIO (for test_ask.py, test_sys_msgs.py, etc.)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="session")
async def task_read_scenario(
    request: pytest.FixtureRequest,
) -> Tuple[TaskScheduler, List[int]]:
    """
    Session-scoped scenario for READ-ONLY tests.

    Uses context: tests/task_scheduler/ReadScenario

    Read-only tests can run fully in parallel since they only read data
    and their rollbacks don't affect mutation tests (separate context).
    """
    return _setup_scenario(
        request,
        ctx="tests/task_scheduler/ReadScenario",
        lock_name="ts_read_scenario",
        commit_hashes=_READ_SCENARIO_COMMIT_HASHES,
    )


@pytest.fixture(scope="function")
def task_scheduler_read_scenario(task_read_scenario):
    """
    Per-test fixture for READ-ONLY tests (e.g., test_ask.py, test_sys_msgs.py).

    Rolls back to committed state before each test. These tests can run
    fully in parallel since they use a separate context from mutation tests.
    """
    ts, task_ids = task_read_scenario

    def rollback_context(ctx):
        unify.rollback_context(
            name=ctx,
            commit_hash=_READ_SCENARIO_COMMIT_HASHES[ctx],
        )

    # Rollback to clean state before test
    ctx_names = list(_READ_SCENARIO_COMMIT_HASHES.keys())
    if ctx_names:
        unify.map(rollback_context, ctx_names, mode="asyncio")

    yield ts, task_ids


# ---------------------------------------------------------------------------
# MUTATION SCENARIO (for test_update_complex.py, etc.)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="session")
async def task_mutation_scenario(
    request: pytest.FixtureRequest,
) -> Tuple[TaskScheduler, List[int]]:
    """
    Session-scoped scenario for MUTATION tests.

    Uses context: tests/task_scheduler/MutationScenario

    Mutation tests use a separate context from read-only tests, ensuring
    that read tests' rollbacks cannot interfere with mutation operations.
    """
    return _setup_scenario(
        request,
        ctx="tests/task_scheduler/MutationScenario",
        lock_name="ts_mutation_scenario",
        commit_hashes=_MUTATION_SCENARIO_COMMIT_HASHES,
    )


@pytest.fixture(scope="function")
def task_scheduler_mutation_scenario(task_mutation_scenario):
    """
    Per-test fixture for tests that MUTATE task data (create, update, delete).

    Uses a SEPARATE context from read-only tests, plus a file lock to serialize
    mutation tests among themselves. This ensures:

    1. Read tests' rollbacks cannot affect mutation tests (different context)
    2. Mutation tests don't race with each other (serialized via lock)
    3. The full sequence (rollback → mutate → verify) is atomic
    """
    ts, task_ids = task_mutation_scenario

    def rollback_context(ctx):
        unify.rollback_context(
            name=ctx,
            commit_hash=_MUTATION_SCENARIO_COMMIT_HASHES[ctx],
        )

    with mutation_test_lock("ts_mutation"):
        # Rollback INSIDE the lock to prevent other mutation tests
        # from rolling back while this test is running
        ctx_names = list(_MUTATION_SCENARIO_COMMIT_HASHES.keys())
        if ctx_names:
            unify.map(rollback_context, ctx_names, mode="asyncio")

        yield ts, task_ids


# ---------------------------------------------------------------------------
# BACKWARDS COMPATIBILITY
# ---------------------------------------------------------------------------
# Keep the old fixture names as aliases


@pytest_asyncio.fixture(scope="session")
async def task_scenario(
    request: pytest.FixtureRequest,
) -> Tuple[TaskScheduler, List[int]]:
    """
    DEPRECATED: Use task_read_scenario or task_mutation_scenario instead.

    This alias exists for backwards compatibility.
    """
    return _setup_scenario(
        request,
        ctx="tests/task_scheduler/ReadScenario",
        lock_name="ts_read_scenario",
        commit_hashes=_READ_SCENARIO_COMMIT_HASHES,
    )


@pytest.fixture(scope="function")
def basic_task_scenario(task_scheduler_mutation_scenario):
    """
    DEPRECATED: Use task_scheduler_read_scenario or task_scheduler_mutation_scenario.

    Maps to mutation scenario for backwards compatibility since most uses
    of basic_task_scenario were in mutation tests (test_update_complex.py).
    """
    yield task_scheduler_mutation_scenario
