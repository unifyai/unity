# --------------------------------------------------------------------------- #
#  Helper to seed a deterministic task set                                   #
# --------------------------------------------------------------------------- #
import os
import pytest
import unify
from unity.task_scheduler.task_scheduler import TaskScheduler
from typing import List, Dict, Any

SCENARIO_COMMIT_HASHES: Dict[str, Any] = {}

# Task data for seeding
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

_TASK_IDS: List[int] = []


class ScenarioBuilderTasks:
    """Populates Unify with initial tasks for TaskScheduler testing."""

    def __init__(self):
        self.ts = TaskScheduler()
        self._populate_task_ids()

    def _populate_task_ids(self):
        """Populate _TASK_IDS by searching for existing tasks."""
        global _TASK_IDS
        _TASK_IDS.clear()

        def search_task_by_name(task_data):
            """Helper function to search for a task by name and return its ID."""
            existing_tasks = self.ts._filter_tasks(
                filter=f"name == '{task_data['name']}'",
            )
            if existing_tasks:
                if existing_tasks[0]["status"] == "primed":
                    self.ts._primed_task = existing_tasks[0]
                return existing_tasks[0]["task_id"]
            return None

        # Wrap each task_data dict in a tuple to avoid unify.map treating dict keys as kwargs
        task_data_tuples = [(task_data,) for task_data in _TASKS_DATA]

        # Use unify.map for parallel task searches
        results = unify.map(
            search_task_by_name,
            task_data_tuples,
            mode="asyncio",
        )

        # Populate the ID list with found tasks (in order)
        for result in sorted(results, key=lambda x: x if x is not None else 0):
            if result is not None:
                _TASK_IDS.append(result)

    def create(self) -> "ScenarioBuilderTasks":
        self._seed_tasks()
        return self

    def _seed_tasks(self) -> None:
        """Create tasks if they don't already exist."""
        global _TASK_IDS

        for i, task_data in enumerate(_TASKS_DATA):
            # Check if task already exists
            existing_tasks = self.ts._filter_tasks(
                filter=f"name == '{task_data['name']}'",
            )
            if existing_tasks:
                # Update _TASK_IDS if we don't have this ID yet
                task_id = existing_tasks[0]["task_id"]
                if len(_TASK_IDS) <= i:
                    _TASK_IDS.append(task_id)
                elif i < len(_TASK_IDS) and _TASK_IDS[i] != task_id:
                    _TASK_IDS[i] = task_id
                continue  # Task already exists, skip

            try:
                task_id = self.ts._create_task(**task_data)["details"]["task_id"]

                # Add to _TASK_IDS list maintaining order
                if len(_TASK_IDS) <= i:
                    _TASK_IDS.append(task_id)
                else:
                    _TASK_IDS[i] = task_id

            except Exception as e:
                print(
                    f"Warning: Could not create task '{task_data['name']}' due to: {e}",
                )


@pytest.fixture(scope="session")
def task_scenario(request: pytest.FixtureRequest):
    """
    Create (and later clean up) a versioned context so that *all* tests share the
    same seeded data. Build scenario once and reuse across tests.
    """
    os.environ["TQDM_DISABLE"] = "1"
    ctx = "tests/test_task_scheduler/Scenario"
    unify.set_context(ctx, relative=False)
    existing_contexts = unify.get_contexts(prefix=ctx)
    no_reuse_scenario = request.config.getoption("--no-reuse-scenario")

    # If --no-reuse-scenario is explicitly set, override reuse_scenario
    if no_reuse_scenario:
        reuse_scenario = False
    else:
        reuse_scenario = True

    if not reuse_scenario:
        # delete all contexts to freshly create the new scenario
        def recreate_contexts(ctx):
            try:
                unify.delete_context(ctx)
                unify.create_context(ctx)
            except Exception as e:
                pass

        _targets = list(existing_contexts.keys())
        if _targets:
            unify.map(
                recreate_contexts,
                _targets,
                mode="asyncio",
            )

    if reuse_scenario and not SCENARIO_COMMIT_HASHES:

        def get_context_commits_and_rollback(ctx):
            history = unify.get_context_commits(ctx)
            if history:
                unify.rollback_context(
                    name=ctx,
                    commit_hash=history[0]["commit_hash"],
                )
                SCENARIO_COMMIT_HASHES[ctx] = history[0]["commit_hash"]

        _targets = list(existing_contexts.keys())
        if _targets:
            unify.map(
                get_context_commits_and_rollback,
                _targets,
                mode="asyncio",
            )

    # --- One-time setup (per session) ---
    sb = ScenarioBuilderTasks()
    if not SCENARIO_COMMIT_HASHES:
        print("Seeding task scheduler scenario...")
        sb.create()

        def commit_context_and_store(ctx):
            commit_info = unify.commit_context(
                name=ctx,
                commit_message="Initial seed data for task scheduler tests",
            )
            SCENARIO_COMMIT_HASHES[ctx] = commit_info["commit_hash"]

        # Refresh contexts after seeding so we commit the ones just created
        existing_contexts = unify.get_contexts(prefix=ctx)
        _targets = list(existing_contexts.keys())
        if _targets:
            unify.map(
                commit_context_and_store,
                _targets,
                mode="asyncio",
            )

    yield sb.ts, _TASK_IDS


@pytest.fixture(scope="function")
def basic_task_scenario(task_scenario):
    """
    Per-test fixture that provides fresh scenario data by rolling back to
    the committed state before each test and after each test completes.
    """
    ts, ids = task_scenario

    def rollback_context(ctx):
        unify.rollback_context(
            name=ctx,
            commit_hash=SCENARIO_COMMIT_HASHES[ctx],
        )

    # Rollback to clean state before test
    _targets = list(SCENARIO_COMMIT_HASHES.keys())
    if _targets:
        unify.map(rollback_context, _targets, mode="asyncio")

    yield ts, ids
