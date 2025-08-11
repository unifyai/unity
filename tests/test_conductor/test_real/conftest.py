# tests/test_conductor/test_real/conftest.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple, Generator

import pytest
import unify
from unity.conductor.conductor import Conductor
from unity.planner.simulated import SimulatedPlanner
from unity.task_scheduler.task_scheduler import TaskScheduler
from unity.transcript_manager.types.message import Message

SCENARIO_COMMIT_HASHES: Dict[str, Any] = {}


# --- Consolidated Data from Other Scenarios ---

# From transcript_manager/conftest.py
_CONTACTS_DATA = [
    {
        "first_name": "Carlos",
        "surname": "Diaz",
        "email_address": "carlos.diaz@example.com",
    },
    {
        "first_name": "Dan",
        "surname": "Turner",
        "email_address": "dan.turner@example.com",
    },
    {
        "first_name": "Julia",
        "surname": "Nguyen",
        "email_address": "julia.nguyen@example.com",
    },
]

# From task_scheduler/conftest.py
_TASKS_DATA = [
    {
        "name": "Write quarterly report",
        "description": "Draft the Q2 report for finance.",
    },
    {
        "name": "Prepare slide deck",
        "description": "Create slides for the board meeting.",
    },
]


class ConductorScenarioBuilder:
    """
    Seeds a rich, multi-manager environment for testing the real Conductor.
    It reuses data and logic patterns from individual manager test scenarios.
    """

    def __init__(self):
        # Inject a simulated planner for predictable and fast task execution tests
        self.planner = SimulatedPlanner(steps=1)
        self.task_scheduler = TaskScheduler(planner=self.planner)
        self.conductor = Conductor(task_scheduler=self.task_scheduler)

        # Direct access to managers for seeding and assertions
        self.cm = self.conductor._contact_manager
        self.tm = self.conductor._transcript_manager
        self.km = self.conductor._knowledge_manager
        self.ts = self.conductor._task_scheduler

        self.id_maps: Dict[str, Dict[str, Any]] = {"contacts": {}, "tasks": {}}

    @classmethod
    def create(cls) -> Tuple[Conductor, Dict[str, Dict[str, Any]]]:
        """Factory method to build and seed the full scenario."""
        self = cls()
        self._seed_contacts()
        self._seed_transcripts()
        self._seed_tasks()
        self._seed_knowledge()
        return self.conductor, self.id_maps

    def _seed_contacts(self):
        """Seed contacts and populate the ID map."""
        for contact_data in _CONTACTS_DATA:
            result = self.cm._create_contact(**contact_data)
            contact_id = result["details"]["contact_id"]
            name_key = contact_data["first_name"].lower()
            self.id_maps["contacts"][name_key] = contact_id

    def _seed_transcripts(self):
        """Seed a meaningful transcript exchange."""
        cid = self.id_maps["contacts"]
        now = datetime(2025, 4, 20, 15, 0, tzinfo=timezone.utc)
        messages = [
            (
                cid["dan"],
                cid["julia"],
                now,
                "Finalising the London event agenda today.",
            ),
            (
                cid["julia"],
                cid["dan"],
                now + timedelta(seconds=45),
                "Great. Let's confirm the speaker list.",
            ),
        ]
        log_entries = [
            Message(
                medium="phone_call",
                sender_id=s,
                receiver_ids=[r],
                timestamp=ts,
                content=txt,
                exchange_id=1,
            ).to_post_json()
            for s, r, ts, txt in messages
        ]
        self.tm.log_messages(log_entries)

    def _seed_tasks(self):
        """Seed tasks and populate the ID map."""
        for task_data in _TASKS_DATA:
            result = self.ts._create_task(**task_data)
            task_id = result["details"]["task_id"]
            name_key = task_data["name"].lower().replace(" ", "_")
            self.id_maps["tasks"][name_key] = task_id

    def _seed_knowledge(self):
        """Seed the knowledge manager with some facts."""
        self.km._create_table(
            name="CompanyInfo",
            columns={"company_name": "str", "hq_location": "str"},
            description="Basic info about companies.",
        )
        self.km._add_rows(
            table="CompanyInfo",
            rows=[{"company_name": "GlobalCorp", "hq_location": "London"}],
        )


@pytest.fixture(scope="session")
def conductor_scenario(
    request: pytest.FixtureRequest,
) -> Generator[Tuple[Conductor, Dict[str, Any]], None, None]:
    """
    Session-scoped fixture to create a versioned, seeded environment for Conductor tests.
    """
    os.environ["TQDM_DISABLE"] = "1"
    context_prefix = "test_real_conductor/Scenario"
    unify.set_context(context_prefix)
    existing_contexts = unify.get_contexts(prefix=context_prefix)

    if not request.config.getoption("--no-reuse-scenario"):
        for ctx in existing_contexts:
            history = unify.get_context_commits(ctx)
            if history:
                SCENARIO_COMMIT_HASHES[ctx] = history[0]["commit_hash"]

    # One-time setup if no cached commits are found
    if not SCENARIO_COMMIT_HASHES:
        print("\nSeeding REAL CONDUCTOR scenario...")
        builder = ConductorScenarioBuilder()

        conductor, id_maps = builder.create()

        for ctx in unify.get_contexts(prefix=context_prefix):
            commit_info = unify.commit_context(
                name=ctx,
                commit_message="Initial seed for Conductor tests",
            )
            SCENARIO_COMMIT_HASHES[ctx] = commit_info["commit_hash"]
    else:
        # If reusing, just create the builder instance to get access to the objects
        builder = ConductorScenarioBuilder()
        conductor, id_maps = builder.conductor, builder.id_maps
        # Manually populate id_maps by searching, since we didn't run create()
        for contact in builder.cm._filter_contacts():
            id_maps["contacts"][contact.first_name] = contact.contact_id
        # ... similar logic for tasks if needed ...

    # Yield the fully configured Conductor and ID maps
    unify.unset_context()
    yield conductor, id_maps


@pytest.fixture(scope="function")
def real_conductor_scenario(conductor_scenario):
    """
    Per-test fixture that rolls back all contexts to the clean, seeded state.
    """
    conductor, id_maps = conductor_scenario
    # Rollback all contexts to the clean state before the test runs
    for ctx, commit_hash in SCENARIO_COMMIT_HASHES.items():
        unify.rollback_context(name=ctx, commit_hash=commit_hash)

    # Also reset the in-memory state of the task scheduler
    conductor._task_scheduler._active_task = None
    conductor._task_scheduler._primed_task = None
    primed_tasks = conductor._task_scheduler._search_tasks(filter="status == 'primed'")
    if primed_tasks:
        conductor._task_scheduler._primed_task = primed_tasks[0]

    yield conductor, id_maps
