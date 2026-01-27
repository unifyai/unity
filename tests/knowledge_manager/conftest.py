from __future__ import annotations

import pytest
import pytest_asyncio
from typing import List, Dict, Tuple, Any
import os

import unify
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.manager_registry import ManagerRegistry
from unity.common.context_registry import ContextRegistry

SCENARIO_COMMIT_HASHES: Dict[str, Any] = {}

# Initial knowledge data for seeding
_KNOWLEDGE_DATA: List[Dict[str, str]] = [
    {
        "content": "The ZX-99 gizmo was released in 1994 by TechCorp Industries.",
        "category": "products",
    },
    {
        "content": "The QuantumDrive unit produces 35 megawatts and weighs 180 kilograms.",
        "category": "products",
    },
    {
        "content": "The OrbitalDrone X99 costs 999 credits and is manufactured by SpaceTech Ltd.",
        "category": "products",
    },
    {
        "content": "The StorageVault contains components named AlphaCore and BetaModule.",
        "category": "inventory",
    },
    {
        "content": "Point P has coordinates x = 3 and y = 4, located in the first quadrant.",
        "category": "geometry",
    },
    {
        "content": "Point Q has coordinates x = 1 and y = 10, also in the first quadrant.",
        "category": "geometry",
    },
    {
        "content": "Unit 42 weighs 30 kilograms and is stored in Bay A of the facility.",
        "category": "inventory",
    },
]

_KNOWLEDGE_IDS: Dict[str, int] = {}
_SEED_TABLE_PREFIX: str = "KB_Seed"


def _category_to_table(category: str) -> str:
    cat = (category or "general").strip().lower().replace(" ", "_")
    return f"{_SEED_TABLE_PREFIX}_{cat}"


class ScenarioBuilderKnowledge:
    """Populates Unify with initial knowledge data for KnowledgeManager testing."""

    def __init__(self):
        self.km = KnowledgeManager()
        self._ensure_seed_tables()
        self._populate_id_mapping()

    def _ensure_seed_tables(self) -> None:
        """Idempotently create one seed table per category in _KNOWLEDGE_DATA."""
        seen: set[str] = set()
        for kd in _KNOWLEDGE_DATA:
            table = _category_to_table(kd.get("category", "general"))
            if table in seen:
                continue
            seen.add(table)
            try:
                self.km._create_table(name=table, description=f"Seed table for {table}")
            except Exception:
                # Table likely exists already; proceed
                pass

    def _populate_id_mapping(self):
        """Populate _KNOWLEDGE_IDS by searching for existing knowledge entries."""
        global _KNOWLEDGE_IDS
        _KNOWLEDGE_IDS.clear()

        def search_and_map_knowledge(knowledge_data):
            """Helper function to search for knowledge and return mapping tuple."""
            content = knowledge_data.get("content", "")
            category = knowledge_data.get("category", "general")

            # Create a search key based on first few words
            search_words = content.split()[:3]  # First 3 words as identifier
            search_key = (
                "_".join(search_words).lower().replace(",", "").replace(".", "")
            )

            # Prefer scanning only the relevant seed table for mapping
            try:
                seed_table = _category_to_table(category)
                results = self.km._filter(tables=[seed_table], limit=1000)
                rows = results.get(seed_table, []) or []
                for row in rows:
                    row_text = str(row).lower()
                    if all(word.lower() in row_text for word in search_words):
                        return (search_key, seed_table, len(rows))
            except Exception:
                pass

            return None

        # Wrap each knowledge_data dict in a tuple
        knowledge_data_tuples = [
            (knowledge_data,) for knowledge_data in _KNOWLEDGE_DATA
        ]

        results = unify.map(
            search_and_map_knowledge,
            knowledge_data_tuples,
            mode="asyncio",
        )

        for result in results:
            if result is not None:
                search_key, table_name, row_count = result
                _KNOWLEDGE_IDS[search_key] = {"table": table_name, "count": row_count}

    @classmethod
    async def create(cls) -> "ScenarioBuilderKnowledge":
        self = cls()
        await self._seed_knowledge()
        return self

    async def _seed_knowledge(self) -> None:
        """Create knowledge entries if they don't already exist."""
        for knowledge_data in _KNOWLEDGE_DATA:
            content = knowledge_data.get("content", "")
            category = knowledge_data.get("category", "general")
            seed_table = _category_to_table(category)

            # Check if knowledge already exists in its category seed table
            try:
                results = self.km._filter(
                    tables=[seed_table],
                    filter=f"content == '{content}'",
                    limit=1,
                )
                rows = results.get(seed_table, []) or []
                if rows:
                    continue
            except Exception:
                pass

            # Insert directly via low-level API into category-specific table
            try:
                self.km._add_rows(table=seed_table, rows=[knowledge_data])
            except Exception as e:
                print(
                    f"Warning: Could not create knowledge entry '{content[:50]}...' due to: {e}",
                )


@pytest_asyncio.fixture(scope="session")
async def knowledge_scenario(
    request: pytest.FixtureRequest,
) -> Tuple[KnowledgeManager, Dict[str, Any]]:
    """
    Create (and later clean up) a versioned context so that *all* tests share the
    same seeded data. Build scenario once and reuse across tests.
    """
    ManagerRegistry.clear()
    ContextRegistry.clear()

    os.environ["TQDM_DISABLE"] = "1"

    ctx = "tests/knowledge/Scenario"
    unify.set_context(ctx, relative=False)
    existing_contexts = unify.get_contexts(prefix=ctx)
    overwrite_scenarios = request.config.getoption("--overwrite-scenarios")

    # If --overwrite-scenarios is set, delete and recreate scenarios
    if overwrite_scenarios:
        reuse_scenario = False
    else:
        reuse_scenario = True

    if not reuse_scenario:
        # delete all contexts to freshly create the new scenario
        def recreate_contexts(ctx):
            unify.delete_context(ctx)
            unify.create_context(ctx)

        existing_ctx_names = list(existing_contexts.keys())
        if existing_ctx_names:
            unify.map(
                recreate_contexts,
                existing_ctx_names,
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

        existing_ctx_names = list(existing_contexts.keys())
        if existing_ctx_names:
            unify.map(
                get_context_commits_and_rollback,
                existing_ctx_names,
                mode="asyncio",
            )

    # --- One-time setup (per session) ---
    builder = ScenarioBuilderKnowledge()
    existing_contexts = unify.get_contexts(
        prefix=ctx,
    )  # fetch newly created contexts by builder

    if not SCENARIO_COMMIT_HASHES:
        print("Seeding knowledge manager scenario...")
        await builder.create()

        def commit_context_and_store(ctx):
            commit_info = unify.commit_context(
                name=ctx,
                commit_message="Initial seed data for knowledge manager tests",
            )
            SCENARIO_COMMIT_HASHES[ctx] = commit_info["commit_hash"]

        existing_ctx_names = list(existing_contexts.keys())
        if existing_ctx_names:
            unify.map(
                commit_context_and_store,
                existing_ctx_names,
                mode="asyncio",
            )

    unify.unset_context()
    return builder.km, _KNOWLEDGE_IDS


@pytest.fixture(scope="function")
def knowledge_manager_scenario(knowledge_scenario):
    """
    Per-test fixture that provides fresh scenario data by rolling back to
    the committed state before each test and after each test completes.
    """
    km, knowledge_map = knowledge_scenario

    def rollback_context(ctx):
        unify.rollback_context(
            name=ctx,
            commit_hash=SCENARIO_COMMIT_HASHES[ctx],
        )

    # Rollback to clean state before test
    ctx_names = list(SCENARIO_COMMIT_HASHES.keys())
    if ctx_names:
        unify.map(rollback_context, ctx_names, mode="asyncio")

    yield km, knowledge_map
