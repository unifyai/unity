from __future__ import annotations

import asyncio
import pytest

from unity.knowledge_manager.simulated import (
    SimulatedKnowledgeManager,
)

# helper that wraps each test in its own Unify project / trace context
from tests.helpers import (
    _handle_project,
    DEFAULT_TIMEOUT,
)


# ────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_simulated_km_docstrings_match_base():
    """
    Public methods in SimulatedKnowledgeManager should copy the real
    BaseKnowledgeManager doc-strings one-for-one (via functools.wraps).
    """
    from unity.knowledge_manager.base import BaseKnowledgeManager
    from unity.knowledge_manager.simulated import SimulatedKnowledgeManager

    assert (
        BaseKnowledgeManager.ask.__doc__.strip()
        in SimulatedKnowledgeManager.ask.__doc__.strip()
    ), ".retrieve doc-string was not copied correctly"

    assert (
        BaseKnowledgeManager.update.__doc__.strip()
        in SimulatedKnowledgeManager.update.__doc__.strip()
    ), ".store doc-string was not copied correctly"

    assert (
        BaseKnowledgeManager.refactor.__doc__.strip()
        in SimulatedKnowledgeManager.refactor.__doc__.strip()
    ), ".refactor doc-string was not copied correctly"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Basic start-and-ask                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_start_and_ask_simulated_km():
    km = SimulatedKnowledgeManager("Demo KB for unit-tests.")
    handle = await km.ask("What do we already know about Zebulon?")
    answer = await handle.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial retrieves                                     #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_km_stateful_serial_retrieves():
    """
    Two consecutive .retrieve() calls should share context.
    """
    km = SimulatedKnowledgeManager()

    # first question – ask for a single‐word theme of the KB
    h1 = await km.ask(
        "Using one word only, how would you describe the overall theme of our knowledge base?",
    )
    theme = (await h1.result()).strip()
    assert theme, "Theme word should not be empty"

    # follow-up question
    h2 = await km.ask(
        "What single word did you just use to describe the knowledge base?",
    )
    ans2 = (await h2.result()).lower()
    assert theme.lower() in ans2, "LLM should recall the theme it produced earlier"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Update then ask – state carries through                                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_km_stateful_update_then_retrieve():
    """
    A fact stored via .store() should be recalled by a later .retrieve().
    """
    km = SimulatedKnowledgeManager()
    fact = "The flagship product of Acme Corp is the Quantum Widget."

    # store a new fact
    h_store = await km.update(fact)
    await h_store.result()

    # retrieve it
    h_ret = await km.ask("What is the flagship product of Acme Corp?")
    answer = (await h_ret.result()).lower()
    assert "quantum" in answer and "widget" in answer


# ────────────────────────────────────────────────────────────────────────────
# 5.  Basic refactor                                                         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_refactor_simulated_km():
    """
    The simulated KM should return a non-empty migration plan in response to
    .refactor().  We do **not** verify the content in detail – only that the
    string is present and mentions something schema-related (e.g. "column").
    """
    km = SimulatedKnowledgeManager(
        "Tiny demo KB where Contacts duplicate company opening hours.",
    )

    handle = await km.refactor(
        "Please remove duplicated columns and introduce proper primary keys.",
    )
    migration_plan = await handle.result()

    assert (
        isinstance(migration_plan, str) and migration_plan.strip()
    ), "Migration plan should be non-empty"
    assert "column" in migration_plan.lower() or "table" in migration_plan.lower(), (
        "Plan should mention schema elements (columns/tables).",
        migration_plan,
    )


# ────────────────────────────────────────────────────────────────────────────
# 11. Clear – reset and remain usable                                         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_simulated_clear():
    """
    SimulatedKnowledgeManager.clear should reset the manager (hard-coded completion)
    and remain usable afterwards.
    """
    km = SimulatedKnowledgeManager()
    # Seed some prior state via an update call
    h_store = await km.update("Add a temporary fact about Project Phoenix.")
    await asyncio.wait_for(h_store.result(), timeout=DEFAULT_TIMEOUT)

    # Clear should not raise and should be quick (no LLM roundtrip requirement)
    km.clear()

    # Post-clear, an ask should still work
    h_q = await km.ask("List any knowledge stored today.")
    answer = await asyncio.wait_for(h_q.result(), timeout=DEFAULT_TIMEOUT)
    assert (
        isinstance(answer, str) and answer.strip()
    ), "Answer should be non-empty after clear()"


@_handle_project
def test_simulated_knowledge_manager_reduce_shapes():
    km = SimulatedKnowledgeManager()

    scalar = km.reduce(table="Content", metric="sum", keys="row_id")
    assert isinstance(scalar, (int, float))

    multi = km.reduce(table="Content", metric="max", keys=["row_id"])
    assert isinstance(multi, dict)
    assert set(multi.keys()) == {"row_id"}

    grouped = km.reduce(
        table="Content",
        metric="sum",
        keys="row_id",
        group_by="row_id",
    )
    assert isinstance(grouped, dict)
