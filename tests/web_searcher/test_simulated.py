from __future__ import annotations

from dotenv import load_dotenv

load_dotenv(override=True)

import pytest

from unity.web_searcher.simulated import (
    SimulatedWebSearcher,
)

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import (
    _handle_project,
)


# ────────────────────────────────────────────────────────────────────────────
# 0.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_docstrings_match_base():
    """
    Public methods in SimulatedWebSearcher should copy the real
    BaseWebSearcher doc-strings one-for-one (via functools.wraps).
    """
    from unity.web_searcher.base import BaseWebSearcher
    from unity.web_searcher.simulated import SimulatedWebSearcher

    assert (
        BaseWebSearcher.ask.__doc__.strip() in SimulatedWebSearcher.ask.__doc__.strip()
    ), ".ask doc-string was not copied correctly"


# ────────────────────────────────────────────────────────────────────────────
# 1.  Basic start-and-ask                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_start_and_ask():
    ws = SimulatedWebSearcher("Demo web-search for unit-tests.")
    h = await ws.ask("What happened in vector DBs in Q1 2025?")
    answer = await h.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Stateful memory – serial asks                                         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_stateful_memory_serial_asks():
    """
    Two consecutive .ask() calls share context because the manager keeps a
    stateful LLM.
    """
    ws = SimulatedWebSearcher()

    h1 = await ws.ask(
        "Please propose a short unique report code for my research, "
        "and reply with only that code.",
    )
    code = (await h1.result()).strip()
    assert code, "Code should not be empty"

    h2 = await ws.ask("Great. What code did you just propose?")
    answer2 = (await h2.result()).lower()
    assert code.lower() in answer2, "LLM should recall the code it generated"


@_handle_project
def test_clear_reinitialises():
    """
    Ensure SimulatedWebSearcher.clear re-runs the constructor (fresh stateful LLM
    and tools mapping stays provisioned).
    """
    from unity.web_searcher.simulated import SimulatedWebSearcher

    sim = SimulatedWebSearcher()
    old_llm = getattr(sim, "_llm", None)
    assert old_llm is not None
    assert isinstance(sim._ask_tools, dict) and sim._ask_tools

    sim.clear()

    # After clear, llm handle should be replaced and tools still present
    assert getattr(sim, "_llm", None) is not None and sim._llm is not old_llm
    assert isinstance(sim._ask_tools, dict) and sim._ask_tools
