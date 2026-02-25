"""Eval tests for CodeActActor routing between WebSearcher, desktop, and web sessions.

Verifies the LLM makes reasonable tool choices for different types of queries:
- Simple text-only queries, non-gated sites -> WebSearcher
- Using a desktop application -> primitives.computer.desktop
- Multi-site or isolated-state web tasks -> primitives.computer.web.new_session()
- Interactive user-visible browsing -> NOT visible=False
"""

import pytest

from contextlib import asynccontextmanager

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    extract_code_act_execute_code_snippets,
)
from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments import StateManagerEnvironment
from unity.actor.environments.computer import ComputerEnvironment
from unity.function_manager.primitives import ComputerPrimitives, Primitives
from unity.manager_registry import ManagerRegistry

pytestmark = pytest.mark.eval


def _join_snippets(handle) -> str:
    return "\n\n".join(extract_code_act_execute_code_snippets(handle))


@asynccontextmanager
async def _make_actor_with_computer():
    """Create a CodeActActor with both state managers and ComputerEnvironment (mock)."""
    ManagerRegistry.clear()
    primitives = Primitives()
    cp = ComputerPrimitives(computer_mode="mock")
    state_env = StateManagerEnvironment(primitives)
    computer_env = ComputerEnvironment(cp)

    actor = CodeActActor(environments=[state_env, computer_env])
    act_tools = actor.get_tools("act")
    keep = {
        k: v
        for k, v in act_tools.items()
        if not k.startswith("FunctionManager_") and not k.startswith("GuidanceManager_")
    }
    actor.add_tools("act", keep)

    try:
        yield actor
    finally:
        try:
            await actor.close()
        except Exception:
            pass
        ManagerRegistry.clear()


# ---------------------------------------------------------------------------
# Test 1: Simple factual query -> WebSearcher (not browser)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@_handle_project
async def test_simple_factual_query_uses_web_searcher():
    """A basic factual question about a public topic should use the text-based
    WebSearcher (primitives.web.ask), not launch a browser session."""
    async with _make_actor_with_computer() as actor:
        handle = await actor.act(
            "What is the current population of Tokyo?",
            clarification_enabled=False,
        )
        await handle.result()

        snippets = _join_snippets(handle)
        assert (
            "primitives.web.ask" in snippets
        ), f"Expected WebSearcher for a simple factual query. Snippets:\n{snippets}"
        assert "new_session" not in snippets, (
            f"Simple factual queries should not launch browser sessions. "
            f"Snippets:\n{snippets}"
        )


# ---------------------------------------------------------------------------
# Test 2: Desktop application -> primitives.computer.desktop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@_handle_project
async def test_desktop_app_uses_desktop_namespace():
    """A request to interact with a desktop application should use
    primitives.computer.desktop, not a web session factory."""
    async with _make_actor_with_computer() as actor:
        handle = await actor.act(
            "Open the Terminal application on the desktop and run 'ls -la'.",
            clarification_enabled=False,
        )
        await handle.result()

        snippets = _join_snippets(handle)
        assert "primitives.computer.desktop" in snippets, (
            f"Desktop app tasks should use primitives.computer.desktop. "
            f"Snippets:\n{snippets}"
        )
        assert (
            "new_session" not in snippets
        ), "Desktop app tasks should NOT create web sessions"


# ---------------------------------------------------------------------------
# Test 3: Multi-site parallel task -> web.new_session()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@_handle_project
async def test_multi_site_task_uses_web_sessions():
    """When the task involves visiting multiple independent sites, the LLM
    should create web sessions (which provide isolated browser state per site)
    rather than using the singleton desktop."""
    async with _make_actor_with_computer() as actor:
        handle = await actor.act(
            "I need you to check three websites at the same time: "
            "get the top headline from https://news.ycombinator.com, "
            "the current Bitcoin price from https://coinmarketcap.com, "
            "and the weather in London from https://weather.com. "
            "Do these in parallel if possible.",
            clarification_enabled=False,
        )
        await handle.result()

        snippets = _join_snippets(handle)
        assert "new_session" in snippets, (
            f"Multi-site parallel tasks should create web sessions. "
            f"Snippets:\n{snippets}"
        )


# ---------------------------------------------------------------------------
# Test 4: User-visible browsing must not be headless
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@_handle_project
async def test_interactive_browsing_not_headless():
    """When the user is watching via screenshare and asks for browser work,
    the LLM should not use visible=False (headless). Either desktop or
    visible web sessions are acceptable."""
    async with _make_actor_with_computer() as actor:
        handle = await actor.act(
            "I'm watching your screen right now via screenshare. Please go to "
            "https://app.example.com and walk me through the new dashboard "
            "features. Show me each section.",
            clarification_enabled=False,
        )
        await handle.result()

        snippets = _join_snippets(handle)
        uses_computer = "primitives.computer" in snippets
        assert uses_computer, (
            f"Should use computer primitives for interactive browsing. "
            f"Snippets:\n{snippets}"
        )
        assert "visible=False" not in snippets, (
            f"When the user is watching, browser sessions must not be headless. "
            f"Snippets:\n{snippets}"
        )
