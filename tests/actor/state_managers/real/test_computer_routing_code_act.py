"""Eval tests for CodeActActor routing between WebSearcher, desktop, and web sessions.

Verifies the LLM makes reasonable tool choices for different types of queries:
- Simple text-only queries, non-gated sites -> WebSearcher
- Using a desktop application -> primitives.computer.desktop
- Multi-site or isolated-state web tasks -> primitives.computer.web.new_session()
- Interactive user-visible browsing -> NOT visible=False
- Simple desktop actions -> verify=False (or omitted)
- Complex multi-step desktop tasks -> verify=True
- Live demo context -> verify=False even for moderate complexity
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


# ---------------------------------------------------------------------------
# Test 5: Simple single-action desktop task -> verify=False (or omitted)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@_handle_project
async def test_simple_desktop_click_does_not_verify():
    """A simple single-action desktop task (clicking a button, opening an app)
    should NOT use verify=True, since single-pass execution is sufficient."""
    async with _make_actor_with_computer() as actor:
        handle = await actor.act(
            "Click the 'Settings' icon on the desktop.",
            clarification_enabled=False,
        )
        await handle.result()

        snippets = _join_snippets(handle)
        assert (
            "primitives.computer.desktop" in snippets
        ), f"Should use desktop primitives. Snippets:\n{snippets}"
        assert "verify=True" not in snippets, (
            f"Simple click tasks should NOT use verify=True. " f"Snippets:\n{snippets}"
        )


# ---------------------------------------------------------------------------
# Test 6: Form filling (moderate complexity) -> verify=False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@_handle_project
async def test_short_form_fill_does_not_verify():
    """Filling a short form (a few fields + submit) is well within
    single-pass capability and should use verify=False."""
    async with _make_actor_with_computer() as actor:
        handle = await actor.act(
            "In the open sign-up form, fill in the name as 'Jane Doe', "
            "the email as 'jane@example.com', and click 'Submit'.",
            clarification_enabled=False,
        )
        await handle.result()

        snippets = _join_snippets(handle)
        assert (
            "primitives.computer.desktop" in snippets
        ), f"Should use desktop primitives. Snippets:\n{snippets}"
        assert "verify=True" not in snippets, (
            f"Short form fills should NOT use verify=True. " f"Snippets:\n{snippets}"
        )


# ---------------------------------------------------------------------------
# Test 7: Complex multi-step wizard -> verify=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@_handle_project
async def test_complex_multistep_wizard_uses_verify():
    """A long multi-page wizard with many steps across different screens
    should use verify=True to ensure each stage completes correctly."""
    async with _make_actor_with_computer() as actor:
        handle = await actor.act(
            "Complete the full account setup wizard in the open application. "
            "It has 5 pages: personal details (name, DOB, address, phone), "
            "employment information (employer, job title, salary, start date), "
            "financial profile (income sources, investment experience, risk "
            "tolerance), document upload (ID and proof of address), and a "
            "final review page where you must check all the confirmation "
            "boxes and click 'Finish Setup'. Use these details:\n"
            "- Name: John Smith, DOB: 1990-05-15\n"
            "- Address: 123 Main St, Springfield, IL 62701\n"
            "- Phone: (555) 123-4567\n"
            "- Employer: Acme Corp, Job: Engineer, Salary: $95,000\n"
            "- Income: salary only, Experience: moderate, Risk: balanced\n",
            clarification_enabled=False,
        )
        await handle.result()

        snippets = _join_snippets(handle)
        assert (
            "primitives.computer.desktop" in snippets
        ), f"Should use desktop primitives. Snippets:\n{snippets}"
        assert "verify=True" in snippets, (
            f"Complex multi-step wizards should use verify=True. "
            f"Snippets:\n{snippets}"
        )


# ---------------------------------------------------------------------------
# Test 8: Live demo context -> verify=False even for moderate task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@_handle_project
async def test_live_demo_prefers_no_verify():
    """During a live demo where latency matters, even a moderately complex
    desktop task should avoid verify=True to keep responsiveness high."""
    async with _make_actor_with_computer() as actor:
        handle = await actor.act(
            "I'm doing a live demo right now and the audience is watching "
            "my screen. Please open the CRM application, navigate to the "
            "Contacts section, and create a new contact with name 'Demo "
            "User' and email 'demo@example.com'.",
            clarification_enabled=False,
        )
        await handle.result()

        snippets = _join_snippets(handle)
        assert (
            "primitives.computer.desktop" in snippets
        ), f"Should use desktop primitives. Snippets:\n{snippets}"
        assert "verify=True" not in snippets, (
            f"During live demos, should NOT use verify=True to minimize "
            f"latency. Snippets:\n{snippets}"
        )
