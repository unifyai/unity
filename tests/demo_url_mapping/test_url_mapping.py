"""End-to-end test for URL mapping (demo-site rerouting).

Verifies the full pipeline: URL mappings on ComputerPrimitives flow through
MagnitudeBackend -> agent-service /start -> BrowserConnector context.route(),
causing Playwright to transparently reroute a fake domain to a local demo site.

The LLM navigates to https://www.pawsomerescue.com (which doesn't exist) and
reports what it sees. The content is served by a local static file server
containing a dog rescue website, confirming the rerouting works end-to-end.
"""

import asyncio

import pytest

from tests.helpers import _handle_project
from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments.computer import ComputerEnvironment
from unity.function_manager.primitives import ComputerPrimitives
from unity.manager_registry import ManagerRegistry

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_url_mapping_routes_to_demo_site(agent_service_url, demo_site_url):
    """Navigate to a fake domain and verify the LLM sees demo site content."""
    ManagerRegistry.clear()
    actor = None
    try:
        cp = ComputerPrimitives(
            computer_mode="magnitude",
            container_url=agent_service_url,
            local_url=agent_service_url,
        )
        ComputerPrimitives.mark_ready()
        cp.url_mappings = {
            "https://www.pawsomerescue.com": demo_site_url,
        }

        computer_env = ComputerEnvironment(cp)
        actor = CodeActActor(environments=[computer_env], timeout=120)

        handle = await actor.act(
            "Create a headless web session (visible=False), navigate to "
            "https://www.pawsomerescue.com, get the page content, and tell "
            "me what this website is about. List any specific names you see.",
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=180)
        result_lower = result.lower()

        assert any(
            keyword in result_lower
            for keyword in (
                "pawsome",
                "dog",
                "rescue",
                "biscuit",
                "luna",
                "thunderpaws",
            )
        ), (
            f"Expected the LLM to describe content from the demo dog rescue site, "
            f"but got:\n{result}"
        )
    finally:
        if actor is not None:
            try:
                await actor.close()
            except Exception:
                pass
        ManagerRegistry.clear()
