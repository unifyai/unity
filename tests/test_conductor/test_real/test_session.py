from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock

from unity.conductor.simulated import SimulatedConductor
from unity.actor.hierarchical_actor import HierarchicalActor

from tests.helpers import _handle_project
from tests.test_conductor.utils import tool_names_from_messages


@pytest.mark.asyncio
@_handle_project
async def test_enforces_single_session(
    monkeypatch,
):
    """
    Launch two concurrent Conductor.request loops that both trigger Actor_act.
    Verify that only one interactive session is adopted, and the other call
    is gracefully rejected with a helpful message.
    """

    real_actor = HierarchicalActor(
        browser_mode="legacy",
        headless=True,
        connect_now=False,
    )

    # Keep the actor lightweight and deterministic
    real_actor.action_provider.navigate = AsyncMock(return_value=None)
    real_actor.action_provider.act = AsyncMock(return_value=None)
    real_actor.action_provider.observe = AsyncMock(return_value="Mocked Page Heading")

    class _NoKeychainBrowser:
        def __init__(self):
            self.backend = object()

        async def get_current_url(self) -> str:
            return ""

        async def get_screenshot(self) -> str:
            return ""

    real_actor.action_provider._browser = _NoKeychainBrowser()

    # Wrap HierarchicalActor.act to signal once scheduled so we can coordinate.
    # IMPORTANT: Patch before Conductor is constructed so the tool mapping captures it.
    _orig_act = HierarchicalActor.act
    tool_started_evt = asyncio.Event()

    async def _wrapped_act(self, *a, **kw):
        h = await _orig_act(self, *a, **kw)
        tool_started_evt.set()
        return h

    monkeypatch.setattr(HierarchicalActor, "act", _wrapped_act, raising=True)

    cond = SimulatedConductor(actor=real_actor)

    # Start the first request and wait until Actor_act is scheduled
    h1 = await cond.request(
        "Open a browser window so we can walk through the setup together.",
        _return_reasoning_steps=True,
    )
    await asyncio.wait_for(tool_started_evt.wait(), timeout=120)

    # Now start the second request while the first session is active
    h2 = await cond.request(
        "Open a browser window so we can walk through the setup together.",
        _return_reasoning_steps=True,
    )

    # Stop both quickly after scheduling
    h1.stop("done")
    h2.stop("done")

    a1, m1 = await asyncio.wait_for(h1.result(), timeout=300)
    a2, m2 = await asyncio.wait_for(h2.result(), timeout=300)

    # Exactly one Actor_act tool call should appear overall
    names_1 = tool_names_from_messages(m1, "Actor")
    names_2 = tool_names_from_messages(m2, "Actor")
    total_actor_calls = names_1.count("Actor_act") + names_2.count("Actor_act")
    assert total_actor_calls == 1
