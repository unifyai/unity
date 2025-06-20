import pytest
import functools
import asyncio

from unity.conductor.simulated import SimulatedConductor
from unity.contact_manager.simulated import SimulatedContactManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_ask_calls_contact_manager_ask(monkeypatch):
    """
    SimulatedConductor.ask should consult SimulatedContactManager.ask once
    when the user’s question is clearly contact-related.
    """
    calls = {"count": 0}
    original = SimulatedContactManager.ask

    @functools.wraps(original)
    async def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return await original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedContactManager, "ask", spy, raising=True)

    tm = SimulatedConductor("Unit-test scenario – small team CRM.")
    handle = await tm.ask("What's Alice Reynolds' mobile number so I can ping her?")
    await asyncio.wait_for(handle.result(), timeout=60)

    assert calls["count"] == 1, "ContactManager.ask must be called exactly once."
