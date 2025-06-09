import pytest
import functools
import asyncio

from unity.task_manager.simulated import SimulatedTaskManager
from unity.contact_manager.simulated import SimulatedContactManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_ask_calls_contact_manager_ask(monkeypatch):
    """
    SimulatedTaskManager.ask should consult SimulatedContactManager.ask once
    when the user’s question is clearly contact-related.
    """
    calls = {"count": 0}
    original = SimulatedContactManager.ask

    @functools.wraps(original)
    def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedContactManager, "ask", spy, raising=True)

    tm = SimulatedTaskManager("Unit-test scenario – small team CRM.")
    handle = await tm.ask("What's Alice Reynolds' mobile number so I can ping her?")
    await asyncio.wait_for(handle.result(), timeout=60)

    assert calls["count"] == 1, "ContactManager.ask must be called exactly once."
