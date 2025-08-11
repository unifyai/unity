import pytest
import functools

from unity.conductor.simulated import SimulatedConductor
from unity.contact_manager.simulated import SimulatedContactManager
from tests.helpers import _handle_project


@pytest.mark.asyncio
@_handle_project
async def test_request_calls_contact_manager_ask(monkeypatch):
    """
    A write request that needs contact info should invoke ContactManager.ask once.
    """
    calls = {"count": 0}
    original = SimulatedContactManager.ask

    @functools.wraps(original)
    async def spy(self, text: str, **kwargs):
        calls["count"] += 1
        return await original(self, text, **kwargs)

    monkeypatch.setattr(SimulatedContactManager, "ask", spy, raising=True)

    cond = SimulatedConductor("CRM demo – add reminder tasks.")
    handle = await cond.request(
        "Create a reminder task to call Alice Reynolds next Wednesday; "
        "look up her direct mobile number and include it in the task notes.",
    )
    await handle.result()

    assert calls["count"] == 1, "ContactManager.ask must be called exactly once."
