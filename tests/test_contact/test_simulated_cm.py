from __future__ import annotations

import asyncio
import pytest

from unity.contact_manager.simulated import (
    SimulatedContactManager,
    _SimulatedContactHandle,
)

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import _handle_project


# ────────────────────────────────────────────────────────────────────────────
# 1.  Basic start-and-ask                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_ask_simulated_cm():
    cm = SimulatedContactManager("Demo CRM for unit-tests.")
    h = cm.ask("List all my contacts.")
    answer = await h.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Interject                                                             #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_interject_simulated_cm(monkeypatch):
    calls = {"interject": 0}
    orig = _SimulatedContactHandle.interject

    def wrapped(self, msg: str) -> str:  # type: ignore[override]
        calls["interject"] += 1
        return orig(self, msg)

    monkeypatch.setattr(_SimulatedContactHandle, "interject", wrapped, raising=True)

    cm = SimulatedContactManager()
    h = cm.ask("Show me all contacts created this quarter.")
    await asyncio.sleep(0.05)
    reply = h.interject("Filter only VIP customers.")
    assert "ack" in reply.lower() or "noted" in reply.lower()
    await h.result()
    assert calls["interject"] == 1, ".interject should be invoked exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 3.  Stop                                                                  #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_simulated_cm():
    cm = SimulatedContactManager()
    h = cm.ask("Generate a full CRM export.")
    await asyncio.sleep(0.05)
    h.stop()
    with pytest.raises(asyncio.CancelledError):
        await h.result()
    assert h.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Clarification handshake                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_cm_requests_clarification():
    cm = SimulatedContactManager()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    h = cm.ask(
        "Please update my client list.",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=30)
    assert "clarify" in question.lower()
    await down_q.put("Yes – focus on European clients.")

    answer = await h.result()
    assert isinstance(answer, str) and answer.strip()
    assert "europe" in answer.lower()


# ────────────────────────────────────────────────────────────────────────────
# 5.  Stateful memory – serial asks                                         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_cm_stateful_memory_serial_asks():
    """
    Two consecutive .ask() calls share context because the manager keeps a
    stateful LLM.
    """
    cm = SimulatedContactManager()

    h1 = cm.ask(
        "Please suggest a unique reference code for a new prospect, "
        "and reply with *only* that code.",
    )
    ref_code = (await h1.result()).strip()
    assert ref_code, "Reference code should not be empty"

    h2 = cm.ask("Great. What reference code did you just propose?")
    answer2 = (await h2.result()).lower()
    assert ref_code.lower() in answer2, "LLM should recall the code it generated"


# ────────────────────────────────────────────────────────────────────────────
# 6.  Update then ask – state carries through                                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_cm_stateful_update_then_ask():
    cm = SimulatedContactManager()
    full_name = "Johnathan Doe"
    email = "john.doe@example.com"

    # create a fictitious contact
    upd = cm.update(
        f"Create a new contact: {full_name}, email {email}, mark as high priority.",
    )
    await upd.result()

    # ask about it
    hq = cm.ask("Do we have Johnathan's contact details on file?")
    ans = (await hq.result()).lower()
    assert (
        "john" in ans and "email" in ans
    ), "Contact created via update should be recalled"


# ────────────────────────────────────────────────────────────────────────────
# 7.  Doc-string inheritance                                                 #
# ────────────────────────────────────────────────────────────────────────────
def test_simulated_cm_docstrings_match_real():
    """
    Public methods in SimulatedContactManager should copy the real
    ContactManager doc-strings one-for-one (via functools.wraps).
    """
    from unity.contact_manager.contact_manager import ContactManager
    from unity.contact_manager.simulated import SimulatedContactManager

    assert (
        SimulatedContactManager.ask.__doc__.strip()
        == ContactManager.ask.__doc__.strip()
    ), ".store doc-string was not copied correctly"

    assert (
        SimulatedContactManager.update.__doc__.strip()
        == ContactManager.update.__doc__.strip()
    ), ".retrieve doc-string was not copied correctly"
