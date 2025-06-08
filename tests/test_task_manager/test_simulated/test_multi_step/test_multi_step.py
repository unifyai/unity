"""
Multi-step integration tests for SimulatedTaskManager.

Each test:

• monkey-patches one (or several) subordinate simulated-manager methods,
  incrementing a counter and delegating to the original implementation;
• spins up a fresh SimulatedTaskManager (so the patches are active);
• performs two-or-more serial calls to `.ask()` / `.request()`;
• awaits each handle to ensure full completion; and
• finally asserts the patched method(s) were invoked the expected number
  of times – nothing more, nothing less.
"""

import asyncio
import functools

import pytest

from unity.task_manager.simulated import SimulatedTaskManager
from unity.contact_manager.simulated import SimulatedContactManager
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from tests.helpers import _handle_project


# --------------------------------------------------------------------------- #
# 1. Update Phone number and Make a Call                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_update_phone_number_then_call(monkeypatch):
    """
    • 1st turn: asks for Alice's phone number → needs ContactManager.ask
    • 2nd turn: change this → needs ContactManager.update
    • 3rd turn: call here → needs start_task
    Expected: ContactManager.ask called exactly twice.
    """

    counts = {}

    # Check + Update Contact

    # ask phone number via contact manager
    counts["cm_ask"] = 0
    original_cm_ask = SimulatedContactManager.ask

    @functools.wraps(original_cm_ask)
    def spy_cm_ask(self, text: str, **kw):
        counts["cm_ask"] += 1
        return original_cm_ask(self, text, **kw)

    monkeypatch.setattr(SimulatedContactManager, "ask", spy_cm_ask, raising=True)

    # update phone number via contact manager
    counts["cm_update"] = 0
    original_cm_update = SimulatedContactManager.update

    @functools.wraps(original_cm_update)
    def spy_cm_update(self, text: str, **kw):
        counts["cm_update"] += 1
        return original_cm_update(self, text, **kw)

    monkeypatch.setattr(SimulatedContactManager, "update", spy_cm_update, raising=True)

    # Task Check, Maybe Create, then Start

    # check if the task already exists
    counts["ts_ask"] = 0
    original_ts_ask = SimulatedTaskScheduler.ask

    @functools.wraps(original_ts_ask)
    def spy_ts_ask(self, text: str, **kw):
        counts["ts_ask"] += 1
        return original_ts_ask(self, text, **kw)

    monkeypatch.setattr(SimulatedTaskScheduler, "ask", spy_ts_ask, raising=True)

    # *maybe* create if the simulation says it does not exist yet
    counts["ts_update"] = 0
    original_ts_update = SimulatedTaskScheduler.update

    @functools.wraps(original_ts_update)
    def spy_ts_update(self, text: str, **kw):
        counts["ts_update"] += 1
        return original_ts_update(self, text, **kw)

    monkeypatch.setattr(SimulatedTaskScheduler, "update", spy_ts_update, raising=True)

    # start phonecall task via task scheduler
    counts["ts_start_task"] = 0
    original_ts_start_task = SimulatedTaskScheduler.start_task

    @functools.wraps(original_ts_start_task)
    def spy_ts_start_task(self, text: str, **kw):
        counts["ts_start_task"] += 1
        return original_ts_start_task(self, text, **kw)

    monkeypatch.setattr(
        SimulatedTaskScheduler,
        "start_task",
        spy_ts_start_task,
        raising=True,
    )

    # task manager
    tm = SimulatedTaskManager("CRM scenario – follow-up meeting scheduling.")

    # Read-only lookup
    usr_msg = "What is Alice Reynolds phone number?"
    h1 = tm.ask(usr_msg)
    assistant_resp = await asyncio.wait_for(h1.result(), timeout=60000)
    chat = [{"user": usr_msg}, {"assistant": assistant_resp}]

    # Update the number
    usr_msg = "Please update it to '+123456789', she recently changed it."
    h2 = tm.request(usr_msg, parent_chat_context=chat)
    assistant_resp = await asyncio.wait_for(h2.result(), timeout=60000)
    chat += [{"user": usr_msg}, {"assistant": assistant_resp}]

    # create task to call her and start it
    usr_msg = "Give Alice a call and ask when she is next free."
    h3 = tm.request(usr_msg, parent_chat_context=chat)
    assistant_resp = await asyncio.wait_for(h3.result(), timeout=60000)

    # check + update contact
    assert counts["cm_ask"] == 1, "ContactManager.ask should be called once."
    assert counts["cm_update"] == 1, "ContactManager.update should be called once."

    assert counts["ts_ask"] == 1, "TaskScheduler.ask should be called once."
    assert counts["ts_update"] in (
        0,
        1,
    ), "TaskScheduler.update should be called either no times or once."
    assert (
        counts["ts_start_task"] == 1
    ), "TaskScheduler.start_task should be called once."


# --------------------------------------------------------------------------- #
# 2. Transcript summary, follow-up Q&A, then unrelated mutation               #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_transcript_summary_followups(monkeypatch):
    """
    • request → TranscriptManager.summarize once
    • ask     → TranscriptManager.ask       once
    • final request does not need transcript calls
    """
    counts = {"sum": 0, "t_ask": 0, "ts_update": 0}

    orig_sum = SimulatedTranscriptManager.summarize
    orig_t_ask = SimulatedTranscriptManager.ask
    orig_ts_update = SimulatedTaskScheduler.update

    @functools.wraps(orig_sum)
    async def spy_sum(self, **kw):
        counts["sum"] += 1
        return await orig_sum(self, **kw)

    @functools.wraps(orig_t_ask)
    def spy_t_ask(self, text: str, **kw):
        counts["t_ask"] += 1
        return orig_t_ask(self, text, **kw)

    @functools.wraps(orig_ts_update)
    def spy_ts_update(self, text: str, **kw):
        counts["ts_update"] += 1
        return orig_ts_update(self, text, **kw)

    monkeypatch.setattr(SimulatedTranscriptManager, "summarize", spy_sum, raising=True)
    monkeypatch.setattr(SimulatedTranscriptManager, "ask", spy_t_ask, raising=True)
    monkeypatch.setattr(SimulatedTaskScheduler, "update", spy_ts_update, raising=True)

    tm = SimulatedTaskManager("Support-call archive demo.")

    # 1️⃣ Summarise & store
    usr_msg = (
        "Summarise support call with exchange_id == 123 from yesterday and store it."
    )
    r1 = tm.request(usr_msg)
    assistant_resp = await asyncio.wait_for(r1.result(), timeout=60)
    chat = [{"user": usr_msg}, {"assistant": assistant_resp}]

    # 2️⃣ Follow-up read query
    usr_msg = "What was the main action item in that summary?"
    q2 = tm.ask(usr_msg)
    assistant_resp = await asyncio.wait_for(q2.result(), timeout=60)
    chat += [{"user": usr_msg}, {"assistant": assistant_resp}]

    # 3️⃣ Unrelated mutation (no additional transcript calls required)
    r3 = tm.request(
        "Create a high-priority task for that action item and assign it to DevOps.",
        parent_chat_context=chat,
    )
    await asyncio.wait_for(r3.result(), timeout=60)

    assert counts == {
        "sum": 1,
        "t_ask": 1,
        "ts_update": 1,
    }, "Unexpected transcript-tool call count."
