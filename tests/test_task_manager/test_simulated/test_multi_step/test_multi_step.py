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

from unity.contact_manager.simulated import SimulatedContactManager
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from unity.planner.simulated import SimulatedActiveTask
from unity.task_scheduler.simulated import SimulatedTaskScheduler
from unity.task_manager.simulated import SimulatedTaskManager
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
    • 3rd turn: call here → needs execute_task
    Expected: ContactManager.ask called exactly twice.
    """

    counts = {}

    # Check + Update Contact

    # ask phone number via contact manager
    counts["cm_ask"] = 0
    original_cm_ask = SimulatedContactManager.ask

    @functools.wraps(original_cm_ask)
    async def spy_cm_ask(self, text: str, **kw):
        counts["cm_ask"] += 1
        return await original_cm_ask(self, text, **kw)

    monkeypatch.setattr(SimulatedContactManager, "ask", spy_cm_ask, raising=True)

    # update phone number via contact manager
    counts["cm_update"] = 0
    original_cm_update = SimulatedContactManager.update

    @functools.wraps(original_cm_update)
    async def spy_cm_update(self, text: str, **kw):
        counts["cm_update"] += 1
        return await original_cm_update(self, text, **kw)

    monkeypatch.setattr(SimulatedContactManager, "update", spy_cm_update, raising=True)

    # Task Check, Maybe Create, then Start

    # check if the task already exists
    counts["ts_ask"] = 0
    original_ts_ask = SimulatedTaskScheduler.ask

    @functools.wraps(original_ts_ask)
    async def spy_ts_ask(self, text: str, **kw):
        counts["ts_ask"] += 1
        return await original_ts_ask(self, text, **kw)

    monkeypatch.setattr(SimulatedTaskScheduler, "ask", spy_ts_ask, raising=True)

    # *maybe* create if the simulation says it does not exist yet
    counts["ts_update"] = 0
    original_ts_update = SimulatedTaskScheduler.update

    @functools.wraps(original_ts_update)
    async def spy_ts_update(self, text: str, **kw):
        counts["ts_update"] += 1
        return await original_ts_update(self, text, **kw)

    monkeypatch.setattr(SimulatedTaskScheduler, "update", spy_ts_update, raising=True)

    # start phonecall task via task scheduler
    counts["ts_execute_task"] = 0
    original_ts_execute_task = SimulatedTaskScheduler.execute_task

    @functools.wraps(original_ts_execute_task)
    async def spy_ts_execute_task(self, text: str, **kw):
        counts["ts_execute_task"] += 1
        return await original_ts_execute_task(self, text, **kw)

    monkeypatch.setattr(
        SimulatedTaskScheduler,
        "execute_task",
        spy_ts_execute_task,
        raising=True,
    )

    # task manager
    tm = SimulatedTaskManager("CRM scenario – follow-up meeting scheduling.")

    # Read-only lookup
    usr_msg = "What is Alice Reynolds phone number?"
    h1 = await tm.ask(usr_msg)
    assistant_resp = await asyncio.wait_for(h1.result(), timeout=60000)
    chat = [{"user": usr_msg}, {"assistant": assistant_resp}]

    # Update the number
    usr_msg = "Please update it to '+123456789', she recently changed it."
    h2 = await tm.request(usr_msg, parent_chat_context=chat)
    assistant_resp = await asyncio.wait_for(h2.result(), timeout=60000)
    chat += [{"user": usr_msg}, {"assistant": assistant_resp}]

    # create task to call her and start it
    usr_msg = "Give Alice a call and ask when she is next free."
    h3 = await tm.request(usr_msg, parent_chat_context=chat)
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
        counts["ts_execute_task"] == 1
    ), "TaskScheduler.execute_task should be called once."


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
    async def spy_t_ask(self, text: str, **kw):
        counts["t_ask"] += 1
        return await orig_t_ask(self, text, **kw)

    @functools.wraps(orig_ts_update)
    async def spy_ts_update(self, text: str, **kw):
        counts["ts_update"] += 1
        return await orig_ts_update(self, text, **kw)

    monkeypatch.setattr(SimulatedTranscriptManager, "summarize", spy_sum, raising=True)
    monkeypatch.setattr(SimulatedTranscriptManager, "ask", spy_t_ask, raising=True)
    monkeypatch.setattr(SimulatedTaskScheduler, "update", spy_ts_update, raising=True)

    tm = SimulatedTaskManager("Support-call archive demo.")

    # 1️⃣ Summarise & store
    usr_msg = (
        "Summarise support call with exchange_id == 123 from yesterday and store it."
    )
    r1 = await tm.request(usr_msg)
    assistant_resp = await asyncio.wait_for(r1.result(), timeout=60)
    chat = [{"user": usr_msg}, {"assistant": assistant_resp}]

    # 2️⃣ Follow-up read query
    usr_msg = "What was the main action item in that summary?"
    q2 = await tm.ask(usr_msg)
    assistant_resp = await asyncio.wait_for(q2.result(), timeout=60)
    chat += [{"user": usr_msg}, {"assistant": assistant_resp}]

    # 3️⃣ Unrelated mutation (no additional transcript calls required)
    r3 = await tm.request(
        "Create a high-priority task for that action item and assign it to DevOps.",
        parent_chat_context=chat,
    )
    await asyncio.wait_for(r3.result(), timeout=60)

    assert counts == {
        "sum": 1,
        "t_ask": 1,
        "ts_update": 1,
    }, "Unexpected transcript-tool call count."


# --------------------------------------------------------------------------- #
# 3. Knowledge-base change audit: read → maybe-write → read                   #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_knowledge_change_audit(monkeypatch):
    """
    • First ask: retrieve once
    • Conditional update: retrieve again, then store once
    """
    counts = {"km_retrieve": 0, "km_store": 0, "ts_update": 0}
    orig_ret = SimulatedKnowledgeManager.ask
    orig_store = SimulatedKnowledgeManager.update
    orig_ts_update = SimulatedTaskScheduler.update

    @functools.wraps(orig_ret)
    async def spy_ret(self, text: str, **kw):
        counts["km_retrieve"] += 1
        return await orig_ret(self, text, **kw)

    @functools.wraps(orig_store)
    async def spy_store(self, text: str, **kw):
        counts["km_store"] += 1
        return await orig_store(self, text, **kw)

    @functools.wraps(orig_ts_update)
    async def spy_ts_update(self, text: str, **kw):
        counts["ts_update"] += 1
        return await orig_ts_update(self, text, **kw)

    monkeypatch.setattr(SimulatedKnowledgeManager, "ask", spy_ret, raising=True)
    monkeypatch.setattr(SimulatedKnowledgeManager, "update", spy_store, raising=True)
    monkeypatch.setattr(SimulatedTaskScheduler, "update", spy_ts_update, raising=True)

    tm = SimulatedTaskManager("HR policy KB audit.")

    # 1️⃣ Initial read
    usr_msg = "How many months of severance do we record for exec layoffs?"
    q1 = await tm.ask(usr_msg)
    assistant_msg = await asyncio.wait_for(q1.result(), timeout=60)
    chat = [{"user": usr_msg}, {"assistant": assistant_msg}]

    # 2️⃣ Conditional write + read
    usr_msg = (
        "If it isn't recorded as six months, update it to six months and "
        "create a task noting the previous value."
    )
    r2 = await tm.request(usr_msg, parent_chat_context=chat)
    assistant_msg = await asyncio.wait_for(r2.result(), timeout=60)
    chat += [{"user": usr_msg}, {"assistant": assistant_msg}]

    assert (
        counts["km_retrieve"] >= 1
    ), "KnowledgeManager.retrieve should be called at least once."
    assert counts["km_store"] == 1, "KnowledgeManager.store should be called once."
    assert counts["ts_update"] == 1, "TaskScheduler.update should be called once."


# --------------------------------------------------------------------------- #
# 4. Sprint rollover: ask → update → ask                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_task_scheduler_rollover(monkeypatch):
    counts = {"ask": 0, "update": 0}
    o_ask = SimulatedTaskScheduler.ask
    o_upd = SimulatedTaskScheduler.update

    @functools.wraps(o_ask)
    async def spy_ask(self, text: str, **kw):
        counts["ask"] += 1
        return await o_ask(self, text, **kw)

    @functools.wraps(o_upd)
    async def spy_upd(self, text: str, **kw):
        counts["update"] += 1
        return await o_upd(self, text, **kw)

    monkeypatch.setattr(SimulatedTaskScheduler, "ask", spy_ask, raising=True)
    monkeypatch.setattr(SimulatedTaskScheduler, "update", spy_upd, raising=True)

    tm = SimulatedTaskManager("Engineering sprint rollover.")

    # 1️⃣ Query backlog
    usr_msg = "Which tasks are currently 'queued'?"
    q1 = await tm.ask(usr_msg)
    assistant_resp = await asyncio.wait_for(q1.result(), timeout=60)
    chat = [{"user": usr_msg}, {"assistant": assistant_resp}]

    # 2️⃣ Bulk carry-over
    usr_msg = "Mark all of these tasks as cancelled."
    r2 = await tm.request(
        usr_msg,
        parent_chat_context=chat,
    )
    assistant_resp = await asyncio.wait_for(r2.result(), timeout=60)
    chat += [{"user": usr_msg}, {"assistant": assistant_resp}]

    # 3️⃣ Confirm empty
    q3 = await tm.ask(
        "Double-check the queue backlog is now empty.",
        parent_chat_context=chat,
    )
    await asyncio.wait_for(q3.result(), timeout=60)

    assert counts == {"ask": 2, "update": 1}, "Unexpected TaskScheduler call count."


# --------------------------------------------------------------------------- #
# 5. Start task then interject                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_plan_activation_and_interjection(monkeypatch):
    counts = {"start": 0, "plan_interject": 0}
    start_called = asyncio.Event()

    # --- patch execute_task -------------------------------------------------- #
    orig_start = SimulatedTaskScheduler.execute_task

    @functools.wraps(orig_start)
    async def spy_start(self, task_id: int, **kw):
        counts["start"] += 1
        start_called.set()
        return await orig_start(self, task_id, **kw)

    monkeypatch.setattr(SimulatedTaskScheduler, "execute_task", spy_start, raising=True)

    # --- patch SimulatedPlan.interject (called via _interject_plan_call_) -- #
    orig_plan_interject = SimulatedActiveTask.interject

    @functools.wraps(orig_plan_interject)
    async def spy_plan_interject(self, instruction: str):
        counts["plan_interject"] += 1
        return await orig_plan_interject(self, instruction)

    monkeypatch.setattr(
        SimulatedActiveTask,
        "interject",
        spy_plan_interject,
        raising=True,
    )

    tm = SimulatedTaskManager("Nightly data-sync demo.")

    # 1️⃣ Kick-off the task (this spawns _execute_task_call_)
    r1 = await tm.request(
        "Run task with task_id == 123 (nightly data sync) immediately.",
    )

    # 2️⃣ Wait until we are *sure* execute_task has been invoked
    await asyncio.wait_for(start_called.wait(), timeout=60)

    # 3️⃣ Now interject – guaranteed to hit the running plan
    await asyncio.wait_for(
        r1.interject(
            "Please make sure we sync the data across all servers, "
            "not only those in the US.",
        ),
        timeout=60,
    )

    # 4️⃣ Let the outer loop finish gracefully
    await asyncio.wait_for(r1.result(), timeout=120)

    assert counts == {
        "start": 1,
        "plan_interject": 1,
    }, "Plan activation/interjection counts off."


# --------------------------------------------------------------------------- #
# 6. Interleaved use of four different tools within one mutation request      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_interleaved_tools(monkeypatch):
    counts = {
        "km_ret": 0,
        "cm_ask": 0,
        "tm_ask": 0,
        "ts_upd": 0,
    }

    # -- Knowledge retrieve
    orig_kb_ret = SimulatedKnowledgeManager.ask

    @functools.wraps(orig_kb_ret)
    async def spy_kb_ret(self, text: str, **kw):
        counts["km_ret"] += 1
        return await orig_kb_ret(self, text, **kw)

    # -- Contact ask
    orig_cm_ask = SimulatedContactManager.ask

    @functools.wraps(orig_cm_ask)
    async def spy_cm_ask(self, text: str, **kw):
        counts["cm_ask"] += 1
        return await orig_cm_ask(self, text, **kw)

    # -- Transcript ask
    orig_tm_ask = SimulatedTranscriptManager.ask

    @functools.wraps(orig_tm_ask)
    async def spy_tm_ask(self, text: str, **kw):
        counts["tm_ask"] += 1
        return await orig_tm_ask(self, text, **kw)

    # -- Task update
    orig_ts_upd = SimulatedTaskScheduler.update

    @functools.wraps(orig_ts_upd)
    async def spy_ts_upd(self, text: str, **kw):
        counts["ts_upd"] += 1
        return await orig_ts_upd(self, text, **kw)

    monkeypatch.setattr(SimulatedKnowledgeManager, "ask", spy_kb_ret, raising=True)
    monkeypatch.setattr(SimulatedContactManager, "ask", spy_cm_ask, raising=True)
    monkeypatch.setattr(SimulatedTranscriptManager, "ask", spy_tm_ask, raising=True)
    monkeypatch.setattr(SimulatedTaskScheduler, "update", spy_ts_upd, raising=True)

    tm = SimulatedTaskManager("Contract-renewal campaign demo.")

    h = await tm.request(
        "Create a task for updating client contracts. "
        "Include the latest contract template from the knowledge-base, "
        "tag every contact we currently have, "
        "and attach a short summary of the last email thread. "
        "Set the due date two weeks from today, and schedule the task for next Monday. "
        "Do not make a start on it yet.",
    )
    await asyncio.wait_for(h.result(), timeout=120)

    expected = {"km_ret": 1, "cm_ask": 1, "tm_ask": 1, "ts_upd": 1}
    assert counts == expected, "Interleaved tool-call counts do not match."
