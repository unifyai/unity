from __future__ import annotations
import asyncio, pytest, re, json

from unity.task_scheduler.task_scheduler import TaskScheduler
from tests.assertion_helpers import assertion_failed
from tests.helpers import _handle_project


# ----------------------------------------------------------------------------
# shared helper
# ----------------------------------------------------------------------------
def _contains(txt: str, *needles: str) -> bool:
    return all(re.search(n, txt, re.I) for n in needles)


# ----------------------------------------------------------------------------
# 1) parent-context disambiguation
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
@_handle_project
async def test_ts_ask_uses_parent_context():
    """
    The user previously agreed to call the “Hotfix security vulnerability”
    task **Thunderbolt**.  That context is passed in; no clarification should
    be needed.
    """
    ts = TaskScheduler()
    # seed tasks (ScenarioBuilder logic condensed)
    ts._create_task(
        name="Hotfix security vulnerability",
        description="Apply CVE-2025-1234 patch to all services.",
        status="primed",
        priority="high",
    )

    parent_ctx = [
        {
            "role": "user",
            "content": "From now on let's nickname the hotfix task 'Thunderbolt'.",
        },
        {"role": "assistant", "content": "Understood – Hotfix ⇢ Thunderbolt."},
    ]

    handle = await ts.ask(
        "What is the priority level of the Thunderbolt task?",
        parent_chat_context=parent_ctx,
    )
    answer = await handle.result()
    assert _contains(answer, "high"), assertion_failed(
        "Answer containing 'high'",
        answer,
        "Parent-context nickname not respected",
    )


# ----------------------------------------------------------------------------
# 2) clarification bubble-up
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
@_handle_project
async def test_ts_ask_requests_clarification():
    """
    There are **two** queued tasks.  Asking “What is the description of the
    queued task?” is ambiguous; the model must request clarification, then
    finish with the correct description once we specify which one.
    """
    ts = TaskScheduler()

    # two queued tasks
    ts._create_task(
        name="Prepare slide deck",
        description="Create slides for the upcoming board meeting.",
        status="primed",
        priority="high",
    )
    ts._create_task(
        name="Hotfix security vulnerability",
        description="Apply CVE-2025-1234 patch to all services.",
        status="queued",
        priority="high",
    )

    up_q, down_q = asyncio.Queue(), asyncio.Queue()

    # run ask in background (loop starts immediately)
    handle = await ts.ask(
        "What is the description of the high priority task? Please request clarification if there is more than one.",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    # expect a clarification question
    question = await asyncio.wait_for(up_q.get(), timeout=300)
    assert _contains(question.lower(), "high") and _contains(
        question.lower(),
        "priority",
    ), "No clarification question"

    # user clarifies we mean the hotfix
    await down_q.put("I mean the Hotfix task.")

    # final answer
    answer = await handle.result()
    assert _contains(answer, "CVE-2025-1234"), assertion_failed(
        "Answer mentioning CVE-2025-1234 patch",
        answer,
        "Clarification answer not propagated",
    )


# --------------------------------------------------------------------------- #
# 3) update – parent-context disambiguation                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_ts_update_uses_parent_context():
    """
    User nicknames 'Hotfix security vulnerability' as *Thunderbolt* in a
    prior exchange.  update() must interpret that nickname without asking.
    """
    ts = TaskScheduler()

    # seed task
    tid = ts._create_task(
        name="Hotfix security vulnerability",
        description="Apply CVE-2025-1234 patch to all services.",
        status="primed",
        priority="high",
    )["details"]["task_id"]

    parent_ctx = [
        {
            "role": "user",
            "content": "Remember: 'Thunderbolt' is just codename for the task named 'Hotfix security vulnerability'.",
        },
        {"role": "assistant", "content": "Acknowledged."},
    ]

    # ask to mark Thunderbolt completed
    await (
        await ts.update(
            "Mark the 'Thunderbolt' task as completed.",
            parent_chat_context=parent_ctx,
        )
    ).result()

    row = ts._filter_tasks(filter=f"task_id == {tid}", limit=1)[0]
    assert row["status"] == "completed", assertion_failed(
        "Task status 'completed'",
        json.dumps(row, indent=2),
        "Parent-context nickname not respected by update()",
    )


# --------------------------------------------------------------------------- #
# 4) update – clarification bubble-up                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_ts_update_requests_clarification():
    """
    Two queued tasks – user says “Set the queued task priority to high”.
    update() should ask *which* queued task, wait for answer, then update.
    """
    ts = TaskScheduler()

    # two queued tasks
    ts._create_task(
        name="Prepare slide deck",
        description="Create slides for the upcoming board meeting.",
        status="primed",
        priority="normal",
    )
    tid_report = ts._create_task(
        name="Write quarterly report",
        description="Compile and draft the Q2 report.",
        status="queued",
        priority="normal",
    )["details"]["task_id"]
    ts._create_task(
        name="Interview candidate",
        description="Interview the recent sales applicant.",
        status="queued",
        priority="normal",
    )

    up_q, down_q = asyncio.Queue(), asyncio.Queue()

    task = asyncio.create_task(
        (
            await ts.update(
                "Set the queued task's priority to high. Please request clarification if there is more than one.",
                clarification_up_q=up_q,
                clarification_down_q=down_q,
            )
        ).result(),
    )

    # clarification expected
    q_text = await asyncio.wait_for(up_q.get(), timeout=300)
    assert _contains(q_text, "queued"), "No clarification question"

    # clarify we mean the slide-deck task
    await down_q.put("I mean the Write quarterly report task.")

    await asyncio.wait_for(task, timeout=300)

    row = ts._filter_tasks(filter=f"task_id == {tid_report}", limit=1)[0]
    assert row["priority"] == "high", assertion_failed(
        "Task priority 'high'",
        json.dumps(row, indent=2),
        "Priority not updated after clarification",
    )
