from __future__ import annotations
import asyncio, pytest, re

from unity.task_list_manager.task_list_manager import TaskListManager
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
async def test_tlm_ask_uses_parent_context():
    """
    The user previously agreed to call the “Hotfix security vulnerability”
    task **Thunderbolt**.  That context is passed in; no clarification should
    be needed.
    """
    tlm = TaskListManager()
    # seed tasks (ScenarioBuilder logic condensed)
    tlm._create_task(
        name="Hotfix security vulnerability",
        description="Apply CVE-2025-1234 patch to all services.",
        status="queued",
        priority="high",
    )

    parent_ctx = [
        {
            "role": "user",
            "content": "From now on let's nickname the hotfix task 'Thunderbolt'.",
        },
        {"role": "assistant", "content": "Understood – Hotfix ⇢ Thunderbolt."},
    ]

    handle = tlm.ask(
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
async def test_tlm_ask_requests_clarification():
    """
    There are **two** queued tasks.  Asking “What is the description of the
    queued task?” is ambiguous; the model must request clarification, then
    finish with the correct description once we specify which one.
    """
    tlm = TaskListManager()

    # two queued tasks
    tlm._create_task(
        name="Prepare slide deck",
        description="Create slides for the upcoming board meeting.",
        status="queued",
        priority="normal",
    )
    tlm._create_task(
        name="Hotfix security vulnerability",
        description="Apply CVE-2025-1234 patch to all services.",
        status="queued",
        priority="high",
    )

    up_q, down_q = asyncio.Queue(), asyncio.Queue()

    # run ask in background (loop starts immediately)
    handle = tlm.ask(
        "What is the description of the queued task?",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    # expect a clarification question
    question = await asyncio.wait_for(up_q.get(), timeout=30)
    assert _contains(question, "which", "queued"), "No clarification question"

    # user clarifies we mean the hotfix
    await down_q.put("I mean the Hotfix task.")

    # final answer
    answer = await handle.result()
    assert _contains(answer, "CVE-2025-1234"), assertion_failed(
        "Answer mentioning CVE-2025-1234 patch",
        answer,
        "Clarification answer not propagated",
    )
