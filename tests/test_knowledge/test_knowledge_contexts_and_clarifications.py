from __future__ import annotations

import asyncio
import json
import re
import pytest

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from tests.helpers import _handle_project
from tests.assertion_helpers import assertion_failed


# --------------------------------------------------------------------------- #
# small helper – case-insensitive contains                                    #
# --------------------------------------------------------------------------- #
def _contains(text: str, *needles: str) -> bool:
    return all(re.search(n, text, re.I) for n in needles)


# --------------------------------------------------------------------------- #
# 1.  Parent-context test                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.timeout(300)
@_handle_project
async def test_update_uses_parent_context():
    """
    A prior conversation instructs the assistant to call 'Carlos' by the
    codename 'Alpha'.  We pass that *parent_chat_context* to .store() and
    verify that the resulting row contains **Alpha** and not **Carlos**.
    """
    km = KnowledgeManager()

    parent_ctx = [
        {
            "role": "user",
            "content": "Whenever you store anything about Project Nova, please refer to it as 'Alpha' and no need to mention Nova anymore.",
        },
        {"role": "assistant", "content": "Understood – Project Nova → Alpha."},
    ]

    handle = await km.update(
        "Project Nova was initiated in 1990.",
        parent_chat_context=parent_ctx,  # ← will be threaded into the loop
    )
    await handle.result()

    # the raw knowledge dump should contain Alpha but NOT Carlos
    all_data_json = json.dumps(
        km._filter(),
    )  # private helper OK for assertions
    assert ("Alpha" in all_data_json and "Project Nova" not in all_data_json) or (
        "Alpha" in all_data_json
        and "alias" in all_data_json
        and "Project Nova" in all_data_json
    ), assertion_failed(
        "Row mentioning 'Alpha' but not 'Project Nova'",
        all_data_json,
        "Parent-context instruction was not applied",
    )


# --------------------------------------------------------------------------- #
# 2.  Clarification-bubble test                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.timeout(300)
@_handle_project
async def test_update_requests_clarification():
    """
    The instruction is ambiguous ("store Carlos' birth year under his
    *surname*") – since the surname is unknown the tool must ask a
    clarification via `clarification_up_q`, wait for the answer, then finish.
    """
    km = KnowledgeManager()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await km.update(
        "Please store Project Nova's initiation year (1990) using its *registry code* as the key.",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    # ➊ the very first thing should be a clarification question
    question = await asyncio.wait_for(up_q.get(), timeout=300)
    assert _contains(
        question,
        "registry",
    ), "No clarification question about the registry code"

    # ➋ provide the missing detail
    await down_q.put("Project Nova's registry code is NV-1990.")

    # ➌ wait for completion and verify the data was stored correctly
    await handle.result()
    data_json = json.dumps(km._filter())
    assert _contains(data_json, "NV-1990", "1990"), assertion_failed(
        "Row containing registry code 'NV-1990' and initiation year '1990'",
        data_json,
        "Clarification answer did not propagate into stored data",
    )


# --------------------------------------------------------------------------- #
# 3.  Retrieve – parent-context disambiguation                                #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.timeout(300)
@_handle_project
async def test_ask_uses_parent_context():
    """
    We stored data about *Carlos*.  The user later calls him "Alpha".
    The mapping is provided only via the parent chat context, therefore
    retrieval must rely on it (no clarifications needed).
    """
    km = KnowledgeManager()

    # ➊ store a simple fact under the original name
    handle = await km.update("Project Nova was initiated in 1990.")
    await handle.result()

    # ➋ build parent-level mapping
    parent_ctx = [
        {
            "role": "user",
            "content": "Remember that 'Alpha' is another name for Project Nova.",
        },
        {"role": "assistant", "content": "Got it – Project Nova ≡ Alpha."},
    ]

    # ➌ ask about Alpha – model must translate via context
    handle = await km.ask(
        "When was Alpha initiated?",
        parent_chat_context=parent_ctx,
        _return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "1990"), assertion_failed(
        "Answer containing '1990' (initiation year)",
        answer,
        reasoning,
        "Parent-context mapping not respected",
        {"Knowledge Data": km._filter()},
    )


# --------------------------------------------------------------------------- #
# 4.  Retrieve – clarification bubble-up                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.timeout(300)
@_handle_project
async def test_ask_requests_clarification():
    """
    We have *four* cars in a garage, each with a different colour.
    When the user asks "What colour is the car in the garage?” without disambiguation,
    retrieve() should request clarification; supplying the car model must
    let it finish with the correct answer.
    """
    km = KnowledgeManager()

    # ➊ seed four distinct coloured vehiles
    await (
        await km.update(
            "There is a red Citroen, a blue Volkswagen, a green BMW, and a silver Porsche in the garage.",
        )
    ).result()

    # ➋ clarification channels
    up_q, down_q = asyncio.Queue(), asyncio.Queue()

    # ➌ run retrieve in background
    handle = await km.ask(
        "What colour is the car in the garage? I'm looking for one colour, request clarification if you're not sure.",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )
    task = asyncio.create_task(handle.result())

    # ➍ expect a clarification question
    await asyncio.wait_for(up_q.get(), timeout=60)

    # ➎ answer the question
    await down_q.put("I mean the Porsche.")

    # ➏ await final answer
    answer = await asyncio.wait_for(task, timeout=60)
    assert _contains(answer, "silver"), assertion_failed(
        "Answer containing 'silver' (silver Porsche)",
        answer,
        "Clarification flow failed",
        {"Knowledge Data": km._filter()},
    )
