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
@pytest.mark.timeout(120)
@_handle_project
async def test_store_uses_parent_context():
    """
    A prior conversation instructs the assistant to call 'Carlos' by the
    codename 'Alpha'.  We pass that *parent_chat_context* to .store() and
    verify that the resulting row contains **Alpha** and not **Carlos**.
    """
    km = KnowledgeManager()

    parent_ctx = [
        {
            "role": "user",
            "content": "Whenever you store anything about Carlos, please refer to him as 'Alpha'.",
        },
        {"role": "assistant", "content": "Understood – Carlos → Alpha."},
    ]

    handle = km.store(
        "Carlos was born in 1990.",
        parent_chat_context=parent_ctx,  # ← will be threaded into the loop
    )
    await handle.result()

    # the raw knowledge dump should contain Alpha but NOT Carlos
    all_data_json = json.dumps(
        km._search_knowledge(),
    )  # private helper OK for assertions
    assert "Alpha" in all_data_json and "Carlos" not in all_data_json, assertion_failed(
        "Row mentioning 'Alpha' but not 'Carlos'",
        all_data_json,
        "Parent-context instruction was not applied",
    )


# --------------------------------------------------------------------------- #
# 2.  Clarification-bubble test                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.timeout(120)
@_handle_project
async def test_store_requests_clarification():
    """
    The instruction is ambiguous (“store Carlos' birth year under his
    *surname*”) – since the surname is unknown the tool must ask a
    clarification via `clarification_up_q`, wait for the answer, then finish.
    """
    km = KnowledgeManager()

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = km.store(
        "Please store Carlos' birth year (1990) using his *surname* as the key.",
        clarification_up_q=up_q,
        clarification_down_q=down_q,
    )

    # ➊ the very first thing should be a clarification question
    question = await asyncio.wait_for(up_q.get(), timeout=30)
    assert _contains(question, "surname"), "No clarification question about the surname"

    # ➋ provide the missing detail
    await down_q.put("Carlos' surname is Rodriguez.")

    # ➌ wait for completion and verify the data was stored correctly
    await handle.result()
    data_json = json.dumps(km._search_knowledge())
    assert _contains(data_json, "Rodriguez", "1990"), assertion_failed(
        "Row containing surname 'Rodriguez' and birth year '1990'",
        data_json,
        "Clarification answer did not propagate into stored data",
    )


# --------------------------------------------------------------------------- #
# 3.  Retrieve – parent-context disambiguation                                #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.timeout(120)
@_handle_project
async def test_retrieve_uses_parent_context():
    """
    We stored data about *Carlos*.  The user later calls him “Alpha”.
    The mapping is provided only via the parent chat context, therefore
    retrieval must rely on it (no clarifications needed).
    """
    km = KnowledgeManager()

    # ➊ store a simple fact under the original name
    await km.store("Carlos was born in 1990.").result()

    # ➋ build parent-level mapping
    parent_ctx = [
        {
            "role": "user",
            "content": "Remember that 'Alpha' is another name for Carlos.",
        },
        {"role": "assistant", "content": "Got it – Carlos ≡ Alpha."},
    ]

    # ➌ ask about Alpha – model must translate via context
    handle = km.retrieve(
        "When was Alpha born?",
        parent_chat_context=parent_ctx,
        _return_reasoning_steps=True,
    )
    answer, reasoning = await handle.result()
    assert _contains(answer, "1990"), assertion_failed(
        "Answer containing '1990' (birth year)",
        answer,
        reasoning,
        "Parent-context mapping not respected",
        {"Knowledge Data": km._search_knowledge()},
    )


# --------------------------------------------------------------------------- #
# 4.  Retrieve – clarification bubble-up                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.timeout(120)
@_handle_project
async def test_retrieve_requests_clarification():
    """
    We have *two* people named Alex in storage.
    When the user asks “When was Alex born?” without disambiguation,
    retrieve() should request clarification; supplying the surname must
    let it finish with the correct answer.
    """
    km = KnowledgeManager()

    # ➊ seed two distinct Alex rows
    await km.store("Alex Johnson was born in 1990.").result()
    await km.store("Alex Lee was born in 1985.").result()

    # ➋ clarification channels
    up_q, down_q = asyncio.Queue(), asyncio.Queue()

    # ➌ run retrieve in background
    task = asyncio.create_task(
        km.retrieve(
            "When was Alex born?",
            clarification_up_q=up_q,
            clarification_down_q=down_q,
        ).result(),  # .result() returns the final string
    )

    # ➍ expect a clarification question
    question = await asyncio.wait_for(up_q.get(), timeout=30)
    assert _contains(question, "which", "Alex"), "No disambiguation question"

    # ➎ answer the question
    await down_q.put("I mean Alex Lee.")

    # ➏ await final answer
    answer = await asyncio.wait_for(task, timeout=30)
    assert _contains(answer, "1985"), assertion_failed(
        "Answer containing '1985' (Alex Lee)",
        answer,
        "Clarification flow failed",
        {"Knowledge Data": km._search_knowledge()},
    )
