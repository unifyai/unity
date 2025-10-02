from __future__ import annotations

import asyncio
from typing import Optional

import pytest
import unify
from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project, SETTINGS

# ──────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────


def make_llm(system_message: Optional[str] = None) -> unify.AsyncUnify:
    return unify.AsyncUnify(
        endpoint="o4-mini@openai",
        system_message=system_message,
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )


# ──────────────────────────────────────────────────────────────────────────
# 1.  DUMMY TOOLS – send_email immediately needs clarification
# ──────────────────────────────────────────────────────────────────────────
@unify.traced
async def send_email(
    address: str,
    description: str,
    *,
    clarification_up_q: asyncio.Queue[str] | None = None,
    clarification_down_q: asyncio.Queue[str] | None = None,
) -> str:
    """Send an email, based on the general description provided."""
    if clarification_up_q is None or clarification_down_q is None:
        raise RuntimeError("clarification queues missing")

    # Dummy code
    await clarification_up_q.put(
        "It's best that we also inform him what we're bringing. Will you be bringing anything with you (food/drink)?",
    )
    await clarification_down_q.get()
    return f"Email sent!"


@unify.traced
async def send_text(
    number: str,
    description: str,
    *,
    clarification_up_q: asyncio.Queue[str] | None = None,
    clarification_down_q: asyncio.Queue[str] | None = None,
) -> str:
    """Send a text message, based on the general description provided."""
    if clarification_up_q is None or clarification_down_q is None:
        raise RuntimeError("clarification queues missing")


# ──────────────────────────────────────────────────────────────────────────
# 2.  OUTER get_clarification – simulates the user
# ──────────────────────────────────────────────────────────────────────────
asked_questions: list[str] = []  # for assertions


# ──────────────────────────────────────────────────────────────────────────
# 3.  The test
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_clarification_bubbles_up_two_tiers() -> None:
    """
    Verifies that the clarification travels up & the answer travels down two
    levels of the call stack.
    """

    outer_client = make_llm(
        "You are coordinating internal tools. If any pending tool asks a clarification question: "
        "(1) Do not start unrelated base tools until that pending call is unblocked; "
        "(2) Obtain the answer by any suitable means (you may call other tools, or call request_clarification to ask the user); "
        "(3) As soon as you have the answer, immediately call the generated clarify_{toolName}_{id} helper for that exact pending call to provide the answer so it can resume. "
        "After the original call resumes or completes, you may continue with further tools as needed. "
        "When the email has been sent successfully, end your final assistant message with an explicit confirmation using the word 'sent' (e.g., 'Email sent.'). "
        "Do not hallucinate any details; if unknown, ask. Keep responses concise.",
    )

    clar_up_q = asyncio.Queue()
    clar_down_q = asyncio.Queue()

    @unify.traced
    async def request_clarification(
        question: str,
    ) -> str:
        """Ask the user **question** and return their reply."""
        # Bubble the request up …
        await clar_up_q.put(question)
        # … then block until the answer comes back down.
        return await clar_down_q.get()

    request_clarification.__name__ = "request_clarification"
    request_clarification.__qualname__ = "request_clarification"

    outer_tools = {
        "send_email": send_email,
        "send_text": send_text,
        "request_clarification": request_clarification,
    }

    outer_handle = start_async_tool_loop(  # type: ignore[attr-defined]
        outer_client,
        message="Please email jonathan.smith123@gmail.com and politely tell him I (Dan) will be arriving at the BBQ around 5pm.",
        tools=outer_tools,
        log_steps=False,
    )

    await clar_up_q.get()
    await clar_down_q.put("I'll be bringing sausages and a pack of beer")

    await outer_handle.result()

    # ─────────────────────────
    # Assertions
    # ─────────────────────────

    import json

    # 0️⃣ basic shape – we always end with 8 chat entries
    assert len(outer_client.messages) == 8

    # 1️⃣ original user request ------------------------------------------------
    assert outer_client.messages[0]["role"] == "user"
    assert outer_client.messages[0]["content"] == (
        "Please email jonathan.smith123@gmail.com and politely tell him I (Dan) "
        "will be arriving at the BBQ around 5pm."
    )

    # 2️⃣ assistant chooses `send_email` --------------------------------------
    m1 = outer_client.messages[1]
    assert m1["role"] == "assistant"
    assert len(m1["tool_calls"]) == 1
    call1 = m1["tool_calls"][0]
    assert call1["function"]["name"] == "send_email"
    args1 = json.loads(call1["function"]["arguments"])
    assert args1["address"] == "jonathan.smith123@gmail.com"

    # 3️⃣ tool asks a clarification question ----------------------------------
    clar_req = outer_client.messages[2]
    assert clar_req["role"] == "tool"
    assert clar_req["name"].startswith("clarification_request_")
    assert "Will you be bringing anything" in clar_req["content"]

    # 4️⃣ assistant calls `request_clarification` -----------------------------
    m3 = outer_client.messages[3]
    assert m3["role"] == "assistant"
    assert m3["tool_calls"][0]["function"]["name"] == "request_clarification"

    # 5️⃣ tool returns the user’s answer --------------------------------------
    clar_ans = outer_client.messages[4]
    assert clar_ans["role"] == "tool"
    assert clar_ans["name"] == "request_clarification"
    assert "sausages" in clar_ans["content"]
    assert "beer" in clar_ans["content"]

    # 6️⃣ assistant forwards the answer via `_clarify_send_email…` ------------
    m5 = outer_client.messages[5]
    assert m5["role"] == "assistant"
    assert m5["tool_calls"][0]["function"]["name"].startswith("clarify_send_email")

    # 7️⃣ final tool message contains the real result -------------------------
    final_tool = outer_client.messages[6]
    assert final_tool["role"] == "tool"
    assert final_tool["name"].startswith("clarify_send_email")
    assert "Email sent" in final_tool["content"]

    # 8️⃣ assistant wraps up ---------------------------------------------------
    closing = outer_client.messages[7]
    assert closing["role"] == "assistant"
    content = closing["content"].lower()
    assert any(["email" in content, "message" in content]) and "sent" in content


# ---------------------------------------------------------------------------
# inner tool  ➟  always asks one clarification
# ---------------------------------------------------------------------------
async def inner_tool(
    *,
    clarification_up_q: asyncio.Queue[str] | None = None,
    clarification_down_q: asyncio.Queue[str] | None = None,
) -> str:
    if clarification_up_q is None or clarification_down_q is None:
        raise RuntimeError("queues missing")

    await clarification_up_q.put("Inner loop: what colour should the widget be?")
    color = await clarification_down_q.get()
    return f"✅ inner finished, color: {color}"


# ---------------------------------------------------------------------------
# outer tool  ➟  immediately spawns an async-tool loop and RETURNS its handle
# ---------------------------------------------------------------------------
async def delegating_tool(
    *,
    clarification_up_q: asyncio.Queue[str] | None = None,
    clarification_down_q: asyncio.Queue[str] | None = None,
) -> str:  # return type misleading on purpose
    inner_llm = make_llm(
        "If any internal tool needs information, you may call request_clarification to ask the user. "
        "When a running tool is waiting for an answer, first provide that answer via the appropriate clarify_{toolName}_{id} helper "
        "before starting unrelated new tools. You may use other tools to determine the answer if helpful.",
    )

    async def request_clarification(question: str) -> str:
        await clarification_up_q.put(question)
        return await clarification_down_q.get()

    handle = start_async_tool_loop(  # <-- returns AsyncToolLoopHandle
        inner_llm,
        message="Run inner_tool please.",
        tools={
            "inner_tool": inner_tool,
            "request_clarification": request_clarification,
        },
        log_steps=False,
    )
    return handle  # outer tool finishes instantly


# ---------------------------------------------------------------------------
# regression test
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_clarification_bubbles_through_returned_handle() -> None:
    """Clarification raised *inside* the returned handle must still reach the user."""

    clar_up_q: asyncio.Queue[str] = asyncio.Queue()
    clar_down_q: asyncio.Queue[str] = asyncio.Queue()

    async def request_clarification(question: str) -> str:
        await clar_up_q.put(question)
        return await clar_down_q.get()

    outer_llm = make_llm(
        "If any internal tool needs information, call `request_clarification` to ask the user, or use other tools to find the answer. "
        "When a delegated inner tool asks a question, provide the answer via the corresponding clarify_{toolName}_{id} helper as soon as you have it, "
        "then continue. Do not start unrelated new tools until the pending call is unblocked.",
    )

    handle = start_async_tool_loop(
        outer_llm,
        message="Run delegating_tool please.",
        tools={
            "delegating_tool": delegating_tool,
            "request_clarification": request_clarification,
        },
        log_steps=False,
    )

    # ── satisfy the clarification that should bubble up ──────────────────
    question = await asyncio.wait_for(clar_up_q.get(), timeout=300)
    assert "what colour" in question.lower() or "what color" in question.lower()

    await clar_down_q.put("Blue, please")

    # ── loop must now complete successfully ───────────────────────────────
    await asyncio.wait_for(handle.result(), timeout=300)

    # final sanity-check: assistant ends with the confirmation from inner_tool
    assert "blue" in outer_llm.messages[-1]["content"].lower()
