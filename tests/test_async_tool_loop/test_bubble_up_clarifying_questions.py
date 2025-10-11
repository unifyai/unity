from __future__ import annotations

import asyncio
from typing import Optional

import pytest
import unify
from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project, SETTINGS
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_tool_message_prefix,
    _wait_for_assistant_call_prefix,
    first_user_message,
    first_assistant_tool_call,
    first_assistant_tool_call_by_prefix,
    first_tool_message_by_name_prefix,
    first_tool_message_by_name,
    last_plain_assistant_message,
)

# ──────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────


def make_llm(system_message: Optional[str] = None) -> unify.AsyncUnify:
    return unify.AsyncUnify(
        endpoint="gpt-5@openai",
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
        "You are coordinating internal tools.\n"
        "When any pending tool asks a clarification question, assume the missing detail is a user-specific fact or preference "
        "that is NOT inferable from the current tool context or prior messages in this loop. Do not guess or invent.\n"
        "Therefore: (1) Do not start unrelated base tools until that pending call is unblocked; "
        "(2) Call `request_clarification` to ask the user the exact question and wait for their answer (you may call other tools only to fetch that answer); "
        "(3) As soon as you have the answer, immediately call the generated clarify_{toolName}_{id} helper for that exact pending call to provide the answer so it can resume.\n"
        "After the original call resumes or completes, you may continue with further tools as needed.\n"
        "When the email has been sent successfully, end your final assistant message with an explicit confirmation using the word 'sent' (e.g., 'Email sent.').\n"
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
    )

    # Deterministic ordering using triggers:
    # 1) Wait until assistant schedules send_email
    await _wait_for_tool_request(outer_client, "send_email", timeout=120.0)
    # 2) Wait until the clarification request tool message appears
    await _wait_for_tool_message_prefix(
        outer_client,
        "clarification_request_",
        timeout=120.0,
    )

    # 3) The request_clarification tool will bubble the question up – capture it
    await clar_up_q.get()
    # 4) Provide the answer; assistant should then call a clarify_* helper
    await clar_down_q.put("I'll be bringing sausages and a pack of beer")

    # 5) Ensure the assistant has invoked a clarify_* helper
    await _wait_for_assistant_call_prefix(
        outer_client,
        "clarify_send_email",
        timeout=120.0,
    )

    await outer_handle.result()

    # ─────────────────────────
    # Assertions
    # ─────────────────────────

    import json

    # 1️⃣ original user request – robust to clients that don't persist system messages
    _first_user = first_user_message(outer_client.messages)
    assert _first_user["content"] == (
        "Please email jonathan.smith123@gmail.com and politely tell him I (Dan) "
        "will be arriving at the BBQ around 5pm."
    )

    # 2️⃣ assistant chooses `send_email` --------------------------------------
    m1, call1 = first_assistant_tool_call(outer_client.messages, "send_email")
    args1 = json.loads(call1["function"]["arguments"])
    assert args1["address"] == "jonathan.smith123@gmail.com"

    # 3️⃣ tool asks a clarification question ----------------------------------
    clar_req = first_tool_message_by_name_prefix(
        outer_client.messages,
        "clarification_request_",
    )
    assert "Will you be bringing anything" in clar_req["content"]

    # 4️⃣ assistant calls `request_clarification` -----------------------------
    m3, req_call = first_assistant_tool_call(
        outer_client.messages,
        "request_clarification",
    )

    # 5️⃣ tool returns the user’s answer --------------------------------------
    clar_ans = first_tool_message_by_name(
        outer_client.messages,
        "request_clarification",
    )
    assert "sausages" in clar_ans["content"]
    assert "beer" in clar_ans["content"]

    # 6️⃣ assistant forwards the answer via `_clarify_send_email…` ------------
    _m5, _clar = first_assistant_tool_call_by_prefix(
        outer_client.messages,
        "clarify_send_email",
    )

    # 7️⃣ final tool message contains the real result -------------------------
    final_tool = first_tool_message_by_name_prefix(
        outer_client.messages,
        "clarify_send_email",
    )
    assert "Email sent" in final_tool["content"]

    # 8️⃣ assistant wraps up ---------------------------------------------------
    closing = last_plain_assistant_message(outer_client.messages)
    content = (closing.get("content") or "").lower()
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
        "You are coordinating internal tools in a nested loop.\n"
        "CRITICAL: When any internal tool requests clarification, the missing information is a user-specific preference or fact "
        "that is NOT available to you from tool context or prior messages in this loop. Do not infer, assume, or guess.\n"
        "Therefore, you MUST first call `request_clarification` to ask the user the exact question, wait for the user's answer, "
        "and ONLY THEN call the corresponding clarify_{toolName}_{id} helper to provide that answer so the tool can resume.\n"
        "Never call a clarify_* helper unless you have just obtained the answer via `request_clarification` in this conversation. "
        "For example, for a question like 'what colour should the widget be?', treat it as a personal preference unknown to you; ask the user. "
        "Keep responses concise.",
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
        "You are the TOP-LEVEL coordinator. When any pending tool (including nested delegated tools) asks a clarification "
        "question via a clarification_request_* tool message, you MUST:\n"
        "(1) Call `request_clarification` to ask the user that exact question;\n"
        "(2) Wait for the user's answer;\n"
        "(3) Forward that answer to the pending tool via the clarify_* helper.\n"
        "Treat these as user preferences or facts unknown to you; do NOT answer them yourself, guess, or infer from unrelated context.\n"
        "Do NOT start unrelated tools until pending clarifications are resolved.",
    )

    handle = start_async_tool_loop(
        outer_llm,
        message="Run delegating_tool please.",
        tools={
            "delegating_tool": delegating_tool,
            "request_clarification": request_clarification,
        },
    )

    # ── satisfy the clarification that should bubble up ──────────────────
    question = await asyncio.wait_for(clar_up_q.get(), timeout=300)
    assert "what colour" in question.lower() or "what color" in question.lower()

    await clar_down_q.put("Blue, please")

    # ── loop must now complete successfully ───────────────────────────────
    await asyncio.wait_for(handle.result(), timeout=300)

    # final sanity-check: assistant ends with the confirmation from inner_tool
    assert "blue" in outer_llm.messages[-1]["content"].lower()
