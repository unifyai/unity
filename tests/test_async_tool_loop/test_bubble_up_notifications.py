from __future__ import annotations

import asyncio
from typing import Optional
import json

import pytest
import unify
from unity.common.async_tool_loop import start_async_tool_use_loop
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
# 1.  DUMMY TOOLS – send_email emits notifications
# ──────────────────────────────────────────────────────────────────────────
@unify.traced
async def send_email(
    address: str,
    description: str,
    *,
    notification_up_q: asyncio.Queue | None = None,
) -> str:
    """Send an email, emitting notifications along the way."""
    if notification_up_q is None:
        raise RuntimeError("notification queue missing")

    # Emit notifications; loop will surface them and allow the assistant to react
    await notification_up_q.put({"message": "Composing email…"})
    await asyncio.sleep(0)  # allow the loop to surface the first update
    await notification_up_q.put({"message": "Sending email…"})
    return "Email sent!"


@unify.traced
async def send_text(
    number: str,
    description: str,
    *,
    notification_up_q: asyncio.Queue | None = None,
) -> str:
    """Send a text message (unused in this test)."""
    # Silently do nothing; this tool is present only to mirror the clarifying test shape
    return "Text queued!"


# ──────────────────────────────────────────────────────────────────────────
# 2.  The test (two tiers)
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_notification_bubbles_up_two_tiers() -> None:
    """
    Verifies that notifications emitted by a running tool are surfaced upstream
    and allow the assistant to react; the tool then completes successfully.
    """

    outer_client = make_llm(
        "When long-running internal tools make progress, surface concise, non-blocking updates. "
        "Continue and finish the task without waiting for acknowledgement.",
    )

    outer_tools = {
        "send_email": send_email,
        "send_text": send_text,
    }

    outer_handle = start_async_tool_use_loop(  # type: ignore[attr-defined]
        outer_client,
        message="Please email jonathan.smith123@gmail.com and politely tell him I (Dan) will be arriving at the BBQ around 5pm.",
        tools=outer_tools,
        log_steps=False,
    )

    # Await the first bubbled notification (fire-and-forget)
    notification_event = await asyncio.wait_for(
        outer_handle.next_notification(),
        timeout=60,
    )
    assert notification_event["type"] == "notification"
    assert notification_event["tool_name"] == "send_email"
    # The payload may contain 'message' (string) or richer fields – check message when present
    if "message" in notification_event and isinstance(
        notification_event["message"],
        str,
    ):
        assert any(
            k in notification_event["message"].lower() for k in ["compos", "send"]
        )

    await asyncio.wait_for(outer_handle.result(), timeout=60)

    # ─────────────────────────
    # Assertions (chat transcript shape)
    # ─────────────────────────
    msgs = outer_client.messages

    # 1️⃣ original user request ------------------------------------------------
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == (
        "Please email jonathan.smith123@gmail.com and politely tell him I (Dan) "
        "will be arriving at the BBQ around 5pm."
    )

    # 2️⃣ assistant chooses `send_email` --------------------------------------
    m1 = msgs[1]
    assert m1["role"] == "assistant"
    assert len(m1.get("tool_calls", [])) == 1
    call1 = m1["tool_calls"][0]
    assert call1["function"]["name"] == "send_email"
    args1 = json.loads(call1["function"]["arguments"])
    assert args1["address"] == "jonathan.smith123@gmail.com"

    # 3️⃣ a tool message acknowledges progress (attached to the original call) --
    # Progress updates are emitted as a tool reply to the original tool_call,
    # with name equal to the base tool (e.g. "send_email").
    notification_tool_msgs = [
        m for m in msgs if m.get("role") == "tool" and m.get("name") == "send_email"
    ]
    assert len(notification_tool_msgs) >= 1
    first_prog = (notification_tool_msgs[0].get("content") or "").lower()
    # The content is a JSON-ish string containing {"tool": "send_email", ...}
    assert (
        ("send_email" in first_prog)
        or ("compos" in first_prog)
        or ("send" in first_prog)
    )

    # 4️⃣ final tool message contains the real result -------------------------
    final_tool = next(m for m in reversed(msgs) if m.get("role") == "tool")
    assert "email sent" in (final_tool.get("content") or "").lower()

    # 5️⃣ assistant wraps up ---------------------------------------------------
    closing = msgs[-1]
    assert closing["role"] == "assistant"
    content = (closing.get("content") or "").lower()
    assert any(["email" in content, "message" in content]) and "sent" in content


# ---------------------------------------------------------------------------
# inner tool  ➟  emits notifications (fire-and-forget) and completes
# ---------------------------------------------------------------------------
async def inner_tool(
    *,
    notification_up_q: asyncio.Queue | None = None,
) -> str:
    if notification_up_q is None:
        raise RuntimeError("notification queue missing")

    await notification_up_q.put({"message": "Inner loop: preparing widget"})
    await asyncio.sleep(0)
    await notification_up_q.put({"message": "Inner loop: halfway"})
    return "✅ inner finished"


# ---------------------------------------------------------------------------
# outer tool  ➟  immediately spawns an async-tool loop and RETURNS its handle
#               notification events from the inner loop bubble via the parent's queue
# ---------------------------------------------------------------------------
async def delegating_tool(
    *,
    notification_up_q: asyncio.Queue | None = None,
) -> str:  # return type misleading on purpose
    inner_llm = make_llm(
        "Surface any internal notifications as they occur; continue to completion.",
    )

    # Bridge notifications by closing over the parent notification queue
    async def inner_tool_bridge() -> str:
        return await inner_tool(notification_up_q=notification_up_q)

    handle = start_async_tool_use_loop(  # <-- returns AsyncToolUseLoopHandle
        inner_llm,
        message="Run inner_tool please.",
        tools={
            "inner_tool": inner_tool_bridge,
        },
        log_steps=False,
    )
    return handle  # outer tool finishes instantly


# ---------------------------------------------------------------------------
# regression test
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_notification_bubbles_through_returned_handle() -> None:
    """Notification raised inside the returned handle must still reach the user."""

    outer_llm = make_llm(
        "If any internal work makes progress, you may acknowledge it briefly but continue to completion.",
    )

    handle = start_async_tool_use_loop(
        outer_llm,
        message="Run delegating_tool please.",
        tools={
            "delegating_tool": delegating_tool,
        },
        log_steps=False,
    )

    # ── satisfy: we should receive a bubbled notification event from the INNER loop ──
    event = await asyncio.wait_for(handle.next_notification(), timeout=60)
    assert event["type"] == "notification"
    assert event["tool_name"] == "delegating_tool"
    assert "widget" in (event.get("message") or "").lower()

    # ── loop must now complete successfully ───────────────────────────────
    await asyncio.wait_for(handle.result(), timeout=60)

    # final sanity-check: assistant ends with the confirmation from inner_tool
    assert "finished" in (outer_llm.messages[-1]["content"] or "").lower()
