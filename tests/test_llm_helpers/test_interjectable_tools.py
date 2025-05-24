"""
End-to-end test for *interjectable* tools.

Flow we enforce:

1. Assistant calls `long_running(topic="cats")`.
2. Test code sees that call, then injects the user turn
   “Actually, please switch to dogs instead.”
3. Assistant calls the auto-generated helper `interject_<id>`
   which drops “dogs” on the tool’s private queue.
4. `long_running` returns a value that reflects the steer.
5. Assistant outputs one final plain-text answer.

We assert that the tool logs the steer and the final assistant reply
mentions “dogs”.
"""

from __future__ import annotations

import asyncio
import os
from typing import List

import pytest
import unify
from unity.common.llm_helpers import start_async_tool_use_loop

MODEL_NAME = os.getenv("UNIFY_MODEL", "gpt-4o@openai")


def new_client() -> unify.AsyncUnify:
    """Fresh client with caching enabled so the run becomes deterministic."""
    return unify.AsyncUnify(MODEL_NAME, cache=True)


@pytest.mark.asyncio
async def test_interjectable_tool_roundtrip() -> None:
    client = new_client()

    # ── 1.  Dummy long-running tool ───────────────────────────────────
    exec_log: List[str] = []

    async def long_running(
        topic: str,
        *,
        interject_queue: asyncio.Queue[str],
    ) -> str:
        """
        Wait up to 2 s for a steer; echo whichever topic we end up with.
        """
        try:
            steer = await asyncio.wait_for(interject_queue.get(), timeout=2.0)
            exec_log.append(f"steered→{steer}")
            return f"Topic switched to: {steer}"
        except asyncio.TimeoutError:
            exec_log.append("no-steer")
            return f"Final topic: {topic}"

    # ── 2.  Kick off the async-tool loop ──────────────────────────────
    handle = start_async_tool_use_loop(
        client=client,
        message=(
            "Follow STRICTLY these steps:\n"
            "1️⃣  Call `long_running` with `{ \"topic\": \"cats\" }`.\n"
            "2️⃣  WAIT for my next instruction.\n"
            "3️⃣  When I say “Actually, please switch to X instead.” "
            "call the helper `interject_<id>` with `{ \"content\": \"X\" }`.\n"
            "4️⃣  After the tool finishes, reply with ONE sentence "
            "mentioning the final topic.\n"
            "Do NOT add extra text between steps."
        ),
        tools={"long_running": long_running},
        interjectable_tools={"long_running"},
    )

    # ── 3.  Wait until the model has really scheduled the first call ──
    cats_call_seen = False
    for _ in range(40):  # up to ~4 s
        await asyncio.sleep(0.1)
        for m in client.messages:
            if (
                m.get("role") == "assistant"
                and m.get("tool_calls")
                and '"topic":"cats"' in m["tool_calls"][0]["function"]["arguments"]
            ):
                cats_call_seen = True
                break
        if cats_call_seen:
            break
    assert cats_call_seen, "LLM never called long_running with cats."

    # ── 4.  Now inject the steer while the tool is still running ──────
    await handle.interject("Actually, please switch to dogs instead.")

    # ── 5.  Wait for completion ───────────────────────────────────────
    final_answer: str = await handle.result()

    # ── 6.  Assertions ────────────────────────────────────────────────
    assert exec_log == ["steered→dogs"], "Tool must receive the 'dogs' steer."
    assert "dogs" in final_answer.lower(), "Assistant reply must mention dogs."

    # There will be two assistant turns (one with the tool call, one final)
    assistant_msgs = [m for m in client.messages if m["role"] == "assistant"]
    assert assistant_msgs[-1]["tool_calls"] is None, (
        "The last assistant message should be plain-text – "
        "all tool calls should precede it."
    )
