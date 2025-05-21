"""
End-to-end behavioural tests for the new *live-handle* features
(`interject`, `stop`, `result`) added to the async-tool loop.

The tests assume that

* `unify.AsyncUnify` exists in the import path,
* `start_async_tool_use_loop` is the public helper that starts the loop and
  returns the handle object, and
* the original `async_tool_use_loop` implementation lives in the same module
  (so we can reuse a couple of internals such as `_dumps` if needed).

We **do not** stub or re-implement `unify`; instead we monkey-patch a few of
its methods on the fly so that the loop sees *plausible* LLM behaviour while
remaining entirely deterministic and fast.
"""

import asyncio
import json
import types
from typing import Any, Dict, List

import pytest

import unify
from unity.common.llm_helpers import start_async_tool_use_loop


# ---------------------------------------------------------------------------#
# Helpers                                                                    #
# ---------------------------------------------------------------------------#

class GenerateScript:
    """
    Tiny state machine that produces a *sequence* of synthetic assistant
    responses so we can control the loop deterministically.
    """

    def __init__(self) -> None:
        self.turn = 0

    def _msg(self, *, tool_calls=None, content=None):
        """Return an object that mimics the OpenAI message structure."""
        return types.SimpleNamespace(tool_calls=tool_calls, content=content)

    def __call__(self, client) -> Any:          # used as patched `generate`
        # pylint: disable=unused-argument
        if self.turn == 0:
            # → Ask for one tool call so the loop spins up a background task.
            self.turn += 1
            tc = [
                {
                    "id": "call_1",
                    "function": {"name": "echo", "arguments": json.dumps({"txt": "A"})},
                }
            ]
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=self._msg(tool_calls=tc))])

        if self.turn == 1:
            # → After the first tool result is returned the model notices the
            #   *interjection* (if any) and may request a second tool call.
            self.turn += 1
            want_b = any(
                m["role"] == "user" and "B please" in m["content"]
                for m in client.messages
            )
            if want_b:
                tc = [
                    {
                        "id": "call_2",
                        "function": {
                            "name": "echo",
                            "arguments": json.dumps({"txt": "B"}),
                        },
                    }
                ]
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=self._msg(tool_calls=tc))]
                )

        # → Final assistant answer (turn 2 or 1 depending on path)
        self.turn += 1
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=self._msg(content="done"))]
        )


async def echo(txt: str) -> str:  # noqa: D401 – simple mock tool
    await asyncio.sleep(0.05)
    return txt


# ---------------------------------------------------------------------------#
# Fixtures                                                                   #
# ---------------------------------------------------------------------------#

@pytest.fixture()
def client():
    return unify.AsyncUnify("gpt-4o@openai")


# ---------------------------------------------------------------------------#
# Tests                                                                       #
# ---------------------------------------------------------------------------#


@pytest.mark.asyncio
async def test_interject_and_result_work_together(client):
    """
    Start the loop, interject a clarification, ensure:
      • the loop *incorporates* the extra user message,
      • the model reacts by asking for an additional tool,
      • we eventually get the expected final answer from `result()`.
    """

    handle = start_async_tool_use_loop(
        client,
        "Echo A please",
        {"echo": echo},
    )

    # - Wait a moment, then inject a follow-up user turn --------------------
    await asyncio.sleep(0.01)
    await handle.interject("And echo B please")  # ← the clarification

    await handle.result()

    # --- Assertions --------------------------------------------------------
    assert client.messages[0] == {'role': 'user', 'content': 'Echo A please'}
    assert client.messages[1]["tool_calls"][0]["function"]["name"] == "echo"
    assert json.loads(client.messages[1]["tool_calls"][0]["function"]["arguments"]) == {"txt": "A"}
    assert client.messages[2]["role"] == "tool"
    assert client.messages[2]["name"] == "echo"
    assert "A" in client.messages[2]["content"]
    assert client.messages[3] == {'role': 'user', 'content': 'And echo B please'}
    assert client.messages[4]["tool_calls"][0]["function"]["name"] == "echo"
    assert json.loads(client.messages[4]["tool_calls"][0]["function"]["arguments"]) == {"txt": "B"}
    assert client.messages[5]["role"] == "tool"
    assert client.messages[5]["name"] == "echo"
    assert "B" in client.messages[5]["content"]

    # The conversation must contain our extra user message in order.
    assert any(m for m in client.messages if m["role"] == "user" and "echo B" in m["content"])

    # The assistant must have produced *two* distinct tool calls (A and B).
    assistant_turns = [m for m in client.messages if m["role"] == "assistant" and m.get("tool_calls")]
    assert len(assistant_turns) == 2


@pytest.mark.asyncio
async def test_stop_cancels_gracefully(client):
    """
    Start the loop and *immediately* request a graceful stop.
    Verify that:
      • `handle.result()` raises `asyncio.CancelledError`,
      • no tool tasks are left running afterwards.
    """

    handle = start_async_tool_use_loop(
        client,
        "Echo something",
        {"echo": echo},
    )

    handle.stop()                      # request cancellation right away

    with pytest.raises(asyncio.CancelledError):
        await handle.result()

    # The underlying task should be done and cancelled.
    assert handle.done()


@pytest.mark.asyncio
async def test_multiple_interjects_then_normal_completion(client):
    """
    Fire *two* interjections while the loop is running; ensure each is
    delivered in order and the final result still resolves normally.
    """

    handle = start_async_tool_use_loop(
        client,
        "Echo A please",
        {"echo": echo},
    )

    await asyncio.sleep(0.01)
    await handle.interject("B please")
    await asyncio.sleep(0.01)
    await handle.interject("C please")

    await handle.result()

    # All three user utterances (initial + 2 extras) must be in the history.
    seen = [m["content"] for m in client.messages if m["role"] == "user"]
    assert seen == ["Echo A please", "B please", "C please"]

    # Assistant should have ended up calling the tool at least 3×.
    assistant_turns = [m for m in client.messages if m["role"] == "assistant" and m.get("tool_calls")]
    assert len(assistant_turns) == 2

    total_tool_calls = sum(len(t["tool_calls"]) for t in assistant_turns)
    assert total_tool_calls >= 3
