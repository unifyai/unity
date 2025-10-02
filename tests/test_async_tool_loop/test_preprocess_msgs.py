import asyncio
import copy
from typing import List

import pytest
import unify

from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project


class SpyAsyncUnify:
    """A minimal AsyncUnify stub that records the *visible* message list each
    time ``generate`` is invoked so that we can assert that the
    ``preprocess_msgs`` callback patched placeholders *before* the LLM saw
    them.
    """

    def __init__(self):
        # The conversation state the tool-loop mutates.
        self.messages: List[dict] = []
        # Snapshots of what *generate* saw on every call.
        self.seen_messages: List[List[dict]] = []
        self._step = 0  # 0 → ask for tool, 1 → final answer

    # ------------------------------------------------------------------ #
    # Minimal surface expected by _async_tool_use_loop_inner             #
    # ------------------------------------------------------------------ #
    def append_messages(self, msgs):
        self.messages.extend(msgs)

    async def generate(self, **_):  # noqa: D401 – minimal stub
        # Record a *deep* copy of the messages visible to the model.
        self.seen_messages.append(copy.deepcopy(self.messages))

        # Step-wise canned replies to drive the outer loop.
        if self._step == 0:
            self._step += 1
            assistant_msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "dummy_tool",
                            "arguments": "{}",
                        },
                    },
                ],
            }
        else:
            self._step += 1
            assistant_msg = {
                "role": "assistant",
                "content": "all done",
                "tool_calls": [],
            }

        self.messages.append(assistant_msg)
        return assistant_msg

    # Property used only for logging in our loop, safe to stub.
    @property
    def system_message(self) -> str:  # noqa: D401
        return ""

    # Convenience helpers for tests ----------------------------------- #
    def seen_contexts(self) -> List[str]:
        """Return the *unique* substituted context tokens the stub saw."""
        out: List[str] = []
        for snap in self.seen_messages:
            for m in snap:
                if isinstance(m.get("content"), str) and "context_" in m["content"]:
                    # extract the token – e.g. "context_0" / "context_1"
                    parts = [p for p in m["content"].split() if "context_" in p]
                    out.extend(parts)
        # keep order of appearance but drop duplicates
        seen: List[str] = []
        for p in out:
            if p not in seen:
                seen.append(p.replace(".", ""))
        return seen


# --------------------------------------------------------------------------- #
#  Test: preprocess_msgs dynamically patches {context} placeholder            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_preprocess_msgs_dynamic_placeholder():
    """Verify that the preprocess hook patches placeholders *per-LLM-call* and
    that the modifications never leak into the persistent chat history."""

    client = SpyAsyncUnify()

    # Counter so each invocation produces a fresh replacement value.
    counter = {"n": 0}

    def preprocess(msgs: List[dict]) -> List[dict]:
        """Replace every occurrence of ``{context}`` with a unique token."""
        new_msgs = copy.deepcopy(msgs)  # operate on our own copy
        replacement = f"context_{counter['n']}"
        counter["n"] += 1
        for m in new_msgs:
            if isinstance(m.get("content"), str):
                m["content"] = m["content"].replace("{context}", replacement)
        return new_msgs

    # ------------------------------------------------------------------ #
    # Dummy tool – returns almost instantly so the loop needs two LLM     #
    # turns (request tool → result → final answer).                       #
    # ------------------------------------------------------------------ #

    @unify.traced  # no-op decorator from real library
    async def dummy_tool():  # noqa: D401
        await asyncio.sleep(0.01)
        return "OK"

    # Kick off the async-tool loop.
    handle = start_async_tool_loop(
        client=client,  # type: ignore[arg-type]
        message="Please run dummy_tool then respond with {context}.",
        tools={"dummy_tool": dummy_tool},
        preprocess_msgs=preprocess,
        log_steps=False,
    )

    final = await handle.result()

    # ------------------------------------------------------------------ #
    # Assertions                                                         #
    # ------------------------------------------------------------------ #

    # 1️⃣  The placeholder was substituted differently on successive calls.
    assert client.seen_contexts()[:2] == [
        "context_0",
        "context_1",
    ], "Preprocessed contexts not observed in order or missing."

    # 2️⃣  The *stored* chat history must still contain the literal placeholder.
    assert any(
        m.get("role") == "user" and "{context}" in m.get("content", "")
        for m in client.messages
    ), "Original placeholder should remain in persistent transcript."

    # 3️⃣  Loop completes and returns assistant reply.
    assert (
        "all done" in final.lower() or final.strip()
    ), "Loop did not finish correctly."
