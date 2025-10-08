import asyncio
import copy
from typing import List

import pytest
import unify

from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project, SETTINGS


# --------------------------------------------------------------------------- #
#  Test: preprocess_msgs dynamically patches {context} placeholder            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_preprocess_msgs_dynamic_placeholder(monkeypatch):
    """Verify that the preprocess hook patches placeholders *per-LLM-call* and
    that the modifications never leak into the persistent chat history."""

    client = unify.AsyncUnify(
        "o4-mini@openai",
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )

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

    # Spy the preprocess application at the exact callsite while still hitting the real LLM.
    from unity.common._async_tool import loop as _loop

    seen_contexts: List[str] = []

    orig_gwp = _loop.generate_with_preprocess

    async def _spy_gwp(_client, _preprocess, **gen_kwargs):
        def _spy_preprocess(msgs: List[dict]) -> List[dict]:
            out = _preprocess(msgs)
            # Extract any context_ tokens for assertions
            for m in out:
                try:
                    if isinstance(m.get("content"), str) and "context_" in m["content"]:
                        parts = [p for p in m["content"].split() if "context_" in p]
                        for p in parts:
                            tok = p.replace(".", "")
                            if tok not in seen_contexts:
                                seen_contexts.append(tok)
                except Exception:
                    pass
            return out

        return await orig_gwp(_client, _spy_preprocess, **gen_kwargs)

    monkeypatch.setattr(_loop, "generate_with_preprocess", _spy_gwp, raising=True)

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
    )

    final = await handle.result()

    # ------------------------------------------------------------------ #
    # Assertions                                                         #
    # ------------------------------------------------------------------ #

    # 1️⃣  The placeholder was substituted differently on successive calls.
    assert seen_contexts[:2] == [
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
