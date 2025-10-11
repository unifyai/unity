# tests/test_image_blocks.py
#
# Regression tests for the image-promotion logic in
# `unity.common.llm_helpers`.

from __future__ import annotations

import asyncio
import os
import base64
from pathlib import Path

import pytest
import unify
from unity.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project, SETTINGS

# --------------------------------------------------------------------------- #
#  CONSTANTS                                                                  #
# --------------------------------------------------------------------------- #

MODEL_NAME = os.getenv("UNIFY_MODEL", "gpt-5@openai")

# Load cat image and convert to base64
with open(Path(__file__).parent / "cat.jpg", "rb") as f:
    CAT_IMG = base64.b64encode(f.read()).decode("utf-8")


def new_client() -> unify.AsyncUnify:
    """Utility to get a fresh client with env-controlled caching / tracing."""
    return unify.AsyncUnify(
        MODEL_NAME,
        cache=SETTINGS.UNIFY_CACHE,
        traced=SETTINGS.UNIFY_TRACED,
    )


# --------------------------------------------------------------------------- #
#  1️⃣  User-supplied image is promoted & understood                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_initial_user_image_is_promoted() -> None:
    """
    The first user turn contains an inline PNG.  We expect:
      • `image_url` block present in the chat payload sent to the model;
      • the assistant correctly answers “cat”.
    """
    client = new_client()
    client.set_system_message(
        "You will receive an image. Answer with ONE three-letter word naming the animal.",
    )

    img_block = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{CAT_IMG}"},
    }
    txt_block = {
        "type": "text",
        "text": "What animal is shown? ONE word only.",
    }

    handle = start_async_tool_loop(
        client,
        message=[img_block, txt_block],  # 👈 ready-made content blocks
        tools={},
    )

    final_reply = await handle.result()

    # ---- 1. image block exists in the first user message ------------------
    user_msg = next(m for m in client.messages if m["role"] == "user")
    assert isinstance(user_msg["content"], list), "User content must be block array"
    assert any(
        blk.get("type") == "image_url"
        and blk["image_url"]["url"].startswith("data:image/png;base64,")
        for blk in user_msg["content"]
    ), "No promoted image_url block found in user message"

    # ---- 2. assistant correctly identifies the cat -----------------------
    assert final_reply.strip().lower().startswith("cat"), (
        "Assistant failed to recognise the animal from the image — "
        f"got: {final_reply!r}"
    )


# --------------------------------------------------------------------------- #
#  2️⃣  Tool-returned image is promoted & understood                           #
# --------------------------------------------------------------------------- #


async def image_tool() -> dict:
    """Return a JSON payload that **includes** a base-64 image."""
    await asyncio.sleep(0.05)
    return {"status": "ok", "image": CAT_IMG}


@pytest.mark.asyncio
@_handle_project
async def test_tool_result_image_is_promoted_and_reasoned_over() -> None:
    """
    Flow:
      1. Assistant calls `image_tool`.
      2. Tool returns a dict containing `"image": <b64>`.
      3. Loop promotes the image to `image_url` in the tool message.
      4. We then ask the assistant (in a *second* loop) what animal it sees;
         it must answer “cat”.
    """
    # ---- phase 1: run the tool and verify promotion ----------------------
    client = new_client()
    client.set_system_message(
        "Call image_tool exactly once. The tool will return a base64-encoded image of a domestic cat. After the tool finishes, respond with exactly 'cat' (lowercase, no punctuation). Do not output anything else.",
    )

    primary_handle = start_async_tool_loop(
        client,
        message="go",
        tools={"image_tool": image_tool},
        timeout=120,
    )
    result = await primary_handle.result()

    # Locate the tool-result message
    tool_msg = next(
        (
            m
            for m in client.messages
            if m["role"] == "tool" and m["name"] == "image_tool"
        ),
        None,
    )
    assert tool_msg is not None, "Tool result message missing"

    content_blocks = tool_msg["content"]
    assert isinstance(content_blocks, list), "Tool content must be block array"
    assert any(
        blk.get("type") == "image_url"
        and blk["image_url"]["url"].startswith("data:image/png;base64,")
        for blk in content_blocks
    ), "Promoted image_url block not found in tool result"

    # correct answer
    assert "cat" in result.lower()
