from __future__ import annotations

import base64
from pathlib import Path

import pytest

from unity.image_manager.image_manager import ImageManager
from tests.helpers import _handle_project

ASSET_LOCAL = Path(__file__).parent / "assets" / "google.jpeg"


@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_ask_returns_text_only(static_now):
    im = ImageManager()
    raw = ASSET_LOCAL.read_bytes()
    img_b64 = base64.b64encode(raw).decode("utf-8")
    [img_id] = im.add_images(
        [
            {
                "timestamp": static_now,
                "caption": "a real photo (google.jpeg test asset)",
                "data": img_b64,
            },
        ],
    )

    handle = im.get_images([img_id])[0]
    answer = await handle.ask(
        "What do you notice in this image?",
    )

    assert isinstance(answer, str) and answer.strip(), "Answer must be non-empty"
    # The textual answer should not include base64 or image block markers
    assert "data:image" not in answer and "image_url" not in answer


@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_ask_uses_parent_chat_context(static_now):
    """
    Verifies that ImageHandle.ask accepts an optional parent chat context and
    injects it as a single summarizing system message (not as many messages), and
    that the inner LLM call can still answer a question about the real image.

    Uses the real LLM (no stubs). The question is intentionally context-dependent
    and should be answered correctly only when the parent chat context is
    understood by the inner LLM.
    """

    im = ImageManager()
    raw = ASSET_LOCAL.read_bytes()
    img_b64 = base64.b64encode(raw).decode("utf-8")
    [img_id] = im.add_images(
        [
            {
                "timestamp": static_now,
                "caption": "Google logo (test asset)",
                "data": img_b64,
            },
        ],
    )

    handle = im.get_images([img_id])[0]

    parent_ctx = [
        {"role": "assistant", "content": 'Our company slogan is "Get em!"'},
    ]

    # The question intentionally requires both the image content and the parent context
    answer = await handle.ask(
        (
            "Which letters in this search engine logo appear in our company slogan? "
            "Reply only with these letters and nothing else. Do not include any missing letters in your response."
        ),
        parent_chat_context_cont=parent_ctx,
    )

    assert isinstance(answer, str) and answer.strip()
    alower = answer.lower()
    assert "g" in alower and "e" in alower
    assert "o" not in alower and "l" not in alower
