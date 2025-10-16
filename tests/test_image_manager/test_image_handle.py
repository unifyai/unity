from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path

import pytest

from unity.image_manager.image_manager import ImageManager
from tests.helpers import _handle_project


ASSET_LOCAL = Path("tests/test_image_manager/assets/google.jpeg")


@pytest.mark.eval
@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_image_handle_ask_returns_text_only():
    im = ImageManager()
    raw = ASSET_LOCAL.read_bytes()
    img_b64 = base64.b64encode(raw).decode("utf-8")
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
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
