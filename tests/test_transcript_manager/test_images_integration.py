from __future__ import annotations

import base64
from datetime import datetime, timezone, timedelta

import pytest

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from unity.image_manager.image_manager import ImageManager
from tests.helpers import _handle_project


# 1x1 PNG (opaque) – small valid image payload (blue)
PNG_1x1_BLUE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMB9j3v1S0AAAAASUVORK5CYII="


@pytest.mark.unit
@_handle_project
def test_get_images_for_message_returns_metadata_only_tm():
    tm = TranscriptManager()
    im = ImageManager()

    # Seed a single blue image
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "blue screen capture",
                "data": PNG_1x1_BLUE,
            },
        ],
    )

    # Log a message that references the image via images mapping
    msg = Message(
        medium="whatsapp_call",
        sender_id=101,
        receiver_ids=[202],
        timestamp=datetime.now(timezone.utc),
        content="Video conference: screen looks one colour",
        exchange_id=424242,
        images={"[0:1]": int(img_id)},
    )
    tm.log_messages(msg)
    tm.join_published()

    # Fetch message_id back then query image metadata via private tool
    stored = tm._filter_messages(filter=f"exchange_id == {msg.exchange_id}")
    assert stored and len(stored) == 1
    mid = int(stored[0].message_id)

    items = tm._get_images_for_message(message_id=mid)
    assert isinstance(items, list) and items, "Expected at least one image entry"
    entry = items[0]
    assert entry.get("image_id") == int(img_id)
    assert entry.get("caption") == "blue screen capture"
    assert isinstance(entry.get("timestamp"), str)
    # Ensure no raw image/base64 field is present
    assert "image" not in entry


@pytest.mark.unit
@_handle_project
def test_attach_image_to_context_promotes_image_block_tm():
    tm = TranscriptManager()
    im = ImageManager()

    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "tiny blue pixel",
                "data": PNG_1x1_BLUE,
            },
        ],
    )

    payload = tm._attach_image_to_context(image_id=int(img_id), note="see screen")
    assert isinstance(payload, dict)
    assert "image" in payload and isinstance(payload["image"], str)
    # Sanity: looks like base64 (decoding should not raise)
    base64.b64decode(payload["image"])  # will raise if invalid


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.requires_real_unify
@_handle_project
async def test_ask_can_use_images_for_color_question_tm():
    """
    Contrived scenario:
    - Create an image (blue pixel) and log a message that references it, framed as a
      video-conference issue ("my screen is one colour").
    - Ask TranscriptManager.ask a question that naturally invites image reasoning.
    - Expect a non-empty textual answer mentioning a blue-ish colour.
    - Also verify the tool trajectory shows either an image attachment or an image
      question tool being used (or an image block in messages).
    """
    tm = TranscriptManager()
    im = ImageManager()

    # Seed image
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc) - timedelta(days=7),
                "caption": "zoe video-conference blue screen",
                "data": PNG_1x1_BLUE,
            },
        ],
    )

    # Log message linked to the image – note: keep content suggestive of a VC context
    tm.log_messages(
        Message(
            medium="whatsapp_call",
            sender_id=301,  # assume 'Zoe' in external context; ID is fine for this test
            receiver_ids=[302],
            timestamp=datetime.now(timezone.utc) - timedelta(days=7),
            content=(
                "Zoe on video conference: my screen is one colour, what is happening?"
            ),
            exchange_id=777001,
            images={"[0:1]": int(img_id)},
        ),
    )
    tm.join_published()

    # Ask the higher-level question; request reasoning steps for inspection
    h = await tm.ask(
        "What colour did Zoe's screen turn on the video conference last week?",
        _return_reasoning_steps=True,
    )
    answer, steps = await h.result()

    assert isinstance(answer, str) and answer.strip(), "Expected a textual description"
    # Heuristic: The tiny asset is blue; allow synonyms or general color mention
    assert any(
        kw in answer.lower() for kw in ("blue", "azure", "navy", "cyan")
    ), f"Answer does not reference blue-ish color: {answer!r}"

    # Validate that the trajectory reflects image usage in at least one of these ways:
    # - an image block promoted in messages (image_url / data:image)
    # - or explicit tool selection of ask_image/attach_image_to_context/attach_message_images_to_context
    serialized = str(steps)
    assert (
        ("image_url" in serialized)
        or ("data:image" in serialized)
        or ("ask_image" in serialized)
        or ("attach_image_to_context" in serialized)
        or ("attach_message_images_to_context" in serialized)
    ), "Expected image-aware reasoning (image block or image tools) to appear in steps"

    # The textual answer itself should not include raw image data
    assert "data:image" not in answer and "image_url" not in answer
