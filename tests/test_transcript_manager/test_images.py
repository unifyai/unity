from __future__ import annotations

import pytest
from datetime import datetime, UTC
import unify

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from unity.image_manager.image_manager import ImageManager
from tests.helpers import _handle_project


# 1x1 PNG (opaque) – tiny valid image payload (blue)
PNG_1x1_BLUE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMB9j3v1S0AAAAASUVORK5CYII="


@pytest.mark.unit
@_handle_project
def test_images_schema_and_roundtrip():
    tm = TranscriptManager()

    # Valid images mapping: supports negative and open-ended bounds
    images = {
        "[6:]": 101,
        "[-10:-5]": 202,
        "[2:]": 303,
        "[4:-4]": 404,
        "[:10]": 505,
        "[-5:]": 606,
    }

    msg = Message(
        medium="email",
        sender_id=0,
        receiver_ids=[1],
        timestamp=datetime.now(UTC),
        content="click this button to open the modal",
        exchange_id=880011,
        images=images,
    )

    tm.log_messages(msg)
    tm.join_published()

    # 1) Column exists in Transcripts context
    fields = unify.get_fields(context=tm._transcripts_ctx)
    assert "images" in fields, "images column should exist in Transcripts"
    # Optional: best-effort type check when backend exposes data_type
    dtype = fields.get("images", {}).get("data_type")
    if dtype is not None:
        assert dtype == "dict"

    # 2) Round-trip retrieval preserves mapping
    stored = tm._filter_messages(filter=f"exchange_id == {msg.exchange_id}")
    assert len(stored) == 1
    assert stored[0].images == images


@pytest.mark.unit
@_handle_project
def test_images_validation_rejects_bad_keys():
    tm = TranscriptManager()

    # Invalid key formats (no colon, triple slice, non-numeric bounds)
    bad_maps = [
        {"[0]": 1},
        {"[0:2:10]": 1},
        {"[a:b]": 1},
    ]

    for bad in bad_maps:
        with pytest.raises(ValueError):
            Message(
                medium="email",
                sender_id=0,
                receiver_ids=[1],
                timestamp=datetime.now(UTC),
                content="bad images mapping",
                exchange_id=777001,
                images=bad,
            )


@pytest.mark.unit
@_handle_project
def test_images_value_coercion_to_int():
    """Values should be storable as ints; strings convertible to int are coerced."""
    m = Message(
        medium="sms_message",
        sender_id=1,
        receiver_ids=[2],
        timestamp=datetime.now(UTC),
        content="coercion test",
        exchange_id=99001,
        images={"[0:10]": "101"},
    )
    assert isinstance(m.images["[0:10]"], int) and m.images["[0:10]"] == 101


@pytest.mark.unit
@_handle_project
def test_get_images_for_message_includes_substring():
    tm = TranscriptManager()
    im = ImageManager()

    # Seed a small valid image
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(UTC),
                "caption": "blue pixel",
                "data": PNG_1x1_BLUE,
            },
        ],
    )

    content = "click this button to open the modal"
    #            012345 678901234567890123456789012345
    # pick a span that extracts "this button"
    images_map = {"[6:18]": int(img_id)}

    msg = Message(
        medium="email",
        sender_id=0,
        receiver_ids=[1],
        timestamp=datetime.now(UTC),
        content=content,
        exchange_id=13579,
        images=images_map,
    )

    tm.log_messages(msg)
    tm.join_published()

    stored = tm._filter_messages(filter=f"exchange_id == {msg.exchange_id}")
    mid = stored[0].message_id

    items = tm._get_images_for_message(message_id=int(mid))
    assert items and isinstance(items[0].get("substring"), str)
    assert items[0]["substring"].strip() == "this button"
