from __future__ import annotations

import pytest
from datetime import datetime, UTC
import unify

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.utils import make_solid_png_base64
from tests.helpers import _handle_project
from unity.image_manager.types import ImageRefs, RawImageRef, AnnotatedImageRef


PNG_1x1_BLUE = make_solid_png_base64(8, 8, (0, 0, 255))


@pytest.mark.unit
@_handle_project
def test_images_schema_and_roundtrip():
    tm = TranscriptManager()

    refs = ImageRefs.model_validate(
        [
            RawImageRef(image_id=101),
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=202),
                annotation="Screenshot of the modal open state",
            ),
        ],
    )

    msg = Message(
        medium="email",
        sender_id=0,
        receiver_ids=[1],
        timestamp=datetime.now(UTC),
        content="click this button to open the modal",
        exchange_id=880011,
        images=refs,
    )

    tm.log_messages(msg)
    tm.join_published()

    # 1) Column exists in Transcripts context
    fields = unify.get_fields(context=tm._transcripts_ctx)
    assert "images" in fields, "images column should exist in Transcripts"

    # 2) Round-trip retrieval preserves references
    stored = tm._filter_messages(filter=f"exchange_id == {msg.exchange_id}")
    assert len(stored) == 1
    got = stored[0].images
    assert isinstance(got, ImageRefs)
    # Compare by image_ids and presence of annotations
    got_items = getattr(got, "root", [])
    assert len(got_items) == 2
    got_ids = [
        (it.image_id if hasattr(it, "image_id") else it.raw_image_ref.image_id)
        for it in got_items
    ]
    assert got_ids == [101, 202]
    ann = getattr(got_items[1], "annotation", None)
    assert isinstance(ann, str) and "modal" in ann.lower()


@pytest.mark.unit
@_handle_project
def test_images_accepts_annotated_and_raw_refs():
    # Construct ImageRefs with mixed raw and annotated entries
    refs = ImageRefs.model_validate(
        [
            RawImageRef(image_id=1),
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=2),
                annotation="Relevant to the settings section",
            ),
        ],
    )
    assert isinstance(refs, ImageRefs)
    root = getattr(refs, "root", [])
    assert len(root) == 2 and hasattr(root[1], "annotation")


@pytest.mark.unit
@_handle_project
def test_images_roundtrip_raw_only():
    m = Message(
        medium="sms_message",
        sender_id=1,
        receiver_ids=[2],
        timestamp=datetime.now(UTC),
        content="coercion test",
        exchange_id=99001,
        images=ImageRefs.model_validate([RawImageRef(image_id=101)]),
    )
    assert isinstance(m.images, ImageRefs)


@pytest.mark.unit
@_handle_project
def test_get_images_for_message_includes_annotation():
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
    # Attach one annotated image reference
    image_refs = ImageRefs.model_validate(
        [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=int(img_id)),
                annotation="this button",
            ),
        ],
    )

    msg = Message(
        medium="email",
        sender_id=0,
        receiver_ids=[1],
        timestamp=datetime.now(UTC),
        content=content,
        exchange_id=13579,
        images=image_refs,
    )

    tm.log_messages(msg)
    tm.join_published()

    stored = tm._filter_messages(filter=f"exchange_id == {msg.exchange_id}")
    mid = stored[0].message_id

    items = tm._get_images_for_message(message_id=int(mid))
    assert items and isinstance(items[0].get("annotation"), (str, type(None)))
    assert items[0]["annotation"].strip() == "this button"
