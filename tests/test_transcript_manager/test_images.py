from __future__ import annotations

import pytest
from datetime import datetime, UTC
import unify

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from unity.image_manager.utils import make_solid_png_base64
from tests.helpers import _handle_project
from unity.image_manager.types import AnnotatedImageRefs, RawImageRef, AnnotatedImageRef


PNG_BLUE = make_solid_png_base64(32, 32, (0, 0, 255))


@_handle_project
def test_schema_roundtrip():
    tm = TranscriptManager()

    refs = AnnotatedImageRefs.model_validate(
        [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=101),
                annotation="first test image",
            ),
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
    stored = tm._filter_messages(filter=f"exchange_id == {msg.exchange_id}")["messages"]
    assert len(stored) == 1
    got = stored[0].images
    assert isinstance(got, AnnotatedImageRefs)
    # Compare by image_ids and presence of annotations
    got_items = getattr(got, "root", [])
    assert len(got_items) == 2
    got_ids = [it.raw_image_ref.image_id for it in got_items]
    assert got_ids == [101, 202]
    ann = getattr(got_items[1], "annotation", None)
    assert isinstance(ann, str) and "modal" in ann.lower()


@_handle_project
def test_accepts_annotated_refs_only():
    refs = AnnotatedImageRefs.model_validate(
        [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=1),
                annotation="Relevant to the settings section",
            ),
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=2),
                annotation="Another relevant screenshot",
            ),
        ],
    )
    assert isinstance(refs, AnnotatedImageRefs)
    root = getattr(refs, "root", [])
    assert len(root) == 2 and hasattr(root[1], "annotation")


@_handle_project
def test_roundtrip_annotated_only():
    m = Message(
        medium="sms_message",
        sender_id=1,
        receiver_ids=[2],
        timestamp=datetime.now(UTC),
        content="coercion test",
        exchange_id=99001,
        images=AnnotatedImageRefs.model_validate(
            [
                AnnotatedImageRef(
                    raw_image_ref=RawImageRef(image_id=101),
                    annotation="single reference",
                ),
            ],
        ),
    )
    assert isinstance(m.images, AnnotatedImageRefs)


@_handle_project
def test_images_field_schema_enforced():
    tm = TranscriptManager()

    # 1) The Transcripts context should expose a nested JSON schema for the images field
    fields = unify.get_fields(context=tm._transcripts_ctx)
    assert "images" in fields
    dtype = str(fields["images"].get("data_type"))
    # Expect array/list with object items including raw_image_ref + annotation and nested image_id
    assert "raw_image_ref" in dtype and "annotation" in dtype and "image_id" in dtype

    # 2) Valid nested payload – should succeed
    common = {
        "medium": "email",
        "sender_id": 1,
        "receiver_ids": [2],
        "timestamp": datetime.now(UTC).isoformat(),
        "content": "hello",
    }

    valid_payload = {
        **common,
        "images": [
            {"raw_image_ref": {"image_id": 101}, "annotation": "blue square"},
        ],
    }
    _ = unify.log(context=tm._transcripts_ctx, **valid_payload, new=True, mutable=True)

    # 3) Invalid nested payload – wrong key name for image id → must be rejected
    invalid_payload_bad_key = {
        **common,
        "images": [
            {"raw_image_ref": {"image_idx": 999}, "annotation": "oops"},
        ],
    }
    with pytest.raises(Exception):
        unify.log(
            context=tm._transcripts_ctx,
            **invalid_payload_bad_key,
            new=True,
            mutable=True,
        )

    # 4) Invalid nested payload – wrong type for annotation → must be rejected
    invalid_payload_bad_type = {
        **common,
        "images": [
            {"raw_image_ref": {"image_id": 202}, "annotation": 123},
        ],
    }
    with pytest.raises(Exception):
        unify.log(
            context=tm._transcripts_ctx,
            **invalid_payload_bad_type,
            new=True,
            mutable=True,
        )
