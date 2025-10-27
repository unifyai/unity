from __future__ import annotations

import pytest
from datetime import datetime, UTC
import base64
import os
import unify

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.transcript_manager.types.message import Message
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.utils import make_solid_png_base64
from tests.helpers import _handle_project
from unity.image_manager.types import AnnotatedImageRefs, RawImageRef, AnnotatedImageRef
from unity.contact_manager.types.contact import Contact


PNG_1x1_BLUE = make_solid_png_base64(8, 8, (0, 0, 255))


@pytest.mark.unit
@_handle_project
def test_images_schema_and_roundtrip():
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


@pytest.mark.unit
@_handle_project
def test_images_accepts_annotated_refs_only():
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


@pytest.mark.unit
@_handle_project
def test_images_roundtrip_annotated_only():
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
    image_refs = AnnotatedImageRefs.model_validate(
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

    stored = tm._filter_messages(filter=f"exchange_id == {msg.exchange_id}")["messages"]
    mid = stored[0].message_id

    items = tm._get_images_for_message(message_id=int(mid))
    assert items and isinstance(items[0].get("annotation"), (str, type(None)))
    assert items[0]["annotation"].strip() == "this button"


@pytest.mark.unit
@_handle_project
def test_transcripts_images_field_schema_is_nested_and_enforced_tm():
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


def _load_contact_card_png_b64() -> str:
    here = os.path.dirname(__file__)
    img_path = os.path.abspath(
        os.path.join(here, "..", "test_contact", "contact_details.png"),
    )
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("ascii")


@pytest.mark.asyncio
@_handle_project
async def test_lookup_transcripts_via_image() -> None:
    tm = TranscriptManager()

    # Seed transcripts with noise and one message from David Smith
    david = Contact(
        first_name="David",
        surname="Smith",
        email_address="david.smith@gmail.com",
        contact_id=-1,
    )
    alice = Contact(first_name="Alice", surname="Jones", contact_id=-1)

    unique_phrase = "Project X launch meeting"

    tm.log_messages(
        [
            {
                "medium": "email",
                "sender_id": alice,
                "receiver_ids": [david],
                "timestamp": datetime.now(UTC),
                "content": "Noise: unrelated update",
                "exchange_id": 3001,
            },
            {
                "medium": "email",
                "sender_id": david,
                "receiver_ids": [alice],
                "timestamp": datetime.now(UTC),
                "content": f"Please schedule the {unique_phrase}.",
                "exchange_id": 3002,
            },
            {
                "medium": "sms_message",
                "sender_id": alice,
                "receiver_ids": [david],
                "timestamp": datetime.now(UTC),
                "content": "Noise: FYI only",
                "exchange_id": 3003,
            },
        ],
    )
    tm.join_published()

    # Persist the contact card image and build typed ImageRefs
    manager = ImageManager()
    b64 = _load_contact_card_png_b64()
    [img_id] = manager.add_images(
        [
            {
                "caption": "contact card",
                "data": b64,
            },
        ],
    )

    images = AnnotatedImageRefs.model_validate(
        [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=int(img_id)),
                annotation="contact card",
            ),
        ],
    )

    # Ask TranscriptManager to find messages from the person in the image
    handle = await tm.ask(
        "Do we have messages from this person? Show the latest.",
        images=images,
        _return_reasoning_steps=True,
    )
    answer, messages = await handle.result()

    # 1) Expect image-aware behaviour via ask_image or attach_image_raw
    image_calls = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name")
            if fn in ("ask_image", "attach_image_raw"):
                image_calls.append(tc)
    assert (
        image_calls
    ), "Expected ask_image or attach_image_raw to be used with live images"

    # 2) Verify a transcripts lookup occurred (search_messages or filter_messages)
    lookup_calls = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name")
            if fn in ("search_messages", "filter_messages"):
                lookup_calls.append(tc)
    assert (
        lookup_calls
    ), "Expected a transcripts lookup (search_messages or filter_messages)"

    # 3) The loop should have surfaced the David message somewhere (answer or tool output)
    low_answer = (answer or "").lower()
    found_phrase = unique_phrase.lower() in low_answer
    if not found_phrase:
        # Fallback: inspect tool results content for the phrase
        for m in messages:
            if m.get("role") == "tool" and m.get("name") in (
                "search_messages",
                "filter_messages",
            ):
                content = str(m.get("content") or "").lower()
                if unique_phrase.lower() in content:
                    found_phrase = True
                    break
    assert (
        found_phrase
    ), "Expected the unique David message to be among the retrieved results"
