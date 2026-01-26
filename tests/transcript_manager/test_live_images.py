from __future__ import annotations

import pytest
import base64
import os

from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.image_manager.image_manager import ImageManager
from tests.helpers import _handle_project
from unity.image_manager.types import AnnotatedImageRefs, RawImageRef, AnnotatedImageRef
from unity.contact_manager.types.contact import Contact


def _load_contact_card_png_b64() -> str:
    here = os.path.dirname(__file__)
    img_path = os.path.abspath(
        os.path.join(here, "..", "test_contact_manager", "details.png"),
    )
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("ascii")


@pytest.mark.asyncio
@_handle_project
async def test_lookup_via_image(static_now) -> None:
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
                "timestamp": static_now,
                "content": "Noise: unrelated update",
                "exchange_id": 3001,
            },
            {
                "medium": "email",
                "sender_id": david,
                "receiver_ids": [alice],
                "timestamp": static_now,
                "content": f"Please schedule the {unique_phrase}.",
                "exchange_id": 3002,
            },
            {
                "medium": "sms_message",
                "sender_id": alice,
                "receiver_ids": [david],
                "timestamp": static_now,
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
                "timestamp": static_now,
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

    # Expect image-aware behaviour via ask_image or attach_image_raw
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

    # Verify a transcripts lookup occurred (search_messages or filter_messages)
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

    # Unique David message should surface in the answer or tool outputs
    low_answer = (answer or "").lower()
    found_phrase = unique_phrase.lower() in low_answer
    if not found_phrase:
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
