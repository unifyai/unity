from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Tuple
import pytest

from unity.contact_manager.contact_manager import ContactManager
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.types.image_refs import ImageRefs
from unity.image_manager.types.raw_image_ref import RawImageRef
from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef
from tests.helpers import _handle_project


def _load_contact_card_png_b64() -> str:
    here = os.path.dirname(__file__)
    img_path = os.path.join(here, "details.png")
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("ascii")


def _find_tool_calls_and_results(
    messages: List[Dict[str, Any]],
    tool_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Find all tool calls with the given name and their corresponding results.

    Handles both sync (direct) and async (check_status_*) tool patterns:
    - Sync: tool result has tool_call_id == original call id
    - Async: tool result has tool_call_id == original call id + "_completed"

    Returns:
        Tuple of (tool_calls, tool_results) where tool_calls are the assistant's
        invocations and tool_results are the corresponding results.
    """
    # Find all tool_calls with the given name
    tool_calls = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if (tc.get("function") or {}).get("name") == tool_name:
                tool_calls.append(tc)

    # Find corresponding tool results by matching tool_call_id
    tool_results = []
    for tc in tool_calls:
        call_id = tc.get("id")
        if not call_id:
            continue
        # Look for tool result with matching id (sync) or id_completed (async)
        for m in messages:
            if m.get("role") != "tool":
                continue
            result_id = m.get("tool_call_id", "")
            if result_id == call_id or result_id == f"{call_id}_completed":
                tool_results.append(m)
                break

    return tool_calls, tool_results


@pytest.mark.asyncio
@_handle_project
async def test_lookup_via_image(static_now) -> None:
    cm = ContactManager()

    # Persist a real image row and build typed ImageRefs for the loop
    manager = ImageManager()
    b64 = _load_contact_card_png_b64()
    [ih] = manager.add_images(
        [
            {
                "caption": "contact card",
                "data": b64,
                "timestamp": static_now,
            },
        ],
        synchronous=True,
        return_handles=True,
    )
    assert ih is not None, "Failed to create test image handle"

    images = ImageRefs(
        [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=int(ih.image_id)),
                annotation="contact card",
            ),
        ],
    )

    # Ask: instruct the model to read fields from the image and then check via filter
    user_msg = "Is this person in the contact list already?"

    handle = await cm.ask(
        user_msg,
        images=images,
        _return_reasoning_steps=True,
    )

    _final_answer, messages = await handle.result()

    # 1) Verify the model used ask_image or attach_image_raw to process the image
    ask_image_calls, ask_image_results = _find_tool_calls_and_results(
        messages,
        "ask_image",
    )
    attach_image_calls, _ = _find_tool_calls_and_results(messages, "attach_image_raw")

    assert (
        ask_image_calls or attach_image_calls
    ), "Expected the model to use ask_image or attach_image_raw with live images"

    # 2) If ask_image was used, verify the result contains the expected parsed content
    if ask_image_results:
        all_ask_text = "\n".join(
            str(m.get("content") or "") for m in ask_image_results
        ).lower()
        assert any(
            term in all_ask_text for term in ["david", "smith", "david.smith@gmail.com"]
        ), "Expected ask_image output to include name or email parsed from the card"

    # 3) Verify the assistant attempted a contacts lookup
    filter_calls, _ = _find_tool_calls_and_results(messages, "filter_contacts")
    search_calls, _ = _find_tool_calls_and_results(messages, "search_contacts")
    assert (
        filter_calls or search_calls
    ), "Expected at least one contacts lookup (filter_contacts or search_contacts)"


@pytest.mark.asyncio
@_handle_project
async def test_update_from_image(static_now) -> None:
    cm = ContactManager()

    # Persist image and provide typed ImageRefs
    manager = ImageManager()
    b64 = _load_contact_card_png_b64()
    [ih] = manager.add_images(
        [
            {
                "caption": "contact card",
                "data": b64,
                "timestamp": static_now,
            },
        ],
        synchronous=True,
        return_handles=True,
    )
    assert ih is not None, "Failed to create test image handle"

    images = ImageRefs(
        [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=int(ih.image_id)),
                annotation="contact card",
            ),
        ],
    )

    # Instruct the model to add the person from the image into contacts
    user_msg = "please add this person to the contact list"

    handle = await cm.update(user_msg, images=images)
    await handle.result()

    # Verify David Smith is now present with correct full name and email
    matches = cm.filter_contacts(
        filter="email_address == 'david.smith@gmail.com'",
        limit=1,
    )["contacts"]
    assert matches, "Expected contact to be created from image details"
    c = matches[0]
    assert (c.first_name or "").lower() == "david"
    assert (c.surname or "").lower() == "smith"
    assert (c.email_address or "").lower() == "david.smith@gmail.com"
