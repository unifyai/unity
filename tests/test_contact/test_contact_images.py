from __future__ import annotations

import base64
import json
import os
import pytest

from unity.image_manager.image_manager import ImageManager, ImageHandle
from unity.image_manager.types.image import Image
from tests.helpers import _handle_project


def _load_contact_card_png_b64() -> str:
    here = os.path.dirname(__file__)
    img_path = os.path.join(here, "contact_details.png")
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("ascii")


@pytest.mark.asyncio
@_handle_project
async def test_lookup_contact_via_image(contact_manager_scenario) -> None:
    cm, _ = contact_manager_scenario

    # Build a real ImageHandle from the provided PNG asset
    manager = ImageManager()
    b64 = _load_contact_card_png_b64()
    ih = ImageHandle(
        manager=manager,
        image=Image(image_id=501, caption="contact card", data=b64),
    )

    # Seed the live-images mapping onto the initial user message
    images = {"[0:5]": ih}

    # Ask: instruct the model to read fields from the image and then check via filter
    user_msg = "Is this person in the contact list already?"

    handle = await cm.ask(
        user_msg,
        images=images,
        _return_reasoning_steps=True,
    )

    _final_answer, messages = await handle.result()

    # 1) The image text should be read correctly via ask_image tool results
    ask_msgs = [
        m for m in messages if m.get("role") == "tool" and m.get("name") == "ask_image"
    ]
    assert ask_msgs, "Expected the model to use ask_image to read the card"
    all_ask_text = "\n".join(str(m.get("content") or "") for m in ask_msgs)
    assert "David" in all_ask_text, "First name 'David' not read from image"
    assert "Smith" in all_ask_text, "Surname 'Smith' not read from image"
    assert (
        "david.smith@gmail.com" in all_ask_text
    ), "Email not read from image via ask_image"

    # 2) Verify a filter_contacts tool call was made with an equality using one extracted value
    # Find assistant tool-calls and inspect arguments for filter expressions
    filter_calls = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}).get("name")
            if fn != "filter_contacts":
                continue
            try:
                args = json.loads((tc.get("function") or {}).get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            filter_calls.append(args)

    assert filter_calls, "Expected at least one filter_contacts tool call"

    any_match = False
    for call in filter_calls:
        filt = str(call.get("filter") or "")
        lower = filt.lower()
        if any(term in lower for term in ["david", "smith", "david.smith@gmail.com"]):
            any_match = True
            break

    assert any_match, (
        "filter_contacts was called, but none of the expected substrings ('david', 'smith', "
        "'david.smith@gmail.com') were present in the filter expression."
    )


@pytest.mark.asyncio
@_handle_project
async def test_update_contact_from_image(contact_manager_scenario) -> None:
    cm, _ = contact_manager_scenario

    # Build a real ImageHandle from the provided PNG asset
    manager = ImageManager()
    b64 = _load_contact_card_png_b64()
    ih = ImageHandle(
        manager=manager,
        image=Image(image_id=502, caption="contact card", data=b64),
    )

    # Seed the live-images mapping onto the initial user message
    images = {"[0:5]": ih}

    # Instruct the model to add the person from the image into contacts
    user_msg = "please add this person to the contact list"

    handle = await cm.update(user_msg, images=images)
    await handle.result()

    # Verify David Smith is now present with correct full name and email
    matches = cm._filter_contacts(
        filter="email_address == 'david.smith@gmail.com'",
        limit=1,
    )
    assert matches, "Expected contact to be created from image details"
    c = matches[0]
    assert (c.first_name or "").lower() == "david"
    assert (c.surname or "").lower() == "smith"
    assert (c.email_address or "").lower() == "david.smith@gmail.com"
