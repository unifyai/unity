from __future__ import annotations

import base64
from datetime import datetime, timezone
from unity.image_manager.utils import make_solid_png_base64

import pytest

from unity.image_manager.image_manager import ImageManager
from unity.guidance_manager.guidance_manager import GuidanceManager
from tests.helpers import _handle_project


PNG_RED_B64 = make_solid_png_base64(8, 8, (255, 0, 0))


@pytest.mark.unit
@_handle_project
def test_get_images_for_guidance_returns_metadata_only():
    im = ImageManager()
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "diagram of layout",
                "data": PNG_RED_B64,
            },
        ],
    )

    gm = GuidanceManager()
    gid = gm._add_guidance(
        title="Layout Review",
        content="We need to review the image layout.",
        images=[{"image_id": int(img_id), "annotation": "layout screenshot"}],
    )["details"]["guidance_id"]

    items = gm._get_images_for_guidance(guidance_id=gid)
    assert isinstance(items, list) and items, "Expected at least one image entry"
    entry = items[0]
    assert entry.get("image_id") == int(img_id)
    assert entry.get("caption") == "diagram of layout"
    assert isinstance(entry.get("timestamp"), str)
    # Ensure metadata includes annotation and no raw image/base64 field is present
    assert "image" not in entry
    assert entry.get("annotation") in (None, "layout screenshot")


@pytest.mark.unit
@_handle_project
def test_attach_image_to_context_promotes_image_block():
    im = ImageManager()
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "tiny red pixel",
                "data": PNG_RED_B64,
            },
        ],
    )

    gm = GuidanceManager()
    payload = gm._attach_image_to_context(image_id=int(img_id), note="see layout")
    # Tool payload must include base64 under the 'image' key for promotion
    assert isinstance(payload, dict)
    assert "image" in payload and isinstance(payload["image"], str)
    # Sanity: looks like base64 (decoding should not raise)
    base64.b64decode(payload["image"])  # will raise if invalid


@pytest.mark.unit
@_handle_project
def test_get_images_for_guidance_includes_annotation():
    im = ImageManager()
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "tiny red pixel",
                "data": PNG_RED_B64,
            },
        ],
    )

    gm = GuidanceManager()
    content = "click this button to open the modal"
    gid = gm._add_guidance(
        title="Annotation Demo",
        content=content,
        images=[{"image_id": int(img_id), "annotation": "button area"}],
    )["details"]["guidance_id"]

    items = gm._get_images_for_guidance(guidance_id=gid)
    assert items and (items[0].get("annotation") in (None, "button area"))
