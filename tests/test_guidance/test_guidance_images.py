from __future__ import annotations

import base64
from datetime import datetime, timezone

import pytest

from unity.image_manager.image_manager import ImageManager
from unity.guidance_manager.guidance_manager import GuidanceManager
from tests.helpers import _handle_project


# Tiny valid 1x1 PNG (opaque)
PNG_1x1_RED = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAtMB9q5g3GkAAAAASUVORK5CYII="


@pytest.mark.unit
@_handle_project
def test_get_images_for_guidance_returns_metadata_only():
    im = ImageManager()
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "diagram of layout",
                "data": PNG_1x1_RED,
            },
        ],
    )

    gm = GuidanceManager()
    gid = gm._add_guidance(
        title="Layout Review",
        content="We need to review the image layout.",
        images={"[0:10]": int(img_id)},
    )["details"]["guidance_id"]

    items = gm._get_images_for_guidance(guidance_id=gid)
    assert isinstance(items, list) and items, "Expected at least one image entry"
    entry = items[0]
    assert entry.get("image_id") == int(img_id)
    assert entry.get("caption") == "diagram of layout"
    assert isinstance(entry.get("timestamp"), str)
    # Ensure no raw image/base64 field is present
    assert "image" not in entry


@pytest.mark.unit
@_handle_project
def test_attach_image_to_context_promotes_image_block():
    im = ImageManager()
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "tiny red pixel",
                "data": PNG_1x1_RED,
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
