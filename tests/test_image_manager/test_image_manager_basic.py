from __future__ import annotations

import base64
from datetime import datetime, timezone
from unity.image_manager.utils import make_solid_png_base64

import pytest

from unity.image_manager.image_manager import ImageManager
from tests.helpers import _handle_project


PNG_RED_B64 = make_solid_png_base64(8, 8, (255, 0, 0))
PNG_BLUE_B64 = make_solid_png_base64(8, 8, (0, 0, 255))


@pytest.mark.unit
@_handle_project
def test_add_and_filter_images():
    im = ImageManager()

    ids = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "A small red square",
                "data": PNG_RED_B64,
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "A tiny blue pixel",
                "data": PNG_BLUE_B64,
            },
        ],
    )
    assert all(isinstance(i, int) for i in ids)

    # Filter by id
    row = im.filter_images(filter=f"image_id == {ids[0]}")
    assert row and row[0].caption == "A small red square"

    # Substring filter on caption
    reds = im.filter_images(filter="caption is not None and 'red' in caption.lower()")
    assert any("red" in (r.caption or "").lower() for r in reds)


@pytest.mark.unit
@_handle_project
def test_update_images():
    im = ImageManager()

    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "original",
                "data": PNG_RED_B64,
            },
        ],
    )

    updated_ids = im.update_images(
        [
            {
                "image_id": img_id,
                "caption": "updated caption",
            },
        ],
    )
    assert img_id in updated_ids

    row = im.filter_images(filter=f"image_id == {img_id}")
    assert row and row[0].caption == "updated caption"


@pytest.mark.unit
@_handle_project
def test_get_images_order_and_raw():
    im = ImageManager()

    ids = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "first",
                "data": PNG_RED_B64,
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "second",
                "data": PNG_BLUE_B64,
            },
        ],
    )
    # Request handles in reversed order; verify order preserved
    handles = im.get_images([ids[1], ids[0]])
    assert [h.image_id for h in handles] == [ids[1], ids[0]]

    # Verify raw bytes round-trip for both
    raw0 = handles[0].raw()
    raw1 = handles[1].raw()
    assert raw0 == base64.b64decode(PNG_BLUE_B64)
    assert raw1 == base64.b64decode(PNG_RED_B64)
