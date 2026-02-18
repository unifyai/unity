from __future__ import annotations

import base64
from datetime import datetime, timezone
from unity.image_manager.utils import make_solid_png_base64

import pytest

from unity.image_manager.image_manager import ImageManager
from tests.helpers import _handle_project

PNG_RED_B64 = make_solid_png_base64(32, 32, (255, 0, 0))
PNG_BLUE_B64 = make_solid_png_base64(32, 32, (0, 0, 255))


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
        synchronous=True,
    )
    assert all(isinstance(i, int) for i in ids)

    # Filter by id
    row = im.filter_images(filter=f"image_id == {ids[0]}")
    assert row and row[0].caption == "A small red square"

    # Substring filter on caption
    reds = im.filter_images(filter="caption is not None and 'red' in caption.lower()")
    assert any("red" in (r.caption or "").lower() for r in reds)


@_handle_project
def test_add_images_return_handles_mode():
    im = ImageManager()

    handles = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "handle red",
                "data": PNG_RED_B64,
                "annotation": "ann-red",
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "handle blue",
                "data": PNG_BLUE_B64,
                "annotation": "ann-blue",
            },
        ],
        return_handles=True,
        synchronous=True,
    )

    # Should return two ImageHandle instances
    assert len(handles) == 2
    assert all(h is None or hasattr(h, "raw") for h in handles)

    # Validate annotations are set on returned handles and not persisted
    anns = [h.annotation for h in handles if h is not None]
    assert set(anns) == {"ann-red", "ann-blue"}

    # Validate raw() round-trip at least for the first non-None handle
    for h in handles:
        if h is not None:
            b = h.raw()
            assert isinstance(b, (bytes, bytearray)) and len(b) > 0
            break


@_handle_project
@pytest.mark.asyncio
async def test_add_images_async_mode_returns_handles_and_schedules_uploads():
    im = ImageManager()

    handles = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "async red",
                "data": PNG_RED_B64,
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "async blue",
                "data": PNG_BLUE_B64,
            },
        ],
        synchronous=False,
        return_handles=True,
    )

    assert len(handles) == 2
    assert all(h is None or h.is_pending for h in handles)

    # Await resolution of all pending handles to ensure uploads were scheduled

    pids = [h.image_id for h in handles if h is not None]
    mapping = await im.await_pending(pids)
    assert set(mapping.keys()) == set(pids)


@_handle_project
def test_add_images_async_invalid_combo_raises():
    im = ImageManager()
    with pytest.raises(ValueError):
        _ = im.add_images(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "caption": "invalid",
                    "data": PNG_RED_B64,
                },
            ],
            synchronous=False,
            return_handles=False,
        )


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


@_handle_project
def test_add_images_with_filepath():
    im = ImageManager()

    ids = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "red with path",
                "data": PNG_RED_B64,
                "filepath": "/tmp/images/red.png",
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "blue no path",
                "data": PNG_BLUE_B64,
            },
        ],
        synchronous=True,
    )
    assert len(ids) == 2

    # Verify filepath round-trips through filter_images
    rows = im.filter_images(filter=f"image_id == {ids[0]}")
    assert rows and rows[0].filepath == "/tmp/images/red.png"

    rows_no_fp = im.filter_images(filter=f"image_id == {ids[1]}")
    assert rows_no_fp and rows_no_fp[0].filepath is None


@_handle_project
def test_filepath_uniqueness():
    im = ImageManager()

    [id1] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "first",
                "data": PNG_RED_B64,
                "filepath": "/tmp/images/unique_path.png",
            },
        ],
        synchronous=True,
    )
    assert isinstance(id1, int)

    # Duplicate filepath: add_images swallows per-item errors in its batch
    # fallback, returning None for failed entries instead of raising.
    [id2] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "second",
                "data": PNG_BLUE_B64,
                "filepath": "/tmp/images/unique_path.png",
            },
        ],
        synchronous=True,
    )
    assert id2 is None, "Duplicate filepath should be rejected by backend uniqueness"

    # Multiple None filepaths are allowed (NULL != NULL)
    ids = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "no path 1",
                "data": PNG_RED_B64,
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "no path 2",
                "data": PNG_BLUE_B64,
            },
        ],
        synchronous=True,
    )
    assert all(isinstance(i, int) for i in ids), "NULL filepaths should not conflict"


@_handle_project
def test_update_filepath():
    im = ImageManager()

    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "initially no path",
                "data": PNG_RED_B64,
            },
        ],
    )

    # Filepath starts as None
    rows = im.filter_images(filter=f"image_id == {img_id}")
    assert rows and rows[0].filepath is None

    # Update filepath via update_images
    updated_ids = im.update_images(
        [{"image_id": img_id, "filepath": "/home/user/photo.png"}],
    )
    assert img_id in updated_ids

    rows = im.filter_images(filter=f"image_id == {img_id}")
    assert rows and rows[0].filepath == "/home/user/photo.png"


@_handle_project
def test_handle_filepath_property_and_update_metadata():
    im = ImageManager()

    handles = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "handle fp test",
                "data": PNG_RED_B64,
                "filepath": "/original/path.png",
            },
        ],
        return_handles=True,
        synchronous=True,
    )
    h = handles[0]
    assert h is not None
    assert h.filepath == "/original/path.png"

    # Update filepath via handle's update_metadata
    h.update_metadata(filepath="/updated/path.png")
    assert h.filepath == "/updated/path.png"

    # Verify persisted to backend
    rows = im.filter_images(filter=f"image_id == {h.image_id}")
    assert rows and rows[0].filepath == "/updated/path.png"


@_handle_project
def test_clear():
    im = ImageManager()

    # Seed a couple of images
    ids = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "alpha",
                "data": PNG_RED_B64,
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "beta",
                "data": PNG_BLUE_B64,
            },
        ],
    )
    id1, id2 = ids
    assert id1 != id2

    # Sanity: present before clear
    assert im.filter_images(filter=f"image_id == {id1}")
    assert im.filter_images(filter=f"image_id == {id2}")

    # Execute clear
    im.clear()

    # After clear: prior images should be gone
    assert len(im.filter_images(filter=f"image_id == {id1}")) == 0
    assert len(im.filter_images(filter=f"image_id == {id2}")) == 0

    # Re-provisioning works: can add a new image
    [new_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "after clear",
                "data": PNG_RED_B64,
            },
        ],
    )
    row = im.filter_images(filter=f"image_id == {new_id}")
    assert row and row[0].caption == "after clear"
