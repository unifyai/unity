from __future__ import annotations

from datetime import datetime, timezone

import pytest

from unity.image_manager.image_manager import ImageManager
from unity.image_manager.utils import make_solid_png_base64
from unity.common.data_store import DataStore
from tests.helpers import _handle_project


PNG_GRAY_B64 = make_solid_png_base64(4, 4, (128, 128, 128))


@pytest.mark.unit
@_handle_project
def test_handle_local_annotation_is_not_persisted():
    im = ImageManager()

    handles = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "anno base",
                "data": PNG_GRAY_B64,
            },
        ],
        synchronous=True,
        return_handles=True,
    )

    h = next(h for h in handles if h is not None)

    # Default is None; setting it affects only this handle
    assert getattr(h, "annotation", None) is None
    h.annotation = "note-1"
    assert h.annotation == "note-1"

    # Not present in DataStore row
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))
    row = ds[h.image_id]
    assert "annotation" not in row
    assert row.get("caption") == "anno base"

    # A fresh handle to the same image must not inherit the annotation
    h2 = im.get_images([h.image_id])[0]
    assert getattr(h2, "annotation", None) is None


@pytest.mark.unit
@_handle_project
def test_pending_handle_annotation_stays_local_and_not_persisted():
    im = ImageManager()

    [h] = [
        x
        for x in im.add_images(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "caption": "pending base",
                    "data": PNG_GRAY_B64,
                },
            ],
            synchronous=False,
            return_handles=True,
        )
        if x is not None
    ]

    assert h.is_pending
    h.annotation = "pending-note"
    assert h.annotation == "pending-note"

    # Resolve pending → real id
    import asyncio as _asyncio

    mapping = _asyncio.get_event_loop().run_until_complete(
        im.await_pending([h.image_id]),
    )
    assert h.image_id in mapping
    resolved_id = mapping[h.image_id]

    # Original handle retains local annotation
    assert h.annotation == "pending-note"

    # Fresh handle to resolved id has no annotation
    h_resolved = im.get_images([resolved_id])[0]
    assert getattr(h_resolved, "annotation", None) is None

    # DataStore row must not contain annotation
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))
    row = ds[resolved_id]
    assert "annotation" not in row
    assert row.get("caption") == "pending base"
