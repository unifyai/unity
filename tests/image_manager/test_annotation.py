from __future__ import annotations

from datetime import datetime, timezone
import asyncio as _asyncio

import pytest

from unity.image_manager.image_manager import ImageManager
from unity.image_manager.utils import make_solid_png_base64
from unity.common.data_store import DataStore
from tests.helpers import _handle_project

PNG_GRAY_B64 = make_solid_png_base64(32, 32, (128, 128, 128))


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


@pytest.mark.asyncio
@_handle_project
async def test_pending_handle_annotation_stays_local_and_not_persisted():
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
    mapping = await im.await_pending([h.image_id])
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


@pytest.mark.asyncio
@_handle_project
async def test_wait_for_annotation_immediate():
    im = ImageManager()
    [h] = [
        x
        for x in im.add_images(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "caption": "immediate anno",
                    "data": PNG_GRAY_B64,
                },
            ],
            synchronous=True,
            return_handles=True,
        )
        if x is not None
    ]

    h.annotation = "ready"
    got = await h.wait_for_annotation(timeout=1.0)
    assert got == "ready"


@pytest.mark.asyncio
@_handle_project
async def test_wait_for_annotation_blocks_then_returns():
    im = ImageManager()
    [h] = [
        x
        for x in im.add_images(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "caption": "delayed anno",
                    "data": PNG_GRAY_B64,
                },
            ],
            synchronous=True,
            return_handles=True,
        )
        if x is not None
    ]

    async def _later():
        await _asyncio.sleep(0.05)
        h.annotation = "late"

    waiter = _asyncio.create_task(h.wait_for_annotation(timeout=1.0))
    setter = _asyncio.create_task(_later())
    got, _ = await _asyncio.gather(waiter, setter)
    assert got == "late"


@pytest.mark.asyncio
@_handle_project
async def test_wait_for_annotation_and_resolution_with_gather():
    im = ImageManager()
    [h] = [
        x
        for x in im.add_images(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "caption": "pending both",
                    "data": PNG_GRAY_B64,
                },
            ],
            synchronous=False,
            return_handles=True,
        )
        if x is not None
    ]

    async def _set_annotation():
        # Simulate producer attaching annotation shortly after creation
        await _asyncio.sleep(0.02)
        h.annotation = "both-ready"

    ann_task = _asyncio.create_task(h.wait_for_annotation(timeout=2.0))
    rid_task = _asyncio.create_task(h.wait_until_resolved())
    _ = _asyncio.create_task(_set_annotation())

    ann, rid = await _asyncio.gather(ann_task, rid_task)
    assert isinstance(rid, int)
    assert ann == "both-ready"


@pytest.mark.asyncio
@_handle_project
async def test_wait_for_caption_immediate_and_delayed():
    im = ImageManager()

    # Immediate caption already present (sync create)
    [h1] = [
        x
        for x in im.add_images(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "caption": "ready-now",
                    "data": PNG_GRAY_B64,
                },
            ],
            synchronous=True,
            return_handles=True,
        )
        if x is not None
    ]
    got1 = await h1.wait_for_caption(timeout=0.5)
    assert got1 == "ready-now"

    # Delayed caption set via update_metadata
    [h2] = [
        x
        for x in im.add_images(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "caption": None,
                    "auto_caption": False,
                    "data": PNG_GRAY_B64,
                },
            ],
            synchronous=True,
            return_handles=True,
        )
        if x is not None
    ]

    async def _later():
        await _asyncio.sleep(0.02)
        h2.update_metadata(caption="arrived")

    waiter = _asyncio.create_task(h2.wait_for_caption(timeout=1.0))
    setter = _asyncio.create_task(_later())
    got2, _ = await _asyncio.gather(waiter, setter)
    assert got2 == "arrived"


@pytest.mark.asyncio
@_handle_project
async def test_wait_for_caption_and_resolution_together():
    im = ImageManager()
    [h] = [
        x
        for x in im.add_images(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "caption": None,
                    "auto_caption": False,
                    "data": PNG_GRAY_B64,
                },
            ],
            synchronous=False,
            return_handles=True,
        )
        if x is not None
    ]

    async def _set_caption():
        await _asyncio.sleep(0.03)
        h.update_metadata(caption="label-ready")

    cap_task = _asyncio.create_task(h.wait_for_caption(timeout=2.0))
    rid_task = _asyncio.create_task(h.wait_until_resolved())
    _ = _asyncio.create_task(_set_caption())

    cap, rid = await _asyncio.gather(cap_task, rid_task)
    assert isinstance(rid, int)
    assert cap == "label-ready"


@_handle_project
def test_constructor_annotation_is_set_and_not_persisted():
    from unity.image_manager.image_manager import ImageHandle
    from unity.image_manager.types.image import Image

    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "ctor base",
                "data": PNG_GRAY_B64,
            },
        ],
    )

    row = ds[img_id]
    h = ImageHandle(manager=im, image=Image(**row), annotation="ctor-note")

    # Annotation provided in constructor should be set on this handle only
    assert h.annotation == "ctor-note"

    # A fresh handle from the manager must not inherit the annotation
    fresh = im.get_images([img_id])[0]
    assert getattr(fresh, "annotation", None) is None

    # DataStore must not contain the annotation
    row2 = ds[img_id]
    assert "annotation" not in row2
    assert row2.get("caption") == "ctor base"


@_handle_project
def test_add_images_with_annotation_sets_handle_local_only():
    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    # Sync mode with return_handles=True
    hs = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "ctor via add",
                "data": PNG_GRAY_B64,
                "annotation": "via-add",
            },
        ],
        synchronous=True,
        return_handles=True,
    )
    h = next(h for h in hs if h is not None)
    assert h.annotation == "via-add"

    row = ds[h.image_id]
    assert "annotation" not in row

    # A fresh handle must not inherit annotation
    h2 = ImageManager().get_images([h.image_id])[0]
    assert getattr(h2, "annotation", None) is None


@pytest.mark.asyncio
@_handle_project
async def test_wait_for_annotation_immediate_via_constructor():
    from unity.image_manager.image_manager import ImageHandle
    from unity.image_manager.types.image import Image

    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "ctor wait",
                "data": PNG_GRAY_B64,
            },
        ],
    )

    row = ds[img_id]
    h = ImageHandle(manager=im, image=Image(**row), annotation="ready-now")

    got = await h.wait_for_annotation(timeout=0.5)
    assert got == "ready-now"


# ────────────────────────────────────────────────────────────────────────────
# Cancellation regression tests
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_wait_for_annotation_is_cancellable():
    """
    Regression test: wait_for_annotation() must be cancellable without
    blocking event loop shutdown.

    Previously this method used asyncio.to_thread(_annotation_event.wait) which
    created executor threads that blocked indefinitely. Now it uses polling,
    allowing clean cancellation.
    """
    im = ImageManager()
    [h] = [
        x
        for x in im.add_images(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "caption": "cancel test",
                    "data": PNG_GRAY_B64,
                },
            ],
            synchronous=True,
            return_handles=True,
        )
        if x is not None
    ]

    # Don't set annotation - this would block indefinitely with old implementation
    wait_task = _asyncio.create_task(h.wait_for_annotation())

    # Give task time to start polling
    await _asyncio.sleep(0.2)

    # Task should still be running (waiting for annotation)
    assert not wait_task.done(), "wait_for_annotation() should be waiting"

    # Cancel the task - this should succeed without hanging
    wait_task.cancel()

    # Wait for cancellation to complete (should be immediate)
    with pytest.raises(_asyncio.CancelledError):
        await _asyncio.wait_for(wait_task, timeout=1.0)


@pytest.mark.asyncio
@_handle_project
async def test_wait_for_caption_is_cancellable():
    """
    Regression test: wait_for_caption() must be cancellable without
    blocking event loop shutdown.

    Previously this method used asyncio.to_thread(_caption_event.wait) which
    created executor threads that blocked indefinitely. Now it uses polling,
    allowing clean cancellation.
    """
    im = ImageManager()
    [h] = [
        x
        for x in im.add_images(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "caption": None,
                    "auto_caption": False,
                    "data": PNG_GRAY_B64,
                },
            ],
            synchronous=True,
            return_handles=True,
        )
        if x is not None
    ]

    # Don't set caption - this would block indefinitely with old implementation
    wait_task = _asyncio.create_task(h.wait_for_caption())

    # Give task time to start polling
    await _asyncio.sleep(0.2)

    # Task should still be running (waiting for caption)
    assert not wait_task.done(), "wait_for_caption() should be waiting"

    # Cancel the task - this should succeed without hanging
    wait_task.cancel()

    # Wait for cancellation to complete (should be immediate)
    with pytest.raises(_asyncio.CancelledError):
        await _asyncio.wait_for(wait_task, timeout=1.0)
