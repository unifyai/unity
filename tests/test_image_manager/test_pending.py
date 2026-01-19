from __future__ import annotations

import base64
from datetime import datetime, timezone
import asyncio
import unify

import pytest

from unity.image_manager.image_manager import ImageManager
from unity.image_manager.utils import make_solid_png_base64
from unity.common.data_store import DataStore
from tests.helpers import _handle_project


PNG_RED_B64 = make_solid_png_base64(32, 32, (255, 0, 0))
PNG_BLUE_B64 = make_solid_png_base64(32, 32, (0, 0, 255))


@_handle_project
def test_async_enqueue_immediate_raw_and_pending_flag():
    im = ImageManager()

    # Enqueue using raw bytes
    raw_bytes = base64.b64decode(PNG_RED_B64)
    [h] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "staged",
                "data": raw_bytes,
                "annotation": "staged-ann",
            },
        ],
        synchronous=False,
        return_handles=True,
    )

    assert h.is_pending
    assert h.annotation == "staged-ann"
    assert isinstance(h.image_id, int) and h.image_id >= 10**12

    # Immediate raw access must work and round-trip to original bytes
    out = h.raw()
    assert out == raw_bytes

    # DataStore should have the row under the pending id
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))
    row = ds[h.image_id]
    assert row["image_id"] == h.image_id
    assert row.get("caption") == "staged"
    assert base64.b64decode(row.get("data")) == raw_bytes
    assert "annotation" not in row


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@_handle_project
def test_await_pending_remaps_ids_and_updates_data_store():
    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    # Enqueue using base64 string
    [staged] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "to flush",
                "data": PNG_BLUE_B64,
            },
        ],
        synchronous=False,
        return_handles=True,
    )
    pid = staged.image_id
    assert pid in ds

    # Await and get real id mapping
    mapping = _run(im.await_pending([pid]))
    assert pid in mapping
    real_id = mapping[pid]
    assert isinstance(real_id, int) and real_id < 10**12

    # Pending row should be replaced by real-id row in DataStore
    assert real_id in ds
    # pending id may or may not be present depending on deletion success, but if present, allow
    new_row = ds[real_id]
    assert new_row.get("caption") == "to flush"
    assert base64.b64decode(new_row.get("data")) == base64.b64decode(PNG_BLUE_B64)

    # New handle by real id works
    handle = im.get_images([real_id])[0]
    assert not handle.is_pending
    assert handle.raw() == base64.b64decode(PNG_BLUE_B64)

    # Old handle can be resolved
    staged.resolve(real_id)
    assert not staged.is_pending

    # Update metadata after resolution propagates to backend and cache
    staged.update_metadata(caption="updated")
    row2 = ds[real_id]
    assert row2.get("caption") == "updated"


@_handle_project
def test_update_metadata_while_pending_reflects_locally():
    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    [h] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "xfer",
                "data": PNG_RED_B64,
            },
        ],
        synchronous=False,
        return_handles=True,
    )
    # Update metadata while pending; should be reflected locally
    h.update_metadata(caption="xfer-updated")
    row = ds[h.image_id]
    assert row.get("caption") == "xfer-updated"


@_handle_project
def test_is_pending_id_and_resolution():
    im = ImageManager()

    [staged] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "p",
                "data": PNG_RED_B64,
            },
        ],
        synchronous=False,
        return_handles=True,
    )
    assert im.is_pending_id(staged.image_id) is True
    assert im.is_pending_id(123) is False

    mapping = _run(im.await_pending([staged.image_id]))
    real_id = mapping[staged.image_id]
    assert im.is_pending_id(real_id) is False


@_handle_project
def test_get_images_for_pending_prefers_cache_no_backend(monkeypatch):
    im = ImageManager()

    h1, h2 = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "p1",
                "data": PNG_RED_B64,
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "p2",
                "data": PNG_BLUE_B64,
            },
        ],
        synchronous=False,
        return_handles=True,
    )

    calls = {"count": 0}
    orig_get_logs = unify.get_logs

    def _wrapped_get_logs(*args, **kwargs):
        calls["count"] += 1
        return orig_get_logs(*args, **kwargs)

    monkeypatch.setattr(unify, "get_logs", _wrapped_get_logs)

    handles = im.get_images([h1.image_id, h2.image_id])
    assert [h.image_id for h in handles] == [h1.image_id, h2.image_id]
    assert calls["count"] == 0


@_handle_project
def test_async_enqueue_accepts_base64_and_raw_roundtrip():
    im = ImageManager()
    [h] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "b64",
                "data": PNG_RED_B64,
            },
        ],
        synchronous=False,
        return_handles=True,
    )
    assert h.is_pending
    assert h.raw() == base64.b64decode(PNG_RED_B64)


@_handle_project
def test_await_pending_multiple_and_datastore_updates():
    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    h1, h2 = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "m1",
                "data": PNG_RED_B64,
                "annotation": "m1-ann",
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "m2",
                "data": PNG_BLUE_B64,
                "annotation": "m2-ann",
            },
        ],
        synchronous=False,
        return_handles=True,
    )

    mapping = _run(im.await_pending([h1.image_id, h2.image_id]))
    assert set(mapping.keys()) == {h1.image_id, h2.image_id}
    rid1 = mapping[h1.image_id]
    rid2 = mapping[h2.image_id]
    assert rid1 in ds and rid2 in ds
    assert base64.b64decode(ds[rid1]["data"]) == base64.b64decode(PNG_RED_B64)
    assert base64.b64decode(ds[rid2]["data"]) == base64.b64decode(PNG_BLUE_B64)

    # get_images by real ids returns in requested order
    hs = im.get_images([rid2, rid1])
    assert [h.image_id for h in hs] == [rid2, rid1]
    # New handles should not inherit prior annotations (local-only)
    assert getattr(hs[0], "annotation", None) is None
    assert getattr(hs[1], "annotation", None) is None


@_handle_project
@pytest.mark.asyncio
async def test_temp_image_id_persists_after_resolution():
    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    [h] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "keep-temp",
                "data": PNG_BLUE_B64,
            },
        ],
        synchronous=False,
        return_handles=True,
    )
    pid = h.image_id
    mapping = await im.await_pending([pid])
    rid = mapping[pid]
    row = ds[rid]
    assert row.get("temp_image_id") == pid


@_handle_project
def test_await_pending_omits_missing_rows():
    im = ImageManager()
    [h] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "ok",
                "data": PNG_RED_B64,
            },
        ],
        synchronous=False,
        return_handles=True,
    )
    missing_pid = 10**12 + 999_999

    # Missing only -> empty mapping
    mapping = _run(im.await_pending([missing_pid]))
    assert mapping == {}

    # Mixed -> only existing pending is mapped
    mapping = _run(im.await_pending([h.image_id, missing_pid]))
    assert set(mapping.keys()) == {h.image_id}


@_handle_project
@pytest.mark.asyncio
async def test_pending_update_persists_after_resolution_and_backend_reflects(
    monkeypatch,
):
    im = ImageManager()
    from unity.common.data_store import DataStore as _DS

    ds = _DS.for_context(im._ctx, key_fields=("image_id",))

    # Enqueue without caption, then update caption while pending
    [h] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": None,
                "auto_caption": False,
                "data": PNG_RED_B64,
            },
        ],
        synchronous=False,
        return_handles=True,
    )

    h.update_metadata(caption="label-one")
    h.update_metadata(caption="label-final")

    # Count backend update_images calls (should be exactly one deferred call)
    calls = {"count": 0}
    orig_update_images = im.update_images

    def _wrapped_update_images(updates):
        calls["count"] += 1
        return orig_update_images(updates)

    monkeypatch.setattr(im, "update_images", _wrapped_update_images)

    # Resolve now; deferred persistence should run shortly thereafter
    rid = await h.wait_until_resolved()

    # Poll until backend reflects the final label or timeout
    import time as _time

    deadline = _time.time() + 2.0
    backend_caption = None
    while _time.time() < deadline:
        rows = im.filter_images(filter=f"image_id == {rid}")
        if rows:
            backend_caption = rows[0].caption
            if backend_caption == "label-final":
                break
        await asyncio.sleep(0.05)

    assert backend_caption == "label-final"
    assert rid in ds and ds[rid].get("caption") == "label-final"
    # Exactly one backend update due to coalescing
    assert calls["count"] == 1


@_handle_project
@pytest.mark.asyncio
async def test_multiple_pending_updates_coalesce_and_persist_only_last(monkeypatch):
    im = ImageManager()

    [h] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "seed",
                "data": PNG_BLUE_B64,
            },
        ],
        synchronous=False,
        return_handles=True,
    )

    # Apply several updates while still pending
    h.update_metadata(caption="v1")
    h.update_metadata(caption="v2")
    h.update_metadata(caption="v3")

    captured_payloads = []
    orig_update_images = im.update_images

    def _wrapped_update_images(updates):
        captured_payloads.append(list(updates))
        return orig_update_images(updates)

    monkeypatch.setattr(im, "update_images", _wrapped_update_images)

    # Resolve and wait briefly for deferred persist to run
    rid = await h.wait_until_resolved()

    import time as _time

    # Wait for one call to update_images and backend to reflect v3
    deadline = _time.time() + 2.0
    while _time.time() < deadline and len(captured_payloads) == 0:
        await asyncio.sleep(0.02)

    assert len(captured_payloads) == 1
    last_payload = captured_payloads[0][0]
    assert last_payload.get("image_id") == rid
    assert last_payload.get("caption") == "v3"

    # Verify backend has v3 too
    backend_caption = None
    deadline2 = _time.time() + 2.0
    while _time.time() < deadline2:
        rows = im.filter_images(filter=f"image_id == {rid}")
        if rows:
            backend_caption = rows[0].caption
            if backend_caption == "v3":
                break
        await asyncio.sleep(0.05)

    assert backend_caption == "v3"


@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_ask_on_pending_enqueue_returns_text_only(static_now):
    im = ImageManager()
    [h] = im.add_images(
        [
            {
                "timestamp": static_now,
                "caption": "ask",
                "data": PNG_RED_B64,
            },
        ],
        synchronous=False,
        return_handles=True,
    )
    answer = await h.ask("What do you notice in this image?")
    assert isinstance(answer, str) and answer.strip()
    assert "data:image" not in answer and "image_url" not in answer
