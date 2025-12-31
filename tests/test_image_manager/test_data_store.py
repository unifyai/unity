from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Any

import unify
from unity.image_manager.image_manager import ImageManager
from unity.common.data_store import DataStore
from tests.helpers import _handle_project
from unity.image_manager.utils import make_solid_png_base64


PNG_RED_B64 = make_solid_png_base64(8, 8, (255, 0, 0))
PNG_BLUE_B64 = make_solid_png_base64(8, 8, (0, 0, 255))


# No project hard-coding: rely on the shared test helpers/conftest to manage
# the active project and contexts. DataStore instances are scoped per
# (project, context) and our tests use @_handle_project to guarantee unique
# per-test contexts, so no explicit registry resets are required here.


@_handle_project
def test_data_store_updated_after_add_and_update():
    im = ImageManager()

    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "CacheTest",
                "data": PNG_RED_B64,
            },
        ],
    )

    row = ds[img_id]
    assert row["image_id"] == img_id
    assert row.get("caption") == "CacheTest"

    # Update caption and verify DataStore reflects it
    updated = im.update_images([{"image_id": img_id, "caption": "Updated"}])
    assert img_id in updated

    row2 = ds[img_id]
    assert row2["image_id"] == img_id
    assert row2.get("caption") == "Updated"


@_handle_project
def test_filter_images_repopulates_data_store():
    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "Filter",
                "data": PNG_BLUE_B64,
            },
        ],
    )

    ds.clear()
    rows = im.filter_images(filter=f"image_id == {img_id}")
    assert rows and int(rows[0].image_id) == int(img_id)

    cached = ds[img_id]
    assert cached["image_id"] == img_id
    assert cached.get("caption") == "Filter"


@_handle_project
def test_search_images_repopulates_data_store():
    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "vector search example",
                "data": PNG_RED_B64,
            },
        ],
    )

    ds.clear()
    results = im.search_images(reference_text="vector")
    assert any(int(r.image_id) == int(img_id) for r in results)

    cached = ds[img_id]
    assert cached["image_id"] == img_id
    assert "vector" in (cached.get("caption") or "").lower()


@_handle_project
def test_get_images_prefers_cache_and_falls_back_backend(monkeypatch):
    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    ids = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "A",
                "data": PNG_RED_B64,
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "B",
                "data": PNG_BLUE_B64,
            },
        ],
    )

    # 1) Cache-hit path: both ids present in DataStore → zero backend reads
    calls = {"count": 0}
    orig_get_logs = unify.get_logs

    def _wrapped_get_logs(*args: Any, **kwargs: Any):
        calls["count"] += 1
        return orig_get_logs(*args, **kwargs)

    monkeypatch.setattr(unify, "get_logs", _wrapped_get_logs)
    _ = im.get_images([ids[0]])
    assert calls["count"] == 0

    # 2) Cache-miss path: clear cache then call → one backend read
    ds.clear()
    _ = im.get_images([ids[0]])
    assert calls["count"] >= 1

    # Subsequent call for same id should be a cache hit → no additional read
    pre = calls["count"]
    _ = im.get_images([ids[0]])
    assert calls["count"] == pre

    # 3) Mixed hit/miss: id0 cached, id1 not cached → exactly one backend read
    ds.clear()
    # Seed only first id in cache
    _ = im.get_images([ids[0]])
    pre = calls["count"]
    _ = im.get_images([ids[0], ids[1]])
    assert calls["count"] == pre + 1


@_handle_project
def test_image_handle_raw_caches_gcs_download(monkeypatch):
    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    # Seed a row that points to GCS so ImageHandle.raw() downloads once
    # Avoid a real POST with a GCS URL by faking unify.log
    class _FakeLog:
        def __init__(self, entries: dict):
            self.entries = entries

    counter = {"next": 10001}

    def _fake_unify_log(
        *,
        context: str,
        new: Any = True,
        mutable: Any = None,
        params: Any = None,
        **entries: Any,
    ):
        eid = counter["next"]
        counter["next"] += 1
        ret = dict(entries)
        ret["image_id"] = eid
        return _FakeLog(ret)

    monkeypatch.setattr(unify, "log", _fake_unify_log)

    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "GCS sample",
                "data": "gs://my-bucket/path/to/image.jpg",
            },
        ],
    )

    # Mock unify.download_object to count downloads
    download_count = {"count": 0}

    def _fake_download_object(gcs_uri, *, api_key=None):
        download_count["count"] += 1
        return b"IMG_BYTES"

    monkeypatch.setattr(unify, "download_object", _fake_download_object)

    # First raw() must download and then cache base64 in DataStore
    h1 = im.get_images([img_id])[0]
    raw1 = h1.raw()
    assert isinstance(raw1, (bytes, bytearray)) and raw1

    # DataStore should now contain base64 for this image id
    cached = ds[img_id]
    data_field = cached.get("data")
    assert isinstance(data_field, str) and not data_field.startswith("gs://")
    # Validate that base64 decodes to the same bytes
    assert base64.b64decode(data_field) == raw1

    # Second raw() on a fresh handle should NOT trigger another download
    h2 = im.get_images([img_id])[0]
    _ = h2.raw()

    # Verify only one download happened
    assert download_count["count"] == 1


@_handle_project
def test_clear_empties_data_store():
    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

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
    assert len(ds) == 2

    im.clear()
    assert len(ds) == 0
