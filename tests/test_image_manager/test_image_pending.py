from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import unify

import pytest

from unity.image_manager.image_manager import ImageManager
from unity.image_manager.utils import make_solid_png_base64
from unity.common.data_store import DataStore
from tests.helpers import _handle_project


PNG_RED_B64 = make_solid_png_base64(8, 8, (255, 0, 0))
PNG_BLUE_B64 = make_solid_png_base64(8, 8, (0, 0, 255))


@pytest.mark.unit
@_handle_project
def test_stage_image_immediate_raw_and_pending_flag():
    im = ImageManager()

    # Stage using raw bytes
    raw_bytes = base64.b64decode(PNG_RED_B64)
    h = im.stage_image(
        timestamp=datetime.now(timezone.utc),
        caption="staged",
        data=raw_bytes,
    )

    assert h.is_pending
    assert isinstance(h.image_id, int) and h.image_id >= 10**12

    # Immediate raw access must work and round-trip to original bytes
    out = h.raw()
    assert out == raw_bytes

    # DataStore should have the staged row under the pending id
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))
    row = ds[h.image_id]
    assert row["image_id"] == h.image_id
    assert row.get("caption") == "staged"
    assert base64.b64decode(row.get("data")) == raw_bytes


@pytest.mark.unit
@_handle_project
def test_flush_pending_remaps_ids_and_updates_data_store():
    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    # Stage using base64 string
    staged = im.stage_image(
        timestamp=datetime.now(timezone.utc),
        caption="to flush",
        data=PNG_BLUE_B64,
    )
    pid = staged.image_id
    assert pid in ds

    # Flush and get real id mapping
    mapping = im.flush_pending([pid])
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


@pytest.mark.unit
@_handle_project
def test_to_ref_from_ref_roundtrip_seeds_data_store():
    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    staged = im.stage_image(
        timestamp=datetime.now(timezone.utc),
        caption="xfer",
        data=PNG_RED_B64,
    )
    ref = staged.to_ref()

    # Construct a new manager and import from ref
    im2 = ImageManager()
    h2 = im2.from_ref(ref)

    assert h2.image_id == staged.image_id
    assert h2.is_pending
    # Should be usable immediately via cache seeding
    assert h2.raw() == base64.b64decode(PNG_RED_B64)

    # DataStore must have the row under the same pending id
    row = ds[h2.image_id]
    assert row.get("caption") == "xfer"
    assert base64.b64decode(row.get("data")) == base64.b64decode(PNG_RED_B64)

    # Update metadata while pending; should be reflected locally and included on flush
    h2.update_metadata(caption="xfer-updated")
    ds2 = DataStore.for_context(im2._ctx, key_fields=("image_id",))
    row2 = ds2[h2.image_id]
    assert row2.get("caption") == "xfer-updated"


@pytest.mark.unit
@_handle_project
def test_manager_is_pending_id_and_resolution():
    im = ImageManager()

    staged = im.stage_image(
        timestamp=datetime.now(timezone.utc),
        caption="p",
        data=PNG_RED_B64,
    )
    assert im.is_pending_id(staged.image_id) is True
    assert im.is_pending_id(123) is False

    mapping = im.flush_pending([staged.image_id])
    real_id = mapping[staged.image_id]
    assert im.is_pending_id(real_id) is False


@pytest.mark.unit
@_handle_project
def test_get_images_for_pending_prefers_cache_no_backend(monkeypatch):
    im = ImageManager()

    h1 = im.stage_image(
        timestamp=datetime.now(timezone.utc),
        caption="p1",
        data=PNG_RED_B64,
    )
    h2 = im.stage_image(
        timestamp=datetime.now(timezone.utc),
        caption="p2",
        data=PNG_BLUE_B64,
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


@pytest.mark.unit
@_handle_project
def test_stage_image_accepts_base64_and_raw_roundtrip():
    im = ImageManager()
    h = im.stage_image(
        timestamp=datetime.now(timezone.utc),
        caption="b64",
        data=PNG_RED_B64,
    )
    assert h.is_pending
    assert h.raw() == base64.b64decode(PNG_RED_B64)


@pytest.mark.unit
@_handle_project
def test_flush_pending_multiple_and_datastore_updates():
    im = ImageManager()
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))

    h1 = im.stage_image(
        timestamp=datetime.now(timezone.utc),
        caption="m1",
        data=PNG_RED_B64,
    )
    h2 = im.stage_image(
        timestamp=datetime.now(timezone.utc),
        caption="m2",
        data=PNG_BLUE_B64,
    )

    mapping = im.flush_pending([h1.image_id, h2.image_id])
    assert set(mapping.keys()) == {h1.image_id, h2.image_id}
    rid1 = mapping[h1.image_id]
    rid2 = mapping[h2.image_id]
    assert rid1 in ds and rid2 in ds
    assert base64.b64decode(ds[rid1]["data"]) == base64.b64decode(PNG_RED_B64)
    assert base64.b64decode(ds[rid2]["data"]) == base64.b64decode(PNG_BLUE_B64)

    # get_images by real ids returns in requested order
    hs = im.get_images([rid2, rid1])
    assert [h.image_id for h in hs] == [rid2, rid1]


@pytest.mark.unit
@_handle_project
def test_from_ref_json_string_without_inline_data_fallbacks_to_backend():
    im = ImageManager()

    # Create a real backend image
    [real_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "real",
                "data": PNG_BLUE_B64,
            },
        ],
    )

    # Build a minimal JSON string ref without data
    ref_str = json.dumps({"image_id": real_id})
    im2 = ImageManager()
    h = im2.from_ref(ref_str)
    assert not h.is_pending
    assert h.image_id == real_id


@pytest.mark.unit
@_handle_project
def test_flush_pending_omits_missing_rows():
    im = ImageManager()
    h = im.stage_image(
        timestamp=datetime.now(timezone.utc),
        caption="ok",
        data=PNG_RED_B64,
    )
    missing_pid = 10**12 + 999_999

    # Missing only -> empty mapping
    assert im.flush_pending([missing_pid]) == {}

    # Mixed -> only existing pending is mapped
    mapping = im.flush_pending([h.image_id, missing_pid])
    assert set(mapping.keys()) == {h.image_id}


@pytest.mark.eval
@pytest.mark.requires_real_unify
@pytest.mark.asyncio
@_handle_project
async def test_ask_on_pending_stage_returns_text_only():
    im = ImageManager()
    h = im.stage_image(
        timestamp=datetime.now(timezone.utc),
        caption="ask",
        data=PNG_RED_B64,
    )
    answer = await h.ask("What do you notice in this image?")
    assert isinstance(answer, str) and answer.strip()
    assert "data:image" not in answer and "image_url" not in answer
