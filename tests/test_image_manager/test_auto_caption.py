from __future__ import annotations

import asyncio
import time

import pytest

from datetime import datetime, timezone

from unity.image_manager.image_manager import ImageManager, ImageHandle
from unity.image_manager.utils import make_solid_png_base64
from unity.common.data_store import DataStore
from tests.helpers import _handle_project


PNG_RED_B64 = make_solid_png_base64(32, 32, (255, 0, 0))


@pytest.mark.asyncio
@_handle_project
async def test_sync_sets_caption_and_persists(monkeypatch):
    im = ImageManager()

    async def _fake_ask(self: ImageHandle, question: str) -> str:
        return "auto caption (sync)"

    monkeypatch.setattr(ImageHandle, "ask", _fake_ask, raising=True)

    [h] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": None,
                "data": PNG_RED_B64,
                "auto_caption": True,
            },
        ],
        synchronous=True,
        return_handles=True,
    )

    # Wait until caption is available (set by auto-caption worker)
    cap = await h.wait_for_caption()
    assert cap == "auto caption (sync)"

    # Poll backend until it reflects the caption (update_metadata persists immediately for resolved rows)
    deadline = time.time() + 2.0
    backend_caption = None
    while time.time() < deadline:
        rows = im.filter_images(filter=f"image_id == {h.image_id}")
        if rows:
            backend_caption = rows[0].caption
            if backend_caption == cap:
                break
        time.sleep(0.05)
    assert backend_caption == cap


@pytest.mark.asyncio
@_handle_project
async def test_pending_sets_caption_then_persists_after_resolution(
    monkeypatch,
):
    im = ImageManager()

    async def _fake_ask(self: ImageHandle, question: str) -> str:
        return "auto caption (pending)"

    monkeypatch.setattr(ImageHandle, "ask", _fake_ask, raising=True)

    [h] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": None,
                "data": PNG_RED_B64,
                "auto_caption": True,
            },
        ],
        synchronous=False,
        return_handles=True,
    )

    # Auto-caption should set a local caption promptly
    cap = await h.wait_for_caption()
    assert cap == "auto caption (pending)"

    # Local DataStore should reflect the caption under the pending id
    ds = DataStore.for_context(im._ctx, key_fields=("image_id",))
    assert ds[h.image_id].get("caption") == cap

    # Resolve and then ensure backend reflects the caption after deferred persist
    rid = await h.wait_until_resolved()
    assert isinstance(rid, int) and rid >= 0

    deadline = time.time() + 3.0
    backend_caption = None
    while time.time() < deadline:
        rows = im.filter_images(filter=f"image_id == {rid}")
        if rows:
            backend_caption = rows[0].caption
            if backend_caption == cap:
                break
        await asyncio.sleep(0.05)
    assert backend_caption == cap
