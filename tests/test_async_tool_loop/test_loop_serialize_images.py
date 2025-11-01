from __future__ import annotations

import base64
import pytest

from tests.helpers import _handle_project
from unity.common.async_tool_loop import start_async_tool_loop, AsyncToolLoopHandle
from tests.test_async_tool_loop.async_helpers import _wait_for_tool_message_prefix
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.types import RawImageRef, AnnotatedImageRef, ImageRefs


@pytest.mark.asyncio
@_handle_project
async def test_serialize_deserialize_with_images_overview_injected():
    # 1) Create a tiny in-memory PNG and add as an image to get a real id
    #    (4x4 single-colour)
    def _tiny_png_bytes() -> bytes:
        # Valid minimal PNG header + IHDR + IDAT for 1x1 opaque pixel (precomputed)
        # Keeping it simple and deterministic
        return base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=",
        )

    im = ImageManager()
    [iid] = im.add_images(
        [
            {
                "timestamp": None,
                "caption": "tiny",
                "data": _tiny_png_bytes(),
            },
        ],
        synchronous=True,
        return_handles=False,
    )
    assert isinstance(iid, int)

    # 2) Seed images context with an annotation
    images = ImageRefs.model_validate(
        [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=iid),
                annotation="sample",
            ),
        ],
    )

    # 3) Start a minimal loop with no base tools; live image helpers should be injected
    #    We use a generic system message; we will assert the overview tool is injected.
    handle = start_async_tool_loop(
        client=(
            im._manager
            if hasattr(im, "_manager")
            else __import__(
                "unify",
            ).AsyncUnify(
                "gpt-5@openai",
                reasoning_effort="high",
                service_tier="priority",
                cache=True,
            )
        ),
        message="begin",
        tools={},
        images=images,
    )

    # Snapshot immediately; images should be captured
    snap = handle.serialize()
    assert isinstance(snap, dict)
    imgs = snap.get("images") or []
    assert any(
        isinstance(x.get("image_id"), int) and x.get("annotation") is not None
        for x in imgs
    )

    # 4) Resume from snapshot and assert the overview tool was injected on startup
    resumed: AsyncToolLoopHandle = AsyncToolLoopHandle.deserialize(snap)
    # Wait until the synthetic overview tool message is present
    await _wait_for_tool_message_prefix(resumed._client, "live_images_overview", timeout=120.0)  # type: ignore[attr-defined]
    hist = resumed.get_history() or []
    assert any(
        (m.get("role") == "tool" and m.get("name") == "live_images_overview")
        for m in hist
    )
