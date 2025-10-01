from __future__ import annotations

from datetime import datetime, timezone

import pytest

from unity.guidance_manager.guidance_manager import GuidanceManager
from unity.image_manager.image_manager import ImageManager
from tests.helpers import _handle_project


# 1x1 PNG (opaque) – small valid image payload
PNG_1x1_BLUE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMB9j3v1S0AAAAASUVORK5CYII="


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.requires_real_unify
@_handle_project
async def test_guidance_persistent_image_context_then_reason():
    """
    Flow:
    1) Create an image and a guidance row that references it.
    2) Call GuidanceManager.ask with a request that attaches the image into the loop
       using the new attach tool.
    3) Ask a follow-up that depends on seeing the image.
    4) Expect a non-empty textual answer (no base64 in answer), leveraging persistent context.
    """
    # Seed one image
    im = ImageManager()
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "blue pixel art icon",
                "data": PNG_1x1_BLUE,
            },
        ],
    )

    # Create a guidance entry pointing to that image
    gm = GuidanceManager()
    out = gm._add_guidance(
        title="Pixel Icon",
        content="Review the icon layout and color.",
        images={"[0:1]": int(img_id)},
    )
    gid = out["details"]["guidance_id"]

    # Step 1: Attach the guidance-linked image persistently to the loop context
    h1 = await gm.ask(
        f"For guidance ID {gid}, attach the image so you can see it, then confirm once attached.",
    )
    ans1 = await h1.result()
    assert (
        isinstance(ans1, str) and ans1.strip()
    ), "Attachment confirmation should be text"
    assert "data:image" not in ans1 and "image_url" not in ans1

    # Step 2: Follow-up question that benefits from persistent image context
    h2 = await gm.ask("Now, describe the dominant color visible.")
    ans2 = await h2.result()
    assert isinstance(ans2, str) and ans2.strip(), "Expected a textual description"
    # Heuristic: The tiny asset is blue; allow synonyms or general color mention
    assert any(
        kw in ans2.lower() for kw in ("blue", "azure", "navy", "cyan")
    ), f"Answer does not reference blue-ish color: {ans2!r}"
