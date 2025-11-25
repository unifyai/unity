from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unity.image_manager.utils import make_solid_png_base64

from unity.image_manager.image_manager import ImageManager
from tests.helpers import _handle_project


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_images_by_caption_semantics():
    im = ImageManager()

    # Seed a few images with distinct captions
    im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "a scenic mountain landscape at sunrise",
                "data": make_solid_png_base64(8, 8, (200, 200, 255)),
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "a cat sitting on a red sofa",
                "data": make_solid_png_base64(8, 8, (255, 0, 0)),
            },
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "blue pixel art icon",
                "data": make_solid_png_base64(8, 8, (0, 0, 255)),
            },
        ],
    )

    # Semantic query for 'cat on couch'
    results = im.search_images(reference_text="cat on a couch", k=2)
    assert results and any("cat" in (r.caption or "").lower() for r in results)

    # When reference text is unrelated, backfill should still return recent images
    recent = im.search_images(reference_text="utterly unrelated phrase", k=3)
    assert len(recent) == 3
