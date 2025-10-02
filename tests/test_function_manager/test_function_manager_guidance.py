from __future__ import annotations

from datetime import datetime, timezone
import pytest

from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager
from unity.guidance_manager.guidance_manager import GuidanceManager
from unity.image_manager.image_manager import ImageManager


# Tiny valid 1x1 PNG (opaque)
PNG_1x1_GREEN = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAusB9r4/ARcAAAAASUVORK5CYII="


@pytest.mark.unit
@_handle_project
def test_function_manager_fetch_guidance_for_function_and_limits():
    # Seed functions
    fm = FunctionManager()
    src_a = "def alpha(x: int) -> int:\n    return x + 1\n"
    src_b = "def beta(y: int) -> int:\n    return y * 2\n"
    fm.add_functions(implementations=[src_a, src_b])
    listing = fm.list_functions()
    alpha_id = listing["alpha"]["function_id"]
    beta_id = listing["beta"]["function_id"]

    # Seed guidance that references alpha and beta
    gm = GuidanceManager()
    g1 = gm._add_guidance(
        title="Alpha Notes",
        content="How to use alpha",
        function_ids=[alpha_id],
    )
    g2 = gm._add_guidance(
        title="Beta Notes",
        content="How to use beta",
        function_ids=[beta_id],
    )

    # Fetch guidance for alpha
    alpha_guidance = fm._get_guidance_for_function(function_id=alpha_id)
    titles = {g["title"] for g in alpha_guidance}
    assert titles == {"Alpha Notes"}

    # Limit behavior
    both = fm._get_guidance_for_function(function_id=beta_id, limit=1)
    assert len(both) == 1


@pytest.mark.unit
@_handle_project
def test_function_manager_image_handles_and_attachment():
    # Seed one image
    im = ImageManager()
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "green pixel",
                "data": PNG_1x1_GREEN,
            },
        ],
    )

    # Seed a function
    fm = FunctionManager()
    src = "def gamma(z: int) -> int:\n    return z - 1\n"
    fm.add_functions(implementations=src)
    listing = fm.list_functions()
    gamma_id = listing["gamma"]["function_id"]

    # Guidance referencing the image and the function
    gm = GuidanceManager()
    gid = gm._add_guidance(
        title="Gamma Visual",
        content="Use gamma with visual aid.",
        images={"[0:1]": int(img_id)},
        function_ids=[gamma_id],
    )["details"]["guidance_id"]

    # Image handles from FunctionManager
    handles = fm._get_image_handles_for_function_guidance(function_id=gamma_id)
    assert handles and handles[0].image_id == int(img_id)
    # Validate raw returns image bytes (PNG or JPEG header)
    raw = handles[0].raw()
    assert isinstance(raw, bytes) and len(raw) > 0
    head = raw[:10]
    assert head.startswith(b"\xff\xd8") or head.startswith(b"\x89PNG\r\n\x1a\n")

    # Attach images into loop payload
    payload = fm._attach_guidance_images_for_function_to_context(
        function_id=gamma_id,
        limit=1,
    )
    assert isinstance(payload, dict)
    assert payload.get("attached_count") == 1
    assert isinstance(payload.get("images"), list) and len(payload["images"]) == 1
    first_meta = payload["images"][0]["meta"]
    assert first_meta.get("image_id") == int(img_id)
