import asyncio

import pytest
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock

from unity.actor.code_act_actor import CodeActActor
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.types import AnnotatedImageRef, RawImageRef, ImageRefs
from unity.image_manager.utils import make_solid_png_base64


class ImageCountModel(BaseModel):
    count: int = Field(description="How many images were provided.")


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_code_act_images_available_via_live_images_overview():
    """
    If images are passed to CodeActActor.act(images=...), the loop should expose them
    via the live-images overview (and the model should be able to count them).
    """
    ImageCountModel.model_rebuild()

    im = ImageManager()
    blue = make_solid_png_base64(16, 16, (0, 0, 255))
    red = make_solid_png_base64(16, 16, (255, 0, 0))
    ids = im.add_images(
        [
            {"data": blue, "caption": "blue square"},
            {"data": red, "caption": "red square"},
        ],
        synchronous=True,
        return_handles=False,
    )
    assert isinstance(ids, list) and len(ids) == 2
    iid1, iid2 = int(ids[0]), int(ids[1])

    images = ImageRefs.model_validate(
        [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=iid1),
                annotation="blue",
            ),
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=iid2),
                annotation="red",
            ),
        ],
    )

    actor = CodeActActor(headless=True, computer_mode="mock", timeout=90)
    actor._computer_primitives.navigate = AsyncMock(return_value=None)
    actor._computer_primitives.act = AsyncMock(return_value="Action completed")
    actor._computer_primitives.observe = AsyncMock(return_value="Page content observed")

    handle = await actor.act(
        "How many images were provided to you? Return {count: <int>}.\n"
        "You may answer directly; do not call execute_code.",
        clarification_enabled=False,
        response_format=ImageCountModel,
        images=images,
        persist=False,
    )
    try:
        res = await asyncio.wait_for(handle.result(), timeout=150)
        assert isinstance(res, ImageCountModel)
        assert res.count == 2
    finally:
        try:
            await actor.close()
        except Exception:
            pass
