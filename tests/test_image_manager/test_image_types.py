from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.helpers import _handle_project
from unity.image_manager.types.image import Image, AnnotatedImage, Images
from unity.image_manager.utils import make_solid_png_base64


@pytest.mark.unit
@_handle_project
def test_annotated_image_basic():
    img = Image(
        timestamp=datetime.now(timezone.utc),
        caption="a red square",
        data=make_solid_png_base64(4, 4, (255, 0, 0)),
    )

    ann = AnnotatedImage(image=img, annotation="training set example: high relevance")

    assert ann.image.caption == "a red square"
    assert ann.annotation.startswith("training set example")


@pytest.mark.unit
@_handle_project
def test_images_container_mixed_types():
    base1 = Image(
        timestamp=datetime.now(timezone.utc),
        caption="blue",
        data=make_solid_png_base64(2, 2, (0, 0, 255)),
    )
    base2 = Image(
        timestamp=datetime.now(timezone.utc),
        caption="red",
        data=make_solid_png_base64(2, 2, (255, 0, 0)),
    )
    annotated = AnnotatedImage(image=base2, annotation="used in alerting scenario")

    images = Images.model_validate([base1, annotated, base2])

    assert len(images.root) == 3
    assert isinstance(images.root[0], Image)
    assert isinstance(images.root[1], AnnotatedImage)
    assert isinstance(images.root[2], Image)
