from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.helpers import _handle_project
from unity.image_manager.types import (
    Image,
    RawImageRef,
    AnnotatedImageRef,
    ImageRefs,
)
from unity.image_manager.utils import make_solid_png_base64


@pytest.mark.unit
@_handle_project
def test_annotated_image_basic():
    img = Image(
        timestamp=datetime.now(timezone.utc),
        caption="a red square",
        data=make_solid_png_base64(4, 4, (255, 0, 0)),
    )

    ann = AnnotatedImageRef(
        raw_image_ref=RawImageRef(image=img),
        annotation="training set example: high relevance",
    )

    assert ann.raw_image_ref.image.caption == "a red square"
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
    base1_ref = RawImageRef(image=base1)
    base2_ref = RawImageRef(image=base2)
    annotated = AnnotatedImageRef(
        raw_image_ref=base2_ref,
        annotation="used in alerting scenario",
    )

    images = ImageRefs.model_validate([base1_ref, annotated, base2_ref])

    assert len(images.root) == 3
    assert isinstance(images.root[0], RawImageRef)
    assert isinstance(images.root[1], AnnotatedImageRef)
    assert isinstance(images.root[2], RawImageRef)
