from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.image_manager.types import (
    RawImageRef,
    AnnotatedImageRef,
    ImageRefs,
)


@pytest.mark.unit
@_handle_project
def test_annotated_image_basic():
    ann = AnnotatedImageRef(
        raw_image_ref=RawImageRef(image_id=123),
        annotation="training set example: high relevance",
    )

    assert ann.raw_image_ref.image_id == 123
    assert ann.annotation.startswith("training set example")


@pytest.mark.unit
@_handle_project
def test_images_container_mixed_types():
    base1_ref = RawImageRef(image_id=1)
    base2_ref = RawImageRef(image_id=2)
    annotated = AnnotatedImageRef(
        raw_image_ref=base2_ref,
        annotation="used in alerting scenario",
    )

    images = ImageRefs.model_validate([base1_ref, annotated, base2_ref])

    assert len(images.root) == 3
    assert isinstance(images.root[0], RawImageRef)
    assert isinstance(images.root[1], AnnotatedImageRef)
    assert isinstance(images.root[2], RawImageRef)
