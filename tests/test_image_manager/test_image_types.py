from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from pydantic import ValidationError
from unity.image_manager.types import (
    RawImageRef,
    AnnotatedImageRef,
    ImageRefs,
    RawImageRefs,
    AnnotatedImageRefs,
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


@pytest.mark.unit
@_handle_project
def test_raw_image_refs_accepts_only_raw():
    r1 = RawImageRef(image_id=10)
    r2 = RawImageRef(image_id=11)

    refs = RawImageRefs.model_validate([r1, r2])

    assert len(refs.root) == 2
    assert all(isinstance(x, RawImageRef) for x in refs.root)


@pytest.mark.unit
@_handle_project
def test_raw_image_refs_rejects_annotated():
    ann = AnnotatedImageRef(
        raw_image_ref=RawImageRef(image_id=5),
        annotation="note",
    )

    with pytest.raises(ValidationError):
        RawImageRefs.model_validate([ann])


@pytest.mark.unit
@_handle_project
def test_annotated_image_refs_accepts_only_annotated():
    ann1 = AnnotatedImageRef(
        raw_image_ref=RawImageRef(image_id=21),
        annotation="first",
    )
    ann2 = AnnotatedImageRef(
        raw_image_ref=RawImageRef(image_id=22),
        annotation="second",
    )

    refs = AnnotatedImageRefs.model_validate([ann1, ann2])

    assert len(refs.root) == 2
    assert all(isinstance(x, AnnotatedImageRef) for x in refs.root)


@pytest.mark.unit
@_handle_project
def test_annotated_image_refs_rejects_raw():
    raw = RawImageRef(image_id=99)

    with pytest.raises(ValidationError):
        AnnotatedImageRefs.model_validate([raw])
