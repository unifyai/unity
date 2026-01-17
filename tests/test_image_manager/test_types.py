from __future__ import annotations

import pytest
import unify
from datetime import datetime, UTC
from pydantic import BaseModel, Field
from unity.common.context_store import TableStore
from unity.common.model_to_fields import model_to_fields
from unity.image_manager.utils import make_solid_png_base64
from unity.image_manager.types.image import Image

from tests.helpers import _handle_project
from pydantic import ValidationError
from unity.image_manager.types import (
    RawImageRef,
    AnnotatedImageRef,
    ImageRefs,
    RawImageRefs,
    AnnotatedImageRefs,
)


@_handle_project
def test_annotated_ref_basic():
    ann = AnnotatedImageRef(
        raw_image_ref=RawImageRef(image_id=123),
        annotation="training set example: high relevance",
    )

    assert ann.raw_image_ref.image_id == 123
    assert ann.annotation.startswith("training set example")


@_handle_project
def test_container_mixed_types():
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


@_handle_project
def test_raw_refs_accepts_only_raw():
    r1 = RawImageRef(image_id=10)
    r2 = RawImageRef(image_id=11)

    refs = RawImageRefs.model_validate([r1, r2])

    assert len(refs.root) == 2
    assert all(isinstance(x, RawImageRef) for x in refs.root)


@_handle_project
def test_raw_refs_rejects_annotated():
    ann = AnnotatedImageRef(
        raw_image_ref=RawImageRef(image_id=5),
        annotation="note",
    )

    with pytest.raises(ValidationError):
        RawImageRefs.model_validate([ann])


@_handle_project
def test_annotated_refs_accepts_only_annotated():
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


@_handle_project
def test_annotated_refs_rejects_raw():
    raw = RawImageRef(image_id=99)

    with pytest.raises(ValidationError):
        AnnotatedImageRefs.model_validate([raw])


# --------------------------------------------------------------------------- #
#  Backend schema enforcement for each Pydantic model in types/                #
# --------------------------------------------------------------------------- #


class _RowIdModel(BaseModel):
    row_id: int = Field(default=-1, ge=-1)


@_handle_project
def test_backend_schema_raw_ref_field_enforced():
    class _RawRefRow(_RowIdModel):
        ref: RawImageRef

    # Provision context with nested schema for RawImageRef
    try:
        ctxs = unify.get_active_context()
        base_ctx = ctxs.get("write") if isinstance(ctxs, dict) else None
    except Exception:
        base_ctx = None
    ctx = f"{base_ctx}/SchemaRawImageRef" if base_ctx else "SchemaRawImageRef"
    store = TableStore(
        ctx,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
        description="Schema test for RawImageRef field",
        fields=model_to_fields(_RawRefRow),
    )
    store.ensure_context()

    fields = unify.get_fields(context=ctx)
    assert "ref" in fields and "image_id" in str(fields["ref"].get("data_type"))

    # Valid payload
    valid = {"ref": {"image_id": 123}}
    _ = unify.log(context=ctx, **valid, new=True, mutable=True)

    # Invalid: wrong nested key
    invalid = {"ref": {"image_idx": 123}}
    with pytest.raises(Exception):
        unify.log(context=ctx, **invalid, new=True, mutable=True)


@_handle_project
def test_backend_schema_annotated_ref_field_enforced():
    class _AnnRefRow(_RowIdModel):
        ref: AnnotatedImageRef

    try:
        ctxs = unify.get_active_context()
        base_ctx = ctxs.get("write") if isinstance(ctxs, dict) else None
    except Exception:
        base_ctx = None
    ctx = (
        f"{base_ctx}/SchemaAnnotatedImageRef" if base_ctx else "SchemaAnnotatedImageRef"
    )
    store = TableStore(
        ctx,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
        description="Schema test for AnnotatedImageRef field",
        fields=model_to_fields(_AnnRefRow),
    )
    store.ensure_context()

    fields = unify.get_fields(context=ctx)
    dtype = str(fields["ref"].get("data_type"))
    assert "raw_image_ref" in dtype and "annotation" in dtype and "image_id" in dtype

    # Valid
    valid = {"ref": {"raw_image_ref": {"image_id": 5}, "annotation": "note"}}
    _ = unify.log(context=ctx, **valid, new=True, mutable=True)

    # Invalid key
    bad_key = {"ref": {"raw_image_ref": {"image_idx": 5}, "annotation": "note"}}
    with pytest.raises(Exception):
        unify.log(context=ctx, **bad_key, new=True, mutable=True)

    # Invalid type
    bad_type = {"ref": {"raw_image_ref": {"image_id": 6}, "annotation": 123}}
    with pytest.raises(Exception):
        unify.log(context=ctx, **bad_type, new=True, mutable=True)


@_handle_project
def test_backend_schema_refs_field_enforced():
    class _ImageRefsRow(_RowIdModel):
        refs: ImageRefs

    try:
        ctxs = unify.get_active_context()
        base_ctx = ctxs.get("write") if isinstance(ctxs, dict) else None
    except Exception:
        base_ctx = None
    ctx = f"{base_ctx}/SchemaImageRefs" if base_ctx else "SchemaImageRefs"
    store = TableStore(
        ctx,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
        description="Schema test for ImageRefs field",
        fields=model_to_fields(_ImageRefsRow),
    )
    store.ensure_context()

    fields = unify.get_fields(context=ctx)
    dtype = str(fields["refs"].get("data_type"))
    assert "raw_image_ref" in dtype and "annotation" in dtype and "image_id" in dtype

    # Valid (mixed)
    valid = {
        "refs": [
            {"image_id": 1},
            {"raw_image_ref": {"image_id": 2}, "annotation": "a"},
        ],
    }
    _ = unify.log(context=ctx, **valid, new=True, mutable=True)

    # Invalid element structure
    bad = {"refs": [{"raw_image_ref": {"image_idx": 3}, "annotation": "x"}]}
    with pytest.raises(Exception):
        unify.log(context=ctx, **bad, new=True, mutable=True)


@_handle_project
def test_backend_schema_raw_refs_field_enforced():
    class _RawImageRefsRow(_RowIdModel):
        refs: RawImageRefs

    try:
        ctxs = unify.get_active_context()
        base_ctx = ctxs.get("write") if isinstance(ctxs, dict) else None
    except Exception:
        base_ctx = None
    ctx = f"{base_ctx}/SchemaRawImageRefs" if base_ctx else "SchemaRawImageRefs"
    store = TableStore(
        ctx,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
        description="Schema test for RawImageRefs field",
        fields=model_to_fields(_RawImageRefsRow),
    )
    store.ensure_context()

    fields = unify.get_fields(context=ctx)
    dtype = str(fields["refs"].get("data_type"))
    assert "image_id" in dtype and "raw_image_ref" not in dtype  # raw-only entries

    # Valid raw-only list
    valid = {"refs": [{"image_id": 10}, {"image_id": 11}]}
    _ = unify.log(context=ctx, **valid, new=True, mutable=True)

    # Invalid: annotated entry inside raw-only list
    bad = {"refs": [{"raw_image_ref": {"image_id": 12}, "annotation": "x"}]}
    with pytest.raises(Exception):
        unify.log(context=ctx, **bad, new=True, mutable=True)


@_handle_project
def test_backend_schema_annotated_refs_field_enforced():
    class _AnnotatedImageRefsRow(_RowIdModel):
        refs: AnnotatedImageRefs

    try:
        ctxs = unify.get_active_context()
        base_ctx = ctxs.get("write") if isinstance(ctxs, dict) else None
    except Exception:
        base_ctx = None
    ctx = (
        f"{base_ctx}/SchemaAnnotatedImageRefs"
        if base_ctx
        else "SchemaAnnotatedImageRefs"
    )
    store = TableStore(
        ctx,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
        description="Schema test for AnnotatedImageRefs field",
        fields=model_to_fields(_AnnotatedImageRefsRow),
    )
    store.ensure_context()

    fields = unify.get_fields(context=ctx)
    dtype = str(fields["refs"].get("data_type"))
    assert "raw_image_ref" in dtype and "annotation" in dtype and "image_id" in dtype

    # Valid annotated list
    valid = {"refs": [{"raw_image_ref": {"image_id": 20}, "annotation": "z"}]}
    _ = unify.log(context=ctx, **valid, new=True, mutable=True)

    # Invalid element type for annotation
    bad = {"refs": [{"raw_image_ref": {"image_id": 21}, "annotation": 7}]}
    with pytest.raises(Exception):
        unify.log(context=ctx, **bad, new=True, mutable=True)


@_handle_project
def test_backend_schema_image_field_enforced():
    class _ImageRow(_RowIdModel):
        entry: Image

    try:
        ctxs = unify.get_active_context()
        base_ctx = ctxs.get("write") if isinstance(ctxs, dict) else None
    except Exception:
        base_ctx = None
    ctx = f"{base_ctx}/SchemaImageField" if base_ctx else "SchemaImageField"
    store = TableStore(
        ctx,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
        description="Schema test for Image field",
        fields=model_to_fields(_ImageRow),
    )
    store.ensure_context()

    fields = unify.get_fields(context=ctx)
    dtype = str(fields["entry"].get("data_type"))
    assert "timestamp" in dtype and "data" in dtype

    # Valid payload for Image (use deterministic base64 PNG)
    png_b64 = make_solid_png_base64(32, 32, (1, 2, 3))
    valid = {
        "entry": {
            "timestamp": datetime.now(UTC).isoformat(),
            "caption": "tiny sample",
            "data": png_b64,
        },
    }
    _ = unify.log(context=ctx, **valid, new=True, mutable=True)

    # Invalid: missing required key 'data'
    bad_missing = {
        "entry": {
            "timestamp": datetime.now(UTC).isoformat(),
            "caption": "no data",
        },
    }
    with pytest.raises(Exception):
        unify.log(context=ctx, **bad_missing, new=True, mutable=True)
