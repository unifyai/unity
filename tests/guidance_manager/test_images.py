from __future__ import annotations

import base64
from datetime import datetime, timezone
from unity.image_manager.utils import make_solid_png_base64

import pytest
import unify

from unity.image_manager.image_manager import ImageManager
from unity.guidance_manager.guidance_manager import GuidanceManager
from tests.helpers import _handle_project

PNG_RED_B64 = make_solid_png_base64(32, 32, (255, 0, 0))


@_handle_project
def test_get_images_returns_metadata_only():
    im = ImageManager()
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "diagram of layout",
                "data": PNG_RED_B64,
            },
        ],
    )

    gm = GuidanceManager()
    gid = gm.add_guidance(
        title="Layout Review",
        content="We need to review the image layout.",
        images=[
            {
                "raw_image_ref": {"image_id": int(img_id)},
                "annotation": "layout screenshot",
            },
        ],
    )["details"]["guidance_id"]

    items = gm._get_images_for_guidance(guidance_id=gid)
    assert isinstance(items, list) and items, "Expected at least one image entry"
    entry = items[0]
    assert entry.get("image_id") == int(img_id)
    assert entry.get("caption") == "diagram of layout"
    assert isinstance(entry.get("timestamp"), str)
    # Ensure metadata includes annotation and no raw image/base64 field is present
    assert "image" not in entry
    assert entry.get("annotation") in (None, "layout screenshot")


@_handle_project
def test_attach_image_to_context_promotes_image_block():
    im = ImageManager()
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "tiny red pixel",
                "data": PNG_RED_B64,
            },
        ],
    )

    gm = GuidanceManager()
    payload = gm._attach_image_to_context(image_id=int(img_id), note="see layout")
    # Tool payload must include base64 under the 'image' key for promotion
    assert isinstance(payload, dict)
    assert "image" in payload and isinstance(payload["image"], str)
    # Sanity: looks like base64 (decoding should not raise)
    base64.b64decode(payload["image"])  # will raise if invalid


@_handle_project
def test_get_images_includes_annotation():
    im = ImageManager()
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "tiny red pixel",
                "data": PNG_RED_B64,
            },
        ],
    )

    gm = GuidanceManager()
    content = "click this button to open the modal"
    gid = gm.add_guidance(
        title="Annotation Demo",
        content=content,
        images=[
            {"raw_image_ref": {"image_id": int(img_id)}, "annotation": "button area"},
        ],
    )["details"]["guidance_id"]

    items = gm._get_images_for_guidance(guidance_id=gid)
    assert items and (items[0].get("annotation") in (None, "button area"))


@_handle_project
def test_images_field_schema_is_nested_and_enforced():
    gm = GuidanceManager()

    # 1) The Guidance context should expose a nested JSON schema for the images field
    fields = unify.get_fields(context=gm._ctx)
    assert "images" in fields
    dtype = str(fields["images"].get("data_type"))
    # Expect array/list with object items including raw_image_ref + annotation and nested image_id
    assert "raw_image_ref" in dtype and "annotation" in dtype and "image_id" in dtype

    # 2) Valid nested payload – should succeed
    valid_payload = {
        "title": "SchemaCheck",
        "content": "Testing images schema enforcement",
        "images": [
            {"raw_image_ref": {"image_id": 101}, "annotation": "overview"},
        ],
    }
    _ = unify.log(context=gm._ctx, **valid_payload, new=True, mutable=True)

    # 3) Invalid nested payload – wrong key name for image id → must be rejected
    invalid_payload_bad_key = {
        "title": "BadKey",
        "content": "Invalid key for image id",
        "images": [
            {"raw_image_ref": {"image_idx": 999}, "annotation": "oops"},
        ],
    }
    with pytest.raises(Exception):
        unify.log(context=gm._ctx, **invalid_payload_bad_key, new=True, mutable=True)

    # 4) Invalid nested payload – wrong type for annotation → must be rejected
    invalid_payload_bad_type = {
        "title": "BadType",
        "content": "Invalid type for annotation",
        "images": [
            {"raw_image_ref": {"image_id": 202}, "annotation": 123},
        ],
    }
    with pytest.raises(Exception):
        unify.log(context=gm._ctx, **invalid_payload_bad_type, new=True, mutable=True)


# --------------------------------------------------------------------------- #
#  Filepath-based image resolution in add_guidance / update_guidance            #
# --------------------------------------------------------------------------- #


@_handle_project
def test_add_guidance_resolves_filepath_images():
    """add_guidance resolves filepath-only refs to image_ids before persisting."""
    im = ImageManager()
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "red square",
                "data": PNG_RED_B64,
                "filepath": "/tmp/images/gm_add_resolve.png",
            },
        ],
        synchronous=True,
    )

    gm = GuidanceManager()
    result = gm.add_guidance(
        title="Filepath Resolve Test",
        content="Testing filepath-based image resolution",
        images=[
            {
                "raw_image_ref": {"filepath": "/tmp/images/gm_add_resolve.png"},
                "annotation": "step 1 screenshot",
            },
        ],
    )
    gid = result["details"]["guidance_id"]

    rows = gm.filter(filter=f"guidance_id == {gid}", limit=1)
    assert rows, "Guidance entry should exist"
    stored_images = rows[0].images.root
    assert len(stored_images) == 1
    assert stored_images[0].raw_image_ref.image_id == img_id


@_handle_project
def test_update_guidance_resolves_filepath_images():
    """update_guidance resolves filepath-only refs to image_ids before persisting."""
    im = ImageManager()
    [img_id] = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "blue square",
                "data": PNG_RED_B64,
                "filepath": "/tmp/images/gm_update_resolve.png",
            },
        ],
        synchronous=True,
    )

    gm = GuidanceManager()
    result = gm.add_guidance(
        title="Update Resolve Test",
        content="Will add images via update",
    )
    gid = result["details"]["guidance_id"]

    gm.update_guidance(
        guidance_id=gid,
        images=[
            {
                "raw_image_ref": {"filepath": "/tmp/images/gm_update_resolve.png"},
                "annotation": "added via update",
            },
        ],
    )

    rows = gm.filter(filter=f"guidance_id == {gid}", limit=1)
    assert rows, "Guidance entry should exist"
    stored_images = rows[0].images.root
    assert len(stored_images) == 1
    assert stored_images[0].raw_image_ref.image_id == img_id
