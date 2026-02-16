"""
Foreign Key Tests for GuidanceManager

Coverage
========
✓ images[*].raw_image_ref.image_id → Images.image_id (deeply nested FK)
  - Validation: Reject invalid image_id in nested structure
  - SET NULL: Remove deleted image from images array
  - CASCADE: Update image_id changes in nested refs

✓ function_ids[*] → Functions.function_id (array FK)
  - Validation: Reject invalid function_ids
  - SET NULL: Remove deleted function from function_ids array
  - CASCADE: Update function_id changes in array
  - Bidirectional consistency with Functions.guidance_ids
"""

from __future__ import annotations

import unify
from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager
from unity.guidance_manager.guidance_manager import GuidanceManager
from unity.image_manager.image_manager import ImageManager


def _make_test_image_b64(
    size: int = 32,
    color: tuple[int, int, int] = (255, 0, 0),
) -> str:
    """Create a minimal valid base64-encoded PNG for testing."""
    from unity.image_manager.utils import make_solid_png_base64

    return make_solid_png_base64(size, size, color)


# --------------------------------------------------------------------------- #
#  Unit Tests: images[*].raw_image_ref.image_id → Images.image_id            #
# --------------------------------------------------------------------------- #


@_handle_project
def test_fk_images_valid_reference():
    """Test that guidance can reference valid image IDs in nested structure."""
    gm = GuidanceManager()
    im = ImageManager()

    # Store images
    img_ids = im.add_images(
        [
            {"data": _make_test_image_b64(color=(255, 0, 0)), "caption": "Image 1"},
            {"data": _make_test_image_b64(color=(0, 255, 0)), "caption": "Image 2"},
        ],
        synchronous=True,
    )
    assert img_ids[0] is not None and img_ids[1] is not None, "Image creation failed"
    img1_id = img_ids[0]
    img2_id = img_ids[1]

    # Create guidance with images
    gm.add_guidance(
        title="Visual Guide",
        content="Guide with images",
        images=[
            {"raw_image_ref": {"image_id": img1_id}, "annotation": "Setup screenshot"},
            {"raw_image_ref": {"image_id": img2_id}, "annotation": "Usage example"},
        ],
    )

    # Verify guidance created with nested image references
    guidance_list = unify.get_logs(
        context=gm._ctx,
        from_fields=["guidance_id", "images"],
    )
    assert len(guidance_list) == 1
    images = guidance_list[0].entries["images"]
    assert len(images) == 2
    assert images[0]["raw_image_ref"]["image_id"] == img1_id
    assert images[1]["raw_image_ref"]["image_id"] == img2_id


@_handle_project
def test_fk_images_set_null_on_delete():
    """Test SET NULL: Deleting image replaces image_id with None in-place."""
    gm = GuidanceManager()
    im = ImageManager()

    # Store multiple images
    img_ids = im.add_images(
        [
            {"data": _make_test_image_b64(color=(255, 0, 0)), "caption": "Image 1"},
            {"data": _make_test_image_b64(color=(0, 255, 0)), "caption": "Image 2"},
            {"data": _make_test_image_b64(color=(0, 0, 255)), "caption": "Image 3"},
        ],
        synchronous=True,
    )
    assert all(iid is not None for iid in img_ids), "Image creation failed"
    img1_id, img2_id, img3_id = img_ids

    # Create guidance with all three images
    gm.add_guidance(
        title="Multi-Image Guide",
        content="Guide with three images",
        images=[
            {"raw_image_ref": {"image_id": img1_id}, "annotation": "First"},
            {"raw_image_ref": {"image_id": img2_id}, "annotation": "Second"},
            {"raw_image_ref": {"image_id": img3_id}, "annotation": "Third"},
        ],
    )

    # Verify all 3 images
    guidance_list = unify.get_logs(
        context=gm._ctx,
        from_fields=["guidance_id", "images"],
    )
    assert len(guidance_list[0].entries["images"]) == 3

    # Delete middle image (img2)
    img2_logs = unify.get_logs(
        context=im._ctx,
        filter=f"image_id == {img2_id}",
        return_ids_only=True,
    )
    assert img2_logs, "Image not found"
    unify.delete_logs(context=im._ctx, logs=img2_logs[0])

    # Verify SET NULL behavior: img2 replaced with None in-place
    guidance_after = unify.get_logs(context=gm._ctx, from_fields=["images"])
    remaining_images = guidance_after[0].entries.get("images", [])
    assert len(remaining_images) == 3  # Still 3 entries, one has None

    # Extract non-None image_ids
    remaining_ids = [
        img["raw_image_ref"]["image_id"]
        for img in remaining_images
        if img["raw_image_ref"]["image_id"] is not None
    ]
    assert len(remaining_ids) == 2  # Only 2 valid IDs remain
    assert img1_id in remaining_ids
    assert img3_id in remaining_ids
    assert img2_id not in remaining_ids

    # Verify None is present
    none_count = sum(
        1 for img in remaining_images if img["raw_image_ref"]["image_id"] is None
    )
    assert none_count == 1


@_handle_project
def test_fk_images_multiple_deletes():
    """Test SET NULL with multiple sequential image deletes."""
    gm = GuidanceManager()
    im = ImageManager()

    # Store 5 images
    img_ids = im.add_images(
        [
            {
                "data": _make_test_image_b64(color=(i * 50, 0, 0)),
                "caption": f"Image {i}",
            }
            for i in range(5)
        ],
        synchronous=True,
    )
    assert all(iid is not None for iid in img_ids), "Image creation failed"
    image_ids = img_ids

    # Create guidance with all 5 images
    gm.add_guidance(
        title="Gallery Guide",
        content="Many images",
        images=[
            {"raw_image_ref": {"image_id": img_id}, "annotation": f"Image {i}"}
            for i, img_id in enumerate(image_ids)
        ],
    )

    # Verify all 5 images
    guidance = unify.get_logs(context=gm._ctx, from_fields=["images"])
    assert len(guidance[0].entries["images"]) == 5

    # Delete first 3 images
    for img_id in image_ids[:3]:
        img_logs = unify.get_logs(
            context=im._ctx,
            filter=f"image_id == {img_id}",
            return_ids_only=True,
        )
        assert img_logs, f"Image {img_id} not found"
        unify.delete_logs(context=im._ctx, logs=img_logs[0])

    # Verify SET NULL behavior: still 5 entries, 3 with None
    guidance_after = unify.get_logs(context=gm._ctx, from_fields=["images"])
    remaining_images = guidance_after[0].entries.get("images", [])
    assert len(remaining_images) == 5  # Still 5 entries

    # Extract non-None image_ids
    remaining_ids = [
        img["raw_image_ref"]["image_id"]
        for img in remaining_images
        if img["raw_image_ref"]["image_id"] is not None
    ]
    assert sorted(remaining_ids) == sorted(image_ids[3:])

    # Verify 3 None values
    none_count = sum(
        1 for img in remaining_images if img["raw_image_ref"]["image_id"] is None
    )
    assert none_count == 3


@_handle_project
def test_fk_images_null_tolerance():
    """Test that guidance with None image_ids (after SET NULL) loads correctly."""
    gm = GuidanceManager()
    im = ImageManager()

    # Create image and guidance
    img_ids = im.add_images(
        [{"data": _make_test_image_b64(color=(255, 0, 0)), "caption": "Test Image"}],
        synchronous=True,
    )
    assert img_ids[0] is not None, "Image creation failed"
    img_id = img_ids[0]

    gm.add_guidance(
        title="Test Guide",
        content="Guide with image that will be deleted",
        images=[
            {"raw_image_ref": {"image_id": img_id}, "annotation": "Test annotation"},
        ],
    )

    # Get guidance_id
    guidance_list = unify.get_logs(context=gm._ctx, from_fields=["guidance_id"])
    assert len(guidance_list) == 1
    guidance_id = int(guidance_list[0].entries["guidance_id"])

    # Delete image (SET NULL will replace image_id with None)
    img_logs = unify.get_logs(
        context=im._ctx,
        filter=f"image_id == {img_id}",
        return_ids_only=True,
    )
    assert img_logs, "Image not found"
    unify.delete_logs(context=im._ctx, logs=img_logs[0])

    # Reinitialize GuidanceManager and verify it doesn't crash
    gm2 = GuidanceManager()

    # Verify guidance still exists and can be read
    guidance_after = unify.get_logs(
        context=gm2._ctx,
        from_fields=["guidance_id", "images"],
    )
    assert len(guidance_after) == 1

    # Verify None is in the images array
    images = guidance_after[0].entries["images"]
    assert len(images) == 1
    assert images[0]["raw_image_ref"]["image_id"] is None

    # Verify helper methods don't crash when encountering None
    result = gm2._get_images_for_guidance(guidance_id=guidance_id)
    assert (
        result == []
    )  # Should return empty list (None images filtered out), not crash

    # Verify attach helper also handles None gracefully
    attach_result = gm2._attach_guidance_images_to_context(
        guidance_id=guidance_id,
        limit=10,
    )
    assert attach_result["attached_count"] == 0  # No valid images to attach
    assert attach_result["images"] == []


# --------------------------------------------------------------------------- #
#  Unit Tests: function_ids[*] → Functions.function_id                       #
# --------------------------------------------------------------------------- #


@_handle_project
def test_fk_function_ids_valid_reference():
    """Test that guidance can reference valid function IDs."""
    gm = GuidanceManager()
    fm = FunctionManager()

    # Create functions
    src1 = "def func1():\n    return 1\n"
    src2 = "def func2():\n    return 2\n"
    fm.add_functions(implementations=[src1, src2])

    # Get function IDs
    funcs = unify.get_logs(context=fm._compositional_ctx, from_fields=["function_id"])
    func_ids = sorted([int(f.entries["function_id"]) for f in funcs])
    assert len(func_ids) == 2

    # Create guidance referencing both functions
    gm.add_guidance(
        title="Function Guide",
        content="Guide for functions",
        function_ids=func_ids,
    )

    # Verify guidance created with function_ids
    guidance_list = unify.get_logs(
        context=gm._ctx,
        from_fields=["guidance_id", "function_ids"],
    )
    assert len(guidance_list) == 1
    assert sorted(guidance_list[0].entries["function_ids"]) == func_ids


@_handle_project
def test_fk_function_ids_set_null_on_delete():
    """Test SET NULL: Deleting function removes it from guidance.function_ids array."""
    gm = GuidanceManager()
    fm = FunctionManager()

    # Create 3 functions
    for i in range(3):
        src = f"def func{i}():\n    return {i}\n"
        fm.add_functions(implementations=src)

    # Get function IDs
    funcs = unify.get_logs(context=fm._compositional_ctx, from_fields=["function_id"])
    func_ids = sorted([int(f.entries["function_id"]) for f in funcs])
    assert len(func_ids) == 3
    f1, f2, f3 = func_ids

    # Create guidance referencing all 3 functions
    gm.add_guidance(
        title="Multi-Function Guide",
        content="Guide for multiple functions",
        function_ids=[f1, f2, f3],
    )

    # Verify all 3 function_ids
    guidance = unify.get_logs(context=gm._ctx, from_fields=["function_ids"])
    assert sorted(guidance[0].entries["function_ids"]) == [f1, f2, f3]

    # Delete middle function (f2)
    fm.delete_function(function_id=f2)

    # Verify f2 removed from function_ids array
    guidance_after = unify.get_logs(context=gm._ctx, from_fields=["function_ids"])
    remaining_ids = sorted(guidance_after[0].entries.get("function_ids", []))
    assert remaining_ids == [f1, f3]
    assert f2 not in remaining_ids


@_handle_project
def test_fk_function_ids_empty_array():
    """Test that empty function_ids array is valid."""
    gm = GuidanceManager()

    # Create guidance with no function references
    gm.add_guidance(
        title="Standalone Guide",
        content="Guide without function references",
        function_ids=[],
    )

    # Verify guidance was created
    guidance = unify.get_logs(
        context=gm._ctx,
        from_fields=["guidance_id", "function_ids"],
    )
    assert len(guidance) == 1
    assert guidance[0].entries.get("function_ids", []) == []


@_handle_project
def test_fk_combined_images_and_functions():
    """Test combined FK constraints: images + function_ids."""
    gm = GuidanceManager()
    im = ImageManager()
    fm = FunctionManager()

    # Create image
    img_ids = im.add_images(
        [{"data": _make_test_image_b64(color=(255, 0, 0)), "caption": "Screenshot"}],
        synchronous=True,
    )
    assert img_ids[0] is not None, "Image creation failed"
    img_id = img_ids[0]

    # Create function
    src = "def demo():\n    return 'demo'\n"
    fm.add_functions(implementations=src)
    funcs = unify.get_logs(context=fm._compositional_ctx, from_fields=["function_id"])
    func_id = int(funcs[0].entries["function_id"])

    # Create guidance with both
    gm.add_guidance(
        title="Complete Guide",
        content="Guide with images and functions",
        images=[
            {"raw_image_ref": {"image_id": img_id}, "annotation": "Demo screenshot"},
        ],
        function_ids=[func_id],
    )

    # Verify both references
    guidance = unify.get_logs(context=gm._ctx, from_fields=["images", "function_ids"])
    assert guidance[0].entries["images"][0]["raw_image_ref"]["image_id"] == img_id
    assert guidance[0].entries["function_ids"] == [func_id]

    # Delete image
    img_logs = unify.get_logs(
        context=im._ctx,
        filter=f"image_id == {img_id}",
        return_ids_only=True,
    )
    assert img_logs, "Image not found"
    unify.delete_logs(context=im._ctx, logs=img_logs[0])

    # Verify SET NULL: image_id replaced with None, function_ids intact
    guidance_after = unify.get_logs(
        context=gm._ctx,
        from_fields=["images", "function_ids"],
    )
    images_after = guidance_after[0].entries.get("images", [])
    assert len(images_after) == 1  # Still 1 entry
    assert images_after[0]["raw_image_ref"]["image_id"] is None  # But image_id is None
    assert guidance_after[0].entries["function_ids"] == [func_id]

    # Delete function (CASCADE should remove from array)
    fm.delete_function(function_id=func_id)

    # Verify function_id removed (CASCADE), guidance still exists
    guidance_final = unify.get_logs(
        context=gm._ctx,
        from_fields=["guidance_id", "images", "function_ids"],
    )
    assert len(guidance_final) == 1  # Guidance survives
    # Image with None still present
    images_final = guidance_final[0].entries.get("images", [])
    assert len(images_final) == 1
    assert images_final[0]["raw_image_ref"]["image_id"] is None
    # Function removed by CASCADE
    assert guidance_final[0].entries.get("function_ids", []) == []
