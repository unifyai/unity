"""Tests for aggregation context mirroring and private field injection."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import unify
from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_contexts
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.utils import make_solid_png_base64

PNG_RED_B64 = make_solid_png_base64(32, 32, (255, 0, 0))
PNG_BLUE_B64 = make_solid_png_base64(32, 32, (0, 0, 255))


def _get_raw_log_by_image_id(ctx: str, image_id: int):
    """Get raw log entry including private fields."""
    logs = unify.get_logs(
        context=ctx,
        filter=f"image_id == {image_id}",
        limit=1,
    )
    return logs[0] if logs else None


@_handle_project
def test_log_creates_all_images_entries():
    """Creating an image should mirror to both aggregation contexts."""
    im = ImageManager()

    # Create an image
    ids = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "Test image for All/Ctx",
                "data": PNG_RED_B64,
            },
        ],
        synchronous=True,
    )
    assert len(ids) >= 1 and ids[0] is not None, "Image should be created"
    image_id = ids[0]

    # Derive both aggregation contexts from the manager's context
    all_ctxs = _derive_all_contexts(im._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it was mirrored to both aggregation contexts
    for all_ctx in all_ctxs:
        all_logs = unify.get_logs(
            context=all_ctx,
            filter=f"image_id == {image_id}",
        )
        assert len(all_logs) >= 1, f"Image should be mirrored to {all_ctx}"


@_handle_project
def test_user_field_injected():
    """Logs should have _user field set to user name."""
    test_user_name = "TestUserName"

    with patch(
        "unity.common.log_utils._get_user_name",
        return_value=test_user_name,
    ):
        im = ImageManager()
        ids = im.add_images(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "caption": "User field test",
                    "data": PNG_RED_B64,
                },
            ],
            synchronous=True,
        )
        assert len(ids) >= 1 and ids[0] is not None
        image_id = ids[0]

        log = _get_raw_log_by_image_id(im._ctx, image_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_user") == test_user_name
        ), f"_user should be '{test_user_name}', got {entries.get('_user')}"


@_handle_project
def test_assistant_field_injected():
    """Logs should have _assistant field set to assistant name."""
    test_assistant_name = "TestAssistantName"

    with patch(
        "unity.common.log_utils._get_assistant_name",
        return_value=test_assistant_name,
    ):
        im = ImageManager()
        ids = im.add_images(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "caption": "Assistant field test",
                    "data": PNG_RED_B64,
                },
            ],
            synchronous=True,
        )
        assert len(ids) >= 1 and ids[0] is not None
        image_id = ids[0]

        log = _get_raw_log_by_image_id(im._ctx, image_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_assistant") == test_assistant_name
        ), f"_assistant should be '{test_assistant_name}', got {entries.get('_assistant')}"


@_handle_project
def test_assistant_id_field_injected():
    """Logs should have _assistant_id field set to assistant's agent_id."""
    test_assistant_id = "test-agent-789"

    with patch(
        "unity.common.log_utils._get_assistant_id",
        return_value=test_assistant_id,
    ):
        im = ImageManager()
        ids = im.add_images(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "caption": "Assistant ID field test",
                    "data": PNG_BLUE_B64,
                },
            ],
            synchronous=True,
        )
        assert len(ids) >= 1 and ids[0] is not None
        image_id = ids[0]

        log = _get_raw_log_by_image_id(im._ctx, image_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_assistant_id") == test_assistant_id
        ), f"_assistant_id should be '{test_assistant_id}', got {entries.get('_assistant_id')}"


@_handle_project
def test_user_id_field_injected():
    """Logs should have _user_id field set to user's ID."""
    test_user_id = "test-user-456"

    with patch(
        "unity.common.log_utils._get_user_id",
        return_value=test_user_id,
    ):
        im = ImageManager()
        ids = im.add_images(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "caption": "User ID field test",
                    "data": PNG_RED_B64,
                },
            ],
            synchronous=True,
        )
        assert len(ids) >= 1 and ids[0] is not None
        image_id = ids[0]

        log = _get_raw_log_by_image_id(im._ctx, image_id)
        assert log is not None, "Log should exist"

        entries = log.entries
        assert (
            entries.get("_user_id") == test_user_id
        ), f"_user_id should be '{test_user_id}', got {entries.get('_user_id')}"


@_handle_project
def test_all_contexts_created_on_provision():
    """Aggregation contexts should be created when ImageManager provisions storage."""
    # ImageManager provisions storage via ContextRegistry.get_context() in __init__
    im = ImageManager()

    # Derive the expected aggregation contexts
    all_ctxs = _derive_all_contexts(im._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify both aggregation contexts exist
    contexts = unify.get_contexts()
    for all_ctx in all_ctxs:
        assert all_ctx in contexts, f"{all_ctx} context should be created"


@_handle_project
def test_private_fields_excluded_from_filter_images():
    """Private fields should be excluded when reading images via public API."""
    im = ImageManager()

    ids = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "Private field exclusion test",
                "data": PNG_RED_B64,
            },
        ],
        synchronous=True,
    )
    assert len(ids) >= 1 and ids[0] is not None
    image_id = ids[0]

    # Get image via public filter_images API
    images = im.filter_images(filter=f"image_id == {image_id}")
    assert len(images) >= 1

    image = images[0]
    # Private fields should NOT be in the Image model (they're excluded on read)
    assert not hasattr(image, "_user"), "_user should not be exposed"
    assert not hasattr(image, "_user_id"), "_user_id should not be exposed"
    assert not hasattr(image, "_assistant"), "_assistant should not be exposed"
    assert not hasattr(image, "_assistant_id"), "_assistant_id should not be exposed"


@_handle_project
def test_deleting_image_removes_from_all_ctxs():
    """Deleting an image should also remove it from all aggregation contexts."""
    im = ImageManager()

    # Create an image
    ids = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "Image to be deleted",
                "data": PNG_BLUE_B64,
            },
        ],
        synchronous=True,
    )
    assert len(ids) >= 1 and ids[0] is not None
    image_id = ids[0]

    # Derive the aggregation contexts
    all_ctxs = _derive_all_contexts(im._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify it exists in all aggregation contexts before deletion
    for all_ctx in all_ctxs:
        all_logs_before = unify.get_logs(
            context=all_ctx,
            filter=f"image_id == {image_id}",
        )
        assert (
            len(all_logs_before) >= 1
        ), f"Image should exist in {all_ctx} before deletion"

    # Delete the image using unify.delete_logs
    logs_to_delete = unify.get_logs(
        context=im._ctx,
        filter=f"image_id == {image_id}",
    )
    if logs_to_delete:
        log_ids = [lg.id for lg in logs_to_delete]
        unify.delete_logs(logs=log_ids, context=im._ctx)

    # Verify it's removed from all aggregation contexts after deletion
    for all_ctx in all_ctxs:
        all_logs_after = unify.get_logs(
            context=all_ctx,
            filter=f"image_id == {image_id}",
        )
        assert (
            len(all_logs_after) == 0
        ), f"Image should be removed from {all_ctx} after deletion"


@_handle_project
def test_update_syncs_to_all_aggregation_contexts():
    """Updating an image should be immediately visible in all aggregation contexts."""
    im = ImageManager()

    # Create an image with initial values
    ids = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "Original caption",
                "data": PNG_RED_B64,
            },
        ],
        synchronous=True,
    )
    assert len(ids) >= 1 and ids[0] is not None
    image_id = ids[0]

    # Derive aggregation contexts
    all_ctxs = _derive_all_contexts(im._ctx)
    assert len(all_ctxs) == 2, "Should have user-level and global aggregation contexts"

    # Verify initial caption in all contexts
    for ctx in [im._ctx, *all_ctxs]:
        log = _get_raw_log_by_image_id(ctx, image_id)
        assert log is not None, f"Log should exist in {ctx}"
        assert (
            log.entries.get("caption") == "Original caption"
        ), f"Initial caption in {ctx}"

    # Update the image's caption
    im.update_images([{"image_id": image_id, "caption": "Updated caption"}])

    # Verify the update is immediately visible in ALL contexts (primary + aggregations)
    for ctx in [im._ctx, *all_ctxs]:
        log = _get_raw_log_by_image_id(ctx, image_id)
        assert log is not None, f"Log should exist in {ctx} after update"
        assert log.entries.get("caption") == "Updated caption", (
            f"Updated caption should be visible in {ctx}. "
            f"Expected 'Updated caption', got '{log.entries.get('caption')}'"
        )


@_handle_project
def test_log_id_unchanged_after_update():
    """Updates should modify the existing log entry, not create a new one."""
    im = ImageManager()

    # Create an image
    ids = im.add_images(
        [
            {
                "timestamp": datetime.now(timezone.utc),
                "caption": "Before update",
                "data": PNG_BLUE_B64,
            },
        ],
        synchronous=True,
    )
    assert len(ids) >= 1 and ids[0] is not None
    image_id = ids[0]

    # Get the original log ID
    original_log = _get_raw_log_by_image_id(im._ctx, image_id)
    original_log_id = original_log.id

    # Update the image
    im.update_images([{"image_id": image_id, "caption": "After update"}])

    # Verify the log ID is unchanged (in-place update, not delete+create)
    updated_log = _get_raw_log_by_image_id(im._ctx, image_id)
    assert updated_log.id == original_log_id, (
        f"Log ID should be unchanged after update. "
        f"Original: {original_log_id}, After update: {updated_log.id}"
    )

    # Verify all aggregation contexts still reference the same log ID
    all_ctxs = _derive_all_contexts(im._ctx)
    for all_ctx in all_ctxs:
        agg_log = _get_raw_log_by_image_id(all_ctx, image_id)
        assert agg_log.id == original_log_id, (
            f"Aggregation context {all_ctx} should still reference the same log. "
            f"Expected {original_log_id}, got {agg_log.id}"
        )
