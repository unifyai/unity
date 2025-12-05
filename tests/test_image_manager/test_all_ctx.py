"""Tests for All/Images context mirroring and private field injection."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import patch

import unify
from tests.helpers import _handle_project
from unity.common.log_utils import _derive_all_context
from unity.image_manager.image_manager import ImageManager
from unity.image_manager.utils import make_solid_png_base64


PNG_RED_B64 = make_solid_png_base64(8, 8, (255, 0, 0))
PNG_BLUE_B64 = make_solid_png_base64(8, 8, (0, 0, 255))


def _get_raw_log_by_image_id(ctx: str, image_id: int):
    """Get raw log entry including private fields."""
    logs = unify.get_logs(
        context=ctx,
        filter=f"image_id == {image_id}",
        limit=1,
    )
    return logs[0] if logs else None


@_handle_project
def test_log_creates_all_images_entry():
    """Creating an image should mirror to All/<Ctx>."""
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

    # Derive the All/<Ctx> context from the manager's context
    all_ctx = _derive_all_context(im._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify it was mirrored to All/<Ctx>
    all_logs = unify.get_logs(
        context=all_ctx,
        filter=f"image_id == {image_id}",
    )
    assert len(all_logs) >= 1, f"Image should be mirrored to {all_ctx}"


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
    """Logs should have _user_id field when USER_ID env is set."""
    test_user_id = "test-user-456"

    with patch.dict(os.environ, {"USER_ID": test_user_id}):
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
def test_all_context_created_on_provision():
    """All/<Ctx> context should be created when ImageManager provisions storage."""
    # ImageManager provisions storage via ContextRegistry.get_context() in __init__
    im = ImageManager()

    # Derive the expected All/<Ctx> context
    all_ctx = _derive_all_context(im._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify All/<Ctx> exists
    contexts = unify.get_contexts()
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
    assert not hasattr(image, "_assistant"), "_assistant should not be exposed"
    assert not hasattr(image, "_assistant_id"), "_assistant_id should not be exposed"
    assert not hasattr(image, "_user_id"), "_user_id should not be exposed"


@_handle_project
def test_deleting_image_removes_from_all_ctx():
    """Deleting an image should also remove it from All/<Ctx>."""
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

    # Derive the All/<Ctx> context
    all_ctx = _derive_all_context(im._ctx)
    assert all_ctx is not None, "All context should be derivable"

    # Verify it exists in All/<Ctx> before deletion
    all_logs_before = unify.get_logs(
        context=all_ctx,
        filter=f"image_id == {image_id}",
    )
    assert len(all_logs_before) >= 1, "Image should exist in All/<Ctx> before deletion"

    # Delete the image using unify.delete_logs
    logs_to_delete = unify.get_logs(
        context=im._ctx,
        filter=f"image_id == {image_id}",
    )
    if logs_to_delete:
        log_ids = [lg.id for lg in logs_to_delete]
        unify.delete_logs(logs=log_ids, context=im._ctx)

    # Verify it's removed from All/<Ctx> after deletion
    all_logs_after = unify.get_logs(
        context=all_ctx,
        filter=f"image_id == {image_id}",
    )
    assert (
        len(all_logs_after) == 0
    ), "Image should be removed from All/<Ctx> after deletion"
