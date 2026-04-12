"""Verify screenshot registration (ImageManager + TM) is deferred to background.

The slow brain's ``_run_llm`` must NOT block on ``image_manager.add_images`` or
``transcript_manager.update_message_images`` before making the LLM call. These
are purely persistence bookkeeping and run as a fire-and-forget background task
after the turn succeeds and the screenshot buffer is committed.

These are symbolic tests — the LLM call is patched to return a deterministic
``wait`` decision.  They verify the structural property that ImageManager
registration only happens AFTER the turn succeeds, not during it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from unity.conversation_manager.cm_types.screenshot import ScreenshotEntry


def _make_screenshot(mid: int | None = None) -> ScreenshotEntry:
    return ScreenshotEntry(
        b64="AAAA",
        utterance="test utterance",
        timestamp=datetime.now(timezone.utc),
        source="user",
        local_message_id=mid,
        filepath="Screenshots/User/test.jpg",
    )


def _make_unify_message(content: str):
    from tests.conversation_manager.conftest import BOSS
    from unity.conversation_manager.events import UnifyMessageReceived

    return UnifyMessageReceived(contact=BOSS, content=content)


def _patch_single_shot():
    """Patch single_shot_tool_decision to return 'wait' without calling the LLM."""
    from unity.common.single_shot import SingleShotResult, ToolExecution

    async def _fake_single_shot(client, message, tools, **kwargs):
        wait_fn = tools.get("wait")
        result = None
        if wait_fn:
            fn = wait_fn.fn if hasattr(wait_fn, "fn") else wait_fn
            if asyncio.iscoroutinefunction(fn):
                result = await fn()
            else:
                result = fn()
        return SingleShotResult(
            tools=[ToolExecution(name="wait", args={}, result=result)],
            text_response=None,
            structured_output=None,
        )

    return patch(
        "unity.conversation_manager.conversation_manager.single_shot_tool_decision",
        side_effect=_fake_single_shot,
    )


@pytest.mark.asyncio
async def test_image_registration_deferred_to_background(initialized_cm):
    """image_manager.add_images must be called by a background task, not inline
    during the LLM call path.  We verify this by checking the call happens after
    _run_llm returns and pending tasks drain.
    """
    cm_driver = initialized_cm
    cm = cm_driver.cm
    cm.user_screen_share_active = True
    cm._screenshot_buffer.append(_make_screenshot(mid=42))

    add_images_calls: list[tuple] = []
    mock_image_manager = MagicMock()
    mock_image_manager.add_images = MagicMock(
        side_effect=lambda items, **kw: (
            add_images_calls.append(("called",)),
            list(range(len(items))),
        )[1],
    )

    with (
        _patch_single_shot(),
        patch(
            "unity.manager_registry.ManagerRegistry.get_image_manager",
            return_value=mock_image_manager,
        ),
    ):
        result = await cm_driver.step_until_wait(
            _make_unify_message("What do you see on my screen?"),
        )

        assert result.llm_ran, "LLM should have run"
        assert (
            len(cm._screenshot_buffer) == 0
        ), "Screenshot buffer should be committed after successful turn"

        # Let the background task run.
        await asyncio.sleep(0)

    assert (
        len(add_images_calls) > 0
    ), "image_manager.add_images should have been called by the background task"


@pytest.mark.asyncio
async def test_no_registration_when_no_screenshots(initialized_cm):
    """When the screenshot buffer is empty, no background task should fire."""
    cm_driver = initialized_cm
    cm = cm_driver.cm
    assert len(cm._screenshot_buffer) == 0

    mock_image_manager = MagicMock()

    with (
        _patch_single_shot(),
        patch(
            "unity.manager_registry.ManagerRegistry.get_image_manager",
            return_value=mock_image_manager,
        ),
    ):
        result = await cm_driver.step_until_wait(
            _make_unify_message("Hello, how are you?"),
        )

    assert result.llm_ran
    mock_image_manager.add_images.assert_not_called()


@pytest.mark.asyncio
async def test_failed_registration_does_not_affect_turn(initialized_cm):
    """If image_manager.add_images raises, the turn still succeeds and the
    buffer is still committed.
    """
    cm_driver = initialized_cm
    cm = cm_driver.cm
    cm.user_screen_share_active = True
    cm._screenshot_buffer.append(_make_screenshot(mid=42))

    mock_image_manager = MagicMock()
    mock_image_manager.add_images = MagicMock(
        side_effect=RuntimeError("backend down"),
    )

    with (
        _patch_single_shot(),
        patch(
            "unity.manager_registry.ManagerRegistry.get_image_manager",
            return_value=mock_image_manager,
        ),
    ):
        result = await cm_driver.step_until_wait(
            _make_unify_message("What is on my screen?"),
        )

        assert result.llm_ran, "Turn should succeed despite registration failure"
        assert len(cm._screenshot_buffer) == 0, "Buffer should still be committed"

        # Let background task drain — it should handle the exception gracefully.
        await asyncio.sleep(0)
