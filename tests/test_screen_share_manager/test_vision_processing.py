import pytest
from unittest.mock import patch

from unity.screen_share_manager.screen_share_manager import ScreenShareManager
from tests.helpers import _handle_project
from tests.test_screen_share_manager.conftest import load_asset_image, PNG_BLUE_B64


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_frame_handling_should_drop_frames_when_queue_is_full(caplog):
    """Tests that frames are proactively dropped when the processing queue is backlogged."""
    manager = ScreenShareManager()
    await manager.start()
    # Mock queue size to simulate a backlog
    with patch.object(manager._frame_queue, "qsize") as mock_qsize:
        mock_qsize.return_value = int(manager.settings.frame_queue_size * 0.8) + 1
        await manager.push_frame(PNG_BLUE_B64, 1.0)
    assert "proactively dropping frame" in caplog.text.lower()
    await manager.stop()


@pytest.mark.vision
@_handle_project
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_pair",
    [
        ("modal_before.png", "modal_after.png"),
        ("button_active_before.png", "button_active_after.png"),
    ],
)
async def test_vision_should_detect_significant_changes(
    manager: ScreenShareManager, image_pair
):
    """Tests that the vision pipeline correctly identifies significant UI changes."""
    before_filename, after_filename = image_pair
    img_before = load_asset_image(before_filename)
    img_after = load_asset_image(after_filename)
    assert manager._is_significant_visual_change(img_before, img_after)


@pytest.mark.vision
@_handle_project
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image_pair",
    [
        ("blinking_caret_before.png", "blinking_caret_after.png"),
        ("cursor_move_before.png", "cursor_move_after.png"),
    ],
)
async def test_vision_should_ignore_insignificant_changes(
    manager: ScreenShareManager, image_pair
):
    """Tests that the vision pipeline correctly ignores insignificant visual noise."""
    before_filename, after_filename = image_pair
    img_before = load_asset_image(before_filename)
    img_after = load_asset_image(after_filename)
    assert not manager._is_significant_visual_change(img_before, img_after)


@pytest.mark.vision
@_handle_project
@pytest.mark.asyncio
async def test_vision_should_ignore_identical_images(manager):
    """Tests that identical images are not considered a significant change."""
    img = load_asset_image("modal_before.png")
    assert not manager._is_significant_visual_change(img, img.copy())
