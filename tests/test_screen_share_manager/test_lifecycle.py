import pytest
from unity.screen_share_manager.screen_share_manager import ScreenShareManager
from tests.helpers import _handle_project

@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_lifecycle_should_handle_idempotent_start_stop():
    """Tests that calling start/stop multiple times does not raise errors."""
    manager = ScreenShareManager()
    await manager.start()
    await manager.start()  # Calling start again should be safe
    await manager.stop()
    await manager.stop()  # Calling stop again should be safe


@pytest.mark.unit
@_handle_project
@pytest.mark.asyncio
async def test_configuration_should_update_settings_dynamically():
    """Tests that changing settings on the manager instance is reflected."""
    manager = ScreenShareManager()
    manager.settings.frame_queue_size = 99
    assert manager.settings.frame_queue_size == 99
    manager.settings.inactivity_timeout_sec = 0.42
    assert manager.settings.inactivity_timeout_sec == 0.42
    await manager.start()
    await manager.stop()