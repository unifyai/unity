import asyncio
import base64
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path

import pytest
from PIL import Image

from unity.image_manager.image_manager import ImageHandle
from unity.screen_share_manager.screen_share_manager import ScreenShareManager
from unity.screen_share_manager.types import DetectedEvent

PNG_BLUE_B64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNkYPhfz/w3A5MBA/8/AAYDAL4/7d4eAAAAAElFTSuQmCC"
PNG_RED_B64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8z8AARf4z/A8DMQABAL9M43+gS1dAAAAAAElFTSuQmCC"
PNG_GREEN_B64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5/hP2E8A5MBA/8/AAYDAF4/7d4eAAAAAElFTSuQmCC"

ASSETS_DIR = Path(__file__).parent / "assets"


def load_asset_image(filename: str) -> Image.Image:
    """Loads an image from the assets directory for vision tests."""
    path = ASSETS_DIR / filename
    if not path.exists():
        pytest.fail(f"Required asset for vision test not found: {path}")
    return Image.open(path).convert("L").resize((512, 288))


@pytest.fixture
def manager(event_loop):
    """Provides a clean, started ScreenShareManager instance for each test."""
    ssm = ScreenShareManager()
    event_loop.run_until_complete(ssm.start())
    yield ssm
    event_loop.run_until_complete(ssm.stop())


@pytest.fixture
def mocked_manager(event_loop):
    """Provides a manager with its LLM clients mocked out."""
    ssm = ScreenShareManager()

    patch_detect = patch.object(ssm, "_detection_client", new_callable=AsyncMock)
    patch_annotate = patch.object(ssm, "_analysis_client", new_callable=AsyncMock)
    patch_summary = patch.object(ssm, "_summary_client", new_callable=AsyncMock)

    mock_detect = patch_detect.start()
    mock_annotate = patch_annotate.start()
    mock_summary = patch_summary.start()

    mock_detect.set_system_message = MagicMock()
    mock_annotate.set_system_message = MagicMock()
    mock_summary.set_system_message = MagicMock()

    event_loop.run_until_complete(ssm.start())

    yield ssm, {
        "detect": mock_detect,
        "annotate": mock_annotate,
        "summary": mock_summary,
    }

    event_loop.run_until_complete(ssm.stop())
    patch_detect.stop()
    patch_annotate.stop()
    patch_summary.stop()
