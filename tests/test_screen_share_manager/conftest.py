import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import pytest
from unity.screen_share_manager.screen_share_manager import ScreenShareManager
from unity.transcript_manager.types.message import Message


@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for our test module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mocked_screen_share_manager(event_loop):
    """
    Provides a ScreenShareManager instance with all external dependencies mocked.

    Yields a tuple containing:
    - The ScreenShareManager instance.
    - A dictionary of its mocked dependencies.
    """
    with patch(
        "unity.screen_share_manager.screen_share_manager.get_event_broker",
    ) as mock_get_broker, patch(
        "unity.screen_share_manager.screen_share_manager.AsyncOpenAI",
    ) as mock_openai, patch(
        "unity.screen_share_manager.screen_share_manager.ImageManager",
    ) as mock_image_manager, patch(
        "unity.screen_share_manager.screen_share_manager.TranscriptManager",
    ) as mock_transcript_manager:

        # Mock Event Broker (Redis)
        mock_broker = MagicMock()
        mock_broker.pubsub.return_value.__aenter__.return_value.get_message = AsyncMock(
            return_value=None,
        )
        mock_broker.publish = AsyncMock()
        mock_get_broker.return_value = mock_broker

        # Mock OpenAI Client
        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create = AsyncMock()
        mock_openai.return_value = mock_openai_instance

        # Mock ImageManager
        mock_image_manager_instance = MagicMock()
        mock_image_manager_instance.add_images.return_value = [
            42,
        ]  # Return a predictable image_id
        mock_image_manager.return_value = mock_image_manager_instance

        # Mock TranscriptManager
        mock_transcript_manager_instance = MagicMock()
        # Make log_messages return a mock with an ID to test back-patching
        mock_logged_message = Message(
            message_id=123,
            medium="phone_call",
            sender_id=1,
            receiver_ids=[0],
            timestamp=datetime.now(),
            content="test",
        )
        mock_transcript_manager_instance.log_messages.return_value = [
            mock_logged_message,
        ]
        mock_transcript_manager.return_value = mock_transcript_manager_instance

        # Instantiate the manager with mocks in place
        manager = ScreenShareManager()

        # Override the event loop for the manager's async tasks
        manager._event_broker = mock_broker

        mocks = {
            "event_broker": mock_broker,
            "openai_client": mock_openai_instance,
            "image_manager": mock_image_manager_instance,
            "transcript_manager": mock_transcript_manager_instance,
        }

        yield manager, mocks
