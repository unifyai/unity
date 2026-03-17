"""Tests for SyncManager class."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.file_manager.sync.config import SyncConfig
from unity.file_manager.sync.manager import SyncManager
from unity.file_manager.sync.rclone import SyncResult


@pytest.fixture
def sync_config(tmp_path):
    """Create a test SyncConfig."""
    return SyncConfig(
        enabled=True,
        ssh_host="test.example.com",
        ssh_port=2222,
        ssh_user="testuser",
        ssh_key_path=str(tmp_path / "test_key"),
        local_root=str(tmp_path / "unity"),
        remote_root="/Unity/Local",
    )


@pytest.fixture
def disabled_config():
    """Create a disabled SyncConfig."""
    return SyncConfig(enabled=False)


class TestSyncManagerInit:
    """Tests for SyncManager initialization."""

    def test_init_with_config(self, sync_config):
        """Test initialization with explicit config."""
        manager = SyncManager(config=sync_config)
        assert manager.config == sync_config
        assert manager._started is False
        assert manager._rclone is None

    def test_enabled_property(self, sync_config, disabled_config):
        """Test enabled property reflects config."""
        enabled_manager = SyncManager(config=sync_config)
        assert enabled_manager.enabled is True

        disabled_manager = SyncManager(config=disabled_config)
        assert disabled_manager.enabled is False


class TestSyncManagerStart:
    """Tests for SyncManager.start() method."""

    @pytest.mark.asyncio
    async def test_start_disabled_returns_false(self, disabled_config):
        """Test start returns False when sync disabled."""
        manager = SyncManager(config=disabled_config)
        result = await manager.start()
        assert result is False

    @pytest.mark.asyncio
    async def test_start_already_started_returns_true(self, sync_config):
        """Test start returns True if already started."""
        manager = SyncManager(config=sync_config)
        manager._started = True
        result = await manager.start()
        assert result is True

    @pytest.mark.asyncio
    async def test_start_calls_bisync_with_force_resync(self, sync_config, tmp_path):
        """Test that start() calls bisync(force_resync=True)."""
        manager = SyncManager(config=sync_config)

        # Mock SSH key retrieval
        mock_ssh_key = "fake-ssh-private-key"

        # Track bisync calls
        bisync_calls = []

        async def mock_bisync(force_resync=False):
            bisync_calls.append({"force_resync": force_resync})
            return SyncResult(success=True)

        with patch.object(manager, "_get_ssh_private_key", return_value=mock_ssh_key):
            # Create mock RcloneSync
            mock_rclone = MagicMock()
            mock_rclone.setup = AsyncMock(return_value=True)
            mock_rclone.bisync = mock_bisync

            with patch(
                "unity.file_manager.sync.manager.RcloneSync",
                return_value=mock_rclone,
            ):
                result = await manager.start()

                # Cancel polling task to clean up
                if manager._poll_task:
                    manager._poll_task.cancel()
                    try:
                        await manager._poll_task
                    except asyncio.CancelledError:
                        pass

        # Verify bisync was called with force_resync=True
        assert len(bisync_calls) == 1
        assert bisync_calls[0]["force_resync"] is True


class TestSyncManagerSentinel:
    """Tests for assistant.txt sentinel creation."""

    def test_sentinel_created_on_start(self, sync_config, tmp_path):
        """Test that _ensure_sentinel creates assistant.txt in local_root."""
        manager = SyncManager(config=sync_config)
        local_root = tmp_path / "unity"
        local_root.mkdir(parents=True, exist_ok=True)

        manager._ensure_sentinel()

        sentinel = local_root / "assistant.txt"
        assert sentinel.exists()
        assert sentinel.read_text() == "unity assistant\n"

    def test_sentinel_not_overwritten(self, sync_config, tmp_path):
        """Test that _ensure_sentinel preserves existing assistant.txt."""
        manager = SyncManager(config=sync_config)
        local_root = tmp_path / "unity"
        local_root.mkdir(parents=True, exist_ok=True)

        sentinel = local_root / "assistant.txt"
        sentinel.write_text("custom content")

        manager._ensure_sentinel()

        assert sentinel.read_text() == "custom content"


class TestSyncManagerPolling:
    """Tests for SyncManager polling behavior."""

    @pytest.mark.asyncio
    async def test_poll_calls_bisync_without_force_resync(self, sync_config):
        """Test that polling calls bisync() without force_resync."""
        manager = SyncManager(config=sync_config)

        bisync_calls = []

        async def mock_bisync(force_resync=False):
            bisync_calls.append({"force_resync": force_resync})
            return SyncResult(success=True)

        mock_rclone = MagicMock()
        mock_rclone.bisync = mock_bisync

        manager._rclone = mock_rclone
        manager._started = True

        # Call sync_remote_changes which is what polling uses
        await manager.sync_remote_changes()

        assert len(bisync_calls) == 1
        assert bisync_calls[0]["force_resync"] is False


class TestSyncManagerStop:
    """Tests for SyncManager.stop() method."""

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self):
        """Test stop does nothing when not started."""
        manager = SyncManager(config=SyncConfig(enabled=False))
        manager._started = False
        await manager.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_stop_cleans_up_without_bisync(self, sync_config):
        """Test stop cleans up rclone without running a final bisync."""
        manager = SyncManager(config=sync_config)
        manager._started = True

        mock_rclone = MagicMock()
        mock_rclone.cleanup = MagicMock()

        manager._rclone = mock_rclone
        manager._poll_task = None

        await manager.stop()

        mock_rclone.bisync.assert_not_called()
        mock_rclone.cleanup.assert_called_once()
        assert manager._rclone is None
        assert manager._started is False


class TestSyncManagerSSHKeyRetrieval:
    """Tests for SSH key retrieval via the admin endpoint."""

    @pytest.mark.asyncio
    async def test_uses_agent_id_endpoint_not_user_scoped(self, sync_config):
        """SSH key retrieval should use GET /admin/assistant?agent_id=X,
        not the user-scoped endpoint that excludes org assistants."""
        manager = SyncManager(config=sync_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "info": [
                {
                    "agent_id": 42,
                    "desktop_filesync_sshkey": "fake-ssh-key-content",
                },
            ],
        }

        with (
            patch(
                "unity.session_details.SESSION_DETAILS",
            ) as mock_sd,
            patch(
                "unity.settings.SETTINGS",
            ) as mock_settings,
            patch(
                "unify.utils.http.get",
                return_value=mock_response,
            ) as mock_get,
        ):
            mock_sd.assistant.agent_id = 42
            mock_settings.ORCHESTRA_URL = "https://api.example.com/v0"
            mock_settings.ORCHESTRA_ADMIN_KEY.get_secret_value.return_value = (
                "test-admin-key"
            )

            key = await manager._get_ssh_private_key()

        assert key == "fake-ssh-key-content"
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        call_params = mock_get.call_args[1].get(
            "params",
            mock_get.call_args.kwargs.get("params"),
        )
        assert call_url == "https://api.example.com/v0/admin/assistant"
        assert call_params == {"agent_id": "42"}
        assert "/user/" not in call_url

    @pytest.mark.asyncio
    async def test_returns_none_when_assistant_not_found(self, sync_config):
        """Should return None when the admin endpoint returns an empty list."""
        manager = SyncManager(config=sync_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"info": []}

        with (
            patch(
                "unity.session_details.SESSION_DETAILS",
            ) as mock_sd,
            patch(
                "unity.settings.SETTINGS",
            ) as mock_settings,
            patch(
                "unify.utils.http.get",
                return_value=mock_response,
            ),
        ):
            mock_sd.assistant.agent_id = 999
            mock_settings.ORCHESTRA_URL = "https://api.example.com/v0"
            mock_settings.ORCHESTRA_ADMIN_KEY.get_secret_value.return_value = (
                "test-admin-key"
            )

            key = await manager._get_ssh_private_key()

        assert key is None
