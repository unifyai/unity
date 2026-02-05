"""Tests for LocalFileSystemAdapter sync functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.file_manager.filesystem_adapters.local_adapter import LocalFileSystemAdapter
from unity.file_manager.sync.rclone import SyncResult


class TestLocalFileSystemAdapterSync:
    """Tests for LocalFileSystemAdapter sync integration."""

    def test_adapter_default_root(self, tmp_path):
        """Test adapter default root is ~/Unity."""
        # With explicit root
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=False)
        assert adapter._root == tmp_path

    def test_adapter_sync_disabled_by_default_no_session(self, tmp_path):
        """Test sync is disabled when no SESSION_DETAILS configured."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=True)
        # Sync manager not created yet (lazy)
        assert adapter._sync_manager is None
        assert adapter.sync_enabled is False
        assert adapter.sync_started is False

    def test_adapter_sync_disabled_flag(self, tmp_path):
        """Test sync can be disabled via constructor flag."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=False)
        assert adapter._enable_sync is False

    def test_save_file_to_downloads_basic(self, tmp_path):
        """Test save_file_to_downloads without sync."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=False)

        result = adapter.save_file_to_downloads("test.txt", b"hello world")

        assert result == "Downloads/test.txt"
        saved_file = tmp_path / "Downloads" / "test.txt"
        assert saved_file.exists()
        assert saved_file.read_bytes() == b"hello world"

    def test_save_file_to_downloads_unique_name(self, tmp_path):
        """Test save_file_to_downloads generates unique names."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=False)

        # Save first file
        result1 = adapter.save_file_to_downloads("test.txt", b"first")
        assert result1 == "Downloads/test.txt"

        # Save second file with same name
        result2 = adapter.save_file_to_downloads("test.txt", b"second")
        assert result2 == "Downloads/test (1).txt"

        # Both files exist with correct content
        assert (tmp_path / "Downloads" / "test.txt").read_bytes() == b"first"
        assert (tmp_path / "Downloads" / "test (1).txt").read_bytes() == b"second"

    def test_save_file_to_downloads_sync_param(self, tmp_path):
        """Test save_file_to_downloads with sync=True (no effect when sync not started)."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=False)

        # sync=True should not fail even when sync not configured
        result = adapter.save_file_to_downloads("test.txt", b"data", sync=True)
        assert result == "Downloads/test.txt"


class TestLocalFileSystemAdapterAsyncSync:
    """Tests for LocalFileSystemAdapter async sync methods."""

    @pytest.mark.asyncio
    async def test_start_sync_returns_false_when_disabled(self, tmp_path):
        """start_sync returns False when enable_sync=False."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=False)

        result = await adapter.start_sync()

        assert result is False
        assert adapter._sync_manager is None

    @pytest.mark.asyncio
    async def test_start_sync_creates_sync_manager_lazily(self, tmp_path):
        """start_sync creates SyncManager lazily."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=True)

        # Before start_sync, no manager
        assert adapter._sync_manager is None

        # Mock SyncManager to avoid real sync
        mock_manager = MagicMock()
        mock_manager.enabled = False  # No desktop_url configured

        with patch(
            "unity.file_manager.filesystem_adapters.local_adapter.SyncManager",
            return_value=mock_manager,
        ):
            result = await adapter.start_sync()

        # Manager was created
        assert adapter._sync_manager is mock_manager
        # Returns False because enabled=False
        assert result is False

    @pytest.mark.asyncio
    async def test_start_sync_delegates_to_manager(self, tmp_path):
        """start_sync delegates to SyncManager.start()."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=True)

        mock_manager = MagicMock()
        mock_manager.enabled = True
        mock_manager.start = AsyncMock(return_value=True)

        with patch(
            "unity.file_manager.filesystem_adapters.local_adapter.SyncManager",
            return_value=mock_manager,
        ):
            result = await adapter.start_sync()

        mock_manager.start.assert_called_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_stop_sync_delegates_to_manager(self, tmp_path):
        """stop_sync delegates to SyncManager.stop()."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=True)

        mock_manager = MagicMock()
        mock_manager.stop = AsyncMock()
        adapter._sync_manager = mock_manager

        await adapter.stop_sync()

        mock_manager.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_sync_no_op_when_no_manager(self, tmp_path):
        """stop_sync does nothing when no SyncManager."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=False)
        assert adapter._sync_manager is None

        # Should not raise
        await adapter.stop_sync()

    @pytest.mark.asyncio
    async def test_notify_file_write_calls_manager(self, tmp_path):
        """notify_file_write delegates to SyncManager.on_file_write()."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=True)

        mock_manager = MagicMock()
        mock_manager._started = True
        mock_manager.on_file_write = AsyncMock()
        adapter._sync_manager = mock_manager

        test_path = str(tmp_path / "test.txt")
        await adapter.notify_file_write(test_path)

        mock_manager.on_file_write.assert_called_once_with(test_path)

    @pytest.mark.asyncio
    async def test_notify_file_write_no_op_when_not_started(self, tmp_path):
        """notify_file_write does nothing when sync not started."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=True)

        mock_manager = MagicMock()
        mock_manager._started = False
        mock_manager.on_file_write = AsyncMock()
        adapter._sync_manager = mock_manager

        await adapter.notify_file_write("/some/path")

        mock_manager.on_file_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_file_write_no_op_when_no_manager(self, tmp_path):
        """notify_file_write does nothing when no SyncManager."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=False)

        # Should not raise
        await adapter.notify_file_write("/some/path")

    @pytest.mark.asyncio
    async def test_notify_file_delete_calls_manager(self, tmp_path):
        """notify_file_delete delegates to SyncManager.on_file_delete()."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=True)

        mock_manager = MagicMock()
        mock_manager._started = True
        mock_manager.on_file_delete = AsyncMock()
        adapter._sync_manager = mock_manager

        test_path = str(tmp_path / "deleted.txt")
        await adapter.notify_file_delete(test_path)

        mock_manager.on_file_delete.assert_called_once_with(test_path)

    @pytest.mark.asyncio
    async def test_notify_file_delete_no_op_when_not_started(self, tmp_path):
        """notify_file_delete does nothing when sync not started."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=True)

        mock_manager = MagicMock()
        mock_manager._started = False
        mock_manager.on_file_delete = AsyncMock()
        adapter._sync_manager = mock_manager

        await adapter.notify_file_delete("/some/path")

        mock_manager.on_file_delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_from_remote_calls_sync_remote_changes(self, tmp_path):
        """refresh_from_remote delegates to SyncManager.sync_remote_changes()."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=True)

        mock_manager = MagicMock()
        mock_manager._started = True
        mock_manager.sync_remote_changes = AsyncMock(
            return_value=SyncResult(success=True),
        )
        adapter._sync_manager = mock_manager

        result = await adapter.refresh_from_remote()

        mock_manager.sync_remote_changes.assert_called_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_refresh_from_remote_returns_false_on_failure(self, tmp_path):
        """refresh_from_remote returns False when sync fails."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=True)

        mock_manager = MagicMock()
        mock_manager._started = True
        mock_manager.sync_remote_changes = AsyncMock(
            return_value=SyncResult(success=False, errors=["Network error"]),
        )
        adapter._sync_manager = mock_manager

        result = await adapter.refresh_from_remote()

        assert result is False

    @pytest.mark.asyncio
    async def test_refresh_from_remote_returns_false_when_not_started(self, tmp_path):
        """refresh_from_remote returns False when sync not started."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=True)

        mock_manager = MagicMock()
        mock_manager._started = False
        adapter._sync_manager = mock_manager

        result = await adapter.refresh_from_remote()

        assert result is False

    @pytest.mark.asyncio
    async def test_refresh_from_remote_returns_false_when_no_manager(self, tmp_path):
        """refresh_from_remote returns False when no SyncManager."""
        adapter = LocalFileSystemAdapter(str(tmp_path), enable_sync=False)

        result = await adapter.refresh_from_remote()

        assert result is False
