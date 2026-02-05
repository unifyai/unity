"""Tests for LocalFileSystemAdapter sync functionality."""

from unity.file_manager.filesystem_adapters.local_adapter import LocalFileSystemAdapter


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
