"""Tests for SyncConfig."""

import pytest
from unity.file_manager.sync.config import SyncConfig


class TestSyncConfig:
    """Tests for SyncConfig dataclass and factory methods."""

    def test_default_values(self):
        """Test default SyncConfig values."""
        config = SyncConfig()

        assert config.enabled is False
        assert config.ssh_host == ""
        assert config.ssh_port == 2222
        assert config.ssh_user == ""
        assert config.remote_root == "/Unity/Local"
        assert config.sync_on_write is True
        assert config.conflict_resolution == "latest"
        assert config.max_retries == 3
        assert config.poll_interval_seconds == 30.0

    def test_local_root_default(self):
        """Test local_root defaults to get_local_root()."""
        from unity.file_manager.settings import get_local_root

        config = SyncConfig()
        assert config.local_root == get_local_root()

    def test_exclude_patterns(self):
        """Test default exclude patterns."""
        config = SyncConfig()
        assert ".git/**" in config.exclude_patterns
        assert "__pycache__/**" in config.exclude_patterns
        assert "*.pyc" in config.exclude_patterns
        assert ".bisync/**" in config.exclude_patterns
        assert "venvs/**" in config.exclude_patterns


class TestExtractHost:
    """Tests for hostname extraction from desktop URL."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://desktop.example.com/vnc", "desktop.example.com"),
            ("http://192.168.1.100:8080/", "192.168.1.100"),
            ("https://my-assistant.unify.ai:6080/vnc.html", "my-assistant.unify.ai"),
            ("", ""),
        ],
    )
    def test_host_extraction(self, url, expected):
        """Test hostname extraction from various URLs."""
        result = SyncConfig._extract_host(url)
        assert result == expected
