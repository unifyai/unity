"""Tests for RcloneSync class."""

from __future__ import annotations

import asyncio

import pytest

from unity.file_manager.sync.config import SyncConfig
from unity.file_manager.sync.rclone import RcloneSync, SyncResult


@pytest.fixture
def sync_config(tmp_path):
    """Create a test SyncConfig."""
    return SyncConfig(
        enabled=True,
        ssh_host="test.example.com",
        ssh_port=2222,
        ssh_user="testuser",
        ssh_key_path=str(tmp_path / "test_key"),
        local_root=str(tmp_path / "local"),
        remote_root="/Unity/Local",
    )


@pytest.fixture
def rclone_sync(sync_config):
    """Create a RcloneSync instance."""
    return RcloneSync(sync_config)


class TestRcloneSyncInit:
    """Tests for RcloneSync initialization."""

    def test_init_creates_lock(self, rclone_sync):
        """Test that init creates an asyncio.Lock for serialization."""
        assert hasattr(rclone_sync, "_op_lock")
        assert isinstance(rclone_sync._op_lock, asyncio.Lock)

    def test_init_sets_config(self, rclone_sync, sync_config):
        """Test that init stores the config."""
        assert rclone_sync.config == sync_config

    def test_init_not_setup(self, rclone_sync):
        """Test that init starts in not-setup state."""
        assert rclone_sync._setup_done is False
        assert rclone_sync._config_path is None


class TestBisync:
    """Tests for bisync method."""

    @pytest.mark.asyncio
    async def test_bisync_force_resync_uses_flag(self, rclone_sync, tmp_path):
        """Test that force_resync=True adds --resync flag."""
        # Setup required state
        rclone_sync._config_path = str(tmp_path / "rclone.conf")
        (tmp_path / "local").mkdir(parents=True, exist_ok=True)

        captured_cmd = []

        async def mock_run_with_retry(cmd, operation):
            captured_cmd.clear()
            captured_cmd.extend(cmd)
            return SyncResult(success=True)

        rclone_sync._run_with_retry = mock_run_with_retry

        await rclone_sync.bisync(force_resync=True)

        assert "--resync" in captured_cmd
        assert "--conflict-resolve" in captured_cmd
        assert "newer" in captured_cmd

    @pytest.mark.asyncio
    async def test_bisync_without_force_resync_no_flag(self, rclone_sync, tmp_path):
        """Test that force_resync=False doesn't add --resync flag initially."""
        rclone_sync._config_path = str(tmp_path / "rclone.conf")
        (tmp_path / "local").mkdir(parents=True, exist_ok=True)

        captured_cmd = []

        async def mock_run_with_retry(cmd, operation):
            captured_cmd.clear()
            captured_cmd.extend(cmd)
            return SyncResult(success=True)

        rclone_sync._run_with_retry = mock_run_with_retry

        await rclone_sync.bisync(force_resync=False)

        assert "--resync" not in captured_cmd
        assert "--conflict-resolve" in captured_cmd

    @pytest.mark.asyncio
    async def test_bisync_auto_recovery_on_resync_error(self, rclone_sync, tmp_path):
        """Test that bisync auto-recovers with --resync on failure."""
        rclone_sync._config_path = str(tmp_path / "rclone.conf")
        (tmp_path / "local").mkdir(parents=True, exist_ok=True)

        call_count = 0
        captured_cmds = []

        async def mock_run_with_retry(cmd, operation):
            nonlocal call_count
            call_count += 1
            captured_cmds.append(list(cmd))

            if call_count == 1:
                # First call fails with resync-needed error
                return SyncResult(
                    success=False,
                    errors=["Bisync aborted. Must run --resync to recover."],
                )
            else:
                # Second call succeeds
                return SyncResult(success=True)

        rclone_sync._run_with_retry = mock_run_with_retry

        result = await rclone_sync.bisync(force_resync=False)

        assert result.success is True
        assert call_count == 2
        # First call without --resync
        assert "--resync" not in captured_cmds[0]
        # Second call (recovery) with --resync
        assert "--resync" in captured_cmds[1]


class TestNeedsResyncRecovery:
    """Tests for _needs_resync_recovery method."""

    def test_detects_must_run_resync(self, rclone_sync):
        """Test detection of 'must run --resync' error."""
        errors = ["Bisync aborted. Must run --resync to recover."]
        assert rclone_sync._needs_resync_recovery(errors) is True

    def test_detects_resync_to_recover(self, rclone_sync):
        """Test detection of 'resync to recover' error."""
        errors = ["Error: resync to recover from this state"]
        assert rclone_sync._needs_resync_recovery(errors) is True

    def test_detects_empty_prior_path(self, rclone_sync):
        """Test detection of 'empty prior path' error."""
        errors = ["Empty prior Path1 listing. Cannot sync to empty directory."]
        assert rclone_sync._needs_resync_recovery(errors) is True

    def test_no_recovery_for_other_errors(self, rclone_sync):
        """Test that other errors don't trigger recovery."""
        errors = ["Connection refused", "Network timeout"]
        assert rclone_sync._needs_resync_recovery(errors) is False

    def test_case_insensitive(self, rclone_sync):
        """Test that detection is case-insensitive."""
        errors = ["MUST RUN --RESYNC TO RECOVER"]
        assert rclone_sync._needs_resync_recovery(errors) is True


class TestOperationLock:
    """Tests for operation serialization via _op_lock."""

    @pytest.mark.asyncio
    async def test_bisync_acquires_lock(self, rclone_sync, tmp_path):
        """Test that bisync acquires the lock."""
        rclone_sync._config_path = str(tmp_path / "rclone.conf")
        (tmp_path / "local").mkdir(parents=True, exist_ok=True)

        lock_was_held = False

        async def mock_run_with_retry(cmd, operation):
            nonlocal lock_was_held
            lock_was_held = rclone_sync._op_lock.locked()
            return SyncResult(success=True)

        rclone_sync._run_with_retry = mock_run_with_retry

        await rclone_sync.bisync()

        assert lock_was_held is True

    @pytest.mark.asyncio
    async def test_operations_are_serialized(self, rclone_sync, tmp_path):
        """Test that concurrent operations are serialized."""
        rclone_sync._config_path = str(tmp_path / "rclone.conf")
        (tmp_path / "local").mkdir(parents=True, exist_ok=True)

        execution_order = []
        operation_active = False

        async def mock_run_with_retry(cmd, operation):
            nonlocal operation_active
            # Check no other operation is active (would indicate race)
            assert operation_active is False, "Operations should be serialized"
            operation_active = True
            execution_order.append(operation)
            await asyncio.sleep(0.01)  # Simulate some work
            operation_active = False
            return SyncResult(success=True)

        rclone_sync._run_with_retry = mock_run_with_retry

        # Run multiple operations concurrently
        await asyncio.gather(
            rclone_sync.bisync(),
            rclone_sync.sync_to_remote(),
            rclone_sync.sync_from_remote(),
        )

        # All operations should have completed
        assert len(execution_order) == 3
