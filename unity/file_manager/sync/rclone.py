"""Wrapper for rclone SFTP operations with retry and debugging."""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import SyncConfig


@dataclass
class SyncResult:
    """Result of a sync operation."""

    success: bool
    files_transferred: int = 0
    bytes_transferred: int = 0
    errors: List[str] = field(default_factory=list)


class RcloneSync:
    """Wrapper for rclone SFTP operations with retry and debugging.

    Uses rclone to sync files between local ~ and remote /home via SFTP.
    Conflict resolution: latest wins (by modification time).
    """

    REMOTE_NAME = "vm_sftp"

    def __init__(self, config: SyncConfig):
        self.config = config
        self._config_path: Optional[str] = None
        self._setup_done = False
        self._op_lock = asyncio.Lock()  # Serializes all rclone operations

    async def setup(self, ssh_private_key: str) -> bool:
        """Setup rclone config and SSH key file.

        Args:
            ssh_private_key: The SSH private key content (from Orchestra secrets)

        Returns:
            True if setup successful, False otherwise
        """
        async with self._op_lock:
            print("[FileSync] Setting up rclone...")

            try:
                # 1. Write SSH private key to temp file with secure permissions
                key_path = Path(self.config.ssh_key_path)
                key_path.parent.mkdir(parents=True, exist_ok=True)
                key_path.write_text(ssh_private_key)
                os.chmod(key_path, 0o600)
                print(f"[FileSync] SSH key written to {key_path}")

                # 2. Ensure local root exists with standard subdirectories
                local_root = Path(self.config.local_root).expanduser()
                local_root.mkdir(parents=True, exist_ok=True)
                print(f"[FileSync] Local root: {local_root}")

                # 3. Create rclone config file
                self._config_path = tempfile.mktemp(suffix=".conf", prefix="rclone_")
                rclone_config = f"""[{self.REMOTE_NAME}]
type = sftp
host = {self.config.ssh_host}
port = {self.config.ssh_port}
user = {self.config.ssh_user}
key_file = {self.config.ssh_key_path}
"""
                Path(self._config_path).write_text(rclone_config)
                print(f"[FileSync] Rclone config written to {self._config_path}")

                # 4. Test connection
                success = await self._test_connection()
                if success:
                    self._setup_done = True
                    print("[FileSync] Setup complete, connection verified")
                else:
                    print("[FileSync] Setup failed: connection test failed")

                return success

            except Exception as e:
                print(f"[FileSync] Setup failed: {e}")
                import traceback

                traceback.print_exc()
                return False

    async def _test_connection(self) -> bool:
        """Test SFTP connection to VM."""
        cmd = self._build_cmd(["lsf", f"{self.REMOTE_NAME}:/", "--max-depth", "1"])
        result = await self._run_with_retry(cmd, operation="connection test")
        return result.success

    async def bisync(self, force_resync: bool = False) -> SyncResult:
        """Bidirectional sync with 'latest wins' conflict resolution.

        Uses rclone bisync which:
        - Propagates new/changed files in both directions
        - Uses modification time for conflict resolution (--conflict-resolve newer)

        Args:
            force_resync: If True, always use --resync flag (for initialization)
        """
        async with self._op_lock:
            remote = f"{self.REMOTE_NAME}:{self.config.remote_root}"
            local = str(Path(self.config.local_root).expanduser())

            print(f"[FileSync] Bisync: {local} ↔ {remote}")

            base_args = [
                "bisync",
                local,
                remote,
                "--conflict-resolve",
                "newer",  # Latest wins
                *self._exclude_args(),
                "-v",
            ]

            if force_resync:
                print("[FileSync] Using --resync for bisync initialization")
                return await self._run_with_retry(
                    self._build_cmd([*base_args, "--resync"]),
                    operation="bisync",
                )

            # Try without --resync, auto-recover if rclone says it needs it
            result = await self._run_with_retry(
                self._build_cmd(base_args),
                operation="bisync",
            )

            if not result.success and self._needs_resync_recovery(result.errors):
                print("[FileSync] Bisync state corrupted, recovering with --resync...")
                result = await self._run_with_retry(
                    self._build_cmd([*base_args, "--resync"]),
                    operation="bisync (recovery)",
                )

            return result

    def _needs_resync_recovery(self, errors: List[str]) -> bool:
        """Check if bisync failure requires --resync recovery."""
        for error in errors:
            error_lower = error.lower()
            if "must run --resync" in error_lower or "resync to recover" in error_lower:
                return True
            # Also catch empty listing errors that precede the resync message
            if "empty prior path" in error_lower:
                return True
        return False

    async def sync_single_file(self, local_path: str) -> SyncResult:
        """Sync a single file to remote (for on-write sync).

        Args:
            local_path: Absolute path to the local file
        """
        async with self._op_lock:
            local_root = Path(self.config.local_root).expanduser()
            local_file = Path(local_path)

            # Expand user paths
            if str(local_file).startswith("~"):
                local_file = local_file.expanduser()

            try:
                rel_path = local_file.relative_to(local_root)
            except ValueError:
                print(f"[FileSync] File {local_path} is outside sync root, skipping")
                return SyncResult(success=True)

            remote_path = f"{self.REMOTE_NAME}:{self.config.remote_root}/{rel_path}"

            print(f"[FileSync] Copying file: {local_file} → {remote_path}")

            cmd = self._build_cmd(["copyto", str(local_file), remote_path, "-v"])
            return await self._run_with_retry(cmd, operation=f"copy {rel_path}")

    async def delete_remote_file(self, local_path: str) -> SyncResult:
        """Delete a file from remote (for on-delete sync).

        Args:
            local_path: Absolute path to the (deleted) local file
        """
        async with self._op_lock:
            local_root = Path(self.config.local_root).expanduser()
            local_file = Path(local_path)

            if str(local_file).startswith("~"):
                local_file = local_file.expanduser()

            try:
                rel_path = local_file.relative_to(local_root)
            except ValueError:
                return SyncResult(success=True)

            remote_path = f"{self.REMOTE_NAME}:{self.config.remote_root}/{rel_path}"

            print(f"[FileSync] Deleting remote: {remote_path}")

            cmd = self._build_cmd(["deletefile", remote_path, "-v"])
            return await self._run_with_retry(cmd, operation=f"delete {rel_path}")

    def _build_cmd(self, args: List[str]) -> List[str]:
        """Build rclone command with config."""
        return ["rclone", "--config", self._config_path, *args]

    def _exclude_args(self) -> List[str]:
        """Build exclude arguments for rclone."""
        args = []
        for pattern in self.config.exclude_patterns:
            args.extend(["--exclude", pattern])
        return args

    async def _run_with_retry(self, cmd: List[str], operation: str) -> SyncResult:
        """Run rclone command with retry logic."""
        last_error = None

        for attempt in range(1, self.config.max_retries + 1):
            print(
                f"[FileSync] {operation} (attempt {attempt}/{self.config.max_retries})",
            )
            print(f"[FileSync] cmd: {' '.join(cmd)}")

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()

                stdout_str = stdout.decode() if stdout else ""
                stderr_str = stderr.decode() if stderr else ""

                if proc.returncode == 0:
                    print(f"[FileSync] {operation} succeeded")
                    if stdout_str:
                        # Truncate long output
                        output = stdout_str[:500]
                        if len(stdout_str) > 500:
                            output += "... (truncated)"
                        print(f"[FileSync] stdout: {output}")
                    return SyncResult(success=True)
                else:
                    last_error = f"Exit code {proc.returncode}: {stderr_str}"
                    print(f"[FileSync] {operation} failed: {last_error}")

            except FileNotFoundError:
                last_error = "rclone not found - is it installed?"
                print(f"[FileSync] {operation} failed: {last_error}")
                # Don't retry if rclone isn't installed
                break
            except Exception as e:
                last_error = str(e)
                print(f"[FileSync] {operation} exception: {last_error}")
                import traceback

                traceback.print_exc()

            if attempt < self.config.max_retries:
                delay = self.config.retry_delay_seconds * attempt
                print(f"[FileSync] Retrying in {delay}s...")
                await asyncio.sleep(delay)

        print(f"[FileSync] {operation} failed after {self.config.max_retries} attempts")
        return SyncResult(success=False, errors=[last_error or "Unknown error"])

    def cleanup(self) -> None:
        """Clean up temp files."""
        for path in [self._config_path, self.config.ssh_key_path]:
            if path and Path(path).exists():
                try:
                    Path(path).unlink()
                    print(f"[FileSync] Cleaned up: {path}")
                except Exception as e:
                    print(f"[FileSync] Failed to clean up {path}: {e}")
