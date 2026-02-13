"""Configuration for managed VM file sync via rclone SFTP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from urllib.parse import urlparse


def _get_local_root() -> str:
    from unity.file_manager.settings import get_local_root

    return get_local_root()


@dataclass
class SyncConfig:
    """Configuration for managed VM file sync via rclone SFTP.

    Paths:
    - local_root: get_local_root() (defaults to ~/Unity/Local)
    - remote_root: /Unity/Local (VM, set up by ubuntu-vm-startup.sh)

    Syncs the dedicated Unity workspace which contains user files
    (Downloads/, functions/, etc.).

    Conflict resolution: Latest wins (by modification time)
    """

    enabled: bool = False

    # SSH connection
    ssh_host: str = ""
    ssh_port: int = 2222
    ssh_user: str = ""
    ssh_key_path: str = ""  # Temp file path for private key

    # Paths
    local_root: str = field(
        default_factory=lambda: _get_local_root(),
    )
    remote_root: str = "/Unity/Local"

    # Sync behavior
    sync_on_write: bool = True
    conflict_resolution: str = "latest"  # "latest" = modification time wins
    max_delete_percent: int = 100  # bisync delete safety threshold (100 = disabled)

    # Exclude patterns (rclone filter syntax)
    exclude_patterns: List[str] = field(
        default_factory=lambda: [
            ".git/**",
            "__pycache__/**",
            "*.pyc",
            ".DS_Store",
            ".bisync/**",  # rclone's own state files
            "venvs/**",  # Virtual environments (managed via HTTP API)
        ],
    )

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 2.0

    # Polling interval for remote changes (seconds)
    poll_interval_seconds: float = 30.0

    @classmethod
    def from_session_details(cls) -> "SyncConfig":
        """Create SyncConfig from SESSION_DETAILS for managed VM mode."""
        from unity.session_details import SESSION_DETAILS

        desktop_url = SESSION_DETAILS.assistant.desktop_url
        assistant_id = SESSION_DETAILS.assistant.id

        if not desktop_url:
            print("[FileSync] No desktop_url configured, sync disabled")
            return cls(enabled=False)

        ssh_host = cls._extract_host(desktop_url)
        if not ssh_host:
            print(f"[FileSync] Could not extract host from desktop_url: {desktop_url}")
            return cls(enabled=False)

        # Use assistant_context directly - matches Unify context naming convention
        ssh_user = SESSION_DETAILS.assistant_context
        if not ssh_user:
            print("[FileSync] Could not derive SSH user from assistant_context")
            return cls(enabled=False)

        # Temp file for SSH key (secure permissions set on write)
        ssh_key_path = f"/tmp/.unity_vm_key_{assistant_id}"

        print(
            f"[FileSync] Config: host={ssh_host}, port=2222, user={ssh_user}, "
            f"local=~/Unity/Local, remote=/Unity/Local",
        )

        return cls(
            enabled=True,
            ssh_host=ssh_host,
            ssh_user=ssh_user,
            ssh_key_path=ssh_key_path,
        )

    @staticmethod
    def _extract_host(desktop_url: str) -> str:
        """Extract hostname from desktop URL."""
        if not desktop_url:
            return ""
        try:
            parsed = urlparse(desktop_url)
            return parsed.hostname or ""
        except Exception:
            return ""
