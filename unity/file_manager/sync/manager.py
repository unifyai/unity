"""Orchestrates file sync lifecycle between assistant and managed VM."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from .config import SyncConfig
from .rclone import RcloneSync, SyncResult


class SyncManager:
    """Orchestrates file sync lifecycle between assistant and managed VM.

    Lifecycle:
    1. start() - Called on job start: setup + initial sync from remote
    2. on_file_write() - Called after file writes: sync file to remote
    3. sync_remote_changes() - Called periodically: bisync for remote changes
    4. stop() - Called on job end: final sync + cleanup

    Conflict resolution: Latest wins (by modification time)
    """

    def __init__(self, config: Optional[SyncConfig] = None):
        self.config = config or SyncConfig.from_session_details()
        self._rclone: Optional[RcloneSync] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._started = False

    @property
    def enabled(self) -> bool:
        """Whether sync is configured and enabled."""
        return self.config.enabled

    async def start(self) -> bool:
        """Initialize sync: setup rclone, pull initial state from remote.

        Returns:
            True if sync started successfully, False otherwise
        """
        if not self.config.enabled:
            print("[FileSync] Sync disabled (no desktop_url configured)")
            return False

        if self._started:
            print("[FileSync] Already started")
            return True

        print("[FileSync] Starting sync manager...")

        # 1. Get SSH private key from Orchestra secrets
        ssh_key = await self._get_ssh_private_key()
        if not ssh_key:
            print("[FileSync] Failed to retrieve SSH key, sync disabled")
            return False

        # 2. Setup rclone
        self._rclone = RcloneSync(self.config)
        if not await self._rclone.setup(ssh_key):
            print("[FileSync] Rclone setup failed")
            return False

        # 3. Initial bisync with --resync to establish bidirectional baseline
        print("[FileSync] Performing initial bisync with --resync...")
        result = await self._rclone.bisync(force_resync=True)
        if not result.success:
            print(f"[FileSync] Initial bisync failed: {result.errors}")
            # Continue anyway - remote might be empty on first run

        # 4. Start background polling for remote changes
        self._poll_task = asyncio.create_task(
            self._poll_remote_changes(),
            name="filesync-poll",
        )

        self._started = True
        print("[FileSync] Sync manager started successfully")
        return True

    async def on_file_write(self, path: str) -> None:
        """Called after file write to sync to remote.

        Args:
            path: Absolute path to the written file
        """
        if not self._started or not self._rclone:
            return

        if not self.config.sync_on_write:
            return

        # Check if path is under our sync root
        try:
            local_root = Path(self.config.local_root).expanduser()
            file_path = Path(path)
            if str(file_path).startswith("~"):
                file_path = file_path.expanduser()
            file_path.relative_to(local_root)
        except ValueError:
            # Not under sync root, ignore
            return

        await self._rclone.sync_single_file(path)

    async def on_file_delete(self, path: str) -> None:
        """Called after file delete to sync deletion to remote.

        Args:
            path: Absolute path to the deleted file
        """
        if not self._started or not self._rclone:
            return

        if not self.config.sync_on_write:
            return

        # Check if path is under our sync root
        try:
            local_root = Path(self.config.local_root).expanduser()
            file_path = Path(path)
            if str(file_path).startswith("~"):
                file_path = file_path.expanduser()
            file_path.relative_to(local_root)
        except ValueError:
            return

        await self._rclone.delete_remote_file(path)

    async def sync_remote_changes(self) -> SyncResult:
        """Manually trigger bisync to pull remote changes.

        Useful for explicit refresh before reading files.
        """
        if not self._started or not self._rclone:
            return SyncResult(success=False, errors=["Sync not started"])

        return await self._rclone.bisync()

    async def stop(self) -> None:
        """Stop sync manager: final sync + cleanup."""
        if not self._started:
            return

        print("[FileSync] Stopping sync manager...")

        # Cancel polling task
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        # Final bisync to push any pending changes and pull remote state
        if self._rclone:
            print("[FileSync] Final bisync...")
            await self._rclone.bisync()
            self._rclone.cleanup()
            self._rclone = None

        self._started = False
        print("[FileSync] Sync manager stopped")

    async def _get_ssh_private_key(self) -> Optional[str]:
        """Retrieve SSH private key from Orchestra assistant secrets."""
        from unity.session_details import SESSION_DETAILS
        from unity.settings import SETTINGS

        assistant_id = SESSION_DETAILS.assistant.id
        user_id = SESSION_DETAILS.user_id
        base_url = SETTINGS.ORCHESTRA_URL
        admin_key = SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()

        if not assistant_id:
            print("[FileSync] No assistant_id configured")
            return None

        if not user_id:
            print("[FileSync] No user_id configured")
            return None

        if not base_url:
            print("[FileSync] No ORCHESTRA_URL configured")
            return None

        if not admin_key:
            print("[FileSync] No ORCHESTRA_ADMIN_KEY configured")
            return None

        url = f"{base_url}/admin/assistant/user/{user_id}"
        headers = {"Authorization": f"Bearer {admin_key}"}

        print(f"[FileSync] Retrieving SSH key from {url}")

        # Retry loop for secret retrieval
        max_retries = self.config.max_retries
        retry_delay = self.config.retry_delay_seconds

        for attempt in range(1, max_retries + 1):
            try:
                from unify.utils import http

                print(
                    f"[FileSync] Fetching secrets (attempt {attempt}/{max_retries})...",
                )
                resp = http.get(url, headers=headers, timeout=30)

                if resp.status_code == 200:
                    data = resp.json()
                    assistants = data.get("info", [])

                    # Find assistant by matching agent_id
                    matched = None
                    for assistant in assistants:
                        if assistant.get("agent_id") == assistant_id:
                            matched = assistant
                            break

                    if not matched:
                        print(
                            f"[FileSync] Assistant {assistant_id} not found in "
                            f"{len(assistants)} assistants for user {user_id}",
                        )
                        return None

                    secrets = matched.get("secrets") or {}
                    key = secrets.get("vm_ssh_private_key")

                    if key:
                        print("[FileSync] SSH key retrieved successfully")
                        return key
                    else:
                        print("[FileSync] No vm_ssh_private_key in secrets")
                        return None
                else:
                    print(
                        f"[FileSync] Failed to get secrets: "
                        f"status={resp.status_code}, body={resp.text[:200]}",
                    )

            except Exception as e:
                print(f"[FileSync] Exception retrieving secrets: {e}")
                import traceback

                traceback.print_exc()

            if attempt < max_retries:
                delay = retry_delay * attempt
                print(f"[FileSync] Retrying in {delay}s...")
                await asyncio.sleep(delay)

        print(f"[FileSync] Failed to retrieve SSH key after {max_retries} attempts")
        return None

    async def _poll_remote_changes(self) -> None:
        """Background task to periodically sync remote changes."""
        interval = self.config.poll_interval_seconds
        print(f"[FileSync] Starting remote change polling (interval={interval}s)")

        while True:
            try:
                await asyncio.sleep(interval)

                if self._rclone:
                    print("[FileSync] Polling: running bisync...")
                    result = await self._rclone.bisync()
                    if result.success:
                        print("[FileSync] Polling: bisync completed successfully")
                    else:
                        print(f"[FileSync] Polling: bisync failed: {result.errors}")

            except asyncio.CancelledError:
                print("[FileSync] Polling task cancelled")
                break
            except Exception as e:
                print(f"[FileSync] Polling error: {e}")
                import traceback

                traceback.print_exc()
                # Continue polling despite errors
