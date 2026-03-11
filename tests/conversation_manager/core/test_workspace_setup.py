"""Tests for workspace directory bootstrap in conversation_manager.main."""

import base64
import os
from datetime import datetime, timezone
from pathlib import Path

from unity.conversation_manager.types import ScreenshotEntry
from unity.conversation_manager.types.screenshot import (
    generate_screenshot_path,
    write_screenshot_to_disk,
)
from unity.file_manager.settings import get_local_root


class TestWorkspaceBootstrap:
    """Verify that the workspace directories are set up correctly.

    These tests exercise the directory creation and Outputs/Screenshots
    clearing logic that runs at the top of run_conversation_manager(),
    extracted here as a pure filesystem operation so it can be validated
    without spinning up the full CM.
    """

    @staticmethod
    def _bootstrap(root: Path) -> None:
        """Reproduce the workspace bootstrap logic from main.py."""
        import shutil

        root.mkdir(parents=True, exist_ok=True)
        (root / "Downloads").mkdir(exist_ok=True)
        (root / "Outputs").mkdir(exist_ok=True)

        screenshots = root / "Screenshots"
        if screenshots.exists():
            shutil.rmtree(screenshots)
        (screenshots / "User").mkdir(parents=True, exist_ok=True)
        (screenshots / "Assistant").mkdir(parents=True, exist_ok=True)

    def test_creates_workspace_directories(self, tmp_path):
        root = tmp_path / "Unity" / "Local"
        self._bootstrap(root)

        assert root.is_dir()
        assert (root / "Downloads").is_dir()
        assert (root / "Outputs").is_dir()
        assert (root / "Screenshots" / "User").is_dir()
        assert (root / "Screenshots" / "Assistant").is_dir()

    def test_outputs_preserved_between_sessions(self, tmp_path):
        root = tmp_path / "Unity" / "Local"
        self._bootstrap(root)

        # Simulate a file generated during a previous session.
        report = root / "Outputs" / "old_report.pdf"
        report.write_bytes(b"report")

        # Second bootstrap should NOT clear Outputs/.
        self._bootstrap(root)

        assert report.exists()

    def test_screenshots_cleared_between_sessions(self, tmp_path):
        root = tmp_path / "Unity" / "Local"
        self._bootstrap(root)

        # Simulate screenshots from a previous session.
        stale_user = root / "Screenshots" / "User" / "old.png"
        stale_asst = root / "Screenshots" / "Assistant" / "old.png"
        stale_user.write_bytes(b"stale")
        stale_asst.write_bytes(b"stale")
        assert stale_user.exists()
        assert stale_asst.exists()

        # Second bootstrap (new session) should clear both.
        self._bootstrap(root)

        assert not stale_user.exists()
        assert not stale_asst.exists()
        assert (root / "Screenshots" / "User").is_dir()
        assert (root / "Screenshots" / "Assistant").is_dir()

    def test_downloads_preserved_between_sessions(self, tmp_path):
        root = tmp_path / "Unity" / "Local"
        self._bootstrap(root)

        # Simulate a downloaded attachment.
        attachment = root / "Downloads" / "invoice.pdf"
        attachment.write_bytes(b"attachment")

        # Second bootstrap should NOT clear Downloads/.
        self._bootstrap(root)

        assert attachment.exists()

    def test_get_local_root_returns_unity_local(self):
        """get_local_root() should resolve to ~/Unity/Local by default."""
        root = get_local_root()
        assert root.endswith(os.path.join("Unity", "Local"))


class TestSaveScreenshot:
    """Verify generate_screenshot_path + write_screenshot_to_disk produce
    correct paths and file content.
    """

    @staticmethod
    def _make_entry(
        source: str = "assistant",
        ts: datetime | None = None,
        pixel: bytes = b"\x89PNG\r\n\x1a\n",
    ) -> ScreenshotEntry:
        b64 = base64.b64encode(pixel).decode()
        return ScreenshotEntry(
            b64=b64,
            utterance="test",
            timestamp=ts
            or datetime(2026, 2, 16, 14, 30, 45, 123456, tzinfo=timezone.utc),
            source=source,
        )

    def test_saves_to_correct_subfolder(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "Screenshots" / "Assistant").mkdir(parents=True)
        (tmp_path / "Screenshots" / "User").mkdir(parents=True)

        asst_entry = self._make_entry(source="assistant")
        user_entry = self._make_entry(source="user")
        asst_path = generate_screenshot_path(asst_entry)
        user_path = generate_screenshot_path(user_entry)
        write_screenshot_to_disk(asst_entry, asst_path)
        write_screenshot_to_disk(user_entry, user_path)

        assert asst_path.startswith("Screenshots/Assistant/")
        assert user_path.startswith("Screenshots/User/")

    def test_saves_webcam_to_correct_subfolder(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "Screenshots" / "Webcam").mkdir(parents=True)

        entry = self._make_entry(source="webcam")
        webcam_path = generate_screenshot_path(entry)
        write_screenshot_to_disk(entry, webcam_path)

        assert webcam_path.startswith("Screenshots/Webcam/")

    def test_timestamp_filename(self):
        ts = datetime(2026, 2, 16, 14, 30, 45, 123456, tzinfo=timezone.utc)
        path = generate_screenshot_path(self._make_entry(source="assistant", ts=ts))

        assert path == "Screenshots/Assistant/2026-02-16T14-30-45.123456.jpg"

    def test_writes_decoded_content(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "Screenshots" / "Assistant").mkdir(parents=True)

        raw = b"\x89PNG\r\n\x1a\nfake_image_data"
        entry = self._make_entry(pixel=raw)
        path = generate_screenshot_path(entry)
        write_screenshot_to_disk(entry, path)

        written = (tmp_path / path).read_bytes()
        assert written == raw
