"""Tests for workspace directory bootstrap in conversation_manager.main."""

import base64
import os
from datetime import datetime, timezone
from pathlib import Path

from unity.conversation_manager.conversation_manager import _save_screenshot
from unity.conversation_manager.types import ScreenshotEntry
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

        outputs = root / "Outputs"
        if outputs.exists():
            shutil.rmtree(outputs)
        outputs.mkdir(exist_ok=True)

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

    def test_outputs_cleared_between_sessions(self, tmp_path):
        root = tmp_path / "Unity" / "Local"
        self._bootstrap(root)

        # Simulate a file generated during a previous session.
        stale = root / "Outputs" / "old_report.pdf"
        stale.write_bytes(b"stale")
        assert stale.exists()

        # Second bootstrap (new session) should clear Outputs/.
        self._bootstrap(root)

        assert not stale.exists()
        assert (root / "Outputs").is_dir()

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
    """Verify _save_screenshot writes files with correct names and content."""

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

        asst_path = _save_screenshot(self._make_entry(source="assistant"))
        user_path = _save_screenshot(self._make_entry(source="user"))

        assert asst_path.startswith("Screenshots/Assistant/")
        assert user_path.startswith("Screenshots/User/")

    def test_saves_webcam_to_correct_subfolder(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "Screenshots" / "Webcam").mkdir(parents=True)

        webcam_path = _save_screenshot(self._make_entry(source="webcam"))

        assert webcam_path.startswith("Screenshots/Webcam/")

    def test_timestamp_filename(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "Screenshots" / "Assistant").mkdir(parents=True)

        ts = datetime(2026, 2, 16, 14, 30, 45, 123456, tzinfo=timezone.utc)
        path = _save_screenshot(self._make_entry(source="assistant", ts=ts))

        assert path == "Screenshots/Assistant/2026-02-16T14-30-45.123456.jpg"

    def test_writes_decoded_content(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "Screenshots" / "Assistant").mkdir(parents=True)

        raw = b"\x89PNG\r\n\x1a\nfake_image_data"
        path = _save_screenshot(self._make_entry(pixel=raw))

        written = (tmp_path / path).read_bytes()
        assert written == raw

    def test_collision_avoidance(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "Screenshots" / "Assistant").mkdir(parents=True)

        ts = datetime(2026, 2, 16, 14, 30, 45, 123456, tzinfo=timezone.utc)
        entry = self._make_entry(source="assistant", ts=ts)

        path1 = _save_screenshot(entry)
        path2 = _save_screenshot(entry)
        path3 = _save_screenshot(entry)

        assert path1 == "Screenshots/Assistant/2026-02-16T14-30-45.123456.jpg"
        assert path2 == "Screenshots/Assistant/2026-02-16T14-30-45.123456_1.jpg"
        assert path3 == "Screenshots/Assistant/2026-02-16T14-30-45.123456_2.jpg"

        # All three files exist with correct content.
        assert (tmp_path / path1).exists()
        assert (tmp_path / path2).exists()
        assert (tmp_path / path3).exists()
