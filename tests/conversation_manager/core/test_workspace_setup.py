"""Tests for workspace directory bootstrap in conversation_manager.main."""

import os
from pathlib import Path


from unity.file_manager.settings import get_local_root


class TestWorkspaceBootstrap:
    """Verify that the workspace directories are set up correctly.

    These tests exercise the directory creation and Outputs/ clearing
    logic that runs at the top of run_conversation_manager(), extracted
    here as a pure filesystem operation so it can be validated without
    spinning up the full CM.
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

    def test_creates_workspace_directories(self, tmp_path):
        root = tmp_path / "Unity" / "Local"
        self._bootstrap(root)

        assert root.is_dir()
        assert (root / "Downloads").is_dir()
        assert (root / "Outputs").is_dir()

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
