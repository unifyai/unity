"""Per-trajectory package overlay for the CodeActActor.

Provides a temporary --target directory per act() trajectory, organized
hierarchically to match sub-agent nesting.  Cleanup reverses all
side-effects (sys.path, sys.modules, disk).
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
from typing import List

logger = logging.getLogger(__name__)


class PackageOverlay:
    """Manages a temporary --target directory for packages installed during a
    single act() trajectory.

    Packages are installed into an isolated directory that is appended to
    sys.path (so system packages always take priority).  When the trajectory
    ends, the directory is removed from sys.path, any modules loaded from it
    are purged from sys.modules, and the directory is deleted from disk.

    Hierarchy
    ---------
    Overlays form a tree that mirrors sub-agent nesting.  Each overlay
    discovers its parent via the ``_CURRENT_PACKAGE_OVERLAY`` ContextVar and
    creates its directory as a child of the parent's directory::

        /tmp/unity_act_pkgs/<root_agent_id>/
        /tmp/unity_act_pkgs/<root_agent_id>/<child_agent_id>/
        /tmp/unity_act_pkgs/<root_agent_id>/<child_agent_id>/<grandchild>/

    Each overlay only manages its own leaf directory.  Children always clean
    up before parents (inner actor cleanup is attached to the handle's
    lifecycle), so the parent's ``shutil.rmtree`` acts as a safety net for
    any orphaned subdirectories.

    Thread-/task-safety: each act() call creates its own overlay with a
    unique agent_id, so concurrent overlays do not interfere.
    """

    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id
        self._parent: PackageOverlay | None = _CURRENT_PACKAGE_OVERLAY.get()
        self._dir: str | None = None

    @property
    def active(self) -> bool:
        return self._dir is not None

    @property
    def _target_dir(self) -> str:
        """The directory path where packages would be installed.

        Computed from the parent hierarchy but not created until the first
        ``install()`` call.
        """
        import tempfile

        if self._parent is not None:
            return os.path.join(self._parent._target_dir, self._agent_id)
        return os.path.join(tempfile.gettempdir(), "unity_act_pkgs", self._agent_id)

    def install(self, packages: List[str], timeout: float = 120) -> dict:
        """Install *packages* into the overlay directory.

        Creates the directory hierarchy lazily on first call and appends
        this overlay's leaf directory to ``sys.path``.

        Returns a dict with ``success``, ``stdout``, ``stderr``, and
        ``packages``.
        """
        import importlib
        import subprocess

        if self._dir is None:
            self._dir = self._target_dir
            os.makedirs(self._dir, exist_ok=True)
            sys.path.append(self._dir)

        result = subprocess.run(
            ["uv", "pip", "install", "--target", self._dir, *packages],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Let Python's import machinery discover newly installed packages.
        importlib.invalidate_caches()

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "packages": packages,
        }

    def cleanup(self) -> None:
        """Remove the overlay directory, its sys.path entry, and any modules
        loaded from it."""
        import shutil as _shutil

        if self._dir is None:
            return

        target = self._dir
        self._dir = None

        # Remove from sys.path.
        try:
            sys.path.remove(target)
        except ValueError:
            pass

        # Purge modules whose files live under the overlay directory so
        # subsequent trajectories get fresh imports.
        to_remove: list[str] = []
        for name, mod in sys.modules.items():
            mod_file = getattr(mod, "__file__", None) or ""
            if mod_file.startswith(target):
                to_remove.append(name)
        for name in to_remove:
            del sys.modules[name]

        # Delete the directory from disk (also removes any orphaned child
        # directories as a safety net).
        _shutil.rmtree(target, ignore_errors=True)


_CURRENT_PACKAGE_OVERLAY: contextvars.ContextVar[PackageOverlay | None] = (
    contextvars.ContextVar("current_package_overlay", default=None)
)
