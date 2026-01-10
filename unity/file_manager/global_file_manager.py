from __future__ import annotations

import functools
from typing import List

import unify

from unity.file_manager.managers.base import BaseFileManager
from unity.file_manager.base import BaseGlobalFileManager


class GlobalFileManager(BaseGlobalFileManager):
    """Single-surface facade over multiple filesystem-specific FileManagers.

    This manager provides a unified view over multiple underlying filesystem
    managers. For filesystem‑wide operations (asking questions across files,
    organizing files), use ``FunctionManager`` to compose bespoke logic
    combining lexical and semantic search with shell scripts.
    """

    def __init__(self, managers: List[BaseFileManager]):
        """
        Construct a GlobalFileManager over multiple FileManager instances.

        Parameters
        ----------
        managers : list[BaseFileManager]
            Concrete FileManager instances to expose.
        """
        super().__init__()
        self._managers: List[BaseFileManager] = list(managers)

    # Helpers

    def list_filesystems(self) -> List[str]:
        """Return the list of manager class names in deterministic order."""
        names = [
            getattr(m.__class__, "__name__", "FileManager") for m in self._managers
        ]
        return sorted(set(names))

    @functools.wraps(BaseGlobalFileManager.clear, updated=())
    def clear(self) -> None:  # type: ignore[override]
        """
        Reset the GlobalFileManager view and all underlying managers.

        Behaviour
        ---------
        - Attempts to delete the GlobalFileManager's own Unify context if one is
          present or can be derived. This manager does not persist aggregated rows
          by default, but this step ensures any future or temporary contexts are
          cleaned up.
        - Calls ``clear()`` on each underlying filesystem‑specific manager so any
          per‑filesystem contexts and local caches are reset.
        - All errors are swallowed to keep ``clear()`` idempotent and safe to call
          in test setup/teardown.
        """
        # Clear a derived global context if present
        ctxs = unify.get_active_context()
        read_ctx = ctxs.get("read")
        global_ctx = f"{read_ctx}/FilesGlobal" if read_ctx else "FilesGlobal"
        unify.delete_context(global_ctx)

        # Fan‑out clear to all underlying managers
        try:
            for mgr in self._managers or []:
                try:
                    mgr.clear()
                except Exception:
                    continue
        except Exception:
            pass
