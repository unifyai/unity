from __future__ import annotations

# Backwards-compat shim: re-export BaseFileManager from managers.base
from unity.common.global_docstrings import CLEAR_METHOD_DOCSTRING
from unity.file_manager.managers.base import BaseFileManager  # noqa: F401
from typing import List
from abc import abstractmethod
from unity.manager_registry import SingletonABCMeta
from ..common.state_managers import BaseStateManager


class BaseGlobalFileManager(BaseStateManager, metaclass=SingletonABCMeta):
    """
    *Public* contract that every concrete **global file‑manager** must satisfy.

    Purpose
    -------
    A global file‑manager presents a unified surface over multiple underlying
    filesystem‑specific file managers (e.g., Local, Interact, CodeSandbox).
    It provides a discovery helper to list configured filesystems.

    For filesystem‑wide operations (asking questions across files, organizing
    files), use ``FunctionManager`` to compose bespoke logic combining lexical
    and semantic search with shell scripts.
    """

    _as_caller_description: str = (
        "the FileManager, managing files on behalf of the end user"
    )

    # ------------------------------ Public API ------------------------------ #
    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    # -------------------------- Discovery helper --------------------------- #
    @abstractmethod
    def list_filesystems(self) -> List[str]:
        """
        Return a list of human‑readable identifiers for the configured
        filesystems. Implementations may return class names of the underlying
        managers (recommended) or any other stable labels suitable for prompts.
        """


# Attach centralised docstring
BaseGlobalFileManager.clear.__doc__ = CLEAR_METHOD_DOCSTRING
