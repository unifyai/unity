"""
FileManager package providing file management abstractions and implementations.
"""

from typing import TYPE_CHECKING
from importlib import import_module

__all__ = [
    "BaseFileManager",
    # adapters
    "BaseFileSystemAdapter",
    "LocalFileSystemAdapter",
    "CodeSandboxFileSystemAdapter",
    "InteractFileSystemAdapter",
    "GoogleDriveAdapter",
    # managers
    "FileManager",
    "LocalFileManager",
    "CodeSandboxFileManager",
    "InteractFileManager",
    "GoogleDriveFileManager",
    # composite
    "GlobalFileManager",
    # types
    "FileRecord",
    "FileSystemCapabilities",
    "FileReference",
    "FolderReference",
]

_lazy_map = {
    "BaseFileManager": "unity.file_manager.base",
    "BaseFileSystemAdapter": "unity.file_manager.filesystem_adapters",
    "LocalFileSystemAdapter": "unity.file_manager.filesystem_adapters",
    "CodeSandboxFileSystemAdapter": "unity.file_manager.filesystem_adapters",
    "InteractFileSystemAdapter": "unity.file_manager.filesystem_adapters",
    "GoogleDriveAdapter": "unity.file_manager.filesystem_adapters",
    "FileManager": "unity.file_manager.managers.file_manager",
    "LocalFileManager": "unity.file_manager.managers.local",
    "CodeSandboxFileManager": "unity.file_manager.managers.codesandbox",
    "InteractFileManager": "unity.file_manager.managers.interact",
    "GoogleDriveFileManager": "unity.file_manager.managers.google_drive",
    "GlobalFileManager": "unity.file_manager.global_file_manager",
    "FileRecord": "unity.file_manager.types",
    "FileSystemCapabilities": "unity.file_manager.types",
    "FileReference": "unity.file_manager.types",
    "FolderReference": "unity.file_manager.types",
}


def __getattr__(name: str):
    if name in _lazy_map:
        module = import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:
    from unity.file_manager.base import BaseFileManager
    from unity.file_manager.filesystem_adapters import (
        BaseFileSystemAdapter,
        LocalFileSystemAdapter,
        CodeSandboxFileSystemAdapter,
        InteractFileSystemAdapter,
        GoogleDriveAdapter,
    )
    from unity.file_manager.managers.file_manager import FileManager
    from unity.file_manager.managers.local import LocalFileManager
    from unity.file_manager.managers.codesandbox import CodeSandboxFileManager
    from unity.file_manager.managers.interact import InteractFileManager
    from unity.file_manager.managers.google_drive import GoogleDriveFileManager
    from unity.file_manager.global_file_manager import GlobalFileManager
    from unity.file_manager.types import (
        FileRecord,
        FileSystemCapabilities,
        FileReference,
        FolderReference,
    )
