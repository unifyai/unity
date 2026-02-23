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
    # managers
    "FileManager",
    "LocalFileManager",
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
    "FileManager": "unity.file_manager.managers.file_manager",
    "LocalFileManager": "unity.file_manager.managers.local",
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
    )
    from unity.file_manager.managers.file_manager import FileManager
    from unity.file_manager.managers.local import LocalFileManager
    from unity.file_manager.types import (
        FileRecord,
        FileSystemCapabilities,
        FileReference,
        FolderReference,
    )
