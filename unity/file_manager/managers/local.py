from __future__ import annotations

from unity.file_manager.managers.file_manager import FileManager
from unity.file_manager.fs_adapters.local_adapter import LocalFileSystemAdapter


class LocalFileManager(FileManager):
    def __init__(self, root: str):
        super().__init__(adapter=LocalFileSystemAdapter(root))
