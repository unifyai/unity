from __future__ import annotations

from unity.file_manager.managers.file_manager import FileManager
from unity.file_manager.fs_adapters.interact_adapter import InteractFileSystemAdapter


class InteractFileManager(FileManager):
    def __init__(self, api_base: str, api_key: str, space: str):
        super().__init__(adapter=InteractFileSystemAdapter(api_base, api_key, space))
