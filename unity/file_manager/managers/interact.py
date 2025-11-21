from __future__ import annotations

from unity.file_manager.managers.file_manager import FileManager
from unity.file_manager.fs_adapters.interact_adapter import InteractFileSystemAdapter
from unity.singleton_registry import SingletonABCMeta


class InteractFileManager(FileManager, metaclass=SingletonABCMeta):
    def __init__(self, api_base: str, api_key: str, space: str):
        super().__init__(adapter=InteractFileSystemAdapter(api_base, api_key, space))
