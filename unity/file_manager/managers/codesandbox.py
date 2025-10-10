from __future__ import annotations

from unity.file_manager.managers.file_manager import FileManager
from unity.file_manager.fs_adapters.codesandbox_adapter import (
    CodeSandboxFileSystemAdapter,
)


class CodeSandboxFileManager(FileManager):
    def __init__(self, sandbox_id: str, auth_token: str):
        super().__init__(adapter=CodeSandboxFileSystemAdapter(sandbox_id, auth_token))
