from __future__ import annotations

from typing import Optional, Any

from unity.file_manager.filesystem_adapters.google_drive_adapter import (
    GoogleDriveAdapter,
)
from unity.file_manager.managers.file_manager import FileManager
from unity.manager_registry import SingletonABCMeta


class GoogleDriveFileManager(FileManager, metaclass=SingletonABCMeta):
    """
    File manager for Google Drive storage.

    This manager provides access to files stored in Google Drive using the
    Google Drive API. It extends the base FileManager with Google Drive-specific
    initialization.

    Examples
    --------
    >>> # Using default OAuth flow
    >>> manager = GoogleDriveFileManager()
    >>> files = manager.list()

    >>> # Using specific credentials
    >>> manager = GoogleDriveFileManager(
    ...     credentials_path="/path/to/credentials.json",
    ...     token_path="/path/to/token.json"
    ... )

    >>> # Restrict to specific folder
    >>> manager = GoogleDriveFileManager(root_folder_id="folder_id_here")
    """

    def __init__(
        self,
        credentials: Optional[Any] = None,
        token_path: Optional[str] = None,
        credentials_path: Optional[str] = None,
        root_folder_id: Optional[str] = None,
    ):
        """
        Initialize Google Drive file manager.

        Parameters
        ----------
        credentials : Credentials, optional
            Google OAuth2 credentials object. If not provided, will attempt
            to authenticate via token_path or credentials_path.
        token_path : str, optional
            Path to stored OAuth token file (default: 'token.json').
        credentials_path : str, optional
            Path to OAuth2 credentials file downloaded from Google Cloud Console
            (default: 'credentials.json').
        root_folder_id : str, optional
            Google Drive folder ID to use as root. If None, uses "My Drive" root.

        Notes
        -----
        To use this manager, you need to:
        1. Create a project in Google Cloud Console
        2. Enable the Google Drive API
        3. Create OAuth 2.0 credentials (Desktop app)
        4. Download credentials.json
        5. On first run, authenticate via web login to generate token.json
        """
        adapter = GoogleDriveAdapter(
            credentials=credentials,
            token_path=token_path,
            credentials_path=credentials_path,
            root_folder_id=root_folder_id,
        )

        # Call parent FileManager constructor with the adapter
        super().__init__(adapter=adapter)
