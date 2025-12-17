from __future__ import annotations

import io
import logging
import mimetypes
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any

from unity.file_manager.filesystem_adapters.base import BaseFileSystemAdapter
from unity.file_manager.types.filesystem import FileSystemCapabilities, FileReference

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow

    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False


class GoogleDriveAdapter(BaseFileSystemAdapter):
    """
    Adapter for Google Drive using the Google Drive API v3.

    This adapter provides read access to Google Drive files and supports
    downloading files to local storage for parsing.

    References:
    - https://developers.google.com/workspace/drive/api/guides/about-files
    - https://developers.google.com/workspace/drive/api/guides/manage-downloads
    - https://developers.google.com/workspace/drive/api/guides/search-files
    """

    # OAuth 2.0 scopes for Drive API access
    SCOPES = [
        "https://www.googleapis.com/auth/drive.readonly",  # Read-only access
        "https://www.googleapis.com/auth/drive.metadata.readonly",  # Metadata access
    ]

    def __init__(
        self,
        credentials: Optional[Any] = None,
        token_path: Optional[str] = None,
        credentials_path: Optional[str] = None,
        root_folder_id: Optional[str] = None,
    ):
        """
        Initialize Google Drive adapter.

        Parameters
        ----------
        credentials : Credentials, optional
            Google OAuth2 credentials object. If not provided, will attempt
            to load from token_path or authenticate via credentials_path.
        token_path : str, optional
            Path to stored token file (default: 'token.json')
        credentials_path : str, optional
            Path to OAuth2 credentials file (default: 'credentials.json')
        root_folder_id : str, optional
            Restrict operations to this folder and its descendants.
            If None, operates on entire "My Drive".
        """
        if not GOOGLE_DRIVE_AVAILABLE:
            raise ImportError(
                "Google Drive API libraries not available. "
                "Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib",
            )

        self._root_folder_id = root_folder_id or "root"  # "root" = My Drive root
        self._token_path = token_path or "token.json"
        self._credentials_path = credentials_path or "credentials.json"

        # Initialize credentials
        if credentials:
            self._creds = credentials
        else:
            self._creds = self._get_credentials()

        # Build Drive API service
        self._service = build("drive", "v3", credentials=self._creds)

        # Set capabilities (read-only by default)
        self._caps = FileSystemCapabilities(
            can_read=True,
            can_rename=False,  # Rename requires write access
            can_move=False,  # Move requires write access
            can_delete=False,  # Delete requires write access
        )

        # Cache for folder structure
        self._folder_cache: Dict[str, str] = {}  # path -> folder_id

    @property
    def name(self) -> str:
        return f"GoogleDrive"

    @property
    def uri_name(self) -> str:
        return "gdrive"

    @property
    def capabilities(self) -> FileSystemCapabilities:
        return self._caps

    def _get_credentials(self) -> Credentials:
        """
        Get or refresh Google OAuth2 credentials.

        Returns
        -------
        Credentials
            Valid Google OAuth2 credentials.
        """
        creds = None

        # Load existing token if available
        if Path(self._token_path).exists():
            try:
                creds = Credentials.from_authorized_user_file(
                    self._token_path,
                    self.SCOPES,
                )
            except Exception as e:
                logging.warning(f"Failed to load token from {self._token_path}: {e}")

        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logging.warning(f"Failed to refresh credentials: {e}")
                    creds = None

            if not creds:
                # Run OAuth flow
                if not Path(self._credentials_path).exists():
                    raise FileNotFoundError(
                        f"Credentials file not found: {self._credentials_path}. "
                        "Download from Google Cloud Console.",
                    )

                flow = InstalledAppFlow.from_client_secrets_file(
                    self._credentials_path,
                    self.SCOPES,
                )
                creds = flow.run_local_server(port=0)

            # Save credentials for future use
            try:
                with open(self._token_path, "w") as token:
                    token.write(creds.to_json())
            except Exception as e:
                logging.warning(f"Failed to save token to {self._token_path}: {e}")

        return creds

    def _get_folder_name(self, folder_id: str) -> str:
        """Get the name of a folder by its ID."""
        if folder_id == "root":
            return "My Drive"

        try:
            file_metadata = (
                self._service.files().get(fileId=folder_id, fields="name").execute()
            )
            return file_metadata.get("name", folder_id)
        except Exception:
            return folder_id

    def _list_files_in_folder(
        self,
        folder_id: str,
        recursive: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        List all files in a folder using the Google Drive API.

        Parameters
        ----------
        folder_id : str
            The Google Drive folder ID to list files from.
        recursive : bool
            Whether to recursively list files in subfolders.

        Returns
        -------
        list[dict]
            List of file metadata dictionaries from Drive API.
        """
        files: List[Dict[str, Any]] = []
        page_token = None

        # Query for all files (not folders) under the specified folder
        query = f"'{folder_id}' in parents and trashed=false"

        try:
            while True:
                response = (
                    self._service.files()
                    .list(
                        q=query,
                        spaces="drive",
                        fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, parents)",
                        pageToken=page_token,
                        pageSize=1000,
                    )
                    .execute()
                )

                items = response.get("files", [])

                for item in items:
                    mime_type = item.get("mimeType", "")

                    # If it's a folder and recursive is True, recurse
                    if mime_type == "application/vnd.google-apps.folder" and recursive:
                        subfolder_files = self._list_files_in_folder(
                            item["id"],
                            recursive=True,
                        )
                        files.extend(subfolder_files)
                    elif mime_type != "application/vnd.google-apps.folder":
                        # It's a file, add it
                        files.append(item)

                page_token = response.get("nextPageToken")
                if not page_token:
                    break

        except Exception as e:
            logging.error(f"Error listing files in folder {folder_id}: {e}")

        return files

    def _get_file_path(self, file_id: str, file_name: str) -> str:
        """
        Construct a virtual path for a file by traversing parent folders.

        Parameters
        ----------
        file_id : str
            The Google Drive file ID.
        file_name : str
            The file name.

        Returns
        -------
        str
            Virtual path like "/folder1/folder2/filename.ext"
        """
        try:
            # Get file metadata with parents
            file_metadata = (
                self._service.files().get(fileId=file_id, fields="parents").execute()
            )
            parents = file_metadata.get("parents", [])

            if not parents or parents[0] == self._root_folder_id:
                return f"/{file_name}"

            # Build path by traversing parents
            path_parts = [file_name]
            current_parent = parents[0]

            while current_parent and current_parent != self._root_folder_id:
                try:
                    parent_metadata = (
                        self._service.files()
                        .get(fileId=current_parent, fields="name, parents")
                        .execute()
                    )
                    path_parts.insert(0, parent_metadata.get("name", ""))
                    parent_parents = parent_metadata.get("parents", [])
                    current_parent = parent_parents[0] if parent_parents else None
                except Exception:
                    break

            return "/" + "/".join(path_parts)

        except Exception as e:
            logging.debug(f"Could not build path for {file_id}: {e}")
            return f"/{file_name}"

    def iter_files(self, root: Optional[str] = None) -> Iterable[FileReference]:
        """
        Iterate over all files in Google Drive.

        Parameters
        ----------
        root : str, optional
            Folder path to list files from. If None, lists from root folder.

        Yields
        ------
        FileReference
            File reference for each file found.
        """
        # Determine which folder to list from
        folder_id = self._root_folder_id
        if root:
            # Try to resolve root path to folder ID
            # For simplicity, we'll just use root_folder_id for now
            # A full implementation would traverse the path
            pass

        files = self._list_files_in_folder(folder_id, recursive=True)

        for file_metadata in files:
            file_id = file_metadata["id"]
            file_name = file_metadata["name"]
            mime_type = file_metadata.get("mimeType")
            size = (
                int(file_metadata.get("size", 0)) if "size" in file_metadata else None
            )
            modified = file_metadata.get("modifiedTime")

            # Build virtual path
            path = self._get_file_path(file_id, file_name)

            yield FileReference(
                path=path,
                name=file_name,
                provider=self.name,
                uri=f"{self.uri_name}://{file_id}",
                size_bytes=size,
                modified_at=modified,
                mime_type=mime_type,
                extra_metadata={"drive_file_id": file_id},
            )

    def get_file(self, path: str) -> FileReference:
        """
        Get a file reference by path or file ID.

        Parameters
        ----------
        path : str
            File path or Google Drive file ID.

        Returns
        -------
        FileReference
            File reference for the requested file.

        Raises
        ------
        FileNotFoundError
            If file not found.
        """
        # Try to interpret path as file ID first
        try:
            file_metadata = (
                self._service.files()
                .get(fileId=path, fields="id, name, mimeType, size, modifiedTime")
                .execute()
            )

            return FileReference(
                path=self._get_file_path(file_metadata["id"], file_metadata["name"]),
                name=file_metadata["name"],
                provider=self.name,
                uri=f"{self.uri_name}://{file_metadata['id']}",
                size_bytes=(
                    int(file_metadata.get("size", 0))
                    if "size" in file_metadata
                    else None
                ),
                modified_at=file_metadata.get("modifiedTime"),
                mime_type=file_metadata.get("mimeType"),
                extra_metadata={"drive_file_id": file_metadata["id"]},
            )
        except Exception:
            pass

        # If not a file ID, search by path name
        file_name = Path(path).name
        query = f"name = '{file_name}' and '{self._root_folder_id}' in parents and trashed=false"

        try:
            response = (
                self._service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="files(id, name, mimeType, size, modifiedTime)",
                    pageSize=10,
                )
                .execute()
            )

            files = response.get("files", [])
            if not files:
                raise FileNotFoundError(f"File not found: {path}")

            # Return first match
            file_metadata = files[0]
            return FileReference(
                path=self._get_file_path(file_metadata["id"], file_metadata["name"]),
                name=file_metadata["name"],
                provider=self.name,
                uri=f"{self.uri_name}://{file_metadata['id']}",
                size_bytes=(
                    int(file_metadata.get("size", 0))
                    if "size" in file_metadata
                    else None
                ),
                modified_at=file_metadata.get("modifiedTime"),
                mime_type=file_metadata.get("mimeType"),
                extra_metadata={"drive_file_id": file_metadata["id"]},
            )

        except Exception as e:
            raise FileNotFoundError(f"File not found: {path}") from e

    def exists(self, path: str) -> bool:
        """Check if a file exists in Google Drive."""
        try:
            self.get_file(path)
            return True
        except FileNotFoundError:
            return False

    def list(self, root: Optional[str] = None) -> List[str]:
        """List all file paths in Google Drive."""
        try:
            return [ref.path.lstrip("/") for ref in self.iter_files(root)]
        except Exception:
            return []

    def open_bytes(self, path: str) -> bytes:
        """
        Download and return file bytes from Google Drive.

        Parameters
        ----------
        path : str
            File path or Google Drive file ID.

        Returns
        -------
        bytes
            File contents as bytes.

        Raises
        ------
        FileNotFoundError
            If file not found.
        """
        # Get file reference to retrieve file ID
        file_ref = self.get_file(path)
        file_id = file_ref.extra_metadata.get("drive_file_id")

        if not file_id:
            raise FileNotFoundError(f"Could not resolve file ID for: {path}")

        try:
            # Check if it's a Google Workspace document (needs export)
            mime_type = file_ref.mime_type or ""

            if mime_type.startswith("application/vnd.google-apps."):
                # Export Google Workspace documents
                export_mime_type = self._get_export_mime_type(mime_type)
                request = self._service.files().export_media(
                    fileId=file_id,
                    mimeType=export_mime_type,
                )
            else:
                # Download regular files
                request = self._service.files().get_media(fileId=file_id)

            # Download to memory
            file_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(file_buffer, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()

            return file_buffer.getvalue()

        except Exception as e:
            raise FileNotFoundError(f"Failed to download file {path}: {e}") from e

    def _get_export_mime_type(self, google_mime_type: str) -> str:
        """
        Get appropriate export MIME type for Google Workspace documents.

        Reference: https://developers.google.com/workspace/drive/api/guides/ref-export-formats
        """
        export_map = {
            "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # Docs -> DOCX
            "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # Sheets -> XLSX
            "application/vnd.google-apps.presentation": "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # Slides -> PPTX
            "application/vnd.google-apps.drawing": "application/pdf",  # Drawings -> PDF
            "application/vnd.google-apps.script": "application/vnd.google-apps.script+json",  # Apps Script -> JSON
        }

        return export_map.get(google_mime_type, "application/pdf")

    def export_file(self, path: str, destination_dir: str) -> str:
        """
        Export a file from Google Drive to a local destination directory.

        Parameters
        ----------
        path : str
            File path or Google Drive file ID.
        destination_dir : str
            Local directory to export the file to.

        Returns
        -------
        str
            Full path to the exported file.
        """
        # Get file content
        file_bytes = self.open_bytes(path)
        file_ref = self.get_file(path)

        # Create destination directory
        dest_dir = Path(destination_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename (handle Google Workspace exports)
        file_name = file_ref.name
        mime_type = file_ref.mime_type or ""

        if mime_type.startswith("application/vnd.google-apps."):
            # Add appropriate extension for exported Google Workspace docs
            export_mime = self._get_export_mime_type(mime_type)
            extension = mimetypes.guess_extension(export_mime) or ""
            if extension and not file_name.endswith(extension):
                file_name = f"{file_name}{extension}"

        # Build destination path preserving relative structure
        rel_path = path.lstrip("/")
        dest_path = dest_dir / rel_path

        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        dest_path.write_bytes(file_bytes)

        return str(dest_path)

    def export_directory(self, path: str, destination_dir: str) -> List[str]:
        """
        Export all files from a Google Drive folder to local destination.

        Parameters
        ----------
        path : str
            Folder path or ID to export from.
        destination_dir : str
            Local directory to export files to.

        Returns
        -------
        list[str]
            List of exported file paths.
        """
        exported: List[str] = []

        try:
            for file_ref in self.iter_files(path):
                try:
                    exported_path = self.export_file(file_ref.path, destination_dir)
                    exported.append(exported_path)
                except Exception as e:
                    logging.warning(f"Failed to export {file_ref.path}: {e}")
                    continue
        except Exception as e:
            logging.error(f"Failed to export directory {path}: {e}")

        return exported
