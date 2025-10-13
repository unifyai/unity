from __future__ import annotations

import json
import logging
import os
import tempfile
import functools
from typing import Any, Callable, Dict, List, Optional, Union
from typing import AsyncIterator

import unify

from unity.file_manager.base import BaseFileManager
from unity.file_manager.parser.base import BaseParser
from unity.file_manager.parser.docling_parser import DoclingParser
from unity.file_manager.types.file import File as FileRow
from unity.file_manager.fs_adapters.base import BaseFileSystemAdapter
from unity.file_manager.prompt_builders import (
    build_file_manager_ask_prompt,
    build_file_manager_ask_about_file_prompt,
    build_file_manager_organize_prompt,
)
from unity.common.llm_helpers import (
    methods_to_tool_dict,
    make_request_clarification_tool,
)
from unity.common.async_tool_loop import (
    TOOL_LOOP_LINEAGE,
    SteerableToolHandle,
    start_async_tool_loop,
)
from unity.constants import is_readonly_ask_guard_enabled
from unity.common.read_only_ask_guard import ReadOnlyAskGuardHandle
from unity.events.manager_event_logging import log_manager_call
from unity.events.event_bus import EVENT_BUS, Event
from unity.common.context_store import TableStore
from unity.common.model_to_fields import model_to_fields
from unity.common.semantic_search import (
    fetch_top_k_by_references,
    backfill_rows,
)
from unity.common.filter_utils import normalize_filter_expr


class FileManager(BaseFileManager):
    """A FileManager bound to exactly one filesystem adapter.

    - Discovery/search index is backed by adapter capabilities
    - Bytes are retrieved from the adapter and parsed on demand
    - Exposes unified tools for read-only inspection and safe rename/move
    """

    def __init__(
        self,
        adapter: Optional[BaseFileSystemAdapter] = None,
        *,
        parser: Optional[BaseParser] = None,
        rolling_summary_in_prompts: bool = True,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._parser: BaseParser = (
            parser
            if parser is not None
            else DoclingParser(
                use_llm_enrichment=True,
                extract_images=True,
                extract_tables=True,
            )
        )
        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # Derive a stable alias and context
        try:
            raw_alias = (
                getattr(self._adapter, "name", "files").strip()
                if self._adapter is not None
                else "files"
            )
        except Exception:
            raw_alias = "files"

        self._fs_alias = self._sanitize_ctx_component(str(raw_alias))

        # Extract clean filesystem type for LLM prompts (without path/details)
        self._fs_type = self._extract_filesystem_type(raw_alias)

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
        if not read_ctx:
            try:
                from .. import ensure_initialised as _ensure_initialised  # type: ignore  # local to avoid cycles

                _ensure_initialised()
                ctxs = unify.get_active_context()
                read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
            except Exception:
                pass
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a FileManager."
        base_ctx = read_ctx or "default"
        # Use a single Files namespace and a per‑filesystem suffix
        self._ctx = f"{base_ctx}/Files__{self._fs_alias}"

        # Ensure context and fields exist
        self._store = TableStore(
            self._ctx,
            unique_keys={"file_id": "int"},
            auto_counting={"file_id": None},
            description=(
                "List of files for a single filesystem; parsed content, metadata, and descriptions."
            ),
            fields=model_to_fields(FileRow),
        )
        self._store.ensure_context()

        # Immutable built-in fields derived from the FileRow model
        self._BUILTIN_FIELDS: tuple[str, ...] = tuple(FileRow.model_fields.keys())

        # Public tool dictionaries, mirroring other managers
        ask_tools: Dict[str, Callable] = methods_to_tool_dict(
            # Unify-backed retrieval helpers
            self._list_columns,
            self._filter_files,
            self._search_files,
            # Basic helpers
            self.list,
            self.exists,
            self.parse,
            include_class_name=False,
        )
        self.add_tools("ask", ask_tools)
        ask_about_file_tools: Dict[str, Callable] = methods_to_tool_dict(
            self.parse,
            # Keep lightweight adapter wrappers for file-specific inspection
            self._adapter_get,
            self._adapter_open_bytes,
            include_class_name=False,
        )
        self.add_tools("ask_about_file", ask_about_file_tools)
        organize_tools: Dict[str, Callable] = methods_to_tool_dict(
            # Retrieval surface
            self._list_columns,
            self._filter_files,
            self._search_files,
            # Inventory helpers
            self.list,
            self.exists,
            self._rename_file,
            self._move_file,
            self._delete_file,
            include_class_name=False,
        )
        self.add_tools("organize", organize_tools)

    @staticmethod
    def _sanitize_ctx_component(value: Any) -> str:
        """
        Uniform sanitizer for any single context path component.

        Rules:
        - Only allow [a-zA-Z0-9_-]
        - Replace everything else with '_'
        - Truncate to 64 chars to keep paths readable
        - Fallback to 'item' when empty
        """
        try:
            import re as _re

            cleaned = _re.sub(r"[^a-zA-Z0-9_-]", "_", str(value))
            cleaned = cleaned[:64]
            return cleaned or "item"
        except Exception:
            return "item"

    @staticmethod
    def _extract_filesystem_type(adapter_name: str) -> str:
        """
        Extract the filesystem type from an adapter name, stripping path details.

        Examples:
            "Local[/tmp/path]" -> "Local"
            "CodeSandbox[sbx-123]" -> "CodeSandbox"
            "Interact" -> "Interact"

        This is useful for LLM prompts where we want the type but not implementation details.
        """
        if not adapter_name:
            return "Unknown"
        # Split on '[' and take the first part (the type)
        return adapter_name.split("[")[0].strip() or adapter_name

    # ---------- Adapter wrappers (bytes + identifiers only) ----------------- #
    def _adapter_list(self) -> List[str]:
        try:
            if self._adapter is None:
                return []
            return [ref.path for ref in self._adapter.iter_files()]
        except Exception:
            return []

    def _adapter_get(self, *, target_id_or_path: str) -> Dict[str, Any]:
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for direct file lookups")
        ref = self._adapter.get_file(target_id_or_path)
        return getattr(ref, "model_dump", lambda: ref.__dict__)()

    def _adapter_open_bytes(self, *, target_id_or_path: str) -> Dict[str, Any]:
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for opening file bytes")
        data = self._adapter.open_bytes(target_id_or_path)
        # Return as base64-ish payload for safety; caller can decide how to use
        try:
            import base64

            return {
                "filename": target_id_or_path,
                "bytes_b64": base64.b64encode(data).decode("utf-8"),
            }
        except Exception:
            return {"filename": target_id_or_path, "length": len(data)}

    # ---------- Update/Delete helpers (private, follow manager patterns) ----- #
    def _update_file(
        self,
        *,
        file_id: int,
        _log_id: Optional[int] = None,
        **updates: Any,
    ) -> Dict[str, Any]:
        """
        Update one or more fields of an existing file record in Unify.

        This is a low-level helper that follows the same pattern as
        _update_contact, _update_rows, _update_secret in other managers.

        Parameters
        ----------
        file_id : int
            The unique file ID to update
        _log_id : int | None
            Optional: The specific log ID if already known (avoids lookup)
        **updates : Any
            Field names and new values to update

        Returns
        -------
        dict
            Outcome with 'outcome' and 'details' keys

        Raises
        ------
        ValueError
            If no file found with the given file_id
        """
        if not updates:
            raise ValueError("At least one field must be provided for update")

        # Find the log ID if not provided
        if _log_id is None:
            try:
                log_ids = unify.get_logs(
                    context=self._ctx,
                    filter=f"file_id == {file_id}",
                    limit=2,
                    return_ids_only=True,
                )
            except Exception:
                log_ids = []

            if not log_ids:
                raise ValueError(f"No file found with file_id {file_id}")
            if len(log_ids) > 1:
                raise ValueError(
                    f"Multiple files found with file_id {file_id}. Data integrity issue.",
                )

            _log_id = log_ids[0]

        # Perform the update
        unify.update_logs(
            logs=[_log_id],
            context=self._ctx,
            entries=updates,
            overwrite=True,
        )

        return {
            "outcome": "file updated",
            "details": {"file_id": file_id, "updated_fields": list(updates.keys())},
        }

    # ---------- Adapter-backed mutators (capability-guarded) --------------- #
    def _rename_file(self, *, target_id_or_path: str, new_name: str) -> Dict[str, Any]:
        if not getattr(self._adapter.capabilities, "can_rename", False):
            raise PermissionError("Rename not permitted by backend policy")

        # Ensure string type (LLM sometimes passes integers)
        target_id_or_path = str(target_id_or_path).lstrip("/")
        new_name = str(new_name)

        # Fetch existing log entry with all fields for comprehensive update
        try:
            logs = unify.get_logs(
                context=self._ctx,
                filter=f"filename == {target_id_or_path!r}",
                limit=2,
                from_fields=["file_id", "filename", "records", "metadata"],
            )
        except Exception:
            logs = []

        if not logs:
            raise ValueError(f"File '{target_id_or_path}' not found in Unify logs.")
        if len(logs) > 1:
            raise ValueError(
                f"Multiple files found with filename '{target_id_or_path}'. Data integrity issue.",
            )

        log_entry = logs[0].entries
        log_id = logs[0].id
        file_id = log_entry.get("file_id")
        old_filename = log_entry.get("filename", target_id_or_path)

        # The filename field IS the filesystem path for all our adapters
        filesystem_path = target_id_or_path

        # Perform the rename via adapter FIRST
        # Only update Unify if this succeeds (doesn't raise exception)
        ref = self._adapter.rename(filesystem_path, new_name)
        new_path = ref.path.lstrip("/")

        # Filesystem operation succeeded, now update ALL path references in Unify
        # This ensures consistency across filename, records[], and metadata
        try:
            # Update records: replace old file_path with new one in all records
            records = log_entry.get("records", [])
            if records:
                for record in records:
                    if record.get("file_path") == old_filename:
                        record["file_path"] = new_path

            # Update metadata: replace file_path if present
            metadata = log_entry.get("metadata", {})
            if isinstance(metadata, dict) and metadata.get("file_path") == old_filename:
                metadata["file_path"] = new_path

            # Write all updates atomically
            self._update_file(
                file_id=file_id,
                _log_id=log_id,
                filename=new_path,
                records=records,
                metadata=metadata,
            )
        except Exception as e:
            # Log update failed after successful filesystem rename
            # This is a data consistency issue but don't fail the operation
            # since the filesystem change was successful
            logging.getLogger(__name__).warning(
                f"Filesystem rename succeeded but Unify update failed: {e}",
            )

        return getattr(
            ref,
            "model_dump",
            lambda: {"path": ref.path, "name": new_name},
        )()

    def _move_file(
        self,
        *,
        target_id_or_path: str,
        new_parent_path: str,
    ) -> Dict[str, Any]:
        if not getattr(self._adapter.capabilities, "can_move", False):
            raise PermissionError("Move not permitted by backend policy")

        # Ensure string type and strip leading slashes (LLM sometimes passes integers or absolute paths)
        target_id_or_path = str(target_id_or_path).lstrip("/")
        new_parent_path = str(new_parent_path).lstrip("/")

        # Fetch existing log entry with all fields for comprehensive update
        try:
            logs = unify.get_logs(
                context=self._ctx,
                filter=f"filename == {target_id_or_path!r}",
                limit=2,
                from_fields=["file_id", "filename", "records", "metadata"],
            )
        except Exception:
            logs = []

        if not logs:
            raise ValueError(f"File '{target_id_or_path}' not found in Unify logs.")
        if len(logs) > 1:
            raise ValueError(
                f"Multiple files found with filename '{target_id_or_path}'. Data integrity issue.",
            )

        log_entry = logs[0].entries
        log_id = logs[0].id
        file_id = log_entry.get("file_id")
        old_filename = log_entry.get("filename", target_id_or_path)

        # The filename field IS the filesystem path for all our adapters
        filesystem_path = target_id_or_path

        # Perform the move via adapter FIRST
        # Only update Unify if this succeeds (doesn't raise exception)
        ref = self._adapter.move(filesystem_path, new_parent_path)
        new_path = ref.path.lstrip("/")

        # Filesystem operation succeeded, now update ALL path references in Unify
        # This ensures consistency across filename, records[], and metadata
        try:
            # Update records: replace old file_path with new one in all records
            records = log_entry.get("records", [])
            if records:
                for record in records:
                    if record.get("file_path") == old_filename:
                        record["file_path"] = new_path

            # Update metadata: replace file_path if present
            metadata = log_entry.get("metadata", {})
            if isinstance(metadata, dict) and metadata.get("file_path") == old_filename:
                metadata["file_path"] = new_path

            # Write all updates atomically
            self._update_file(
                file_id=file_id,
                _log_id=log_id,
                filename=new_path,
                records=records,
                metadata=metadata,
            )
        except Exception as e:
            # Log update failed after successful filesystem move
            # This is a data consistency issue but don't fail the operation
            # since the filesystem change was successful
            logging.getLogger(__name__).warning(
                f"Filesystem move succeeded but Unify update failed: {e}",
            )

        return getattr(
            ref,
            "model_dump",
            lambda: {"path": ref.path, "parent": new_parent_path},
        )()

    def _delete_file(
        self,
        *,
        file_id: int,
        _log_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Delete a file record from Unify table and optionally from the underlying filesystem.

        This method follows the same pattern as _delete_contact, _delete_secret in other managers.

        Steps:
        1. Lookup file by file_id to get filename and metadata
        2. Check if the file is protected (raises PermissionError if protected)
        3. If adapter supports deletion (can_delete=True), delete from filesystem
        4. Remove the record from the Unify table using unify.delete_logs with log ID

        Parameters
        ----------
        file_id : int
            Unique file ID from the Unify table
        _log_id : int | None
            Optional: The specific log ID if already known (avoids lookup)

        Returns
        -------
        dict
            Result with 'outcome' and 'details' keys

        Raises
        ------
        ValueError
            If no file with the given file_id exists
        PermissionError
            If the file is protected
        RuntimeError
            If multiple files found with the same file_id (data integrity issue)
        """
        # 1. Lookup file by ID and get log ID if not provided
        if _log_id is None:
            try:
                log_ids = unify.get_logs(
                    context=self._ctx,
                    filter=f"file_id == {file_id}",
                    limit=2,
                    return_ids_only=True,
                )
            except Exception:
                log_ids = []

            if not log_ids:
                raise ValueError(f"No file found with file_id {file_id}")
            if len(log_ids) > 1:
                raise RuntimeError(
                    f"Multiple files found with file_id {file_id}. Data integrity issue.",
                )

            _log_id = log_ids[0]

        # Get filename for protected check and filesystem deletion
        try:
            logs = unify.get_logs(
                context=self._ctx,
                filter=f"file_id == {file_id}",
                limit=1,
                from_fields=["file_id", "filename"],
            )
        except Exception:
            logs = []

        if not logs:
            raise ValueError(f"No file found with file_id {file_id}")

        entry = logs[0].entries
        filename = entry.get("filename", "")

        # 2. Check if protected
        if self.is_protected(filename):
            raise PermissionError(
                f"'{filename}' is protected and cannot be deleted by FileManager.",
            )

        # 3. Delete from filesystem if adapter supports it
        if self._adapter is not None and getattr(
            self._adapter.capabilities,
            "can_delete",
            False,
        ):
            try:
                # The filename field IS the filesystem path - use it directly
                self._adapter.delete(filename)
            except NotImplementedError:
                # Adapter doesn't support deletion, just remove from table
                pass
            except FileNotFoundError:
                # File already deleted from filesystem, just clean up table
                pass

        # 4. Delete from Unify table using log ID (not filter)
        try:
            unify.delete_logs(
                context=self._ctx,
                logs=_log_id,
            )
        except Exception as e:
            raise ValueError(f"Failed to delete file record: {e}")

        return {
            "outcome": "file deleted",
            "details": {
                "file_id": file_id,
                "filename": filename,
            },
        }

    # ---------- Unify-backed retrieval + BaseFileManager API --------------- #
    def exists(self, filename: str) -> bool:  # type: ignore[override]
        """Check if a file exists in the underlying filesystem.

        This method delegates to the adapter's exists method to check
        if the file exists in the filesystem. It does NOT check Unify logs,
        which only contain parsed files.

        Parameters
        ----------
        filename : str
            The display name or path of the file to check.

        Returns
        -------
        bool
            True if the file exists in the filesystem, False otherwise.
        """
        if self._adapter is None:
            return False
        return self._adapter.exists(filename)

    def list(self) -> List[str]:  # type: ignore[override]
        """List all files in the underlying filesystem.

        This method delegates to the adapter's list method to list
        files in the filesystem. It does NOT query Unify logs, which only
        contain parsed files.

        Returns
        -------
        list[str]
            List of file paths/display names in the filesystem.
        """
        if self._adapter is None:
            return []
        return self._adapter.list()

    def _create_result_dict(
        self,
        filename: str,
        document: Any,
        *,
        status: str = "success",
        error: Optional[str] = None,
        auto_counting: Optional[Dict[str, Optional[str]]] = None,
        document_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        if status == "error":
            return {
                "filename": filename,
                "status": "error",
                "error": error or "Unknown error",
                "records": [],
                "full_text": "",
                "metadata": {},
                "description": "",
            }
        try:
            # Prefer schema-aware rows when available
            if hasattr(document, "to_schema_rows"):
                records = document.to_schema_rows(
                    auto_counting=auto_counting,
                    document_index=document_index,
                )
            else:
                records = document.to_flat_records()
            full_text = (
                document.to_plain_text()
                if hasattr(document, "to_plain_text")
                else document.full_text
            )
            # Ensure full_text is always a string to avoid embedding issues
            safe_full_text = full_text if full_text is not None else ""
            meta = getattr(document, "metadata", None)
            return {
                "filename": filename,
                "status": "success",
                "error": None,
                "records": records,
                "full_text": full_text,
                "metadata": {
                    "document_id": getattr(document, "document_id", ""),
                    "total_records": len(records),
                    "processing_time": (
                        getattr(meta, "processing_time", 0.0) if meta else 0.0
                    ),
                    "file_path": str(getattr(meta, "file_path", "")) if meta else "",
                    "file_type": (
                        getattr(meta, "file_type", "unknown") if meta else "unknown"
                    ),
                    "file_size": getattr(meta, "file_size", 0) if meta else 0,
                    "created_at": getattr(meta, "created_at", None) if meta else None,
                    "modified_at": getattr(meta, "modified_at", None) if meta else None,
                },
                "description": getattr(meta, "description", "") if meta else "",
            }
        except Exception as e:
            return {
                "filename": filename,
                "status": "error",
                "error": f"Failed to process parsed document: {e}",
                "records": [],
                "full_text": "",
                "metadata": {},
                "description": "",
            }

    def parse(self, filenames: Union[str, List[str]], **options: Any) -> Dict[str, Dict[str, Any]]:  # type: ignore[override]
        """
        Parse one or more files and return flat records for each.

        Args:
            filenames: Single filename (str) or list of filenames to parse
            **options: Additional parser-specific options

        Returns:
            Dict mapping filename to result dict containing:
                - status: "success" or "error"
                - records: List[Dict] of flat records (if success)
                - error: str error message (if error)
                - metadata: Dict with document metadata
        """
        # Extract flattening options (not forwarded to parser)
        auto_counting: Optional[Dict[str, Optional[str]]] = options.pop(
            "auto_counting",
            None,
        )
        document_index_offset: int = int(options.pop("document_index_offset", 0))

        # Validate and prepare paths
        if isinstance(filenames, str):
            filenames = [filenames]
        results: Dict[str, Dict[str, Any]] = {}
        file_paths: List[str] = []
        filename_to_path: Dict[str, str] = {}
        invalid: List[str] = []
        temp_dir = None

        # Create a temporary directory to hold exported files with their original names
        temp_dir = tempfile.mkdtemp(prefix="filemanager_parse_")

        try:
            # Export files from the underlying filesystem to temp directory
            # Each adapter implements export_file appropriately:
            # - LocalAdapter: copies file
            # - CodeSandboxAdapter: downloads via SDK download API
            # - InteractAdapter: streams via API
            for name in filenames:
                try:
                    # Export file with original filename preserved
                    exported_path = self.export_file(name, temp_dir)
                    file_paths.append(exported_path)
                    filename_to_path[exported_path] = name
                except Exception:
                    invalid.append(name)

            for bad in invalid:
                results[bad] = self._create_result_dict(
                    bad,
                    None,
                    status="error",
                    error=f"File not found: {bad}",
                )

            if file_paths:
                try:
                    if len(file_paths) > 1 and hasattr(self._parser, "parse_batch"):
                        documents = self._parser.parse_batch(file_paths, **options)
                    else:
                        documents = [self._parser.parse(file_paths[0], **options)]

                    for idx, document in enumerate(documents):
                        fp = file_paths[idx]
                        name = filename_to_path[fp]
                        result = self._create_result_dict(
                            name,
                            document,
                            auto_counting=auto_counting,
                            document_index=document_index_offset + idx,
                        )
                        results[name] = result
                        try:
                            self._create_file(
                                filename=name,
                                status=result.get("status"),
                                error=result.get("error"),
                                records=result.get("records"),
                                full_text=result.get("full_text"),
                                metadata=result.get("metadata"),
                                description=result.get("description"),
                            )
                        except Exception as e:
                            print(f"Error creating file: {e}")
                        try:
                            # Per-table ingestion (spreadsheets)
                            self._ingest_tables_for_file(
                                filename=name,
                                document=document,
                            )
                        except Exception as e:
                            print(f"Error ingesting tables for file {name}: {e}")
                except Exception as e:
                    print(f"Error parsing file: {e}")
                    for fp in file_paths:
                        name = filename_to_path[fp]
                        try:
                            document = self._parser.parse(fp, **options)
                            result = self._create_result_dict(
                                name,
                                document,
                                auto_counting=auto_counting,
                                document_index=document_index_offset,
                            )
                            results[name] = result
                            try:
                                self._create_file(
                                    filename=name,
                                    status=result.get("status"),
                                    error=result.get("error"),
                                    records=result.get("records"),
                                    full_text=result.get("full_text"),
                                    metadata=result.get("metadata"),
                                    description=result.get("description"),
                                )
                            except Exception as e:
                                print(f"Error creating file {name}: {e}")
                            try:
                                self._ingest_tables_for_file(
                                    filename=name,
                                    document=document,
                                )
                            except Exception as e:
                                print(f"Error ingesting tables for file {name}: {e}")
                        except Exception as indiv_e:
                            err = self._create_result_dict(
                                name,
                                None,
                                status="error",
                                error=str(indiv_e),
                            )
                            results[name] = err
                            try:
                                self._create_file(
                                    filename=name,
                                    status="error",
                                    error=str(indiv_e),
                                    records=[],
                                    full_text="",
                                    metadata={},
                                    description="",
                                )
                            except Exception as e:
                                print(f"Error creating file {name}: {e}")
        finally:
            # Clean up temporary directory and all files within it
            if temp_dir:
                import shutil

                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass

        return results

    async def parse_async(
        self,
        filenames: Union[str, List[str]],
        batch_size: int = 3,
        **options: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Parse one or more files asynchronously, yielding results as they complete.

        Args:
            filenames: Single filename (str) or list of filenames to parse
            batch_size: Number of files to process in parallel
            **options: Additional parser-specific options

        Yields:
            Dict containing:
                - filename: The filename being processed
                - status: "success" or "error"
                - records: List[Dict] of flat records (if success)
                - error: str error message (if error)
                - metadata: Dict with document metadata
        """
        # Extract flattening options (not forwarded to parser)
        auto_counting: Optional[Dict[str, Optional[str]]] = options.pop(
            "auto_counting",
            None,
        )
        document_index_offset: int = int(options.pop("document_index_offset", 0))

        if isinstance(filenames, str):
            filenames = [filenames]
        file_paths: List[str] = []
        filename_to_path: Dict[str, str] = {}
        invalid: List[str] = []
        temp_dir = None

        # Create a temporary directory to hold exported files with their original names
        temp_dir = tempfile.mkdtemp(prefix="filemanager_parse_async_")

        try:
            # Export files from the underlying filesystem to temp directory
            # Each adapter implements export_file appropriately:
            # - LocalAdapter: copies file
            # - CodeSandboxAdapter: downloads via SDK download API
            # - InteractAdapter: streams via API
            for name in filenames:
                try:
                    # Export file with original filename preserved
                    exported_path = self.export_file(name, temp_dir)
                    file_paths.append(exported_path)
                    filename_to_path[exported_path] = name
                except Exception:
                    invalid.append(name)

            for bad in invalid:
                yield self._create_result_dict(
                    bad,
                    None,
                    status="error",
                    error=f"File not found: {bad}",
                )

            if file_paths:
                if hasattr(self._parser, "parse_batch_async"):
                    async for index, document in self._parser.parse_batch_async(
                        file_paths,
                        batch_size=batch_size,
                        **options,
                    ):
                        fp = file_paths[index]
                        name = filename_to_path[fp]
                        result = self._create_result_dict(
                            name,
                            document,
                            auto_counting=auto_counting,
                            document_index=document_index_offset + index,
                        )
                        try:
                            self._create_file(
                                filename=name,
                                status=result.get("status"),
                                error=result.get("error"),
                                records=result.get("records"),
                                full_text=result.get("full_text"),
                                metadata=result.get("metadata"),
                                description=result.get("description"),
                            )
                        except Exception as e:
                            print(f"Error creating file {name}: {e}")
                        try:
                            self._ingest_tables_for_file(
                                filename=name,
                                document=document,
                            )
                        except Exception as e:
                            print(f"Error ingesting tables for file {name}: {e}")
                        yield result
                else:
                    # Fallback: sequential parse in async generator
                    for fp in file_paths:
                        name = filename_to_path[fp]
                        try:
                            document = self._parser.parse(fp, **options)
                            result = self._create_result_dict(
                                name,
                                document,
                                auto_counting=auto_counting,
                                document_index=document_index_offset,
                            )
                            try:
                                self._create_file(
                                    filename=name,
                                    status=result.get("status"),
                                    error=result.get("error"),
                                    records=result.get("records"),
                                    full_text=result.get("full_text"),
                                    metadata=result.get("metadata"),
                                    description=result.get("description"),
                                )
                            except Exception as e:
                                print(f"Error creating file {name}: {e}")
                            yield result
                        except Exception as indiv_e:
                            print(f"Error creating file {name}: {indiv_e}")
                            yield self._create_result_dict(
                                name,
                                None,
                                status="error",
                                error=str(indiv_e),
                            )
        finally:
            # Clean up temporary directory and all files within it
            if temp_dir:
                import shutil

                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Error cleaning up temporary directory: {e}")

    def _open_bytes_by_filename(self, filename: str) -> bytes:
        """Return file bytes by consulting Unify metadata first, falling back to adapter."""
        # Try Unify metadata.source_path
        try:
            rows = unify.get_logs(
                context=self._ctx,
                filter=f"filename == {filename!r}",
                limit=1,
                from_fields=["filename", "metadata"],
            )
        except Exception as e:
            print(f"Error opening bytes by filename: {e}")
            rows = []
        if rows:
            try:
                e = rows[0].entries
                meta = e.get("metadata") or {}
                src = meta.get("source_path")
                if src:
                    from pathlib import Path as _Path

                    p = _Path(str(src))
                    if p.exists() and p.is_file():
                        return p.read_bytes()
            except Exception as e:
                print(f"Error reading bytes by filename: {e}")
        # Fallback to adapter if present
        if self._adapter is not None:
            return self._adapter.open_bytes(filename)
        raise FileNotFoundError(f"Unable to resolve file bytes for '{filename}'")

    # ---------- Unify table helpers (schema + retrieval) ------------------ #
    def _get_columns(self) -> Dict[str, str]:
        return self._store.get_columns()

    def _list_columns(
        self,
        *,
        include_types: bool = True,
    ) -> Dict[str, Any] | List[str]:
        cols = self._get_columns()
        return cols if include_types else list(cols)

    def _allowed_fields(self) -> List[str]:
        return list(self._BUILTIN_FIELDS)

    def _filter_files(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[FileRow]:
        """
        Filter files using a boolean Python expression evaluated per row.
        Mirrors ContactManager.filter_contacts.

        Parameters
        ----------
        filter : str | None, default None
            A Python boolean expression evaluated with column names in scope. Examples:
            - "filename.endswith('.pdf')"
            - "status == 'success'"
            - "metadata['file_size'] > 1000000"
            When None, returns all files. String comparisons are case‑sensitive unless
            your expression applies a case‑normalisation.
        offset : int, default 0
            Zero-based index of the first result to include.
        limit : int, default 100
            Maximum number of records to return. Must be <= 1000.

        Returns
        -------
        List[FileRow]
            Matching files as File objects in creation order.
        
        Notes
        -----
        - Be careful with quoting inside the expression. Use single quotes to delimit string
          literals inside the filter string.
        - This tool is brittle for substring searches across text; prefer ``_search_files``
          for that purpose.
        """
        from unity.common.embed_utils import list_private_fields

        try:
            normalized = normalize_filter_expr(filter)
            logs = unify.get_logs(
                context=self._ctx,
                filter=normalized,
                offset=offset,
                limit=limit,
                from_fields=self._allowed_fields(),
                exclude_fields=list_private_fields(self._ctx),
            )
        except Exception as e:
            print(f"Error filtering files: {e}")
            logs = []
        rows = [FileRow(**lg.entries) for lg in logs]
        return rows

    def _search_files(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[FileRow]:
        """
        Semantic search over files using one or more reference texts.

        Parameters
        ----------
        references : Dict[str, str] | None, default None
            Mapping of ``source_expr → reference_text`` terms that define the search space.
            - ``source_expr`` can be either a simple column name (e.g. ``"bio"``,
              ``"first_name"``) or a full Unify derived‑expression (e.g.
              ``"str({first_name}) + ' ' + str({surname})"``). For expressions, a stable
              derived source column is created automatically if needed.
            - ``reference_text`` is free‑form text which will be embedded using the
              configured embedding model.
              When None or empty dict, returns the most recent files.
        k : int, default 10
            Maximum number of files to return. Must be a positive integer. Must be <= 1000.

        Returns
        -------
        List[File]
            Up to k File objects. When semantic references are provided,
            results are sorted by similarity. When references are omitted,
            returns the most recent files.
        """
        from unity.common.search_utils import table_search_top_k

        # Restrict payload to the File schema to avoid fetching private/vector fields
        allowed_fields = list(FileRow.model_fields.keys())

        rows = table_search_top_k(
            context=self._ctx,
            references=references,
            k=k,
            allowed_fields=allowed_fields,
            unique_id_field="file_id",
        )
        return [FileRow(**r) for r in rows]

    # Internal: client factory mirroring other managers
    def _new_llm_client(self, model: str) -> "unify.AsyncUnify":
        return unify.AsyncUnify(
            model,
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "false")),
            reasoning_effort="high",
            service_tier="priority",
        )

    # ---------- Row creation (private) ------------------------------------ #
    def _create_file(
        self,
        *,
        filename: str,
        status: Optional[str] = None,
        error: Optional[str] = None,
        records: Optional[List[Dict[str, Any]]] = None,
        full_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        entries: Dict[str, Any] = {
            "filename": filename,
        }
        if status is not None:
            entries["status"] = status
        if error is not None:
            entries["error"] = error
        if records is not None:
            entries["records"] = records
        if full_text is not None:
            entries["full_text"] = full_text
        if metadata is not None:
            entries["metadata"] = metadata
        if description is not None:
            entries["description"] = description

        log = unify.log(
            context=self._ctx,
            **entries,
            new=True,
            mutable=True,
        )
        return {
            "outcome": "file created successfully",
            "details": {
                "file_id": log.entries.get("file_id"),
                "filename": filename,
            },
        }

    # ---------- Per-table ingestion for spreadsheets (CSV/XLSX/Sheets) ----- #
    def _ingest_tables_for_file(self, *, filename: str, document: Any) -> None:
        """
        Create one sub-context per extracted table and log its rows.

        Context naming: <FilesCtx>/Tables__<fs_alias>/<safe_filename>/<safe_table_label>
        Schema: dynamic per table – inferred from detected column names (str), with
        an auto-incrementing unique key `row_id`.
        """
        try:
            import unify
            from unity.knowledge_manager.types import ColumnType
        except Exception:
            ColumnType = None  # type: ignore

        safe = self._sanitize_ctx_component

        tables = getattr(getattr(document, "metadata", None), "tables", []) or []
        if not tables:
            return

        # Only ingest when structured rows/columns are available
        for idx, tbl in enumerate(tables, start=1):
            columns = getattr(tbl, "columns", None)
            rows = getattr(tbl, "rows", None)
            if not rows:
                continue

            # Derive column names. If missing, synthesize generic headers
            if not columns:
                # Try to infer number of cols from first row
                num_cols = len(rows[0]) if rows and len(rows) > 0 else 0
                columns = [f"col_{i+1}" for i in range(num_cols)]

            # Build a stable table context name
            sheet_name = getattr(tbl, "sheet_name", None)
            table_label = f"{idx:02d}_{sheet_name}" if sheet_name else f"{idx:02d}"
            table_ctx = f"{self._ctx}/Tables__{self._fs_alias}/{safe(filename)}/{safe(table_label)}"
            try:
                unify.create_context(
                    table_ctx,
                    unique_keys={"row_id": "int"},
                    auto_counting={"row_id": None},
                    description=(
                        f"Rows for table #{idx} from file '{filename}'"
                        + (f" (sheet: {sheet_name})" if sheet_name else "")
                    ),
                )
            except Exception as e:
                print(f"Error creating table context: {e}")

            # Rely on backend to infer fields/types from logged rows; do not create fields explicitly
            # since we cannot differentiate between ambiguous entries e.g. "2025-01-01" can be a date or a string

            for r in rows:
                try:
                    entry = {
                        str(col): (str(val) if val is not None else "")
                        for col, val in zip(columns, r)
                    }
                    unify.log(context=table_ctx, **entry, new=True, mutable=True)
                except Exception as e:
                    print(f"Error logging table row: {e}")
                    continue

    # ---------- High-level importers (delegated to adapter) ---------------- #
    def import_file(self, file_path: Any) -> str:
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for import_file")
        return self._adapter.import_file(str(file_path))

    def import_directory(self, directory: Any) -> List[str]:
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for import_directory")
        return self._adapter.import_directory(str(directory))

    def export_file(self, filename: str, destination_dir: str) -> str:  # type: ignore[override]
        """Export a file from the underlying filesystem to a local destination directory.

        This method delegates to the adapter's export_file method, which:
        - LocalAdapter: copies file preserving metadata
        - CodeSandboxAdapter: downloads via SDK (with download API)
        - InteractAdapter: streams via API

        Parameters
        ----------
        filename : str
            The display name or path of the file to export.
        destination_dir : str
            Local directory path where the file should be exported.

        Returns
        -------
        str
            Full path to the exported file in the destination directory.
        """
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for export_file")
        return self._adapter.export_file(filename, destination_dir)

    def export_directory(self, directory: str, destination_dir: str) -> List[str]:  # type: ignore[override]
        """Export all files from a directory to a local destination directory.

        This method delegates to the adapter's export_directory method, which
        optimizes for batch operations where possible (e.g., zip downloads for CodeSandbox).

        Parameters
        ----------
        directory : str
            The directory path to export files from.
        destination_dir : str
            Local directory path where files should be exported.

        Returns
        -------
        list[str]
            List of full paths to exported files in the destination directory.
        """
        if self._adapter is None:
            raise NotImplementedError("No adapter configured for export_directory")
        return self._adapter.export_directory(directory, destination_dir)

    def register_existing_file(
        self,
        path: Any,
        *,
        display_name: Optional[str] = None,
        protected: bool = False,
    ) -> str:
        if self._adapter is None:
            raise NotImplementedError(
                "No adapter configured for register_existing_file",
            )
        return self._adapter.register_existing_file(
            str(path),
            display_name=display_name,
            protected=protected,
        )

    def is_protected(self, filename: str) -> bool:
        if self._adapter is None:
            return False
        return self._adapter.is_protected(filename)

    def save_file_to_downloads(self, filename: str, contents: bytes) -> str:
        if self._adapter is None:
            raise NotImplementedError(
                "No adapter configured for save_file_to_downloads",
            )
        return self._adapter.save_file_to_downloads(filename, contents)

    # Filesystem-level Q&A
    @functools.wraps(BaseFileManager.ask, updated=())
    @log_manager_call("FileManager", "ask", payload_key="question")
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[Any] = None,
        _clarification_down_q: Optional[Any] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:  # type: ignore[override]
        client = self._new_llm_client("gpt-5@openai")
        tools = dict(self.get_tools("ask"))
        if _clarification_up_q is not None and _clarification_down_q is not None:

            async def _on_request(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "FileManager",
                                "method": "ask",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_answer(ans: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "FileManager",
                                "method": "ask",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

            tools["request_clarification"] = make_request_clarification_tool(
                _clarification_up_q,
                _clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        system_msg = build_file_manager_ask_prompt(
            tools=tools,
            num_files=len(self.list()),
            columns={},
            include_activity=include_activity,
        )
        client.set_system_message(system_msg)
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.ask",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=lambda i, t: ("required", t) if i < 1 else ("auto", t),
            handle_cls=(
                ReadOnlyAskGuardHandle if is_readonly_ask_guard_enabled() else None
            ),
        )
        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result  # type: ignore[attr-defined]
        return handle

    # File-specific Q&A
    @functools.wraps(BaseFileManager.ask_about_file, updated=())
    @log_manager_call("FileManager", "ask_about_file", payload_key="question")
    async def ask_about_file(
        self,
        filename: str,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[Any] = None,
        _clarification_down_q: Optional[Any] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:  # type: ignore[override]
        if not self.exists(filename):
            raise FileNotFoundError(filename)
        client = self._new_llm_client("gpt-5@openai")
        tools = dict(self.get_tools("ask_about_file"))
        if _clarification_up_q is not None and _clarification_down_q is not None:

            async def _on_request(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "FileManager",
                                "method": "ask_about_file",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_answer(ans: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "FileManager",
                                "method": "ask_about_file",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

            tools["request_clarification"] = make_request_clarification_tool(
                _clarification_up_q,
                _clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        system_msg = build_file_manager_ask_about_file_prompt(
            tools=tools,
            include_activity=include_activity,
        )
        client.set_system_message(system_msg)
        # Use filesystem type without exposing absolute paths to LLM
        user_blob = json.dumps(
            {"filesystem": self._fs_type, "filename": filename, "question": question},
            indent=2,
        )
        handle = start_async_tool_loop(
            client,
            user_blob,
            tools,
            loop_id=f"{self.__class__.__name__}.ask_about_file",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=lambda i, t: ("required", t) if i < 1 else ("auto", t),
            handle_cls=(
                ReadOnlyAskGuardHandle if is_readonly_ask_guard_enabled() else None
            ),
        )
        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result  # type: ignore[attr-defined]
        return handle

    # Filesystem reorganization
    @functools.wraps(BaseFileManager.organize, updated=())
    @log_manager_call("FileManager", "organize", payload_key="text")
    async def organize(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _clarification_up_q: Optional[Any] = None,
        _clarification_down_q: Optional[Any] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:  # type: ignore[override]
        client = self._new_llm_client("gpt-5@openai")
        tools = dict(self.get_tools("organize"))
        if _clarification_up_q is not None and _clarification_down_q is not None:

            async def _on_request(q: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "FileManager",
                                "method": "organize",
                                "action": "clarification_request",
                                "question": q,
                            },
                        ),
                    )
                except Exception:
                    pass

            async def _on_answer(ans: str):
                try:
                    await EVENT_BUS.publish(
                        Event(
                            type="ManagerMethod",
                            calling_id=_call_id,
                            payload={
                                "manager": "FileManager",
                                "method": "organize",
                                "action": "clarification_answer",
                                "answer": ans,
                            },
                        ),
                    )
                except Exception:
                    pass

            tools["request_clarification"] = make_request_clarification_tool(
                _clarification_up_q,
                _clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )
        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )
        system_msg = build_file_manager_organize_prompt(
            tools=tools,
            num_files=len(self.list()),
            columns={},
            include_activity=include_activity,
        )
        client.set_system_message(system_msg)
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.organize",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=_parent_chat_context,
            tool_policy=lambda i, t: ("required", t) if i < 1 else ("auto", t),
            handle_cls=(
                ReadOnlyAskGuardHandle if is_readonly_ask_guard_enabled() else None
            ),
        )
        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result  # type: ignore[attr-defined]
        return handle
