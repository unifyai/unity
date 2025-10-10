from __future__ import annotations

import atexit
import asyncio
import functools
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Tuple, Callable
from .types.file import File

import unify

from ..common.llm_helpers import (
    methods_to_tool_dict,
    inject_broader_context,
    make_request_clarification_tool,
)
from ..common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..events.manager_event_logging import (
    log_manager_call,
)
from ..events.event_bus import EVENT_BUS, Event
from .base import BaseFileManager
from .parser import BaseParser, DoclingParser
from .prompt_builders import build_ask_prompt


def _unique_name(existing: set[str], desired: str) -> str:
    base = Path(desired).stem
    ext = Path(desired).suffix
    name = f"{base}{ext}"
    if name not in existing:
        return name
    i = 1
    while True:
        candidate = f"{base} ({i}){ext}"
        if candidate not in existing:
            return candidate
        i += 1


class FileManager(BaseFileManager):
    """
    Concrete in-session file registry.

    - Stores managed files in a private temp directory.
    - Provides read-only tools to list/inspect contents.
    - Cleans up all managed files on process exit.
    """

    def __init__(
        self,
        *,
        parser: Optional[BaseParser] = None,
        rolling_summary_in_prompts: bool = True,
    ) -> None:
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="unity_files_"))
        self._display_to_path: Dict[str, Path] = {}
        # Track display names that are registered as protected (read-only from the FileManager's perspective)
        self._protected_display_names: set[str] = set()
        # Use provided parser or create default DoclingParser
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

        # ------------------------------------------------------------------ #
        #  Unify context – replicate managers' pattern                        #
        # ------------------------------------------------------------------ #
        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        if not read_ctx:
            # Ensure the global assistant/context is selected before we derive our sub-context
            try:
                from .. import (
                    ensure_initialised as _ensure_initialised,
                )  # local to avoid cycles

                _ensure_initialised()
                ctxs = unify.get_active_context()
                read_ctx, write_ctx = ctxs["read"], ctxs["write"]
            except Exception as e:
                # If ensure fails (e.g. offline tests), proceed; downstream will fall back safely
                logging.warning(f"Failed to get active context: {e}")
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a FileManager."

        self._ctx = f"{read_ctx}/Files" if read_ctx else "Files"

        if self._ctx not in unify.get_contexts():
            unify.create_context(
                self._ctx,
                unique_keys={"file_id": "int"},
                auto_counting={"file_id": None},
                description="Registry of files received or downloaded during a session.",
            )

            # Derive column specs from the Pydantic File model so schema changes
            # automatically propagate.
            from ..common.model_to_fields import model_to_fields
            from .types.file import File as _FileModel

            unify.create_fields(
                model_to_fields(_FileModel),
                context=self._ctx,
            )

        # ------------------------------------------------------------------ #
        #  Tools exposed to LLM                                               #
        # ------------------------------------------------------------------ #
        # ask-side tools are read-only, so they never change
        self._ask_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self.list,
                self.exists,
                self.parse,
                self.import_file,
                self.import_directory,
                self._list_columns,
                self._search_files,
                self._filter_files,
                include_class_name=False,
            ),
        }

        # All tools are read-only for FileManager
        self._tools = dict(**self._ask_tools)

        atexit.register(self._cleanup)

    @property
    def supported_formats(self) -> List[str]:
        """Get list of supported file formats from the parser."""
        return self._parser.supported_formats

    # Internal helpers ---------------------------------------------------- #
    def _cleanup(self) -> None:
        try:
            if self._tmp_dir.exists():
                shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception as e:
            logging.warning(f"Failed to cleanup temporary directory: {e}")

    def _add_file(
        self,
        source_path: Path,
        *,
        display_name: Optional[str] = None,
    ) -> str:
        source_path = source_path.expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(str(source_path))
        display_name = display_name or source_path.name
        unique = _unique_name(set(self._display_to_path.keys()), display_name)
        dest = self._tmp_dir / unique
        shutil.copy2(source_path, dest)
        self._display_to_path[unique] = dest
        return unique

    def import_directory(self, directory: str | os.PathLike[str]) -> List[str]:
        """Import all files in a directory (non-recursive). Returns display names."""
        p = Path(directory).expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise NotADirectoryError(str(directory))
        added: List[str] = []
        for child in sorted(p.iterdir()):
            if child.is_file():
                try:
                    added.append(self._add_file(child))
                except Exception:
                    continue
        return added

    def import_file(self, file_path: str | os.PathLike[str]) -> str:
        """Import a single file from the filesystem. Returns display name."""
        return self._add_file(Path(file_path))

    def save_file_to_downloads(self, filename: str, contents: bytes) -> str:
        """
        Save provided contents as a file into the session "Downloads" directory,
        ensuring a unique filename within that directory, then register it using
        the internal _add_file helper.

        Returns the registered display name.
        """
        downloads_dir = self._tmp_dir / "Downloads"
        try:
            downloads_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Sanitize the incoming filename and ensure uniqueness within Downloads
        desired = Path(filename).name or "downloaded_file"
        try:
            existing = {p.name for p in downloads_dir.iterdir() if p.is_file()}
        except Exception:
            existing = set()
        unique_name = _unique_name(existing, desired)

        target_path = downloads_dir / unique_name

        # Write contents to the unique target path
        try:
            with open(target_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            raise RuntimeError(f"Failed to write to Downloads: {e}")

        # Register the file to display_to_path
        return self.register_existing_file(
            str(target_path),
            display_name=f"downloads/{unique_name}",
        )

    # Protected/registration helpers ------------------------------------- #
    def register_existing_file(
        self,
        path: str | os.PathLike[str],
        *,
        display_name: Optional[str] = None,
        protected: bool = False,
    ) -> str:
        """
        Register an already-existing file on disk for read-only access.

        - Does not copy the file into the temp directory
        - Makes the file visible via FileManager.list()/exists()/parse()
        - When protected=True, the file cannot be deleted/mutated by FileManager APIs
        """
        p = Path(path).expanduser().resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(str(p))

        # Use provided display name verbatim when given, otherwise default to the filename
        name = display_name or p.name
        if name in self._display_to_path and self._display_to_path[name] != p:
            # Avoid accidental collision; make a stable unique variant
            name = _unique_name(set(self._display_to_path.keys()), name)

        self._display_to_path[name] = p
        if protected:
            self._protected_display_names.add(name)
        return name

    def is_protected(self, filename: str) -> bool:
        """Return True if the registered display name is marked protected."""
        return filename in self._protected_display_names

    # Helper methods ------------------------------------------------------ #
    def _validate_and_prepare_files(
        self,
        filenames: Union[str, List[str]],
    ) -> Tuple[List[Path], Dict[str, str], List[str]]:
        """
        Validate files exist and prepare for parsing.

        Args:
            filenames: Single filename or list of filenames

        Returns:
            Tuple of:
                - file_paths: List of valid file paths
                - filename_to_path: Dict mapping path strings to filenames
                - invalid_files: List of filenames that don't exist
        """
        # Normalize input to always be a list
        if isinstance(filenames, str):
            filenames = [filenames]

        file_paths = []
        filename_to_path = {}
        invalid_files = []

        for filename in filenames:
            if filename not in self._display_to_path:
                invalid_files.append(filename)
            else:
                file_path = self._display_to_path[filename]
                file_paths.append(file_path)
                filename_to_path[str(file_path)] = filename

        return file_paths, filename_to_path, invalid_files

    def _create_result_dict(
        self,
        filename: str,
        document: Any,
        status: str = "success",
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a standardized result dictionary for a parsed document.
        This output is directly compatible with the File pydantic model.

        Args:
            filename: The filename
            document: The parsed document (or None if error)
            status: "success" or "error"
            error: Error message if status is "error"

        Returns:
            Standardized result dictionary compatible with File model
        """
        if status == "error":
            return {
                "filename": filename,
                "status": "error",
                "error": error or "Unknown error",
                "records": [],
                "full_text": "",  # Ensure full_text is always a string
                "metadata": {},
                "description": "",
            }

        try:
            records = document.to_flat_records()
            full_text = (
                document.to_plain_text()
                if hasattr(document, "to_plain_text")
                else document.full_text
            )
            # Ensure full_text is always a string to avoid embedding issues
            safe_full_text = full_text if full_text is not None else ""

            return {
                "filename": filename,
                "status": "success",
                "error": None,
                "records": records,
                "full_text": safe_full_text,
                "metadata": {
                    "document_id": document.document_id,
                    "total_records": len(records),
                    "processing_time": getattr(
                        document.metadata,
                        "processing_time",
                        0.0,
                    ),
                    "file_path": (
                        str(document.metadata.file_path)
                        if hasattr(document.metadata, "file_path")
                        else ""
                    ),
                    "file_type": getattr(document.metadata, "file_type", "unknown"),
                    "file_size": getattr(document.metadata, "file_size", 0),
                    "created_at": getattr(document.metadata, "created_at", None),
                    "modified_at": getattr(document.metadata, "modified_at", None),
                },
                "description": getattr(document.metadata, "description", ""),
            }
        except Exception as e:
            return {
                "filename": filename,
                "status": "error",
                "error": f"Failed to process parsed document: {str(e)}",
                "records": [],
                "full_text": "",  # Ensure full_text is always a string
                "metadata": {},
                "description": "",
            }

    def _create_file(
        self,
        *,
        filename: str,
        status: str = "success",
        error: Optional[str] = None,
        records: Optional[List[Dict[str, Any]]] = None,
        full_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Create and persist a new file record in the context table.

        This mirrors ContactManager._create_contact but for files.
        IMPORTANT: This method only creates new records. Filenames should be
        unique due to the _unique_name() logic in import operations.

        Parameters
        ----------
        filename : str
            Display filename unique within the session.
        status : str, default "success"
            Processing status: "success" or "error".
        error : str | None
            Error message if status is "error".
        records : List[Dict[str, Any]] | None
            Flat records from document parsing.
        full_text : str | None
            Complete plain text content of the parsed file.
        metadata : Dict[str, Any] | None
            File metadata including document_id, processing_time, etc.
        **kwargs
            Additional custom fields for the file record.

        Returns
        -------
        Dict[str, Any]
            Standard outcome dict with file_id.
        """
        # Ensure full_text is always a string to avoid embedding issues
        # When full_text is None, embeddings will fail with "'NoneType' object is not iterable"
        safe_full_text = full_text if full_text is not None else ""

        # Build the file record dictionary
        file_details = {
            "filename": filename,
            "status": status,
            "error": error,
            "records": records or [],
            "full_text": safe_full_text,
            "metadata": metadata or {},
            "description": description or "",
        }

        # Merge any additional custom fields
        if kwargs:
            file_details.update(kwargs)

        # Create new file record (filenames should be unique due to _unique_name logic)
        log = unify.log(
            context=self._ctx,
            **file_details,
            new=True,
            mutable=True,
        )
        return {
            "outcome": "file created successfully",
            "details": {"file_id": log.entries["file_id"], "filename": filename},
        }

    def _search_files(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[File]:
        """
        Semantic search over files using one or more reference texts.

        Parameters
        ----------
        references : Dict[str, str] | None, default None
            Mapping of source_expr → reference_text terms that define the search space.
            - source_expr can be a column name (e.g. "full_text", "filename") or a
              Unify derived-expression.
            - reference_text is free-form text which will be embedded.
            When None or empty dict, returns the most recent files.
        k : int, default 10
            Maximum number of files to return.

        Returns
        -------
        List[File]
            Up to k File objects. When semantic references are provided,
            results are sorted by similarity. When references are omitted,
            returns the most recent files.
        """
        from ..common.semantic_search import (
            fetch_top_k_by_references,
            backfill_rows,
        )
        from .types.file import File

        rows = fetch_top_k_by_references(
            self._ctx,
            references,
            k=k,
            row_filter="status == 'success'",
        )
        filled = backfill_rows(
            self._ctx,
            rows,
            k,
            unique_id_field="file_id",
            row_filter="status == 'success'",
        )
        return [File(**r) for r in filled]

    def _filter_files(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[File]:
        """
        Filter files using a boolean Python expression evaluated per row.
        Mirrors ContactManager._filter_contacts.

        Parameters
        ----------
        filter : str | None, default None
            A Python boolean expression evaluated with column names in scope. Examples:
            - "filename.endswith('.pdf')"
            - "status == 'success'"
            - "metadata['file_size'] > 1000000"
            When None, returns all files.
        offset : int, default 0
            Zero-based index of the first result to include.
        limit : int, default 100
            Maximum number of records to return.

        Returns
        -------
        List[File]
            Matching files as File objects in creation order.
        """
        from ..common.embed_utils import list_private_fields
        from .types.file import File

        logs = unify.get_logs(
            context=self._ctx,
            filter=filter,
            offset=offset,
            limit=limit,
            exclude_fields=list_private_fields(self._ctx),
        )
        return [File(**lg.entries) for lg in logs]

    def _delete_file(
        self,
        *,
        file_id: int,
    ) -> Dict[str, Any]:
        """
        Permanently delete a file record from the context table.

        Parameters
        ----------
        file_id : int
            The identifier of the file record to remove.

        Returns
        -------
        Dict[str, Any]
            Standard outcome dict: {"outcome": "file deleted", "details": {"file_id": <int>}}.

        Raises
        ------
        ValueError
            If the file does not exist, or if multiple records share the same file_id
            (indicates data integrity issues).

        Notes
        -----
        - This operation only deletes the log record from the context table.
        - The actual file in the temporary directory is NOT removed by this method.
        - This operation cannot be undone.
        """
        logs = unify.get_logs(
            context=self._ctx,
            filter=f"file_id == {file_id}",
        )
        if not logs:
            raise ValueError(
                f"No file found with file_id {file_id} to delete.",
            )
        if len(logs) > 1:
            raise RuntimeError(
                f"Multiple files found with file_id {file_id}. Data integrity issue.",
            )

        # Guard against deleting protected files
        try:
            filename = logs[0].entries.get("filename")
            if isinstance(filename, str) and filename in self._protected_display_names:
                raise PermissionError(
                    f"'{filename}' is protected and cannot be deleted by FileManager.",
                )
        except Exception:
            pass

        unify.delete_logs(
            context=self._ctx,
            logs=[logs[0].id],
        )
        return {
            "outcome": "file deleted",
            "details": {"file_id": file_id},
        }

    # Helpers #
    # --------#

    def _get_columns(self) -> Dict[str, str]:
        """
        Return {column_name: column_type} for the files table.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping column names to their types.
        """
        ret = unify.get_fields(context=self._ctx)
        return {k: v["data_type"] for k, v in ret.items()}

    def _list_columns(
        self,
        *,
        include_types: bool = True,
    ) -> Dict[str, str] | List[str]:
        """
        Return the list of available columns in the files table, optionally with types.

        Parameters
        ----------
        include_types : bool, default True
            Controls the shape of the returned value:
            - When True: returns a mapping {column_name: column_type}.
            - When False: returns a list of column names.
        """
        cols = self._get_columns()
        return cols if include_types else list(cols)

    def _num_files(self) -> int:
        """Return the total number of files registered."""
        ret = unify.get_logs_metric(
            metric="count",
            key="file_id",
            context=self._ctx,
        )
        if ret is None:
            return 0
        return int(ret)

    def _new_llm_client(self, model: str) -> "unify.AsyncUnify":
        """Construct a configured AsyncUnify client for the given model."""
        return unify.AsyncUnify(
            model,
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "false")),
            reasoning_effort="high",
            service_tier="priority",
        )

    # Public API ---------------------------------------------------------- #
    def exists(self, filename: str) -> bool:  # type: ignore[override]
        return filename in self._display_to_path

    def list(self) -> List[str]:  # type: ignore[override]
        return list(self._display_to_path.keys())

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
        # Validate and prepare files
        file_paths, filename_to_path, invalid_files = self._validate_and_prepare_files(
            filenames,
        )

        results = {}

        # Add error results for invalid files
        for filename in invalid_files:
            results[filename] = self._create_result_dict(
                filename,
                None,
                status="error",
                error=f"File not found: {filename}",
            )

        # If we have files to parse, use the parser's batch capability
        if file_paths:
            try:
                # Use the parser's optimized batch parsing
                if len(file_paths) > 1:
                    # Use batch parsing for multiple files
                    documents = self._parser.parse_batch(file_paths, **options)
                else:
                    # For single file, just call parse directly
                    documents = [self._parser.parse(file_paths[0], **options)]

                # Process each parsed document
                for i, document in enumerate(documents):
                    file_path = file_paths[i]
                    filename = filename_to_path[str(file_path)]
                    result_dict = self._create_result_dict(filename, document)
                    results[filename] = result_dict

                    # Log the parsed file to the context table
                    try:
                        self._create_file(**result_dict)
                    except Exception as e:
                        # Log the error but don't fail the entire operation
                        logging.warning(
                            f"Failed to log parsed file '{filename}' to context table: {e}",
                        )

            except Exception as e:
                # If batch parsing fails, fall back to individual parsing
                for file_path in file_paths:
                    filename = filename_to_path[str(file_path)]
                    try:
                        document = self._parser.parse(file_path, **options)
                        result_dict = self._create_result_dict(filename, document)
                        results[filename] = result_dict

                        # Log the parsed file to the context table
                        try:
                            self._create_file(**result_dict)
                        except Exception as e:
                            # Log the error but don't fail the entire operation
                            logging.warning(
                                f"Failed to log parsed file '{filename}' to context table: {e}",
                            )
                    except Exception as individual_e:
                        result_dict = self._create_result_dict(
                            filename,
                            None,
                            status="error",
                            error=str(individual_e),
                        )
                        results[filename] = result_dict

                        # Log the error result to the context table
                        try:
                            self._create_file(**result_dict)
                        except Exception as e:
                            # Log the error but don't fail the entire operation
                            logging.warning(
                                f"Failed to log error result for '{filename}' to context table: {e}",
                            )

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
        # Validate and prepare files
        file_paths, filename_to_path, invalid_files = self._validate_and_prepare_files(
            filenames,
        )

        # Yield error results for invalid files immediately
        for filename in invalid_files:
            yield self._create_result_dict(
                filename,
                None,
                status="error",
                error=f"File not found: {filename}",
            )

        # If we have files to parse, use the parser's async batch capability
        if file_paths:
            async for index, document in self._parser.parse_batch_async(
                file_paths,
                batch_size=batch_size,
                **options,
            ):
                file_path = file_paths[index]
                filename = filename_to_path[str(file_path)]
                result_dict = self._create_result_dict(filename, document)

                # Log the parsed file to the context table
                try:
                    self._create_file(**result_dict)
                except Exception as e:
                    # Log the error but don't fail the entire operation
                    logging.warning(
                        f"Failed to log parsed file '{filename}' to context table: {e}",
                    )

                yield result_dict

    @functools.wraps(BaseFileManager.ask, updated=())
    @log_manager_call("FileManager", "ask", payload_key="question")
    async def ask(
        self,
        filename: str,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:  # type: ignore[override]
        if not self.exists(filename):
            raise FileNotFoundError(filename)

        client = self._new_llm_client("gpt-5@openai")

        # Tools available inside the loop
        tools = dict(self._ask_tools)

        # Optional live clarification helper with event publishing
        if clarification_up_q is not None and clarification_down_q is not None:

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
                except Exception as e:
                    logging.warning(
                        f"Failed to publish clarification request event: {e}",
                    )

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
                except Exception as e:
                    logging.warning(
                        f"Failed to publish clarification answer event: {e}",
                    )

            tools["request_clarification"] = make_request_clarification_tool(
                clarification_up_q,
                clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        # Use the build_ask_prompt function to generate the system message
        system_msg = build_ask_prompt(
            tools=tools,
            num_files=self._num_files(),
            columns=self._list_columns(),
            include_activity=include_activity,
        )

        client.set_system_message(system_msg)

        # Launch tool loop with the question formatted to reference the file
        user_blob = json.dumps({"filename": filename, "question": question}, indent=2)

        handle = start_async_tool_loop(
            client,
            user_blob,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            tool_policy=lambda i, t: ("required", t) if i < 1 else ("auto", t),
            preprocess_msgs=inject_broader_context,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def _wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = _wrapped_result  # type: ignore[attr-defined]

        return handle
