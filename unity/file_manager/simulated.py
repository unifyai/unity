# unity/file_manager/simulated.py
from __future__ import annotations

import asyncio
import json
import functools
import threading
from typing import List, Dict, Any, Optional, Type, Union, TYPE_CHECKING

import unillm
from pydantic import BaseModel

if TYPE_CHECKING:
    from unity.file_manager.types.ingest import IngestPipelineResult
    from unity.data_manager.base import BaseDataManager as DataManager

from .base import BaseFileManager, BaseGlobalFileManager
from unity.data_manager.types import PlotResult as _VizPlotResult
from ..common.async_tool_loop import SteerableToolHandle
from ..common.llm_client import new_llm_client
from .prompt_builders import (
    build_file_manager_ask_about_file_prompt,
    build_simulated_method_prompt,
)
from ..common.simulated import (
    mirror_file_manager_tools,
    SimulatedLineage,
    SimulatedLog,
    simulated_llm_roundtrip,
    SimulatedHandleMixin,
)
from ..constants import LOGGER


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper handle
# ─────────────────────────────────────────────────────────────────────────────
class _SimulatedFileHandle(SteerableToolHandle, SimulatedHandleMixin):
    """
    Handle returned by SimulatedFileManager.ask_about_file.
    """

    def __init__(
        self,
        llm: unillm.Unify,
        initial_text: str,
        *,
        _return_reasoning_steps: bool,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
        response_format: Optional[Type[BaseModel]] = None,
    ):
        self._llm = llm
        self._initial = initial_text
        self._want_steps = _return_reasoning_steps
        self._clar_up_q = clarification_up_q
        self._clar_down_q = clarification_down_q
        if _requests_clarification and (
            not clarification_up_q or not clarification_down_q
        ):
            raise ValueError(
                "Clarification queues must be provided when _requests_clarification is True",
            )
        self._needs_clar = _requests_clarification

        # Human-friendly log label derived from current lineage, mirroring other simulated managers:
        # "<outer...>->SimulatedFileManager.ask_about_file(abcd)"
        self._log_label = SimulatedLineage.make_label(
            "SimulatedFileManager.ask_about_file",
        )

        # fire clarification question immediately if queues supplied
        if self._needs_clar:
            try:
                q_text = "Could you clarify your file-related request?"
                self._clar_up_q.put_nowait(q_text)
                try:
                    SimulatedLog.log_clarification_request(self._log_label, q_text)
                except Exception:
                    pass
                try:
                    LOGGER.info(f"❓ [{self._log_label}] Clarification requested")
                except Exception:
                    pass
            except asyncio.QueueFull:
                pass

        self._extra_msgs: List[str] = []
        self._response_format = response_format

        self._done_event = threading.Event()
        self._cancelled = False
        self._answer: str | None = None
        self._messages: List[Dict[str, Any]] = []
        self._paused = False
        # Async cancellation signal to break clarification waits
        self._cancel_event: asyncio.Event = asyncio.Event()

    # --------------------------------------------------------------------- #
    # SteerableToolHandle API
    # --------------------------------------------------------------------- #
    async def result(self):
        if self._cancelled:
            return "processed stopped early, no result"

        # honour pauses injected by an outer loop
        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done_event.is_set():
            if self._needs_clar:
                try:
                    LOGGER.info(
                        f"⏳ [{self._log_label}] Waiting for clarification answer…",
                    )
                except Exception:
                    pass
                # Race clarification against cancellation
                clar: str | None = None
                get_task = asyncio.create_task(self._clar_down_q.get())
                cancel_task = asyncio.create_task(self._cancel_event.wait())
                done, pending = await asyncio.wait(
                    {get_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                if cancel_task in done:
                    self._done_event.set()
                    return "processed stopped early, no result"
                try:
                    clar = get_task.result()
                except Exception:
                    clar = None
                if clar is None:
                    self._done_event.set()
                    return "processed stopped early, no result"
                self._extra_msgs.append(f"Clarification: {clar}")
                try:
                    SimulatedLog.log_clarification_answer(self._log_label, clar)
                except Exception:
                    pass
                try:
                    LOGGER.info(f"💬 [{self._log_label}] Clarification answer received")
                except Exception:
                    pass

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_msgs)
            # Unified simulated LLM roundtrip with lineage-aware logging and gated response preview
            try:
                sys_msg = getattr(self._llm, "system_message", None)
            except Exception:
                sys_msg = None
            answer = await simulated_llm_roundtrip(
                self._llm,
                label=self._log_label,
                prompt=prompt,
                response_format=self._response_format,
            )
            self._answer = answer
            self._messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            self._done_event.set()

        # If cancellation happened after the coroutine started, return a stable post-cancel value.
        if self._cancelled:
            return "processed stopped early, no result"
        if self._want_steps:
            return self._answer, self._messages
        return self._answer

    def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> str:
        """Interject a message into the in-flight handle.

        Args:
            message: The interjection message to inject.
            parent_chat_context_cont: Optional continuation of parent chat context.
                Accepted for API parity with real handles but not currently used.
            images: Optional image references. Accepted for API parity with real handles
                but not currently used.
        """
        if self._cancelled:
            return "Interaction stopped."
        self._log_interject(message)
        self._extra_msgs.append(message)
        return "Acknowledged."

    def stop(
        self,
        reason: str | None = None,
    ) -> str:
        """Stop the in-flight handle.

        Args:
            reason: Optional reason for stopping.
        """
        self._log_stop(reason)
        self._cancelled = True
        try:
            self._cancel_event.set()
        except Exception:
            pass
        self._done_event.set()
        return "Stopped." if reason is None else f"Stopped: {reason}"

    async def pause(self) -> str:
        if self._paused:
            return "Already paused."
        self._log_pause()
        self._paused = True
        return "Paused."

    async def resume(self) -> str:
        if not self._paused:
            return "Already running."
        self._log_resume()
        self._paused = False
        return "Resumed."

    def done(self) -> bool:
        return self._done_event.is_set()

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> "SteerableToolHandle":
        """Ask a follow-up question about the current operation.

        Args:
            question: The question to ask.
            parent_chat_context: Optional parent chat context for the inspection loop.
                Accepted for API parity with real handles but not currently used.
            images: Optional image references. Accepted for API parity with real handles
                but not currently used.
        """
        q_msg = (
            f"Your only task is to simulate an answer to the following question: {question}\n\n"
            "However, there is a also ongoing simulated process which had the instructions given below. "
            "Please make your answer realastic and conceivable given the provided context of the simulated task."
        )
        follow_up_prompt = "\n\n---\n\n".join(
            [q_msg]
            + [self._initial]
            + self._extra_msgs
            + [f"Question to answer (as a reminder!): {question}"],
        )

        handle = _SimulatedFileHandle(
            self._llm,
            follow_up_prompt,
            _return_reasoning_steps=self._want_steps,
            _requests_clarification=False,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
            response_format=self._response_format,
        )
        # Align with other simulated components: concise "Question(<parent_label>)" label
        try:
            handle._log_label = SimulatedLineage.question_label(self._log_label)  # type: ignore[attr-defined]
        except Exception:
            pass
        # Emit a human-facing log for the nested ask
        try:
            SimulatedLog.log_request("ask", getattr(handle, "_log_label", ""), question)  # type: ignore[arg-type]
        except Exception:
            pass
        return handle

    # --- event APIs required by SteerableToolHandle ---------------------
    async def next_clarification(self) -> dict:
        """Retrieve the next clarification request, if any.

        Only surfaces clarification events when this handle explicitly requested
        clarification. This prevents cross-handle consumption of shared clarification
        queues that may be injected by external processes.
        """
        if not getattr(self, "_needs_clar", False):
            return {}
        try:
            if self._clar_up_q is not None:
                msg = await self._clar_up_q.get()
                return {
                    "type": "clarification",
                    "call_id": "unknown",
                    "tool_name": "unknown",
                    "question": msg,
                }
        except Exception:
            pass
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        try:
            if self._clar_down_q is not None:
                await self._clar_down_q.put(answer)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Public simulated FileManager
# ─────────────────────────────────────────────────────────────────────────────
class SimulatedFileManager(BaseFileManager):
    """
    Drop-in replacement for FileManager that only uses an LLM to invent
    plausible answers about files. Suitable for offline demos and tests
    where real file storage is unnecessary.
    """

    def __init__(
        self,
        description: str = "nothing fixed, make up some imaginary scenario",
        *,
        log_events: bool = False,
        rolling_summary_in_prompts: bool = True,
        simulation_guidance: Optional[str] = None,
        # Accept but ignore extra parameters for compatibility
        **kwargs: Any,
    ) -> None:
        self._description = description
        self._log_events = log_events
        self._rolling_summary_in_prompts = rolling_summary_in_prompts
        self._simulation_guidance = simulation_guidance
        # In-memory storage for simulated files
        self._files: Dict[str, Dict[str, Any]] = {}
        # Track protected files by display name (read-only from simulated API)
        self._protected: set[str] = set()

        # Counter for simulated file IDs
        self._next_file_id = 1

        # Lazy-initialized simulated data manager
        self.__data_manager: Optional["DataManager"] = None

        # Shared, *stateful* **asynchronous** LLM
        self._llm = new_llm_client(stateful=True)

        # Mirror the real file manager's tool exposure programmatically
        try:
            ask_about_file_tools = mirror_file_manager_tools("ask_about_file")
        except (ImportError, AttributeError):
            ask_about_file_tools = {
                "describe": {
                    "description": "Get comprehensive file storage map with contexts and schemas",
                },
                "list_columns": {"description": "List available table columns"},
                "filter_files": {
                    "description": "Filter files using boolean expressions",
                },
                "search_files": {"description": "Semantic search over file contents"},
                "reduce": {"description": "Compute aggregates over rows"},
                "filter_join": {"description": "Filter-based join across tables"},
                "search_join": {"description": "Search-based join across tables"},
                "filter_multi_join": {"description": "Multi-table filter-based join"},
                "search_multi_join": {"description": "Multi-table search-based join"},
            }

        # Build prompt using the new prompt builders
        about_msg = build_file_manager_ask_about_file_prompt(
            ask_about_file_tools,
            include_activity=self._rolling_summary_in_prompts,
        )

        self._llm.set_system_message(
            "You are a *simulated* file manager assistant. "
            "There is no real file storage; invent plausible file records and "
            "keep your story consistent across turns.\n\n"
            "As reference, here is the system message used by the *real* file manager. "
            "You do not have access to tools – produce the final answer only.\n\n"
            f"'ask_about_file' system message:\n{about_msg}\n\n"
            f"Back-story: {self._description}",
        )

    @property
    def _data_manager(self) -> "DataManager":
        """Return a simulated DataManager for delegation tests."""
        if self.__data_manager is None:
            from unity.manager_registry import ManagerRegistry

            self.__data_manager = ManagerRegistry.get_data_manager()
        return self.__data_manager

    # --------------------------------------------------------------------- #
    # Synchronous methods that don't need handles                          #
    # --------------------------------------------------------------------- #
    def list(self) -> List[str]:
        """List all available files in simulated storage."""
        return list(self._files.keys())

    def exists(self, filename: str) -> bool:
        """Check if a file exists in simulated storage."""
        return filename in self._files

    def ingest_files(self, filenames, **options) -> "IngestPipelineResult":
        """
        Run the complete file processing pipeline: parse, ingest, and embed (simulated).

        This method simulates the full file processing workflow for testing.
        Returns IngestPipelineResult consistent with the real FileManager.
        """
        from unity.file_manager.types.ingest import (
            IngestPipelineResult,
            BaseIngestedFile,
            ContentRef,
            FileMetrics,
        )

        if isinstance(filenames, str):
            filenames = [filenames]

        results: Dict[str, BaseIngestedFile] = {}
        for filename in filenames:
            if filename in self._files:
                file_data = self._files[filename]
                meta = file_data.get("metadata", {}) or {}
                # Derive file_format from mime when not provided
                mime = meta.get("mime_type") or meta.get("file_type")
                file_format = meta.get("file_format")
                if not file_format and isinstance(mime, str):
                    fmt_map = {
                        "text/plain": "txt",
                        "text/markdown": "txt",
                        "text/csv": "csv",
                        "text/html": "html",
                        "application/json": "json",
                        "application/pdf": "pdf",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
                        "application/msword": "doc",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
                        "application/vnd.ms-excel": "xlsx",
                    }
                    file_format = fmt_map.get(mime, "unknown")

                records = file_data.get("records", [])
                full_text = file_data.get("full_text", "")
                results[filename] = BaseIngestedFile(
                    file_path=filename,
                    status="success",
                    error=None,
                    content_ref=ContentRef(
                        context=f"simulated/{filename}/Content",
                        record_count=len(records),
                        text_chars=len(full_text),
                    ),
                    metrics=FileMetrics(processing_time=0.001),
                    file_format=file_format,
                )
            else:
                results[filename] = BaseIngestedFile(
                    file_path=filename,
                    status="error",
                    error=f"File '{filename}' not found",
                    content_ref=ContentRef(context="", record_count=0, text_chars=0),
                    metrics=FileMetrics(processing_time=0.0),
                )

        return IngestPipelineResult.from_results(results)

    # --------------------------------------------------------------------- #
    # ask_about_file                                                        #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseFileManager.ask_about_file, updated=())
    async def ask_about_file(
        self,
        file_path: str,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
        response_format: Optional[Any] = None,
    ) -> SteerableToolHandle:
        if file_path not in self._files:
            raise FileNotFoundError(file_path)
        instruction = build_simulated_method_prompt(
            "ask_about_file",
            f"File: {file_path}\nQuestion: {question}",
            parent_chat_context=_parent_chat_context,
        )
        file_info = self._files[file_path]
        instruction += f"\n\nFile information: {json.dumps(file_info, indent=2)}"
        handle = _SimulatedFileHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            response_format=response_format,
        )
        return handle

    def import_file(self, file_path: str) -> str:
        """Simulate importing a single file from filesystem."""
        # Generate a simulated filename
        from pathlib import Path

        path = Path(file_path)
        filename = path.name

        # Ensure unique filename
        base_name = path.stem
        extension = path.suffix
        counter = 1
        while filename in self._files:
            filename = f"{base_name} ({counter}){extension}"
            counter += 1

        # Create simulated file data
        self.add_simulated_file(
            filename,
            records=[{"content": f"Simulated content from {file_path}"}],
            metadata={
                "file_format": extension.lstrip(".").lower(),
                "mime_type": None,
                "source_path": file_path,
            },
            full_text=f"Simulated content from {file_path}",
            description=f"Imported file: {filename}",
        )

        return filename

    def import_directory(self, directory: str) -> List[str]:
        """Simulate importing all files from a directory."""
        # Generate simulated filenames
        import random

        file_extensions = [".txt", ".pdf", ".docx", ".md", ".csv"]
        num_files = random.randint(2, 5)
        added_files = []

        for i in range(num_files):
            ext = random.choice(file_extensions)
            base_filename = f"file_{i+1}{ext}"
            filename = base_filename

            # Ensure unique filename
            counter = 1
            while filename in self._files:
                stem = base_filename.rsplit(".", 1)[0]
                filename = f"{stem} ({counter}){ext}"
                counter += 1

            # Create simulated file data
            self.add_simulated_file(
                filename,
                records=[{"content": f"Simulated content from {directory}/{filename}"}],
                metadata={
                    "file_format": ext.lstrip(".").lower(),
                    "mime_type": None,
                    "source_directory": directory,
                },
                full_text=f"Simulated content from {directory}/{filename}",
                description=f"File from directory: {filename}",
            )
            added_files.append(filename)

        return added_files

    def export_file(self, filename: str, destination_dir: str) -> str:
        """Simulate exporting a file to a local destination directory."""
        from pathlib import Path

        if filename not in self._files:
            raise FileNotFoundError(f"File '{filename}' not found in simulated storage")

        # Create destination directory
        dest_dir = Path(destination_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Create simulated file at destination with original filename
        dest_path = dest_dir / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Write simulated content to file
        file_data = self._files[filename]
        content = file_data.get("full_text", f"Simulated content for {filename}")
        dest_path.write_text(content)

        return str(dest_path)

    def export_directory(self, directory: str, destination_dir: str) -> List[str]:
        """Simulate exporting all files from a directory to a local destination."""
        from pathlib import Path

        # In simulated mode, "directory" is treated as a prefix filter
        # Export all files that start with the directory path
        exported = []
        dest_dir = Path(destination_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for filename in self._files:
            # Simple prefix matching for simulated directory export
            if (
                directory == ""
                or filename.startswith(directory.rstrip("/") + "/")
                or filename == directory
            ):
                try:
                    exported_path = self.export_file(filename, destination_dir)
                    exported.append(exported_path)
                except Exception:
                    continue

        return exported

    # --------------------------------------------------------------------- #
    # Unify-backed retrieval (public tools)                              #
    # --------------------------------------------------------------------- #
    def filter_files(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        tables: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter files using a boolean expression evaluated per row (Unify)."""
        files = list(self._files.values())

        # Apply filter (simplified simulation)
        filtered_files = []
        for file_data in files:
            match = True
            if filter:
                # Very basic simulation: check if filter string is present in any string field
                # In a real scenario, this would involve parsing and evaluating the expression
                if (
                    "status == 'success'" in filter
                    and file_data.get("status") != "success"
                ):
                    match = False
                elif "endswith('.pdf')" in filter and not file_data.get(
                    "filename",
                    "",
                ).endswith(".pdf"):
                    match = False
                # Add more simulated filter conditions as needed for tests
            if match:
                filtered_files.append(file_data)

        # Apply offset and limit
        return filtered_files[offset : offset + limit]

    def search_files(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        table: Optional[str] = None,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over a (simulated) resolved context.

        Parameters
        ----------
        references : dict[str, str] | None
            Mapping of source_expr → reference text.
        k : int
            Number of rows to return.
        table : str | None
            Ignored in simulation; present for signature parity.
        filter : str | None
            Ignored in simulation; present for signature parity.
        """
        files = list(self._files.values())

        if not references:
            # Return most recent files (simulate by reversing order)
            files = files[-k:]
        else:
            # Simulate ranking by simple keyword matching
            def score_file(file_data):
                score = 0
                for source_expr, reference_text in references.items():
                    text_to_search = ""
                    if source_expr == "full_text":
                        text_to_search = file_data.get("full_text", "")
                    elif source_expr == "description":
                        text_to_search = file_data.get("description", "")
                    elif source_expr == "metadata":
                        text_to_search = str(file_data.get("metadata", {}))
                    else:
                        text_to_search = file_data.get(source_expr, "")

                    # Simple keyword matching for simulation
                    keywords = reference_text.lower().split()
                    for keyword in keywords:
                        if keyword in text_to_search.lower():
                            score += 1
                return score

            # Sort by relevance score
            files.sort(key=score_file, reverse=True)
            files = files[:k]

        return files

    def visualize(
        self,
        *,
        tables: Union[str, List[str]],
        plot_type: str,
        x_axis: str,
        y_axis: Optional[str] = None,
        group_by: Optional[str] = None,
        filter: Optional[str] = None,
        title: Optional[str] = None,
        aggregate: Optional[str] = None,
        scale_x: Optional[str] = None,
        scale_y: Optional[str] = None,
        bin_count: Optional[int] = None,
        show_regression: Optional[bool] = None,
    ) -> Union["_VizPlotResult", List["_VizPlotResult"]]:
        """
        Return simulated plot result(s) with placeholder URLs.

        The simulated manager does not call the real Plot API; this returns
        deterministic placeholder responses with the same shape as the real
        implementation.
        """
        from secrets import token_hex

        # Normalize tables to list
        table_list: List[str] = []
        if isinstance(tables, str):
            if tables:
                table_list = [tables]
        else:
            table_list = [t for t in tables if t]

        if not table_list:
            return _VizPlotResult(error="No tables provided", title=title or "Untitled")

        # Generate placeholder results
        results: List[_VizPlotResult] = []
        for tbl in table_list:
            # Extract table label: simulates what resolved context would produce
            # For paths like "/path/to/file.Tables.July_2025", use the table name
            # For resolved contexts like "User/Asst/Files/.../Tables/July_2025", use last segment
            if ".Tables." in tbl:
                table_label = tbl.rsplit(".Tables.", 1)[-1]
            elif "/" in tbl:
                table_label = tbl.rsplit("/", 1)[-1]
            else:
                table_label = tbl

            # Build title
            base_title = title or f"{plot_type} chart"
            plot_title = (
                f"{base_title} ({table_label})" if len(table_list) > 1 else base_title
            )

            # Generate placeholder URL
            token = token_hex(8)
            results.append(
                _VizPlotResult(
                    url=f"https://console.unify.ai/plot/simulated/{token}",
                    token=token,
                    expires_in_hours=24,
                    title=plot_title,
                    table=tbl,
                ),
            )

        # Return single PlotResult or list based on input
        if len(results) == 1:
            return results[0]
        return results

    def rename_file(
        self,
        *,
        file_id_or_path: Union[str, int],
        new_name: str,
    ) -> Dict[str, Any]:
        """Simulate renaming a file."""
        # Resolve file_id_or_path to filename
        if isinstance(file_id_or_path, int):
            # Find file by ID
            filename = None
            for fname, fdata in self._files.items():
                if fdata.get("file_id", 0) == file_id_or_path:
                    filename = fname
                    break
            if filename is None:
                raise FileNotFoundError(
                    f"File with file_id {file_id_or_path} not found.",
                )
        else:
            filename = str(file_id_or_path).lstrip("/")

        new_name = str(new_name)

        if filename not in self._files:
            raise FileNotFoundError(f"File '{filename}' not found.")
        file_data = self._files.pop(filename)
        file_data["filename"] = new_name
        self._files[new_name] = file_data
        return {"path": new_name, "name": new_name}

    def move_file(
        self,
        *,
        file_id_or_path: Union[str, int],
        new_parent_path: str,
    ) -> Dict[str, Any]:
        """Simulate moving a file."""
        # Resolve file_id_or_path to filename
        if isinstance(file_id_or_path, int):
            # Find file by ID
            filename = None
            for fname, fdata in self._files.items():
                if fdata.get("file_id", 0) == file_id_or_path:
                    filename = fname
                    break
            if filename is None:
                raise FileNotFoundError(
                    f"File with file_id {file_id_or_path} not found.",
                )
        else:
            filename = str(file_id_or_path).lstrip("/")

        new_parent_path = str(new_parent_path).lstrip("/")

        if filename not in self._files:
            raise FileNotFoundError(f"File '{filename}' not found.")
        # In simulation, path is just the filename. Moving is like renaming with a path prefix.
        new_path = f"{new_parent_path}/{filename}" if new_parent_path else filename
        file_data = self._files.pop(filename)
        file_data["filename"] = new_path
        self._files[new_path] = file_data
        return {"path": new_path, "parent": new_parent_path}

    def list_columns(
        self,
        *,
        include_types: bool = True,
        table: Optional[str] = None,
    ) -> Dict[str, Any] | List[str]:
        """Simulate listing table columns (returns index schema for any table)."""
        columns = {
            "file_id": "int",
            "filename": "str",
            "status": "str",
            "error": "str",
            "records": "list",
            "full_text": "str",
            "metadata": "dict",
            "description": "str",
            "file_format": "str",
            "imported_at": "datetime",
        }
        return columns if include_types else list(columns.keys())

    def sync(self, *, file_path: str) -> Dict[str, Any]:
        """Simulate a sync operation (no-op with a plausible summary)."""
        exists = file_path in self._files
        return {
            "outcome": (
                "sync complete (simulated)" if exists else "sync skipped (not found)"
            ),
            "purged": {"content_rows": 0, "table_rows": 0},
            "file_path": file_path,
        }

    # --------------------------------------------------------------------- #
    # Simulation helpers                                                    #
    # --------------------------------------------------------------------- #
    def add_simulated_file(
        self,
        filename: str,
        records: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        full_text: Optional[str] = None,
        description: Optional[str] = None,
        status: str = "success",
    ) -> None:
        """Add a simulated file to the storage."""
        self._files[filename] = {
            "file_id": self._next_file_id,
            "filename": filename,
            "records": records,
            "metadata": metadata or {},
            "full_text": full_text or f"Simulated content for {filename}",
            "description": description or f"Simulated file: {filename}",
            "status": status,
            "error": None,
            "imported_at": "2024-01-01T00:00:00Z",
        }
        self._next_file_id += 1

    def remove_simulated_file(self, filename: str) -> None:
        """Remove a simulated file from storage."""
        if filename in self._files:
            del self._files[filename]

    def clear_simulated_files(self) -> None:
        """Clear all simulated files."""
        self._files.clear()

    def delete_file(self, *, file_id_or_path: Union[str, int]) -> Dict[str, Any]:
        """Simulate deleting a file record."""
        # Resolve file_id_or_path to filename
        if isinstance(file_id_or_path, int):
            # Find file by ID
            filename = None
            for fname, fdata in list(self._files.items()):
                if fdata.get("file_id", 0) == file_id_or_path:
                    filename = fname
                    break
            if filename is None:
                raise ValueError(
                    f"No file found with file_id {file_id_or_path} to delete.",
                )
        else:
            filename = str(file_id_or_path).lstrip("/")
            if filename not in self._files:
                raise ValueError(
                    f"No file found with file_path '{file_id_or_path}' to delete.",
                )

        if filename in self._protected:
            raise PermissionError(
                f"'{filename}' is protected and cannot be deleted by FileManager.",
            )
        file_data = self._files[filename]
        file_id = file_data.get("file_id", 0)
        del self._files[filename]
        return {
            "outcome": "file deleted",
            "details": {"file_id": file_id, "file_path": filename},
        }

    @functools.wraps(BaseFileManager.describe, updated=())
    def describe(
        self,
        *,
        file_path: Optional[str] = None,
        file_id: Optional[int] = None,
    ) -> Any:
        """Simulated counterpart of describe().

        Returns a minimal FileStorageMap-like dict for simulated files.
        """
        from unity.file_manager.types.describe import FileStorageMap

        if file_path is None and file_id is None:
            raise ValueError("Either file_path or file_id must be provided")

        # Find in simulated storage
        target_path = file_path
        if file_id is not None:
            for fp, meta in self._files.items():
                if meta.get("file_id") == file_id:
                    target_path = fp
                    break

        if target_path and target_path in self._files:
            meta = self._files[target_path]
            return FileStorageMap(
                file_path=target_path,
                file_id=meta.get("file_id"),
                storage_id=str(meta.get("file_id", "")),
                source_uri=meta.get("source_uri"),
                source_provider="Simulated",
                filesystem_exists=True,
                indexed_exists=True,
                parsed_status="success",
                file_format=meta.get("file_format"),
                table_ingest=False,
                has_document=False,
                has_tables=False,
            )

        # File not found - return minimal map
        return FileStorageMap(
            file_path=file_path or f"unknown_file_{file_id}",
            file_id=file_id,
            storage_id=str(file_id) if file_id else "",
            filesystem_exists=False,
            indexed_exists=False,
            parsed_status=None,
            table_ingest=False,
            has_document=False,
            has_tables=False,
        )

    @functools.wraps(BaseFileManager.clear, updated=())
    def clear(self) -> None:  # type: ignore[override]
        """Re-initialise the simulated manager and reset stateful LLM."""
        type(self).__init__(
            self,
            description=getattr(
                self,
                "_description",
                "nothing fixed, make up some imaginary scenario",
            ),
            log_events=getattr(self, "_log_events", False),
            rolling_summary_in_prompts=getattr(
                self,
                "_rolling_summary_in_prompts",
                True,
            ),
            simulation_guidance=getattr(self, "_simulation_guidance", None),
        )

    def reduce(
        self,
        *,
        context: Optional[str] = None,
        metric: str,
        columns: str | list[str],
        filter: Optional[str | dict[str, str]] = None,
        group_by: Optional[str | list[str]] = None,
    ) -> Any:
        """
        Simulated counterpart of the FileManager.reduce tool.

        The simulated file manager stores file metadata in-memory only; this
        method computes deterministic placeholder metrics with the same return
        shapes as the concrete implementation:

        * Single column, no grouping  → scalar.
        * Multiple columns, no grouping → ``dict[column -> scalar]``.
        * With grouping             → nested ``dict[group -> value or dict]``.
        """

        def _scalar(k: str) -> float:
            base = len(self._files) or 1
            return float(base + len(str(k)))

        col_list: list[str] = [columns] if isinstance(columns, str) else list(columns)

        if group_by is None:
            if isinstance(columns, str):
                return _scalar(columns)
            return {k: _scalar(k) for k in col_list}

        groups: list[str] = (
            [group_by] if isinstance(group_by, str) else [str(g) for g in group_by]
        )
        if isinstance(columns, str):
            return {g: _scalar(columns) for g in groups}
        return {g: {k: _scalar(k) for k in col_list} for g in groups}

    # --------------------------------------------------------------------- #
    # Join methods (simulated placeholders)                                  #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseFileManager.filter_join, updated=())
    def filter_join(
        self,
        *,
        tables: Union[str, List[str]],
        join_expr: str,
        select: Dict[str, str],
        mode: str = "inner",
        left_where: Optional[str] = None,
        right_where: Optional[str] = None,
        result_where: Optional[str] = None,
        result_limit: int = 100,
        result_offset: int = 0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Simulated filter_join: returns placeholder empty results.

        In simulated mode, no real join is performed. Returns an empty list
        to satisfy the API contract.
        """
        return {"rows": []}

    @functools.wraps(BaseFileManager.search_join, updated=())
    def search_join(
        self,
        *,
        tables: Union[str, List[str]],
        join_expr: str,
        select: Dict[str, str],
        mode: str = "inner",
        left_where: Optional[str] = None,
        right_where: Optional[str] = None,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Simulated search_join: returns placeholder empty results.

        In simulated mode, no real join or semantic search is performed.
        Returns an empty list to satisfy the API contract.
        """
        return []

    @functools.wraps(BaseFileManager.filter_multi_join, updated=())
    def filter_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        result_where: Optional[str] = None,
        result_limit: int = 100,
        result_offset: int = 0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Simulated filter_multi_join: returns placeholder empty results.

        In simulated mode, no real multi-join is performed. Returns an empty
        list to satisfy the API contract.
        """
        return {"rows": []}

    @functools.wraps(BaseFileManager.search_multi_join, updated=())
    def search_multi_join(
        self,
        *,
        joins: List[Dict[str, Any]],
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Simulated search_multi_join: returns placeholder empty results.

        In simulated mode, no real multi-join or semantic search is performed.
        Returns an empty list to satisfy the API contract.
        """
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Simulated GlobalFileManager
# ─────────────────────────────────────────────────────────────────────────────


class SimulatedGlobalFileManager(BaseGlobalFileManager):
    """
    Simulated counterpart to GlobalFileManager. Aggregates multiple FileManagers
    and provides the list_filesystems helper. For filesystem‑wide operations,
    use ``FunctionManager`` to compose bespoke logic.
    """

    def __init__(self, managers: List[BaseFileManager]):
        self._managers: List[BaseFileManager] = list(managers)

    def list_filesystems(self) -> List[str]:
        names = [
            getattr(m.__class__, "__name__", "FileManager") for m in self._managers
        ]
        return sorted(set(names))

    @functools.wraps(BaseGlobalFileManager.clear, updated=())
    def clear(self) -> None:  # type: ignore[override]
        """Re-initialise the simulated manager."""
        type(self).__init__(self, managers=self._managers)
