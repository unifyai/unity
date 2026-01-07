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

from .base import BaseFileManager, BaseGlobalFileManager
from .managers.utils.viz_utils import PlotResult as _VizPlotResult
from ..common.async_tool_loop import SteerableToolHandle
from ..common.llm_client import new_llm_client
from .prompt_builders import (
    build_file_manager_ask_prompt,
    build_file_manager_ask_about_file_prompt,
    build_file_manager_organize_prompt,
    build_global_file_manager_ask_prompt,
    build_global_file_manager_organize_prompt,
    build_simulated_method_prompt,
)
from ..common.simulated import (
    mirror_file_manager_tools,
    SimulatedLineage,
    SimulatedLog,
    simulated_llm_roundtrip,
    SimulatedHandleMixin,
    maybe_tool_log_scheduled,
)
from ..constants import LOGGER


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper handle
# ─────────────────────────────────────────────────────────────────────────────
class _SimulatedFileHandle(SteerableToolHandle, SimulatedHandleMixin):
    """
    Handle returned by SimulatedFileManager.ask.
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
        # "<outer...>->SimulatedFileManager.ask(abcd)"
        self._log_label = SimulatedLineage.make_label("SimulatedFileManager.ask")

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
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ) -> str:
        """Stop the in-flight handle.

        Args:
            reason: Optional reason for stopping.
            parent_chat_context_cont: Optional continuation of parent chat context.
                Accepted for API parity with real handles but not currently used.
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
        parent_chat_context_cont: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> "SteerableToolHandle":
        """Ask a follow-up question about the current operation.

        Args:
            question: The question to ask.
            parent_chat_context_cont: Optional continuation of parent chat context.
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
        try:
            if self._clar_up_q is not None:
                msg = await self._clar_up_q.get()
                return {"message": msg}
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

        # Shared, *stateful* **asynchronous** LLM
        self._llm = new_llm_client(stateful=True)

        # Mirror the real file manager's tool exposure programmatically
        try:
            ask_tools = mirror_file_manager_tools("ask")
        except (ImportError, AttributeError):
            # Fallback if mirror function doesn't exist yet
            ask_tools = {
                "list_columns": {"description": "List available table columns"},
                "tables_overview": {"description": "Get overview of available tables"},
                "schema_explain": {
                    "description": "Get natural-language schema explanation",
                },
                "file_info": {
                    "description": "Get comprehensive file status and identity",
                },
                "filter_files": {
                    "description": "Filter files using boolean expressions",
                },
                "search_files": {"description": "Semantic search over file contents"},
                "reduce": {"description": "Compute aggregates over rows"},
                "list": {"description": "List all available files"},
                "ask_about_file": {
                    "description": "Ask questions about a specific file",
                },
            }

        try:
            ask_about_file_tools = mirror_file_manager_tools("ask_about_file")
        except (ImportError, AttributeError):
            ask_about_file_tools = {
                "file_info": {
                    "description": "Get comprehensive file status and identity",
                },
                "list_columns": {"description": "List available table columns"},
                "tables_overview": {"description": "Get overview of available tables"},
                "schema_explain": {
                    "description": "Get natural-language schema explanation",
                },
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

        try:
            organize_tools = mirror_file_manager_tools("organize")
        except (ImportError, AttributeError):
            organize_tools = {
                "ask": {"description": "Ask questions to discover files"},
                "rename_file": {"description": "Rename a file"},
                "move_file": {"description": "Move a file to a new location"},
                "delete_file": {"description": "Delete a file"},
            }

        # Build prompt using the new prompt builders
        ask_msg = build_file_manager_ask_prompt(
            ask_tools,
            num_files=len(self._files),
            include_activity=self._rolling_summary_in_prompts,
        )
        about_msg = build_file_manager_ask_about_file_prompt(
            ask_about_file_tools,
            include_activity=self._rolling_summary_in_prompts,
        )
        org_msg = build_file_manager_organize_prompt(
            organize_tools,
            num_files=len(self._files),
            include_activity=self._rolling_summary_in_prompts,
        )

        self._llm.set_system_message(
            "You are a *simulated* file manager assistant. "
            "There is no real file storage; invent plausible file records and "
            "keep your story consistent across turns.\n\n"
            "As reference, here are the system messages used by the *real* file manager. "
            "You do not have access to tools – produce the final answer only.\n\n"
            f"'ask' system message:\n{ask_msg}\n\n"
            f"'ask_about_file' system message:\n{about_msg}\n\n"
            f"'organize' system message:\n{org_msg}\n\n"
            f"Back-story: {self._description}",
        )

    # --------------------------------------------------------------------- #
    # ask                                                                   #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseFileManager.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        log_events: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = None  # No EventBus publishing for simulated managers

        # Provide inventory summary to make simulated answers coherent
        inventory = sorted(list(self._files.keys()))
        instruction = build_simulated_method_prompt(
            "ask",
            json.dumps({"question": text, "inventory": inventory}, indent=2),
            parent_chat_context=_parent_chat_context,
        )

        handle = _SimulatedFileHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=_clarification_up_q,
            clarification_down_q=_clarification_down_q,
            response_format=response_format,
        )

        # Tool-style scheduled log (only when no parent lineage)
        maybe_tool_log_scheduled(
            "SimulatedFileManager.ask",
            "ask",
            {
                "question": text,
                "requests_clarification": _requests_clarification,
            },
        )

        return handle

    # Append guidance for outer orchestrators via tool description
    ask.__doc__ = (ask.__doc__ or "") + (
        "\n\nOuter-orchestrator guidance: Avoid invoking this tool repeatedly with the same "
        "arguments within the same conversation. Prefer reusing prior results and "
        "compose the final answer once sufficient information has been gathered."
    )

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
        filename: str,
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
        if filename not in self._files:
            raise FileNotFoundError(filename)
        instruction = build_simulated_method_prompt(
            "ask_about_file",
            f"File: {filename}\nQuestion: {question}",
            parent_chat_context=_parent_chat_context,
        )
        file_info = self._files[filename]
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

    # --------------------------------------------------------------------- #
    # organize                                                               #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseFileManager.organize, updated=())
    async def organize(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _requests_clarification: bool = False,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> SteerableToolHandle:
        # Simulate an organization plan by summarizing current files
        inventory = sorted(list(self._files.keys()))
        prompt_body = {
            "action": "organize",
            "request": text,
            "files": inventory,
        }
        instruction = build_simulated_method_prompt(
            "organize",
            json.dumps(prompt_body, indent=2),
            parent_chat_context=_parent_chat_context,
        )
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

    def file_info(self, *, identifier: Union[str, int]) -> Any:
        """
        Return comprehensive information about a file's status and ingest identity.

        This is a simulated implementation that derives all fields from the
        in-memory `_files` registry.
        """
        from pathlib import Path

        from unity.file_manager.file_parsers.types.formats import (
            FileFormat,
            extension_to_format,
        )
        from unity.file_manager.types.file import FileInfo

        file_data: Optional[Dict[str, Any]] = None
        file_path: str = str(identifier)

        # Resolve file_id -> file_path when possible.
        if isinstance(identifier, int):
            for fname, fdata in self._files.items():
                try:
                    if int(fdata.get("file_id", -1)) == int(identifier):
                        file_path = fname
                        file_data = fdata
                        break
                except Exception:
                    continue
        else:
            # Normalize string identifiers to match simulated keys (no leading '/')
            file_path = str(identifier)
            if file_path.startswith("/"):
                file_path = file_path.lstrip("/")
            file_data = self._files.get(file_path)

        filesystem_exists = file_data is not None
        indexed_exists = file_data is not None
        parsed_status = None
        file_format: Optional[FileFormat] = None

        if file_data is not None:
            parsed_status = file_data.get("status")
            meta = file_data.get("metadata") or {}

            # Best-effort file format inference.
            fmt = meta.get("file_format")
            if isinstance(fmt, FileFormat):
                file_format = fmt
            elif isinstance(fmt, str) and fmt.strip():
                try:
                    file_format = FileFormat(fmt.strip().lower())
                except Exception:
                    file_format = extension_to_format(Path(file_path).suffix.lower())
            else:
                file_format = extension_to_format(Path(file_path).suffix.lower())

        # Keep identity fields stable but clearly simulated.
        source_provider = "Simulated"
        source_uri = f"simulated:///{file_path}"

        return FileInfo(
            file_path=file_path,
            filesystem_exists=filesystem_exists,
            indexed_exists=indexed_exists,
            parsed_status=parsed_status,
            source_provider=source_provider,
            source_uri=source_uri,
            ingest_mode="per_file",
            unified_label=None,
            table_ingest=True,
            file_format=file_format,
        )

    def tables_overview(
        self,
        *,
        include_column_info: bool = True,
        file: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return an overview of available tables/contexts managed by this FileManager.

        This is a simulated implementation that provides stable, deterministic
        context strings derived from the in-memory `_files` registry.
        """
        index = {
            "context": "simulated/FileRecords",
            "description": "Simulated FileRecords index (in-memory only).",
        }
        if include_column_info:
            try:
                index["columns"] = self.list_columns(include_types=True)
            except Exception:
                index["columns"] = {}

        # Global-only view
        if file is None:
            return {"FileRecords": index}

        file_key = str(file)
        if file_key.startswith("/"):
            file_key = file_key.lstrip("/")

        out: Dict[str, Dict[str, Any]] = {"FileRecords": index}
        out[file_key] = {
            "Content": {
                "context": f"simulated/{file_key}/Content",
                "description": f"Simulated per-file Content context for '{file_key}'.",
            },
            # The simulated manager does not create per-table contexts, but we
            # keep the shape consistent with the real manager.
            "Tables": {},
        }
        return out

    def schema_explain(self, *, table: str) -> str:
        """
        Return a natural-language explanation of a table's structure and purpose.

        The simulated manager does not have real Unify contexts; this returns a
        compact explanation based on the requested logical table reference.
        """
        t = str(table or "").strip()
        if not t:
            return "No schema information available (empty table reference)."

        if t.lower() == "filerecords":
            return (
                "FileRecords is the simulated file index (one row per file). "
                "It includes identifiers, status/error fields, lightweight metadata, "
                f"and a short description. Approximate row count: {len(self._files)}."
            )

        # Per-table context reference: "<file>.Tables.<label>"
        if ".tables." in t.lower():
            return (
                f"{t} is a simulated per-file table context. In the real FileManager, "
                "this would contain extracted tabular rows for the given label. "
                "In simulated mode, table contexts are not materialized."
            )

        # Otherwise, treat as per-file Content table reference.
        file_key = t.lstrip("/")
        recs = (self._files.get(file_key) or {}).get("records") or []
        # Infer "columns" from the union of keys in record dicts.
        cols: list[str] = []
        try:
            seen = set()
            for r in list(recs):
                if isinstance(r, dict):
                    for k in r.keys():
                        if k not in seen:
                            seen.add(k)
                            cols.append(str(k))
        except Exception:
            cols = []

        cols_str = ", ".join(cols[:10]) if cols else "unknown"
        return (
            f"{t} refers to the simulated per-file Content context for '{file_key}'. "
            f"It stores flattened content rows derived from the file. "
            f"Approximate row count: {len(recs)}. Example columns: {cols_str}."
        )

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
        table: Optional[str] = None,
        metric: str,
        keys: str | list[str],
        filter: Optional[str | dict[str, str]] = None,
        group_by: Optional[str | list[str]] = None,
    ) -> Any:
        """
        Simulated counterpart of the FileManager.reduce tool.

        The simulated file manager stores file metadata in-memory only; this
        method computes deterministic placeholder metrics with the same return
        shapes as the concrete implementation:

        * Single key, no grouping  → scalar.
        * Multiple keys, no grouping → ``dict[key -> scalar]``.
        * With grouping             → nested ``dict[group -> value or dict]``.
        """

        def _scalar(k: str) -> float:
            base = len(self._files) or 1
            return float(base + len(str(k)))

        key_list: list[str] = [keys] if isinstance(keys, str) else list(keys)

        if group_by is None:
            if isinstance(keys, str):
                return _scalar(keys)
            return {k: _scalar(k) for k in key_list}

        groups: list[str] = (
            [group_by] if isinstance(group_by, str) else [str(g) for g in group_by]
        )
        if isinstance(keys, str):
            return {g: _scalar(keys) for g in groups}
        return {g: {k: _scalar(k) for k in key_list} for g in groups}


# ─────────────────────────────────────────────────────────────────────────────
# Simulated GlobalFileManager
# ─────────────────────────────────────────────────────────────────────────────


class SimulatedGlobalFileManager(BaseGlobalFileManager):
    """
    Simulated counterpart to GlobalFileManager that produces plausible
    answers without invoking real tools. Aggregates multiple FileManagers
    and exposes the BaseGlobalFileManager public contract.
    """

    def __init__(self, managers: List[BaseFileManager]):
        self._managers: List[BaseFileManager] = list(managers)
        self._llm = new_llm_client(stateful=True)

        # Build prompt tool lists to mirror class-named exposure
        def _lf() -> List[str]:
            names = [
                getattr(m.__class__, "__name__", "FileManager") for m in self._managers
            ]
            return sorted(set(names))

        ask_tools: Dict[str, Any] = {"GlobalFileManager_list_filesystems": _lf}
        for mgr in self._managers:
            cname = getattr(mgr.__class__, "__name__", "FileManager")
            ask_tools[f"{cname}_ask"] = (
                lambda text, _c=cname: None
            )  # placeholder signature
            ask_tools[f"{cname}_ask_about_file"] = (
                lambda filename, question, _c=cname: None
            )

        organize_tools: Dict[str, Any] = {"GlobalFileManager_ask": (lambda text: None)}
        for mgr in self._managers:
            cname = getattr(mgr.__class__, "__name__", "FileManager")
            organize_tools[f"{cname}_organize"] = lambda text, _c=cname: None

        ask_sys = build_global_file_manager_ask_prompt(
            ask_tools,
            num_filesystems=len(self._managers),
            include_activity=True,
        )
        org_sys = build_global_file_manager_organize_prompt(
            organize_tools,
            num_filesystems=len(self._managers),
            include_activity=True,
        )
        self._llm.set_system_message(
            "You are a *simulated* global file manager assistant. "
            "Work at the conceptual level across multiple filesystems.\n\n"
            "As reference, here are the system messages used by the *real* global file manager. "
            "You do not have access to tools – produce final answers only.\n\n"
            f"'ask' (global) system message:\n{ask_sys}\n\n"
            f"'organize' (global) system message:\n{org_sys}",
        )

    def list_filesystems(self) -> List[str]:
        names = [
            getattr(m.__class__, "__name__", "FileManager") for m in self._managers
        ]
        return sorted(set(names))

    # ------------------------------ Public API ------------------------------ #
    @functools.wraps(BaseGlobalFileManager.ask, updated=())
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _requests_clarification: bool = False,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> SteerableToolHandle:
        # Provide a compact inventory snapshot to the simulator
        inventory = {
            getattr(m.__class__, "__name__", "FileManager"): getattr(
                m,
                "list",
                lambda: [],
            )()
            for m in self._managers
        }
        body = {
            "action": "global.ask",
            "request": text,
            "filesystems": self.list_filesystems(),
            "inventory": inventory,
        }
        instruction = build_simulated_method_prompt(
            "global_ask",
            json.dumps(body, indent=2),
            parent_chat_context=_parent_chat_context,
        )
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

    @functools.wraps(BaseGlobalFileManager.organize, updated=())
    async def organize(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _requests_clarification: bool = False,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> SteerableToolHandle:
        # Build a simple plan description and return a handle
        plan = {
            "action": "global.organize",
            "request": text,
            "filesystems": self.list_filesystems(),
        }
        instruction = build_simulated_method_prompt(
            "global_organize",
            json.dumps(plan, indent=2),
            parent_chat_context=_parent_chat_context,
        )
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

    @functools.wraps(BaseGlobalFileManager.clear, updated=())
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
