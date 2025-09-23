# unity/file_manager/simulated.py
from __future__ import annotations

import asyncio
import json
import os
import functools
import threading
from typing import List, Dict, Any, Optional

import unify
from .base import BaseFileManager
from ..common.llm_helpers import SteerableToolHandle
from .prompt_builders import (
    build_ask_prompt,
    build_simulated_method_prompt,
)
from ..events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
    wrap_handle_with_logging,
)
from ..common.simulated import mirror_file_manager_tools


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper handle
# ─────────────────────────────────────────────────────────────────────────────
class _SimulatedFileHandle(SteerableToolHandle):
    """
    Handle returned by SimulatedFileManager.ask.
    """

    def __init__(
        self,
        llm: unify.Unify,
        initial_text: str,
        *,
        _return_reasoning_steps: bool,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None,
        clarification_down_q: asyncio.Queue[str] | None,
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

        # fire clarification question immediately if queues supplied
        if self._needs_clar:
            try:
                self._clar_up_q.put_nowait(
                    "Could you clarify your file-related request?",
                )
            except asyncio.QueueFull:
                pass

        self._extra_msgs: List[str] = []

        self._done_event = threading.Event()
        self._cancelled = False
        self._answer: str | None = None
        self._messages: List[Dict[str, Any]] = []
        self._paused = False

    # --------------------------------------------------------------------- #
    # SteerableToolHandle API
    # --------------------------------------------------------------------- #
    async def result(self):
        if self._cancelled:
            raise asyncio.CancelledError()

        # honour pauses injected by an outer loop
        while self._paused and not self._cancelled:
            await asyncio.sleep(0.05)

        if not self._done_event.is_set():
            if self._needs_clar:
                clar = await self._clar_down_q.get()
                self._extra_msgs.append(f"Clarification: {clar}")

            prompt = "\n\n---\n\n".join([self._initial] + self._extra_msgs)
            answer = await self._llm.generate(prompt)
            self._answer = answer
            self._messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            self._done_event.set()

        if self._want_steps:
            return self._answer, self._messages
        return self._answer

    def interject(self, message: str) -> str:
        if self._cancelled:
            return "Interaction stopped."
        self._extra_msgs.append(message)
        return "Acknowledged."

    def stop(self, reason: str | None = None) -> str:
        self._cancelled = True
        self._done_event.set()
        return "Stopped." if reason is None else f"Stopped: {reason}"

    def pause(self) -> str:
        if self._paused:
            return "Already paused."
        self._paused = True
        return "Paused."

    def resume(self) -> str:
        if not self._paused:
            return "Already running."
        self._paused = False
        return "Resumed."

    def done(self) -> bool:
        return self._done_event.is_set()

    @property
    def valid_tools(self):
        tools = {
            self.interject.__name__: self.interject,
            self.stop.__name__: self.stop,
        }
        if self._paused:
            tools[self.resume.__name__] = self.resume
        else:
            tools[self.pause.__name__] = self.pause
        return tools

    async def ask(self, question: str) -> "SteerableToolHandle":
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

        return _SimulatedFileHandle(
            self._llm,
            follow_up_prompt,
            _return_reasoning_steps=self._want_steps,
            _requests_clarification=False,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )


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

        # Counter for simulated file IDs
        self._next_file_id = 1

        # Shared, *stateful* **asynchronous** LLM
        self._llm = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
            traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
            stateful=True,
        )

        # Mirror the real file manager's tool exposure programmatically
        try:
            ask_tools = mirror_file_manager_tools("ask")
        except (ImportError, AttributeError):
            # Fallback if mirror function doesn't exist yet
            ask_tools = {
                "list": {"description": "List all available files"},
                "exists": {"description": "Check if a file exists"},
                "parse": {"description": "Parse file content into structured records"},
                "import_file": {"description": "Import a single file from filesystem"},
                "import_directory": {
                    "description": "Import all files from a directory",
                },
                "_search_files": {"description": "Semantic search over file contents"},
                "_filter_files": {
                    "description": "Filter files using boolean expressions",
                },
                "_list_columns": {"description": "List available table columns"},
            }

        # Build prompt using the same pattern as other managers
        try:
            ask_msg = build_ask_prompt(
                ask_tools,
                num_files=len(self._files),
                columns={
                    "filename": "str",
                    "status": "str",
                    "full_text": "str",
                    "metadata": "dict",
                    "description": "str",
                },
                include_activity=self._rolling_summary_in_prompts,
            )
        except (ImportError, TypeError):
            # Fallback prompt if builder doesn't exist yet
            ask_msg = (
                "You are a file management assistant. You can list files, "
                "check if files exist, parse file contents, search files semantically, "
                "filter files, and import files from filesystem. "
                "Available tools: list(), exists(filename), parse(filename_or_list), "
                "import_file(file_path), import_directory(directory), "
                "_search_files(references, k), _filter_files(filter), _list_columns()"
            )

        self._llm.set_system_message(
            "You are a *simulated* file manager assistant. "
            "There is no real file storage; invent plausible file records and "
            "keep your story consistent across turns.\n\n"
            "As a reference, the system messages for the *real* file-manager 'ask' method is as follows. "
            "You do not have access to any real tools, so you should just create a final answer to the question/request. "
            f"\n\n'ask' system message:\n{ask_msg}\n\n"
            f"Back-story: {self._description}",
        )

    # --------------------------------------------------------------------- #
    # ask                                                                   #
    # --------------------------------------------------------------------- #
    @functools.wraps(BaseFileManager.ask, updated=())
    async def ask(
        self,
        filename: str,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        _requests_clarification: bool = False,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
        log_events: bool = False,
    ) -> SteerableToolHandle:
        should_log = self._log_events or log_events
        call_id = _call_id

        if should_log:
            if call_id is None:
                call_id = new_call_id()
            await publish_manager_method_event(
                call_id,
                "FileManager",
                "ask",
                phase="incoming",
                filename=filename,
                question=question,
            )

        # Check if file exists in simulated storage
        if filename not in self._files:
            raise FileNotFoundError(f"File '{filename}' not found in simulated storage")

        instruction = build_simulated_method_prompt(
            "ask",
            f"File: {filename}\nQuestion: {question}",
            parent_chat_context=parent_chat_context,
        )

        # Add file context
        file_info = self._files[filename]
        instruction += f"\n\nFile information: {json.dumps(file_info, indent=2)}"

        handle = _SimulatedFileHandle(
            self._llm,
            instruction,
            _return_reasoning_steps=_return_reasoning_steps,
            _requests_clarification=_requests_clarification,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

        if should_log and call_id is not None:
            handle = wrap_handle_with_logging(
                handle,
                call_id,
                "FileManager",
                "ask",
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

    def parse(self, filenames) -> Dict[str, Dict[str, Any]]:
        """Parse files from simulated storage."""
        if isinstance(filenames, str):
            filenames = [filenames]

        results = {}
        for filename in filenames:
            if filename in self._files:
                file_data = self._files[filename]
                results[filename] = {
                    "status": "success",
                    "records": file_data.get("records", []),
                    "metadata": file_data.get("metadata", {}),
                    "full_text": file_data.get("full_text", ""),
                    "description": file_data.get("description", ""),
                    "error": None,
                }
            else:
                results[filename] = {
                    "status": "error",
                    "error": f"File '{filename}' not found",
                    "records": [],
                    "metadata": {},
                    "full_text": "",
                    "description": "",
                }

        return results

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
            metadata={"file_type": extension, "source_path": file_path},
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
                metadata={"file_type": ext, "source_directory": directory},
                full_text=f"Simulated content from {directory}/{filename}",
                description=f"File from directory: {filename}",
            )
            added_files.append(filename)

        return added_files

    def _search_files(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Simulate semantic search over files."""
        files = list(self._files.items())

        if not references:
            # Return most recent files (simulate by reversing order)
            files = files[-k:]
        else:
            # Simulate ranking by simple keyword matching
            def score_file(item):
                filename, file_data = item
                score = 0
                for source_expr, reference_text in references.items():
                    if source_expr == "full_text":
                        text = file_data.get("full_text", "")
                    elif source_expr == "description":
                        text = file_data.get("description", "")
                    elif source_expr == "metadata":
                        text = str(file_data.get("metadata", {}))
                    else:
                        text = file_data.get(source_expr, "")

                    # Simple keyword matching for simulation
                    keywords = reference_text.lower().split()
                    for keyword in keywords:
                        if keyword in text.lower():
                            score += 1

                return score

            # Sort by relevance score
            files.sort(key=score_file, reverse=True)
            files = files[:k]

        # Convert to File-like dictionaries
        results = []
        for filename, file_data in files:
            results.append(
                {
                    "file_id": file_data.get("file_id", self._next_file_id),
                    "filename": filename,
                    "status": file_data.get("status", "success"),
                    "error": file_data.get("error", None),
                    "records": file_data.get("records", []),
                    "full_text": file_data.get("full_text", ""),
                    "metadata": file_data.get("metadata", {}),
                    "description": file_data.get("description", ""),
                    "imported_at": file_data.get("imported_at", "2024-01-01T00:00:00Z"),
                },
            )

        return results

    def _filter_files(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Simulate filtering files using boolean expressions."""
        files = list(self._files.items())

        # Apply offset and limit
        files = files[offset : offset + limit]

        # Convert to File-like dictionaries
        results = []
        for filename, file_data in files:
            file_dict = {
                "file_id": file_data.get("file_id", self._next_file_id),
                "filename": filename,
                "status": file_data.get("status", "success"),
                "error": file_data.get("error", None),
                "records": file_data.get("records", []),
                "full_text": file_data.get("full_text", ""),
                "metadata": file_data.get("metadata", {}),
                "description": file_data.get("description", ""),
                "imported_at": file_data.get("imported_at", "2024-01-01T00:00:00Z"),
            }

            # Simple filter simulation - just check if filter string matches
            if filter is None:
                results.append(file_dict)
            else:
                # Simulate basic filtering (very simplified)
                if "status == 'success'" in filter and file_dict["status"] == "success":
                    results.append(file_dict)
                elif "endswith('.pdf')" in filter and filename.endswith(".pdf"):
                    results.append(file_dict)
                elif not any(op in filter for op in ["==", "endswith", ">"]):
                    # If no clear filter operations, include all
                    results.append(file_dict)

        return results

    def _list_columns(
        self,
        *,
        include_types: bool = True,
    ) -> Dict[str, str] | List[str]:
        """Simulate listing table columns."""
        columns = {
            "file_id": "int",
            "filename": "str",
            "status": "str",
            "error": "str",
            "records": "list",
            "full_text": "str",
            "metadata": "dict",
            "description": "str",
            "imported_at": "datetime",
        }

        return columns if include_types else list(columns.keys())

    def _delete_file(self, *, file_id: int) -> Dict[str, Any]:
        """Simulate deleting a file record."""
        # Find file by ID (simplified simulation)
        for filename, file_data in self._files.items():
            if file_data.get("file_id", 0) == file_id:
                del self._files[filename]
                return {
                    "outcome": "file deleted",
                    "details": {"file_id": file_id},
                }

        raise ValueError(f"No file found with file_id {file_id} to delete.")

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
