"""Rich-based progress display manager for FileManager pipeline.

Provides fixed-position progress bars that support concurrent stages per file,
showing ingestion and embedding progress simultaneously when using "along" strategy.
"""

from typing import Dict, Optional, Any
from threading import Lock
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TaskID,
)
from rich.live import Live

try:
    from unity.file_manager.types.config import FilePipelineConfig
except ImportError:
    FilePipelineConfig = Any  # type: ignore


class FileProgressManager:
    """Manages rich progress displays for multiple files with concurrent stage tracking.

    Supports showing ingestion and embedding progress simultaneously when using
    "along" embed strategy, with fixed-position per-file sections.
    """

    def __init__(self, enable: bool = True):
        """Initialize the progress manager.

        Parameters
        ----------
        enable : bool
            Whether to enable progress display. If False, all methods are no-ops.
        """
        self.enable = enable
        self.console = Console() if enable else None
        self.progress: Optional[Progress] = None
        self.live: Optional[Live] = None
        self._lock = Lock()

        # File tracking: file_path -> file_info dict
        self._files: Dict[str, Dict[str, Any]] = {}

        # Task IDs for each file's stages
        self._file_tasks: Dict[str, Dict[str, TaskID]] = {}

        # Concurrent embedding tracking: file_path -> {chunk_id: status}
        self._content_embed_chunks: Dict[str, Dict[str, str]] = (
            {}
        )  # chunk_id -> "started"|"completed"
        self._table_embed_chunks: Dict[str, Dict[str, Dict[str, str]]] = (
            {}
        )  # table_label -> {chunk_id: status}

        # Counters for concurrent operations
        self._content_embed_active: Dict[str, int] = {}  # file_path -> count
        self._content_embed_completed: Dict[str, int] = {}  # file_path -> count
        self._table_embed_active: Dict[str, Dict[str, int]] = (
            {}
        )  # file_path -> {table_label: count}
        self._table_embed_completed: Dict[str, Dict[str, int]] = (
            {}
        )  # file_path -> {table_label: count}

    def start(self) -> None:
        """Start the live progress display."""
        if not self.enable:
            return

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console,
            expand=True,
            refresh_per_second=10,  # Increase refresh rate for better responsiveness
        )
        self.live = Live(self.progress, console=self.console, refresh_per_second=10)
        self.live.start()

    def stop(self) -> None:
        """Stop the live display and finalize."""
        if self.live:
            self.live.stop()
            self.live = None
        if self.progress:
            self.progress = None

    def register_file(self, file_path: str, config: FilePipelineConfig) -> str:
        """Register a new file for progress tracking.

        Parameters
        ----------
        file_path : str
            File path identifier.
        config : FilePipelineConfig
            Pipeline configuration for this file.

        Returns
        -------
        str
            File ID (same as file_path for now).
        """
        if not self.enable or not self.progress:
            return file_path

        with self._lock:
            if file_path in self._files:
                return file_path

            # Extract file name for display
            display_name = file_path.split("/")[-1] if "/" in file_path else file_path

            # Detect embed strategy and ingest mode
            embed_strategy = getattr(getattr(config, "embed", None), "strategy", "off")
            ingest_mode = getattr(getattr(config, "ingest", None), "mode", "per_file")
            table_ingest = bool(
                getattr(getattr(config, "ingest", None), "table_ingest", True),
            )

            # Store file info
            self._files[file_path] = {
                "display_name": display_name,
                "embed_strategy": embed_strategy,
                "ingest_mode": ingest_mode,
                "table_ingest": table_ingest,
                "status": "parsing",
            }

            # Initialize task tracking
            self._file_tasks[file_path] = {}
            self._content_embed_chunks[file_path] = {}
            self._table_embed_chunks[file_path] = {}
            self._content_embed_active[file_path] = 0
            self._content_embed_completed[file_path] = 0
            self._table_embed_active[file_path] = {}
            self._table_embed_completed[file_path] = {}

            # Create parsing task
            task_id = self.progress.add_task(
                f"[cyan]📄 {display_name}[/cyan] - Parsing...",
                total=None,
            )
            self._file_tasks[file_path]["parsing"] = task_id

        return file_path

    def update_parsing(self, file_path: str, status: str) -> None:
        """Update parsing status for a file.

        Parameters
        ----------
        file_path : str
            File path identifier.
        status : str
            Status message (e.g., "parsing", "complete", "failed").
        """
        if not self.enable or not self.progress:
            return

        with self._lock:
            if file_path not in self._file_tasks:
                return

            task_id = self._file_tasks[file_path].get("parsing")
            if task_id is None:
                return

            display_name = self._files[file_path]["display_name"]

            if status == "complete":
                self.progress.update(
                    task_id,
                    description=f"[green]✓ {display_name}[/green] - Parsed",
                    completed=1,
                    total=1,
                )
                self._files[file_path]["status"] = "parsed"
            elif status == "failed":
                self.progress.update(
                    task_id,
                    description=f"[red]✗ {display_name}[/red] - Parse failed",
                    completed=1,
                    total=1,
                )
            else:
                self.progress.update(
                    task_id,
                    description=f"[cyan]📄 {display_name}[/cyan] - {status}",
                )

    def start_content_ingest(self, file_path: str, total: int) -> None:
        """Start content ingestion tracking.

        Parameters
        ----------
        file_path : str
            File path identifier.
        total : int
            Total number of chunks to ingest.
        """
        if not self.enable or not self.progress:
            return

        with self._lock:
            if file_path not in self._file_tasks:
                return

            display_name = self._files[file_path]["display_name"]
            task_id = self.progress.add_task(
                f"[yellow]📦 {display_name}[/yellow] - Content ingestion",
                total=total,
            )
            self._file_tasks[file_path]["content_ingest"] = task_id

    def update_content_ingest(self, file_path: str, current: int) -> None:
        """Update content ingestion progress.

        Parameters
        ----------
        file_path : str
            File path identifier.
        current : int
            Current chunk number (1-indexed).
        """
        if not self.enable or not self.progress:
            return

        with self._lock:
            task_id = self._file_tasks.get(file_path, {}).get("content_ingest")
            if task_id is not None:
                self.progress.update(task_id, completed=current)

    def complete_content_ingest(self, file_path: str) -> None:
        """Mark content ingestion as complete.

        Parameters
        ----------
        file_path : str
            File path identifier.
        """
        if not self.enable or not self.progress:
            return

        with self._lock:
            task_id = self._file_tasks.get(file_path, {}).get("content_ingest")
            if task_id is not None:
                # Get total from progress task
                task = self.progress.tasks[task_id]
                if task.total:
                    self.progress.update(task_id, completed=task.total)

    def start_content_embed(self, file_path: str, total: int) -> None:
        """Start content embedding tracking.

        Parameters
        ----------
        file_path : str
            File path identifier.
        total : int
            Total number of chunks to embed.
        """
        if not self.enable or not self.progress:
            return

        with self._lock:
            if file_path not in self._file_tasks:
                return

            display_name = self._files[file_path]["display_name"]
            embed_strategy = self._files[file_path]["embed_strategy"]

            if embed_strategy == "along":
                # Show concurrent embedding status
                task_id = self.progress.add_task(
                    f"[magenta]🔄 {display_name}[/magenta] - Content embedding (0 active, 0/{total} completed)",
                    total=total,
                )
            else:
                task_id = self.progress.add_task(
                    f"[magenta]🔄 {display_name}[/magenta] - Content embedding",
                    total=total,
                )

            self._file_tasks[file_path]["content_embed"] = task_id
            self._content_embed_chunks[file_path] = {}
            self._content_embed_active[file_path] = 0
            self._content_embed_completed[file_path] = 0

    def update_content_embed_chunk(
        self,
        file_path: str,
        chunk_id: str,
        status: str,
    ) -> None:
        """Update a specific content embedding chunk status.

        Parameters
        ----------
        file_path : str
            File path identifier.
        chunk_id : str
            Unique identifier for the chunk (e.g., "chunk_1").
        status : str
            Status: "started" or "completed".
        """
        if not self.enable or not self.progress:
            return

        with self._lock:
            if file_path not in self._file_tasks:
                return

            task_id = self._file_tasks.get(file_path, {}).get("content_embed")
            if task_id is None:
                return

            chunks = self._content_embed_chunks[file_path]
            embed_strategy = self._files[file_path]["embed_strategy"]

            if status == "started":
                chunks[chunk_id] = "started"
                self._content_embed_active[file_path] = sum(
                    1 for s in chunks.values() if s == "started"
                )
            elif status == "completed":
                chunks[chunk_id] = "completed"
                self._content_embed_active[file_path] = max(
                    0,
                    self._content_embed_active[file_path] - 1,
                )
                self._content_embed_completed[file_path] = sum(
                    1 for s in chunks.values() if s == "completed"
                )

            # Update progress bar description for "along" strategy
            if embed_strategy == "along":
                display_name = self._files[file_path]["display_name"]
                active = self._content_embed_active[file_path]
                completed = self._content_embed_completed[file_path]
                task = self.progress.tasks[task_id]
                total = task.total or 0
                self.progress.update(
                    task_id,
                    description=f"[magenta]🔄 {display_name}[/magenta] - Content embedding ({active} active, {completed}/{total} completed)",
                    completed=completed,
                )
            else:
                # Simple progress update for "after" strategy
                completed = self._content_embed_completed[file_path]
                self.progress.update(task_id, completed=completed)

    def complete_content_embed(self, file_path: str) -> None:
        """Mark content embedding as complete.

        Parameters
        ----------
        file_path : str
            File path identifier.
        """
        if not self.enable or not self.progress:
            return

        with self._lock:
            task_id = self._file_tasks.get(file_path, {}).get("content_embed")
            if task_id is not None:
                task = self.progress.tasks[task_id]
                if task.total:
                    self.progress.update(task_id, completed=task.total)

    def register_table(
        self,
        file_path: str,
        table_label: str,
        ingest_total: int,
        embed_total: int,
    ) -> None:
        """Register a table for progress tracking.

        Parameters
        ----------
        file_path : str
            File path identifier.
        table_label : str
            Table label/name.
        ingest_total : int
            Total chunks for table ingestion.
        embed_total : int
            Total chunks for table embedding.
        """
        if not self.enable or not self.progress:
            return

        with self._lock:
            if file_path not in self._file_tasks:
                return

            display_name = self._files[file_path]["display_name"]
            embed_strategy = self._files[file_path]["embed_strategy"]

            # Create table ingestion task
            ingest_task_id = self.progress.add_task(
                f"[blue]📊 {display_name}[/blue] - Table '{table_label}' ingestion",
                total=ingest_total,
            )

            # Create table embedding task
            if embed_strategy == "along":
                embed_task_id = self.progress.add_task(
                    f"[cyan]🔄 {display_name}[/cyan] - Table '{table_label}' embedding (0 active, 0/{embed_total} completed)",
                    total=embed_total,
                )
            else:
                embed_task_id = self.progress.add_task(
                    f"[cyan]🔄 {display_name}[/cyan] - Table '{table_label}' embedding",
                    total=embed_total,
                )

            # Store task IDs
            if "tables" not in self._file_tasks[file_path]:
                self._file_tasks[file_path]["tables"] = {}
            self._file_tasks[file_path]["tables"][table_label] = {
                "ingest": ingest_task_id,
                "embed": embed_task_id,
            }

            # Initialize embedding tracking
            if table_label not in self._table_embed_chunks[file_path]:
                self._table_embed_chunks[file_path][table_label] = {}
            self._table_embed_active[file_path][table_label] = 0
            self._table_embed_completed[file_path][table_label] = 0

    def update_table_ingest(
        self,
        file_path: str,
        table_label: str,
        current: int,
    ) -> None:
        """Update table ingestion progress.

        Parameters
        ----------
        file_path : str
            File path identifier.
        table_label : str
            Table label/name.
        current : int
            Current chunk number.
        """
        if not self.enable or not self.progress:
            return

        with self._lock:
            table_tasks = self._file_tasks.get(file_path, {}).get("tables", {})
            task_id = table_tasks.get(table_label, {}).get("ingest")
            if task_id is not None:
                self.progress.update(task_id, completed=current)

    def complete_table_ingest(self, file_path: str, table_label: str) -> None:
        """Mark table ingestion as complete.

        Parameters
        ----------
        file_path : str
            File path identifier.
        table_label : str
            Table label/name.
        """
        if not self.enable or not self.progress:
            return

        with self._lock:
            table_tasks = self._file_tasks.get(file_path, {}).get("tables", {})
            task_id = table_tasks.get(table_label, {}).get("ingest")
            if task_id is not None:
                task = self.progress.tasks[task_id]
                if task.total:
                    self.progress.update(task_id, completed=task.total)

    def start_table_embed(self, file_path: str, table_label: str) -> None:
        """Start table embedding tracking (no-op if already started via register_table).

        Parameters
        ----------
        file_path : str
            File path identifier.
        table_label : str
            Table label/name.
        """
        # Already handled in register_table, but kept for API consistency

    def update_table_embed_chunk(
        self,
        file_path: str,
        table_label: str,
        chunk_id: str,
        status: str,
    ) -> None:
        """Update a specific table embedding chunk status.

        Parameters
        ----------
        file_path : str
            File path identifier.
        table_label : str
            Table label/name.
        chunk_id : str
            Unique identifier for the chunk.
        status : str
            Status: "started" or "completed".
        """
        if not self.enable or not self.progress:
            return

        with self._lock:
            if file_path not in self._file_tasks:
                return

            table_tasks = self._file_tasks.get(file_path, {}).get("tables", {})
            task_id = table_tasks.get(table_label, {}).get("embed")
            if task_id is None:
                return

            embed_strategy = self._files[file_path]["embed_strategy"]

            if table_label not in self._table_embed_chunks[file_path]:
                self._table_embed_chunks[file_path][table_label] = {}

            chunks = self._table_embed_chunks[file_path][table_label]

            if status == "started":
                chunks[chunk_id] = "started"
                self._table_embed_active[file_path][table_label] = sum(
                    1 for s in chunks.values() if s == "started"
                )
            elif status == "completed":
                chunks[chunk_id] = "completed"
                self._table_embed_active[file_path][table_label] = max(
                    0,
                    self._table_embed_active[file_path][table_label] - 1,
                )
                self._table_embed_completed[file_path][table_label] = sum(
                    1 for s in chunks.values() if s == "completed"
                )

            # Update progress bar description for "along" strategy
            if embed_strategy == "along":
                display_name = self._files[file_path]["display_name"]
                active = self._table_embed_active[file_path][table_label]
                completed = self._table_embed_completed[file_path][table_label]
                task = self.progress.tasks[task_id]
                total = task.total or 0
                # Update with both description and completed count
                # Note: rich.Progress is thread-safe and will update automatically
                self.progress.update(
                    task_id,
                    description=f"[cyan]🔄 {display_name}[/cyan] - Table '{table_label}' embedding ({active} active, {completed}/{total} completed)",
                    completed=completed,
                )
            else:
                completed = self._table_embed_completed[file_path][table_label]
                self.progress.update(task_id, completed=completed)

    def complete_table_embed(self, file_path: str, table_label: str) -> None:
        """Mark table embedding as complete.

        Parameters
        ----------
        file_path : str
            File path identifier.
        table_label : str
            Table label/name.
        """
        if not self.enable or not self.progress:
            return

        with self._lock:
            table_tasks = self._file_tasks.get(file_path, {}).get("tables", {})
            task_id = table_tasks.get(table_label, {}).get("embed")
            if task_id is not None:
                task = self.progress.tasks[task_id]
                if task.total:
                    self.progress.update(task_id, completed=task.total)

    def complete_file(self, file_path: str) -> None:
        """Mark a file as completely processed.

        Parameters
        ----------
        file_path : str
            File path identifier.
        """
        if not self.enable:
            return

        with self._lock:
            if file_path in self._files:
                self._files[file_path]["status"] = "complete"

    def fail_file(self, file_path: str, error: str) -> None:
        """Mark a file as failed.

        Parameters
        ----------
        file_path : str
            File path identifier.
        error : str
            Error message.
        """
        if not self.enable or not self.progress:
            return

        with self._lock:
            if file_path in self._files:
                self._files[file_path]["status"] = "failed"
                display_name = self._files[file_path]["display_name"]
                # Update all active tasks to show error
                for task_id in self._file_tasks.get(file_path, {}).values():
                    if isinstance(task_id, dict):
                        for sub_task_id in task_id.values():
                            if isinstance(sub_task_id, dict):
                                for final_task_id in sub_task_id.values():
                                    if isinstance(final_task_id, int):
                                        try:
                                            task = self.progress.tasks[final_task_id]
                                            if not task.finished:
                                                self.progress.update(
                                                    final_task_id,
                                                    description=f"[red]✗ {display_name}[/red] - Failed: {error[:50]}",
                                                )
                                        except (IndexError, KeyError):
                                            pass
                            elif isinstance(sub_task_id, int):
                                try:
                                    task = self.progress.tasks[sub_task_id]
                                    if not task.finished:
                                        self.progress.update(
                                            sub_task_id,
                                            description=f"[red]✗ {display_name}[/red] - Failed: {error[:50]}",
                                        )
                                except (IndexError, KeyError):
                                    pass
                    elif isinstance(task_id, int):
                        try:
                            task = self.progress.tasks[task_id]
                            if not task.finished:
                                self.progress.update(
                                    task_id,
                                    description=f"[red]✗ {display_name}[/red] - Failed: {error[:50]}",
                                )
                        except (IndexError, KeyError):
                            pass
