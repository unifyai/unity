"""Task-based pipeline execution engine for FileManager.

This module provides a unified, dependency-aware task execution system for the
file processing pipeline. It replaces the complex async/sync/parallel execution
paths with a clean task graph model.

Key components:
- Task: A single unit of work with dependencies and retry logic
- TaskGraph: A DAG of tasks with topological ordering
- PipelineExecutor: Executes task graphs with configurable parallelism

The executor handles:
- Dependency resolution between tasks
- Parallel execution with configurable worker pools
- Retry logic with exponential backoff
- Progress reporting via the ProgressReporter protocol
- Graceful error handling and fail-fast mode
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
)

from unity.file_manager.file_parsers.types.contracts import FileParseResult
from unity.file_manager.types.ingest import (
    BaseIngestedFile,
    IngestedMinimal,
    IngestPipelineResult,
    ContentRef,
    FileMetrics,
    FileResultType,
)

if TYPE_CHECKING:
    from .progress import ProgressReporter

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a task in the execution pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class TaskType(Enum):
    """Types of tasks in the file pipeline."""

    PARSE = "parse"
    FILE_RECORD = "file_record"
    INGEST_CONTENT = "ingest_content"
    EMBED_CONTENT = "embed_content"
    INGEST_TABLE = "ingest_table"
    EMBED_TABLE = "embed_table"
    FILE_COMPLETE = "file_complete"


T = TypeVar("T")


@dataclass
class TaskResult(Generic[T]):
    """Result of a task execution."""

    success: bool
    value: Optional[T] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    retries: int = 0


@dataclass
class Task:
    """A single unit of work in the pipeline.

    Tasks represent atomic operations like ingesting a batch of rows or
    embedding a chunk. They track their dependencies, status, and results.

    Attributes
    ----------
    id : str
        Unique identifier for the task.
    task_type : TaskType
        The type of operation this task performs.
    file_path : str
        The file this task operates on.
    func : Callable
        The function to execute.
    args : tuple
        Positional arguments for the function.
    kwargs : dict
        Keyword arguments for the function.
    dependencies : set[str]
        Task IDs that must complete before this task can run.
    status : TaskStatus
        Current status of the task.
    result : TaskResult | None
        Result after execution.
    chunk_index : int | None
        For chunked operations, the index of this chunk.
    total_chunks : int | None
        For chunked operations, the total number of chunks.
    table_label : str | None
        For table operations, the table label.
    metadata : dict
        Additional metadata for progress reporting.
    """

    id: str
    task_type: TaskType
    file_path: str
    func: Callable[..., Any]
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[TaskResult[Any]] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    table_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Task):
            return False
        return self.id == other.id


class TaskGraph:
    """A directed acyclic graph of tasks with dependency resolution.

    The graph maintains topological ordering and tracks which tasks
    are ready to execute based on their dependency completion status.

    Attributes
    ----------
    tasks : dict[str, Task]
        All tasks in the graph by ID.
    file_path : str
        The file this graph processes.
    """

    def __init__(self, file_path: str) -> None:
        """Initialize an empty task graph for a file.

        Parameters
        ----------
        file_path : str
            The file this graph will process.
        """
        self.file_path = file_path
        self.tasks: Dict[str, Task] = {}
        self._completed: Set[str] = set()

    def add_task(self, task: Task) -> str:
        """Add a task to the graph.

        Parameters
        ----------
        task : Task
            The task to add.

        Returns
        -------
        str
            The task ID.
        """
        self.tasks[task.id] = task
        return task.id

    def add_dependency(self, task_id: str, depends_on: str) -> None:
        """Add a dependency between two tasks.

        Parameters
        ----------
        task_id : str
            The task that depends on another.
        depends_on : str
            The task that must complete first.
        """
        if task_id in self.tasks:
            self.tasks[task_id].dependencies.add(depends_on)

    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that are ready to execute.

        A task is ready when all its dependencies are completed.

        Returns
        -------
        list[Task]
            Tasks ready for execution.
        """
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            if task.dependencies.issubset(self._completed):
                ready.append(task)
        return ready

    def mark_completed(self, task_id: str, result: TaskResult[Any]) -> None:
        """Mark a task as completed with its result.

        Parameters
        ----------
        task_id : str
            The task ID.
        result : TaskResult
            The execution result.
        """
        if task_id in self.tasks:
            self.tasks[task_id].status = (
                TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            )
            self.tasks[task_id].result = result
            if result.success:
                self._completed.add(task_id)

    def mark_cancelled(self, task_id: str) -> None:
        """Mark a task as cancelled.

        Parameters
        ----------
        task_id : str
            The task ID.
        """
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.CANCELLED

    def is_complete(self) -> bool:
        """Check if all tasks have finished (completed, failed, or cancelled).

        Returns
        -------
        bool
            True if no tasks are pending or running.
        """
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            for t in self.tasks.values()
        )

    def has_failures(self) -> bool:
        """Check if any task failed.

        Returns
        -------
        bool
            True if at least one task failed.
        """
        return any(t.status == TaskStatus.FAILED for t in self.tasks.values())

    def get_failed_tasks(self) -> List[Task]:
        """Get all failed tasks.

        Returns
        -------
        list[Task]
            Failed tasks.
        """
        return [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of task statuses.

        Returns
        -------
        dict
            Status counts and overall success.
        """
        counts = {s.value: 0 for s in TaskStatus}
        for t in self.tasks.values():
            counts[t.status.value] += 1
        return {
            "total": len(self.tasks),
            "statuses": counts,
            "success": not self.has_failures() and counts["pending"] == 0,
        }


@dataclass
class ExecutionConfig:
    """Configuration for task execution.

    Attributes
    ----------
    parallel_files : bool
        Whether to process multiple files in parallel.
    max_file_workers : int
        Maximum concurrent file processing tasks.
    max_embed_workers : int
        Maximum concurrent embedding tasks per file.
    max_retries : int
        Maximum retry attempts for failed tasks.
    retry_delay_seconds : float
        Base delay between retries (with exponential backoff).
    fail_fast : bool
        If True, stop pipeline on first failure.
    """

    parallel_files: bool = False
    max_file_workers: int = 4
    max_embed_workers: int = 8
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    fail_fast: bool = False


class PipelineExecutor:
    """Executes task graphs with configurable parallelism and retries.

    The executor processes tasks from one or more TaskGraphs, handling
    dependencies, retries, and progress reporting.

    Attributes
    ----------
    config : ExecutionConfig
        Execution configuration.
    reporter : ProgressReporter | None
        Optional progress reporter for status updates.
    verbosity : str
        Verbosity level for progress events: "low", "medium", or "high".
    """

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        reporter: Optional["ProgressReporter"] = None,
        verbosity: str = "low",
    ) -> None:
        """Initialize the executor.

        Parameters
        ----------
        config : ExecutionConfig | None
            Execution configuration. Defaults to ExecutionConfig().
        reporter : ProgressReporter | None
            Optional progress reporter.
        verbosity : str
            Verbosity level for progress events: "low", "medium", "high".
            Controls the amount of metadata included in progress events.
        """
        self.config = config or ExecutionConfig()
        self.reporter = reporter
        self._verbosity = verbosity
        self._executor: Optional[ThreadPoolExecutor] = None
        self._stop_requested = False

    def execute_graph(self, graph: TaskGraph) -> Dict[str, TaskResult[Any]]:
        """Execute all tasks in a graph synchronously.

        Tasks are executed in topological order, respecting dependencies.
        Parallelism is used for independent tasks when configured.

        Parameters
        ----------
        graph : TaskGraph
            The task graph to execute.

        Returns
        -------
        dict[str, TaskResult]
            Results for all tasks by ID.
        """
        results: Dict[str, TaskResult[Any]] = {}
        self._stop_requested = False

        # Create thread pool for parallel task execution
        max_workers = max(1, self.config.max_embed_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            self._executor = executor

            while not graph.is_complete() and not self._stop_requested:
                ready_tasks = graph.get_ready_tasks()
                if not ready_tasks:
                    # No tasks ready but graph not complete - check for deadlock
                    pending = [
                        t
                        for t in graph.tasks.values()
                        if t.status == TaskStatus.PENDING
                    ]
                    if pending:
                        # Check if pending tasks have unmet dependencies (failed OR cancelled)
                        # We need to cancel tasks that depend on failed OR cancelled tasks
                        # to properly handle transitive dependency failures
                        failed_or_cancelled = {
                            t.id
                            for t in graph.tasks.values()
                            if t.status in (TaskStatus.FAILED, TaskStatus.CANCELLED)
                        }
                        if failed_or_cancelled:
                            # Cancel tasks that have any failed/cancelled dependency
                            cancelled_any = False
                            for task in pending:
                                if task.dependencies.intersection(failed_or_cancelled):
                                    graph.mark_cancelled(task.id)
                                    results[task.id] = TaskResult(
                                        success=False,
                                        error="Dependency failed or cancelled",
                                    )
                                    self._report_task_cancelled(task)
                                    cancelled_any = True
                            if cancelled_any:
                                continue
                        # If we reach here with pending tasks but no failed/cancelled deps,
                        # there's a true deadlock (shouldn't happen with proper graph)
                        logger.error(
                            f"Deadlock detected in task graph for {graph.file_path}: "
                            f"{len(pending)} pending tasks with no failed/cancelled dependencies",
                        )
                        # Cancel all remaining pending tasks to avoid hanging
                        for task in pending:
                            graph.mark_cancelled(task.id)
                            results[task.id] = TaskResult(
                                success=False,
                                error="Deadlock detected - task cancelled",
                            )
                            self._report_task_cancelled(task)
                        break
                    continue

                # Execute ready tasks (potentially in parallel)
                if len(ready_tasks) == 1:
                    # Single task - execute directly
                    task = ready_tasks[0]
                    result = self._execute_task(task)
                    results[task.id] = result
                    graph.mark_completed(task.id, result)
                    # Propagate inserted_ids from ingest tasks to dependent embed tasks
                    self._propagate_inserted_ids(graph, task, result)
                else:
                    # Multiple ready tasks - execute in parallel
                    futures = {
                        executor.submit(self._execute_task, task): task
                        for task in ready_tasks
                    }
                    for future in futures:
                        task = futures[future]
                        try:
                            result = future.result()
                        except Exception as e:
                            result = TaskResult(success=False, error=str(e))
                        results[task.id] = result
                        graph.mark_completed(task.id, result)
                        # Propagate inserted_ids from ingest tasks to dependent embed tasks
                        self._propagate_inserted_ids(graph, task, result)

                # Check fail-fast
                if self.config.fail_fast and graph.has_failures():
                    logger.info("Fail-fast triggered, stopping execution")
                    self._stop_requested = True

        self._executor = None
        return results

    def _propagate_inserted_ids(
        self,
        graph: TaskGraph,
        completed_task: Task,
        result: TaskResult[Any],
    ) -> None:
        """Propagate inserted_ids from ingest tasks to their dependent embed tasks.

        The task graph encodes the dependency between ingest and embed tasks via
        the ``depends_on_ingest`` entry in the embed task's metadata. When an
        ingest task completes successfully, we take its ``inserted_ids`` from
        the TaskResult and wire them into the kwargs of the corresponding embed
        task. This ensures that embed tasks operate on the correct row IDs.

        This method is intentionally generic: it works for both content and table
        tasks, as long as:

        - The ingest task's result.value is a dict containing ``inserted_ids``.
        - The embed task's metadata contains ``depends_on_ingest`` pointing to
          the ingest task's ID.
        """
        # Only propagate on successful tasks with a dict result value
        if not result.success or not isinstance(result.value, dict):
            return

        inserted_ids = result.value.get("inserted_ids")
        if not inserted_ids:
            return

        ingest_task_id = completed_task.id

        for task in graph.tasks.values():
            # Look for embed tasks that declare a dependency on this ingest task
            if task.metadata.get("depends_on_ingest") != ingest_task_id:
                continue

            # We expect embed tasks to accept an 'inserted_ids' kwarg
            existing_ids = task.kwargs.get("inserted_ids") or []
            # In the current design, each embed task is associated with exactly
            # one ingest chunk, so we simply overwrite any placeholder list.
            # If we ever need to aggregate across multiple ingests, we can
            # change this to ``existing_ids + inserted_ids``.
            if existing_ids:
                # For safety, avoid duplicating IDs if they were partially set
                merged = list(dict.fromkeys(list(existing_ids) + list(inserted_ids)))
                task.kwargs["inserted_ids"] = merged
            else:
                task.kwargs["inserted_ids"] = inserted_ids

            logger.debug(
                "Propagated %d inserted_ids from %s to embed task %s",
                len(inserted_ids),
                ingest_task_id,
                task.id,
            )

    async def execute_graph_async(
        self,
        graph: TaskGraph,
    ) -> Dict[str, TaskResult[Any]]:
        """Execute all tasks in a graph asynchronously.

        Parameters
        ----------
        graph : TaskGraph
            The task graph to execute.

        Returns
        -------
        dict[str, TaskResult]
            Results for all tasks by ID.
        """
        loop = asyncio.get_event_loop()
        results: Dict[str, TaskResult[Any]] = {}
        self._stop_requested = False

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_embed_workers)

        async def run_task(task: Task) -> TaskResult[Any]:
            async with semaphore:
                return await loop.run_in_executor(None, self._execute_task, task)

        pending_futures: Dict[asyncio.Task, Task] = {}

        while not graph.is_complete() and not self._stop_requested:
            # Start ready tasks
            ready_tasks = graph.get_ready_tasks()
            for task in ready_tasks:
                task.status = TaskStatus.RUNNING
                future = asyncio.create_task(run_task(task))
                pending_futures[future] = task

            if not pending_futures:
                # No pending work - check for deadlock
                if not graph.is_complete():
                    failed_deps = graph.get_failed_tasks()
                    if failed_deps:
                        # Cancel downstream tasks
                        for task in graph.tasks.values():
                            if task.status == TaskStatus.PENDING:
                                if task.dependencies.intersection(
                                    {t.id for t in failed_deps},
                                ):
                                    graph.mark_cancelled(task.id)
                                    results[task.id] = TaskResult(
                                        success=False,
                                        error="Dependency failed",
                                    )
                        continue
                break

            # Wait for at least one task to complete
            done, _ = await asyncio.wait(
                pending_futures.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Process completed tasks
            for future in done:
                task = pending_futures.pop(future)
                try:
                    result = future.result()
                except Exception as e:
                    result = TaskResult(success=False, error=str(e))
                results[task.id] = result
                graph.mark_completed(task.id, result)

                # Check fail-fast
                if self.config.fail_fast and not result.success:
                    logger.info("Fail-fast triggered, stopping execution")
                    self._stop_requested = True
                    # Cancel remaining tasks
                    for pending_future in pending_futures:
                        pending_future.cancel()
                    for pending_task in pending_futures.values():
                        graph.mark_cancelled(pending_task.id)
                    pending_futures.clear()
                    break

        return results

    def execute_graphs(
        self,
        graphs: List[TaskGraph],
    ) -> Dict[str, Dict[str, TaskResult[Any]]]:
        """Execute multiple task graphs.

        If parallel_files is True, graphs are processed concurrently.
        Otherwise, they are processed sequentially.

        Parameters
        ----------
        graphs : list[TaskGraph]
            Task graphs to execute.

        Returns
        -------
        dict[str, dict[str, TaskResult]]
            Results by file_path, then by task_id.
        """
        if not graphs:
            return {}

        if not self.config.parallel_files or len(graphs) == 1:
            # Sequential execution
            return {g.file_path: self.execute_graph(g) for g in graphs}

        # Parallel execution
        all_results: Dict[str, Dict[str, TaskResult[Any]]] = {}
        max_workers = min(len(graphs), self.config.max_file_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.execute_graph, g): g for g in graphs}
            for future in futures:
                graph = futures[future]
                try:
                    results = future.result()
                    all_results[graph.file_path] = results
                except Exception as e:
                    logger.error(f"Error executing graph for {graph.file_path}: {e}")
                    all_results[graph.file_path] = {}

        return all_results

    def _execute_task(self, task: Task) -> TaskResult[Any]:
        """Execute a single task with retry logic and comprehensive timing.

        Timing is captured at multiple levels:
        - duration_ms: how long THIS execution attempt took
        - elapsed_ms: cumulative time since file_start_time (from task metadata)

        Parameters
        ----------
        task : Task
            The task to execute.

        Returns
        -------
        TaskResult
            The execution result with timing information.
        """
        # Get file_start_time from task metadata for elapsed calculation
        file_start_time = task.metadata.get("file_start_time", time.perf_counter())

        self._report_task_started(task, file_start_time)

        retries = 0
        last_error: Optional[str] = None
        last_traceback: Optional[str] = None

        while retries <= self.config.max_retries:
            start_time = time.perf_counter()
            try:
                result = task.func(*task.args, **task.kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                elapsed_ms = (time.perf_counter() - file_start_time) * 1000

                task_result = TaskResult(
                    success=True,
                    value=result,
                    duration_ms=duration_ms,
                    retries=retries,
                )
                self._report_task_completed(task, task_result, elapsed_ms)
                return task_result

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                elapsed_ms = (time.perf_counter() - file_start_time) * 1000
                last_error = str(e)
                last_traceback = traceback.format_exc()
                retries += 1

                if retries <= self.config.max_retries:
                    self._report_task_retry(task, retries, last_error, elapsed_ms)
                    # Exponential backoff
                    delay = self.config.retry_delay_seconds * (2 ** (retries - 1))
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Task {task.id} failed after {retries} attempts: {last_error}\n{last_traceback}",
                    )

        # All retries exhausted
        elapsed_ms = (time.perf_counter() - file_start_time) * 1000
        task_result = TaskResult(
            success=False,
            error=last_error,
            duration_ms=0.0,
            retries=retries - 1,
        )
        self._report_task_failed(task, task_result, elapsed_ms, last_traceback)
        return task_result

    def stop(self) -> None:
        """Request graceful shutdown of execution."""
        self._stop_requested = True

    def _report_task_started(self, task: Task, file_start_time: float) -> None:
        """Report task started to progress reporter with timing."""
        if self.reporter is None:
            return
        try:
            from .progress import create_progress_event

            elapsed_ms = (time.perf_counter() - file_start_time) * 1000
            meta = self._build_task_meta(task)

            event = create_progress_event(
                task.file_path,
                task.task_type.value,
                "started",
                duration_ms=0.0,
                elapsed_ms=elapsed_ms,
                meta=meta,
                verbosity=self._verbosity,  # type: ignore[arg-type]
            )
            self.reporter.report(event)
        except Exception as e:
            logger.debug(f"Progress report failed: {e}")

    def _report_task_completed(
        self,
        task: Task,
        result: TaskResult[Any],
        elapsed_ms: float,
    ) -> None:
        """Report task completed to progress reporter with timing."""
        if self.reporter is None:
            return
        try:
            from .progress import create_progress_event

            meta = self._build_task_meta(task)
            # Add result-specific metadata
            meta["retries"] = result.retries
            if result.value and isinstance(result.value, dict):
                if "inserted_ids" in result.value:
                    meta["inserted_ids_count"] = len(result.value["inserted_ids"])
                if "row_count" in result.value:
                    meta["row_count"] = result.value["row_count"]

            event = create_progress_event(
                task.file_path,
                task.task_type.value,
                "completed",
                duration_ms=result.duration_ms,
                elapsed_ms=elapsed_ms,
                meta=meta,
                verbosity=self._verbosity,  # type: ignore[arg-type]
            )
            self.reporter.report(event)
        except Exception as e:
            logger.debug(f"Progress report failed: {e}")

    def _report_task_failed(
        self,
        task: Task,
        result: TaskResult[Any],
        elapsed_ms: float,
        traceback_str: Optional[str] = None,
    ) -> None:
        """Report task failed to progress reporter with timing and traceback."""
        if self.reporter is None:
            return
        try:
            from .progress import create_progress_event

            meta = self._build_task_meta(task)
            meta["retries"] = result.retries

            event = create_progress_event(
                task.file_path,
                task.task_type.value,
                "failed",
                duration_ms=result.duration_ms,
                elapsed_ms=elapsed_ms,
                error=result.error,
                traceback_str=traceback_str,
                meta=meta,
                verbosity=self._verbosity,  # type: ignore[arg-type]
            )
            self.reporter.report(event)
        except Exception as e:
            logger.debug(f"Progress report failed: {e}")

    def _report_task_retry(
        self,
        task: Task,
        retry_num: int,
        error: str,
        elapsed_ms: float,
    ) -> None:
        """Report task retry to progress reporter with timing."""
        if self.reporter is None:
            return
        try:
            from .progress import create_progress_event

            meta = self._build_task_meta(task)
            meta["retry_num"] = retry_num
            meta["max_retries"] = self.config.max_retries

            event = create_progress_event(
                task.file_path,
                task.task_type.value,
                "retry",
                duration_ms=0.0,
                elapsed_ms=elapsed_ms,
                error=f"Retry {retry_num}/{self.config.max_retries}: {error}",
                meta=meta,
                verbosity=self._verbosity,  # type: ignore[arg-type]
            )
            self.reporter.report(event)
        except Exception as e:
            logger.debug(f"Progress report failed: {e}")

    def _report_task_cancelled(self, task: Task) -> None:
        """Report task cancelled to progress reporter."""
        if self.reporter is None:
            return
        try:
            from .progress import create_progress_event

            # Get file_start_time for elapsed calculation
            file_start_time = task.metadata.get("file_start_time", time.perf_counter())
            elapsed_ms = (time.perf_counter() - file_start_time) * 1000

            meta = self._build_task_meta(task)
            meta["reason"] = "dependency_failed"

            event = create_progress_event(
                task.file_path,
                task.task_type.value,
                "cancelled",
                duration_ms=0.0,
                elapsed_ms=elapsed_ms,
                meta=meta,
                verbosity=self._verbosity,  # type: ignore[arg-type]
            )
            self.reporter.report(event)
        except Exception as e:
            logger.debug(f"Progress report failed: {e}")

    def _build_task_meta(self, task: Task) -> Dict[str, Any]:
        """Build verbosity-appropriate metadata for a task.

        Parameters
        ----------
        task : Task
            The task to build metadata for.

        Returns
        -------
        dict
            Metadata dictionary with task-specific fields.
        """
        meta: Dict[str, Any] = {}

        # Always include chunk info if present
        if task.chunk_index is not None:
            meta["chunk"] = task.chunk_index + 1  # 1-indexed for humans
        if task.total_chunks is not None:
            meta["total_chunks"] = task.total_chunks
        if task.table_label is not None:
            meta["table_label"] = task.table_label

        # Include metadata from task
        if "row_count" in task.metadata:
            meta["row_count"] = task.metadata["row_count"]
        if "batch_size" in task.metadata:
            meta["batch_size"] = task.metadata["batch_size"]

        return meta


def create_task_id(
    task_type: TaskType,
    file_path: str,
    chunk_index: Optional[int] = None,
    table_label: Optional[str] = None,
) -> str:
    """Create a unique task ID.

    Parameters
    ----------
    task_type : TaskType
        The type of task.
    file_path : str
        The file path.
    chunk_index : int | None
        Optional chunk index for batched operations.
    table_label : str | None
        Optional table label for table operations.

    Returns
    -------
    str
        A unique task ID.
    """
    parts = [task_type.value, file_path]
    if table_label:
        parts.append(table_label)
    if chunk_index is not None:
        parts.append(str(chunk_index))
    # Add short UUID suffix for uniqueness
    parts.append(uuid.uuid4().hex[:6])
    return "__".join(parts)


# ---------------------------------------------------------------------------
# Result Aggregation
# ---------------------------------------------------------------------------


def build_file_result(
    file_path: str,
    graph: TaskGraph,
    task_results: Dict[str, TaskResult[Any]],
    file_start_time: float,
    parse_result: FileParseResult,
) -> Dict[str, Any]:
    """
    Aggregate all task results into a comprehensive file-level summary.

    This function walks through all tasks in the graph and aggregates
    timing, success/failure counts, and detailed breakdowns.

    Parameters
    ----------
    file_path : str
        The file path being processed.
    graph : TaskGraph
        The task graph containing all tasks.
    task_results : dict[str, TaskResult]
        Results for each task by ID.
    file_start_time : float
        Timestamp when file processing started.
    parse_result : FileParseResult
        Original parse result from the file parser (Pydantic model).

    Returns
    -------
    dict
        Comprehensive file-level result with structure:
        {
            "file_path": str,
            "status": "success" | "partial" | "error",
            "total_duration_ms": float,
            "timing_breakdown": {
                "file_record_ms": float,
                "ingest_content_ms": float,
                "embed_content_ms": float,
                "ingest_table_ms": float,
                "embed_table_ms": float,
            },
            "chunks": {
                "content_ingested": int,
                "content_embedded": int,
                "tables_ingested": int,
                "tables_embedded": int,
            },
            "failures": {
                "ingest_failures": int,
                "embed_failures": int,
                "failed_task_ids": list[str],
            },
            "retries_used": int,
            "parse_result": FileParseResult,
        }
    """
    total_duration_ms = (time.perf_counter() - file_start_time) * 1000

    # Initialize timing breakdown
    timing_breakdown = {
        "file_record_ms": 0.0,
        "ingest_content_ms": 0.0,
        "embed_content_ms": 0.0,
        "ingest_table_ms": 0.0,
        "embed_table_ms": 0.0,
    }

    # Initialize chunk counts
    chunks = {
        "content_ingested": 0,
        "content_embedded": 0,
        "tables_ingested": 0,
        "tables_embedded": 0,
    }

    # Initialize failure tracking
    failures = {
        "ingest_failures": 0,
        "embed_failures": 0,
        "failed_task_ids": [],
    }

    retries_used = 0

    # Process each task result
    for task_id, result in task_results.items():
        task = graph.tasks.get(task_id)
        if task is None:
            continue

        retries_used += result.retries

        task_type = task.task_type

        # Accumulate timing by task type
        if task_type == TaskType.PARSE or task_type == TaskType.FILE_RECORD:
            timing_breakdown["file_record_ms"] += result.duration_ms
        elif task_type == TaskType.INGEST_CONTENT:
            timing_breakdown["ingest_content_ms"] += result.duration_ms
            if result.success:
                chunks["content_ingested"] += 1
            else:
                failures["ingest_failures"] += 1
                failures["failed_task_ids"].append(task_id)
        elif task_type == TaskType.EMBED_CONTENT:
            timing_breakdown["embed_content_ms"] += result.duration_ms
            if result.success:
                chunks["content_embedded"] += 1
            else:
                failures["embed_failures"] += 1
                failures["failed_task_ids"].append(task_id)
        elif task_type == TaskType.INGEST_TABLE:
            timing_breakdown["ingest_table_ms"] += result.duration_ms
            if result.success:
                chunks["tables_ingested"] += 1
            else:
                failures["ingest_failures"] += 1
                failures["failed_task_ids"].append(task_id)
        elif task_type == TaskType.EMBED_TABLE:
            timing_breakdown["embed_table_ms"] += result.duration_ms
            if result.success:
                chunks["tables_embedded"] += 1
            else:
                failures["embed_failures"] += 1
                failures["failed_task_ids"].append(task_id)

    # Determine overall status
    if failures["ingest_failures"] > 0:
        status = "error"  # Ingest failures are critical
    elif failures["embed_failures"] > 0:
        status = "partial"  # Embed failures are recoverable
    else:
        status = "success"

    return {
        "file_path": file_path,
        "status": status,
        "total_duration_ms": total_duration_ms,
        "timing_breakdown": timing_breakdown,
        "chunks": chunks,
        "failures": failures,
        "retries_used": retries_used,
        "parse_result": parse_result,
    }


def report_file_complete(
    reporter: Optional["ProgressReporter"],
    file_path: str,
    file_start_time: float,
    result: Dict[str, Any],
    verbosity: str,
) -> None:
    """Emit final summary event for a file.

    Parameters
    ----------
    reporter : ProgressReporter | None
        The progress reporter (may be None).
    file_path : str
        The file path.
    file_start_time : float
        Timestamp when file processing started.
    result : dict
        The aggregated file result from build_file_result.
    verbosity : str
        Verbosity level for the event.
    """
    if reporter is None:
        return

    try:
        from .progress import create_progress_event

        total_ms = (time.perf_counter() - file_start_time) * 1000

        status = "completed" if result["status"] == "success" else "failed"

        meta = {
            "total_duration_ms": total_ms,
            **result.get("timing_breakdown", {}),
            **result.get("chunks", {}),
            "embed_failures": result.get("failures", {}).get("embed_failures", 0),
            "ingest_failures": result.get("failures", {}).get("ingest_failures", 0),
            "retries_used": result.get("retries_used", 0),
        }

        event = create_progress_event(
            file_path,
            "file_complete",
            status,
            duration_ms=total_ms,
            elapsed_ms=total_ms,
            meta=meta,
            verbosity=verbosity,  # type: ignore[arg-type]
        )
        reporter.report(event)
    except Exception as e:
        logger.debug(f"File complete report failed: {e}")


# ---------------------------------------------------------------------------
# High-level file processing utilities
# ---------------------------------------------------------------------------


def process_single_file(
    file_manager: Any,
    *,
    parse_result: FileParseResult,
    file_path: str,
    config: Any,
    reporter: Optional["ProgressReporter"] = None,
    enable_progress: bool = False,
    verbosity: str = "low",
) -> FileResultType:
    """
    Process a single parsed document through the ingest+embed pipeline using task graphs.

    This is the main entry point for processing a file after parsing.
    It builds a task graph representing all operations and dependencies,
    then executes the graph via the PipelineExecutor.

    The task graph approach provides:
    - Proper dependency handling between ingest and embed tasks
    - Non-blocking embed in "along" strategy (ingest N+1 proceeds while embed N runs)
    - Configurable retry logic with exponential backoff
    - Comprehensive timing and progress reporting
    - Graceful error handling and cancellation of dependent tasks

    Parameters
    ----------
    file_manager : FileManager
        The file manager instance (provides context and storage methods).
    parse_result : FileParseResult
        The FileParseResult from the file parser (graph + tables + trace).
    file_path : str
        The original file path.
    config : FilePipelineConfig
        Pipeline configuration including ingest, embed, retry settings.
    reporter : ProgressReporter | None
        Optional progress reporter for status updates.
    enable_progress : bool
        Whether to emit progress events.
    verbosity : str
        Verbosity level for progress events: "low", "medium", "high".

    Returns
    -------
    FileResultType
        Result depends on config.output.return_mode:
        - "compact" (default): IngestedFileUnion Pydantic model
        - "full": IngestedFullFile Pydantic model (parse artifacts + lowered rows + refs/metrics)
        - "none": IngestedMinimal Pydantic model (just status stub)
    """
    file_start_time = time.perf_counter()

    # Get return mode from config
    return_mode = getattr(getattr(config, "output", None), "return_mode", "compact")

    try:
        # 1. Build task graph using the factory
        from .task_factory import build_file_task_graph

        graph = build_file_task_graph(
            file_manager,
            file_path=file_path,
            parse_result=parse_result,
            config=config,
            file_start_time=file_start_time,
        )

        # 3. Create executor with appropriate configuration
        exec_config = ExecutionConfig(
            max_embed_workers=getattr(config.execution, "max_embed_workers", 8),
            max_retries=getattr(config.retry, "max_retries", 2),
            retry_delay_seconds=getattr(config.retry, "retry_delay_seconds", 1.0),
            fail_fast=getattr(config.retry, "fail_fast", False),
        )

        executor = PipelineExecutor(
            config=exec_config,
            reporter=reporter if enable_progress else None,
            verbosity=verbosity,
        )

        # 4. Execute the task graph
        task_results = executor.execute_graph(graph)

        # 5. Aggregate results into file-level summary (dict for internal use)
        result_dict = build_file_result(
            file_path,
            graph,
            task_results,
            file_start_time,
            parse_result,
        )

        # 6. Report file completion
        if enable_progress and reporter:
            report_file_complete(
                reporter,
                file_path,
                file_start_time,
                result_dict,
                verbosity,
            )

        # 7. Build result based on return mode - ALL returns are Pydantic models
        if return_mode == "full":
            from unity.file_manager.parse_adapter import (
                adapt_parse_result_for_file_manager,
            )
            from unity.file_manager.types.ingest import IngestedFullFile

            adapted = adapt_parse_result_for_file_manager(parse_result, config=config)

            # Build compact refs/metrics for convenience (same as compact mode)
            from .file_ops import build_compact_ingest_model

            compact = build_compact_ingest_model(
                file_manager,
                file_path=file_path,
                parse_result=parse_result,
                config=config,
            )

            return IngestedFullFile(
                file_path=file_path,
                status=parse_result.status,
                error=parse_result.error,
                file_format=parse_result.file_format,
                mime_type=parse_result.mime_type,
                summary=getattr(parse_result, "summary", "") or "",
                full_text=getattr(parse_result, "full_text", "") or "",
                trace=getattr(parse_result, "trace", None),
                metadata=(
                    parse_result.metadata.model_dump(mode="json", exclude_none=True)
                    if getattr(parse_result, "metadata", None) is not None
                    else None
                ),
                graph=getattr(parse_result, "graph", None),
                tables=list(getattr(parse_result, "tables", []) or []),
                content_rows=list(adapted.content_rows or []),
                content_ref=getattr(compact, "content_ref", None),
                tables_ref=list(getattr(compact, "tables_ref", []) or []),
                metrics=getattr(compact, "metrics", None),
            )
        elif return_mode == "none":
            # Return IngestedMinimal Pydantic model (just status stub)
            total_records = 0
            try:
                from unity.file_manager.parse_adapter import (
                    adapt_parse_result_for_file_manager,
                )

                adapted = adapt_parse_result_for_file_manager(
                    parse_result,
                    config=config,
                )
                total_records = len(list(adapted.content_rows or []))
            except Exception:
                total_records = 0
            return IngestedMinimal(
                file_path=file_path,
                status=parse_result.status,
                error=parse_result.error,
                total_records=total_records,
                file_format=(
                    str(parse_result.file_format) if parse_result.file_format else None
                ),
            )
        else:
            # "compact" (default) - Build typed IngestedFileUnion Pydantic model
            from .file_ops import build_compact_ingest_model

            return build_compact_ingest_model(
                file_manager,
                file_path=file_path,
                parse_result=parse_result,
                config=config,
            )

    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Fatal error processing {file_path}: {e}\n{tb_str}")

        elapsed_ms = (time.perf_counter() - file_start_time) * 1000

        # Report fatal failure
        if enable_progress and reporter:
            from .progress import create_progress_event

            event = create_progress_event(
                file_path,
                "file_complete",
                "failed",
                duration_ms=elapsed_ms,
                elapsed_ms=elapsed_ms,
                error=str(e),
                traceback_str=tb_str,
                verbosity=verbosity,  # type: ignore[arg-type]
            )
            reporter.report(event)

        # Return error based on return mode - ALL returns are Pydantic models
        if return_mode == "full":
            from unity.file_manager.types.ingest import IngestedFullFile

            return IngestedFullFile(
                file_path=file_path,
                status="error",
                error=str(e),
                content_rows=[],
                tables=[],
            )
        elif return_mode == "none":
            return IngestedMinimal(
                file_path=file_path,
                status="error",
                error=str(e),
                total_records=0,
                file_format=None,
            )
        else:
            # Return error model with minimal content_ref
            return BaseIngestedFile(
                file_path=file_path,
                status="error",
                error=str(e),
                content_ref=ContentRef(context="", record_count=0, text_chars=0),
                metrics=FileMetrics(processing_time=elapsed_ms / 1000.0),
            )


def run_pipeline(
    file_manager: Any,
    *,
    parse_results: List[FileParseResult],
    file_paths: List[str],
    config: Any,
    reporter: Optional["ProgressReporter"] = None,
    enable_progress: bool = False,
    verbosity: str = "low",
) -> IngestPipelineResult:
    """
    Run the complete file processing pipeline for multiple files.

    Each file is processed through a task graph that handles dependencies,
    retries, and progress reporting. Files can be processed in parallel
    or sequentially based on config.

    Parameters
    ----------
    file_manager : FileManager
        The file manager instance.
    parse_results : list[FileParseResult]
        Parsed file results (FileParseResult).
    file_paths : list[str]
        Corresponding file paths.
    config : FilePipelineConfig
        Pipeline configuration.
    reporter : ProgressReporter | None
        Optional progress reporter.
    enable_progress : bool
        Whether to emit progress events.
    verbosity : str
        Verbosity level for progress events: "low", "medium", "high".

    Returns
    -------
    IngestPipelineResult
        Structured result containing per-file ingest results (all Pydantic models)
        and global statistics. Individual file results depend on config.output.return_mode.
    """
    pipeline_start_time = time.perf_counter()

    if not parse_results or not file_paths:
        return IngestPipelineResult()

    # Get return mode for error handling
    return_mode = getattr(getattr(config, "output", None), "return_mode", "compact")

    results: Dict[str, FileResultType] = {}

    # Get execution config
    parallel = getattr(getattr(config, "execution", None), "parallel_files", False)
    max_workers = getattr(getattr(config, "execution", None), "max_file_workers", 4)

    if enable_progress:
        logger.info(
            f"Processing {len(parse_results)} files "
            f"({'parallel' if parallel else 'sequential'}, max_workers={max_workers})",
        )

    if not parallel or len(parse_results) == 1:
        # Sequential processing
        for idx, (parse_result, path) in enumerate(zip(parse_results, file_paths)):
            if enable_progress:
                logger.info(
                    f"Processing file {idx + 1}/{len(parse_results)}: {path}",
                )
            results[path] = process_single_file(
                file_manager,
                parse_result=parse_result,
                file_path=path,
                config=config,
                reporter=reporter,
                enable_progress=enable_progress,
                verbosity=verbosity,
            )
    else:
        # Parallel processing
        with ThreadPoolExecutor(
            max_workers=min(len(parse_results), max_workers),
        ) as executor:
            futures = {
                executor.submit(
                    process_single_file,
                    file_manager,
                    parse_result=parse_result,
                    file_path=path,
                    config=config,
                    reporter=reporter,
                    enable_progress=enable_progress,
                    verbosity=verbosity,
                ): path
                for parse_result, path in zip(parse_results, file_paths)
            }

            for future in futures:
                path = futures[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    tb_str = traceback.format_exc()
                    logger.error(f"Error processing {path}: {e}\n{tb_str}")
                    # Return error based on return mode - ALL are Pydantic models
                    if return_mode == "full":
                        from unity.file_manager.types.ingest import IngestedFullFile

                        results[path] = IngestedFullFile(
                            file_path=path,
                            status="error",
                            error=str(e),
                            content_rows=[],
                            tables=[],
                        )
                    elif return_mode == "none":
                        results[path] = IngestedMinimal(
                            file_path=path,
                            status="error",
                            error=str(e),
                            total_records=0,
                            file_format=None,
                        )
                    else:
                        results[path] = BaseIngestedFile(
                            file_path=path,
                            status="error",
                            error=str(e),
                            content_ref=ContentRef(
                                context="",
                                record_count=0,
                                text_chars=0,
                            ),
                            metrics=FileMetrics(),
                        )

    if enable_progress and reporter:
        reporter.flush()

    # Calculate total duration
    total_duration_ms = (time.perf_counter() - pipeline_start_time) * 1000

    return IngestPipelineResult.from_results(
        results,
        total_duration_ms=total_duration_ms,
    )
