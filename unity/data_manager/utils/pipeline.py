"""Generic DAG-based pipeline execution engine for DataManager.

This module provides a dependency-aware task execution system used internally
by DataManager for operations that benefit from chunked parallel execution
(e.g. bulk ingestion with optional embedding).

Key components
--------------
TaskStatus : Enum
    Lifecycle states a task can be in.
TaskResult : dataclass
    Outcome of executing a single task, including timing and retry count.
Task : dataclass
    A unit of work with a callable, dependencies, and generic metadata.
TaskGraph : class
    A DAG of tasks with topological ordering and dependency tracking.
PipelineExecutor : class
    Executes a TaskGraph with configurable parallelism, retries, and
    fail-fast behaviour.
ExecutionConfig : dataclass
    Knobs for the executor (workers, retries, backoff, fail-fast).

Design notes
~~~~~~~~~~~~
* ``task_type`` is a plain ``str`` so callers can define their own
  vocabularies without coupling to a shared enum.
* Tasks carry a ``metadata: dict`` bag instead of domain-specific fields,
  keeping the engine reusable.
* No imports from FileManager or any other manager -- this module is
  self-contained.
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task lifecycle
# ---------------------------------------------------------------------------


class TaskStatus(Enum):
    """Lifecycle state of a pipeline task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Result wrapper
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Outcome of a single task execution.

    Attributes
    ----------
    success : bool
        Whether the task completed without error.
    value : Any | None
        Return value of the task callable on success.
    error : str | None
        Error message on failure.
    duration_ms : float
        Wall-clock time for the (final) execution attempt.
    retries : int
        Number of retry attempts that were made before the final outcome.
    """

    success: bool
    value: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    retries: int = 0


# ---------------------------------------------------------------------------
# Task node
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A single unit of work in a pipeline graph.

    Attributes
    ----------
    id : str
        Unique identifier within the graph.
    task_type : str
        Caller-defined label (e.g. ``"create_table"``, ``"insert_chunk"``).
    func : Callable[..., Any]
        The callable to invoke.  Receives ``*args, **kwargs``.
    args : tuple
        Positional arguments forwarded to *func*.
    kwargs : dict
        Keyword arguments forwarded to *func*.
    dependencies : set[str]
        IDs of tasks that must complete successfully before this one starts.
    status : TaskStatus
        Current lifecycle state.
    result : TaskResult | None
        Populated after execution.
    metadata : dict
        Arbitrary caller-supplied data (e.g. chunk index, context path).
    """

    id: str
    task_type: str
    func: Callable[..., Any]
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[TaskResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Task):
            return False
        return self.id == other.id


# ---------------------------------------------------------------------------
# Directed Acyclic Graph of tasks
# ---------------------------------------------------------------------------


class TaskGraph:
    """A DAG of :class:`Task` nodes with dependency-based scheduling.

    The graph tracks which tasks are ready to run (all dependencies met)
    and which have completed, failed, or been cancelled.

    Parameters
    ----------
    name : str, optional
        Human-readable label for the graph (used in log messages).
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.tasks: Dict[str, Task] = {}
        self._completed: Set[str] = set()

    # -- mutators -----------------------------------------------------------

    def add_task(self, task: Task) -> str:
        """Register a task in the graph.

        Returns the task ID for convenience when wiring dependencies.
        """
        self.tasks[task.id] = task
        return task.id

    def add_dependency(self, task_id: str, depends_on: str) -> None:
        """Declare that *task_id* must wait for *depends_on* to complete."""
        if task_id in self.tasks:
            self.tasks[task_id].dependencies.add(depends_on)

    # -- queries ------------------------------------------------------------

    def get_ready_tasks(self) -> List[Task]:
        """Return tasks whose dependencies are all satisfied and status is PENDING."""
        return [
            t
            for t in self.tasks.values()
            if t.status == TaskStatus.PENDING
            and t.dependencies.issubset(self._completed)
        ]

    def get_failed_tasks(self) -> List[Task]:
        """Return all tasks that ended with FAILED status."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]

    def has_failures(self) -> bool:
        """Return ``True`` if any task in the graph has FAILED."""
        return any(t.status == TaskStatus.FAILED for t in self.tasks.values())

    def is_complete(self) -> bool:
        """Return ``True`` when no tasks are PENDING or RUNNING."""
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
        return all(t.status in terminal for t in self.tasks.values())

    def get_summary(self) -> Dict[str, Any]:
        """Return a dict of status counts and an overall ``success`` flag."""
        counts: Dict[str, int] = {s.value: 0 for s in TaskStatus}
        for t in self.tasks.values():
            counts[t.status.value] += 1
        return {
            "total": len(self.tasks),
            "statuses": counts,
            "success": not self.has_failures() and counts["pending"] == 0,
        }

    # -- lifecycle ----------------------------------------------------------

    def mark_completed(self, task_id: str, result: TaskResult) -> None:
        """Record *result* for *task_id* and advance the graph state."""
        task = self.tasks.get(task_id)
        if task is None:
            return
        task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
        task.result = result
        if result.success:
            self._completed.add(task_id)

    def mark_cancelled(self, task_id: str) -> None:
        """Mark *task_id* as CANCELLED (dependency failure or stop request)."""
        task = self.tasks.get(task_id)
        if task is not None:
            task.status = TaskStatus.CANCELLED


# ---------------------------------------------------------------------------
# Execution configuration
# ---------------------------------------------------------------------------


@dataclass
class ExecutionConfig:
    """Knobs for :class:`PipelineExecutor`.

    Attributes
    ----------
    max_workers : int
        Maximum threads used for concurrent task execution.
    max_retries : int
        How many times a failed task is retried before giving up.
    retry_delay_seconds : float
        Base delay between retries; actual delay is
        ``retry_delay_seconds * 2 ** (attempt - 1)`` (exponential backoff).
    fail_fast : bool
        If ``True``, the executor stops scheduling new tasks as soon as any
        task fails (already-running tasks are allowed to finish).
    """

    max_workers: int = 4
    max_retries: int = 3
    retry_delay_seconds: float = 3.0
    fail_fast: bool = False


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class PipelineExecutor:
    """Execute a :class:`TaskGraph` with parallelism, retries, and fail-fast.

    Parameters
    ----------
    config : ExecutionConfig | None
        Execution settings.  Defaults to ``ExecutionConfig()``.
    on_task_complete : callable | None
        Optional ``(Task, TaskResult) -> None`` callback fired after each
        task finishes (success or failure).  Useful for progress reporting.
    """

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        on_task_complete: Optional[Callable[[Task, TaskResult], None]] = None,
    ) -> None:
        self.config = config or ExecutionConfig()
        self._on_task_complete = on_task_complete
        self._stop_requested = False

    # -- public API ---------------------------------------------------------

    def execute(self, graph: TaskGraph) -> Dict[str, TaskResult]:
        """Run all tasks in *graph*, respecting dependencies and parallelism.

        Scheduling policy
        ~~~~~~~~~~~~~~~~~
        Only *max_workers* tasks are in-flight at any time.  When a slot
        opens, ready tasks are sorted so that **downstream / dependent**
        tasks (e.g. ``embed_chunk``) run before new root-level tasks
        (e.g. the next ``insert_chunk``).  This prevents the "along"
        embed strategy from silently degenerating to "after" when the
        thread pool queue is flooded with insert tasks.

        Parameters
        ----------
        graph : TaskGraph
            The graph to execute.

        Returns
        -------
        dict[str, TaskResult]
            Mapping of task ID -> result for every task in the graph.
        """
        results: Dict[str, TaskResult] = {}
        self._stop_requested = False

        max_workers = max(1, self.config.max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            in_flight: dict = {}  # future -> Task

            def _submit_ready() -> None:
                """Fill worker slots with ready tasks, downstream-first."""
                available = max_workers - len(in_flight)
                if available <= 0:
                    return
                ready = graph.get_ready_tasks()
                if not ready:
                    return
                in_flight_ids = {t.id for t in in_flight.values()}
                candidates = [t for t in ready if t.id not in in_flight_ids]
                # Downstream tasks (those with dependencies) before root-level
                candidates.sort(key=lambda t: (len(t.dependencies) == 0, t.id))
                for task in candidates[:available]:
                    fut = pool.submit(self._run_task, task)
                    in_flight[fut] = task

            while not graph.is_complete() and not self._stop_requested:
                _submit_ready()

                if not in_flight:
                    self._cancel_blocked_tasks(graph, results)
                    if not graph.is_complete():
                        self._cancel_all_pending(
                            graph,
                            results,
                            reason="Deadlock detected",
                        )
                    break

                # Wait for at least one to finish, then re-fill slots.
                for future in as_completed(in_flight):
                    task = in_flight.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = TaskResult(success=False, error=str(exc))
                    results[task.id] = result
                    graph.mark_completed(task.id, result)
                    self._notify(task, result)

                    if self.config.fail_fast and graph.has_failures():
                        logger.info(
                            "Fail-fast triggered -- stopping execution of %s",
                            graph.name,
                        )
                        self._stop_requested = True
                        break

                    # Eagerly fill freed slots with newly-ready tasks.
                    _submit_ready()
                    break  # Re-enter the outer loop to pick up state changes

        # Cancel any remaining pending tasks (stop requested or dependency failures).
        if not graph.is_complete():
            self._cancel_blocked_tasks(graph, results)
            self._cancel_all_pending(graph, results, reason="Execution stopped")

        return results

    def stop(self) -> None:
        """Request a graceful shutdown (no new tasks will be started)."""
        self._stop_requested = True

    # -- internal helpers ---------------------------------------------------

    def _run_task(self, task: Task) -> TaskResult:
        """Execute *task* with retry logic and exponential backoff."""
        task.status = TaskStatus.RUNNING
        retries = 0
        last_error: Optional[str] = None

        while retries <= self.config.max_retries:
            start = time.perf_counter()
            try:
                value = task.func(*task.args, **task.kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                return TaskResult(
                    success=True,
                    value=value,
                    duration_ms=duration_ms,
                    retries=retries,
                )
            except Exception as exc:
                last_error = str(exc)
                retries += 1
                if retries <= self.config.max_retries:
                    delay = self.config.retry_delay_seconds * (2 ** (retries - 1))
                    logger.debug(
                        "Task %s attempt %d failed (%s), retrying in %.1fs",
                        task.id,
                        retries,
                        last_error,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Task %s failed after %d attempts: %s\n%s",
                        task.id,
                        retries,
                        last_error,
                        traceback.format_exc(),
                    )

        return TaskResult(
            success=False,
            error=last_error,
            retries=retries - 1,
        )

    def _cancel_blocked_tasks(
        self,
        graph: TaskGraph,
        results: Dict[str, TaskResult],
    ) -> None:
        """Cancel PENDING tasks whose dependencies include a FAILED/CANCELLED task."""
        non_runnable = {
            t.id
            for t in graph.tasks.values()
            if t.status in (TaskStatus.FAILED, TaskStatus.CANCELLED)
        }
        if not non_runnable:
            return
        changed = True
        while changed:
            changed = False
            for task in list(graph.tasks.values()):
                if task.status != TaskStatus.PENDING:
                    continue
                if task.dependencies & non_runnable:
                    graph.mark_cancelled(task.id)
                    results[task.id] = TaskResult(
                        success=False,
                        error="Dependency failed or cancelled",
                    )
                    non_runnable.add(task.id)
                    changed = True

    def _cancel_all_pending(
        self,
        graph: TaskGraph,
        results: Dict[str, TaskResult],
        reason: str = "Cancelled",
    ) -> None:
        """Cancel every remaining PENDING task (used for deadlock recovery)."""
        for task in graph.tasks.values():
            if task.status == TaskStatus.PENDING:
                graph.mark_cancelled(task.id)
                results[task.id] = TaskResult(success=False, error=reason)
                logger.error(
                    "Cancelled task %s in graph %s: %s",
                    task.id,
                    graph.name,
                    reason,
                )

    def _notify(self, task: Task, result: TaskResult) -> None:
        """Fire the optional completion callback."""
        if self._on_task_complete is not None:
            try:
                self._on_task_complete(task, result)
            except Exception:
                logger.debug(
                    "on_task_complete callback failed for %s",
                    task.id,
                    exc_info=True,
                )
