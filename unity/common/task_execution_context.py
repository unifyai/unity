"""
Pure plumbing for **run-scoped task execution delegation**.

This module introduces a small abstraction that allows code that *starts* task
execution (e.g., `TaskScheduler.execute`) to delegate that execution to the
current run's execution environment without explicitly threading parameters
through every call site.

Why this exists
---------------
A single "topmost" execution environment (often an Actor-like
orchestrator) may need to:

- trigger durable task execution via TaskScheduler, and
- ensure that the task is executed *through the same environment* (rather than
  spawning a fresh one).

This is a cyclic routing problem:

    execution environment → TaskScheduler → (delegate back to) execution environment

`ContextVar` is used throughout the repo to propagate async context safely.
Here, it enables run-scoped delegation:

- **Run-scoped**: a delegate is set at the top of an async execution context and
  reset in a `finally` block.
- **Async-safe**: `ContextVar` propagation ensures each async task tree sees the
  correct delegate under concurrency.
- **No leakage**: callers must reset to prevent delegates persisting across runs.

Correct usage pattern
---------------------
Set the delegate at the start of an async run, and always reset it:

```python
token = current_task_execution_delegate.set(delegate)
try:
    # ... run logic that may call into TaskScheduler ...
finally:
    current_task_execution_delegate.reset(token)
```

Anti-patterns
-------------
- Setting the delegate in `__init__` (async context is not established yet).
- Forgetting to reset in a `finally` block (causes cross-run leakage).
- Relying on the delegate outside an async execution context (undefined).
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any, Optional, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    # Typing-only import to avoid creating import-time cycles.
    from unity.common.async_tool_loop import SteerableToolHandle


@runtime_checkable
class TaskExecutionDelegate(Protocol):
    """Protocol for delegating task execution to a run-scoped execution environment.

    Contract
    --------
    - Implementations must be run-scoped: set at the start of an execution context
      and reset in a `finally` block to prevent leakage.
    - Safe under async concurrency: `ContextVar` propagation ensures each async
      task tree gets its own delegate instance.
    - Must not leak across runs: after a run completes, the delegate must be `None`
      in fresh async contexts.

    Usage
    -----
    This protocol is used by task execution routing to run tasks through the
    same execution environment that initiated the task, rather than spawning a
    new one.
    """

    async def start_task_run(
        self,
        *,
        task_description: str,
        entrypoint: int | None,
        parent_chat_context: list[dict] | None,
        clarification_up_q: Optional[asyncio.Queue[str]],
        clarification_down_q: Optional[asyncio.Queue[str]],
        images: Any | None = None,
        **kwargs: Any,
    ) -> "SteerableToolHandle":
        """Start a task execution and return a steerable handle.

        Parameters
        ----------
        task_description : str
            Natural language description of the task to execute.
        entrypoint : int | None
            Optional function ID to execute directly, bypassing plan generation.
        parent_chat_context : list[dict] | None
            Chat context from the parent execution environment.
        clarification_up_q : asyncio.Queue[str] | None
            Queue for sending clarification questions upward.
        clarification_down_q : asyncio.Queue[str] | None
            Queue for receiving clarification answers downward.
        images : Any | None
            Optional image references for the task execution.
        **kwargs : Any
            Additional implementation-specific parameters.

        Returns
        -------
        SteerableToolHandle
            A live steerable handle for controlling the task execution.
        """


# Holds the current run-scoped delegate for task execution routing.
#
# - Set this at the top of an async execution context (e.g., a run method),
#   and reset the token in a `finally` block to prevent leakage across runs.
# - When `None`, callers should fall back to default behavior (typically direct
#   actor/environment instantiation).
current_task_execution_delegate: ContextVar[TaskExecutionDelegate | None] = ContextVar(
    "current_task_execution_delegate",
    default=None,
)
