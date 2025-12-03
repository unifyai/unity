"""FileManager utility modules.

This package contains internal utilities for the FileManager pipeline:
- executor: Task-based pipeline execution engine
- task_factory: Task graph construction for files
- task_functions: Pure task execution functions
- progress: Progress reporting protocol and implementations
- ops: Core batch operations (legacy, being refactored)
- search: Search and join utilities
- storage: Context storage helpers
- ingest_ops: Pure batch ingest functions
- embed_ops: Pure batch embed functions
- file_ops: File operation helpers
"""

from __future__ import annotations

# Executor classes
from .executor import (
    Task,
    TaskGraph,
    TaskType,
    TaskStatus,
    TaskResult,
    PipelineExecutor,
    ExecutionConfig,
    create_task_id,
    build_file_result,
    report_file_complete,
    process_single_file,
    run_pipeline,
)

# Task factory
from .task_factory import (
    build_file_task_graph,
    create_task_id as create_task_id_v2,  # New deterministic ID generator
)

# Task functions (pure execution functions)
from .task_functions import (
    execute_create_file_record,
    execute_ingest_content_chunk,
    execute_embed_content_chunk,
    execute_ingest_table_chunk,
    execute_embed_table_chunk,
)

# Progress reporting
from .progress import (
    ProgressEvent,
    ProgressReporter,
    NoOpReporter,
    ConsoleReporter,
    JsonFileReporter,
    CallbackReporter,
    CompositeReporter,
    create_reporter,
    create_progress_event,
)

# Re-export commonly used items
__all__ = [
    # Executor
    "Task",
    "TaskGraph",
    "TaskType",
    "TaskStatus",
    "TaskResult",
    "PipelineExecutor",
    "ExecutionConfig",
    "create_task_id",
    "build_file_result",
    "report_file_complete",
    "process_single_file",
    "run_pipeline",
    # Task Factory
    "build_file_task_graph",
    "create_task_id_v2",
    # Task Functions
    "execute_create_file_record",
    "execute_ingest_content_chunk",
    "execute_embed_content_chunk",
    "execute_ingest_table_chunk",
    "execute_embed_table_chunk",
    # Progress
    "ProgressEvent",
    "ProgressReporter",
    "NoOpReporter",
    "ConsoleReporter",
    "JsonFileReporter",
    "CallbackReporter",
    "CompositeReporter",
    "create_reporter",
    "create_progress_event",
]
