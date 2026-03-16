"""FileManager utility modules.

This package contains internal utilities for the FileManager pipeline:
- executor: Thin orchestration layer (process_single_file, run_pipeline)
- task_functions: Pure task execution functions
- ingest_ops: Bridge between FM file concepts and DM.ingest()
- progress: Progress reporting protocol and implementations
- ops: Core file record and storage operations
- search: Search and join utilities
- storage: Context storage helpers
- file_ops: File operation helpers
"""

from __future__ import annotations

# Orchestration
from .executor import (
    build_file_result,
    report_file_complete,
    process_single_file,
    run_pipeline,
)

# Task functions (pure execution functions)
from .task_functions import (
    execute_create_file_record,
    execute_ingest_content,
    execute_ingest_table,
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

__all__ = [
    # Orchestration
    "build_file_result",
    "report_file_complete",
    "process_single_file",
    "run_pipeline",
    # Task Functions
    "execute_create_file_record",
    "execute_ingest_content",
    "execute_ingest_table",
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
