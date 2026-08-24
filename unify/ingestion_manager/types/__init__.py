"""Type definitions for IngestionManager."""

from unify.ingestion_manager.types.request import (
    CollectionTarget,
    EmbedSpec,
    FilesSource,
    FolderSource,
    IngestionRequest,
    IngestionSource,
    IngestionTarget,
    RowsSource,
    TableSource,
    TableTarget,
)
from unify.ingestion_manager.types.run import (
    TERMINAL_STATES,
    AttemptState,
    IngestionEventRow,
    IngestionRun,
    IngestionRunRecord,
    IngestionSummary,
    LogEntry,
    RetryResult,
    RetryScope,
    RunState,
    FileProgress,
    TableReconciliation,
    RunStatus,
    StageProgress,
)

__all__ = [
    "AttemptState",
    # Request vocabulary
    "RowsSource",
    "FilesSource",
    "FolderSource",
    "TableSource",
    "IngestionSource",
    "TableTarget",
    "CollectionTarget",
    "IngestionTarget",
    "EmbedSpec",
    "IngestionRequest",
    # Run vocabulary
    "RunState",
    "TERMINAL_STATES",
    "RetryScope",
    "FileProgress",
    "TableReconciliation",
    "StageProgress",
    "LogEntry",
    "IngestionEventRow",
    "IngestionRunRecord",
    "IngestionRun",
    "RunStatus",
    "RetryResult",
    "IngestionSummary",
]
