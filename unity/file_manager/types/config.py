from __future__ import annotations

"""Config models for the FileManager parsing → ingestion → embedding pipeline.

This module intentionally keeps the number of models small and grouped by concern
to remain approachable while still being extensible. The design goals are:

- Single, obvious entry point (`FilePipelineConfig`) that callers provide.
- Clear separation of concerns: parser knobs, ingest/storage layout, embeddings,
  and execution/retry behavior.
- Extensibility: future features (e.g., images ingestion, custom schemas,
  alternate layouts) can be added with minimal churn.

Tip for future contributors:
- Prefer adding options to these grouped models rather than creating many small
  models. Keep defaults sensible so the default behavior matches the current
  FileManager pipeline without configuration.
"""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, model_validator
import json
from pathlib import Path

from unity.file_manager.file_parsers.registry import (
    DEFAULT_BACKEND_CLASS_PATHS_BY_FORMAT,
)

# ------------------------------ Parser ------------------------------------ #


class ParseConfig(BaseModel):
    """Options forwarded to the underlying parser.

    Hot-swapping
    ------------
    Backends can be overridden per file format by providing dotted class paths.
    This enables swapping an entire implementation set OR a single format backend
    (e.g., swap XLSX only) without changing FileParser code.
    """

    # Controls parse-stage parallelism (number of files processed concurrently).
    # The FileParser enforces a conservative upper bound even if the config asks for more.
    max_concurrent_parses: int = 3

    # format -> dotted class path (e.g. "xlsx" -> "pkg.module.CustomExcelBackend")
    backend_class_paths_by_format: Dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_BACKEND_CLASS_PATHS_BY_FORMAT),
    )


# --------------------------- Ingest / Storage ----------------------------- #


class IngestConfig(BaseModel):
    """Controls how parsed rows are stored in Unify contexts.

    Storage Layout
    --------------
    Each file's content is stored under a context path determined by `storage_id`:
    - `Files/<alias>/<storage_id>/Content` for document content
    - `Files/<alias>/<storage_id>/Tables/<table>` for extracted tables

    When `storage_id` is None (default), each file gets its own context using
    `str(file_id)` as the storage_id (auto-assigned after record creation).

    When `storage_id` is set to a custom label, all files in the ingestion run
    share that context, enabling unified storage for related files.
    """

    # Storage identifier. When None, auto-assigned as str(file_id) per file.
    storage_id: Optional[str] = None

    # Chunking controls for ingestion
    # - content_rows_batch_size controls batching for per-file Content rows
    #   (used by along/auto strategies to incrementally ingest+embed).
    content_rows_batch_size: int = 1000

    # Row management
    replace_existing: bool = True

    # Hierarchical identifiers are represented via a single dict field `content_id`
    # on each `/Content/` row, e.g.:
    #   {"document": 0, "section": 2, "paragraph": 1, "sentence": 3}
    #
    # Legacy column-based ID layouts and auto-counting configurations have been
    # removed to keep the schema simple and robust.

    # Table ingestion control
    table_ingest: bool = True
    table_label_strategy: Literal["sheet_name", "section_path", "index"] = "sheet_name"
    table_rows_batch_size: int = 100

    # Type inference control
    # When True, adds infer_untyped_fields=True to each row during ingestion,
    # instructing the backend to infer types for fields that don't have explicit
    # type definitions. This is useful for spreadsheet data where column types
    # may vary (e.g., dates, times, numbers stored as strings).
    infer_untyped_fields: bool = False

    # Business context specifications for enriching table contexts with descriptions
    # Uses BusinessContextsConfig with global_rules, file_contexts (with file_rules), and table_contexts (with table_rules)
    business_contexts: Optional["BusinessContextsConfig"] = None


# --------------------------- Business Context ----------------------------- #


class TableBusinessContextSpec(BaseModel):
    """Table-level business context specification for enriching a single table context with descriptions.

    - table: exact table label (required for matching)
    - table_description: optional description for the table context itself
    - column_descriptions: mapping of column name → description
    - table_rules: rules about interpreting multiple columns within this table
    """

    table: str
    table_rules: List[str] = Field(default_factory=list)
    table_description: Optional[str] = None
    column_descriptions: Dict[str, str] = Field(default_factory=dict)


class FileBusinessContextSpec(BaseModel):
    """File-level business context specification for enriching table contexts with descriptions.

    - file_path: exact file path (required for matching)
    - file_rules: rules about interpreting data across multiple tables in this file
    - table_contexts: list of table specs for this file (at least one required)
    """

    file_path: str
    file_rules: List[str] = Field(default_factory=list)
    table_contexts: List[TableBusinessContextSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_table_contexts_not_empty(self) -> "FileBusinessContextSpec":
        """Ensure table_contexts list is not empty."""
        if not self.table_contexts:
            raise ValueError(
                "FileBusinessContextSpec must have at least one table specification",
            )
        return self


class BusinessContextsConfig(BaseModel):
    """Top-level business contexts configuration with hierarchical rules support.

    - global_rules: rules about interpreting data across multiple files
    - file_contexts: list of file-level business context specs
    """

    global_rules: List[str] = Field(default_factory=list)
    file_contexts: List[FileBusinessContextSpec] = Field(default_factory=list)


# ------------------------------ Embeddings -------------------------------- #


class TableEmbeddingSpec(BaseModel):
    """Table-level embedding specification for a single table.

    - table: exact table label (required for matching)
    - source_columns and target_columns are parallel lists; one spec can embed multiple columns.
    """

    table: str
    source_columns: List[str]
    target_columns: List[str]

    @model_validator(mode="after")
    def validate_column_lists_match(self) -> "TableEmbeddingSpec":
        """Ensure source_columns and target_columns have the same length."""
        if len(self.source_columns) != len(self.target_columns):
            raise ValueError(
                f"source_columns ({len(self.source_columns)}) and target_columns ({len(self.target_columns)}) must have the same length",
            )
        return self


class FileEmbeddingSpec(BaseModel):
    """File-level embedding specification for enriching tables/contexts with embeddings.

    - file_path: exact file path (required for matching). Use "*" to match all files.
    - context: selects where to embed ("per_file", "per_file_table", "unified")
    - tables: list of table specs for this file (at least one required for per_file_table context)
    """

    file_path: str
    context: Literal["per_file", "per_file_table", "unified"]
    tables: List[TableEmbeddingSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tables_not_empty(self) -> "FileEmbeddingSpec":
        """Ensure tables list is not empty."""
        if not self.tables:
            raise ValueError(
                "FileEmbeddingSpec must have at least one table specification",
            )
        return self


class EmbeddingsConfig(BaseModel):
    """Embedding behavior for content and per-file tables.

    strategy:
        - "off": disable embeddings
        - "after": run embeddings after ingest (classic flow)
        - "along": ingest and embed in chunks within a single file loop (non-blocking)
    file_specs:
        File-level embedding specifications describing where and what to embed, organized by file_path.
        Each FileEmbeddingSpec can target multiple tables within a file.
    """

    strategy: Literal["off", "after", "along"] = "after"
    file_specs: List[FileEmbeddingSpec] = Field(default_factory=list)


# ------------------------------ Output modes ------------------------------- #


class OutputConfig(BaseModel):
    """Controls what parse returns to the LLM.

    Modes:
    - "compact" (default): return typed, reference-first model (no heavy fields)
    - "full": return the full parse dict from Document.to_parse_result (verbatim)
    - "none": return only a minimal status stub
    """

    return_mode: Literal["compact", "full", "none"] = "compact"


class DiagnosticsConfig(BaseModel):
    """Controls optional pipeline diagnostics output.

    - enable_progress: when True, emit progress events for ingest/embed steps.
    - progress_mode: selects the progress reporter type:
        - "json_file": append JSON-lines to progress_file (auto-generated if not provided)
        - "callback": invoke a user-provided callback
        - "off": disable progress reporting
    - progress_file: path for JSON-lines output when progress_mode is "json_file".
      If not provided, auto-generates: ./pipeline_progress_{timestamp}.jsonl
    - verbosity: controls the detail level of progress events:
        - "low" (default): minimal events (file_path, phase, status, timestamp)
        - "medium": detailed (include chunk numbers, row counts, table labels, durations)
        - "high": verbose (all metadata plus intermediate step details)
    """

    enable_progress: bool = False
    progress_mode: Literal["json_file", "callback", "off"] = "json_file"
    progress_file: Optional[str] = None
    verbosity: Literal["low", "medium", "high"] = "low"


class ExecutionConfig(BaseModel):
    """Controls pipeline execution behavior.

    - parallel_files: when True, process multiple files concurrently.
      Defaults to False for safe sequential processing.
    - max_file_workers: maximum concurrent file processing tasks when parallel.
    - max_embed_workers: maximum concurrent embedding tasks per file.
    """

    parallel_files: bool = False
    max_file_workers: int = 4
    max_embed_workers: int = 8


class RetryConfig(BaseModel):
    """Controls retry behavior for failed operations.

    - max_retries: maximum retry attempts for failed tasks (0 = no retries).
    - retry_delay_seconds: base delay between retries (with exponential backoff).
    - fail_fast: when True, stop pipeline on first failure without processing
      remaining files/tasks.
    """

    max_retries: int = 3
    retry_delay_seconds: float = 3.0
    fail_fast: bool = False


# ------------------------------ Entry point ------------------------------- #


class FilePipelineConfig(BaseModel):
    """Top-level configuration for the FileManager pipeline.

    Keep this as the single entry point. Extend the grouped sub-configs as
    needed instead of growing many small models. Defaults preserve current
    behavior (per-file layout, sequential processing, modest parser batch size).

    Can be instantiated with defaults or loaded from a JSON file using `from_file()`.
    Supports partial configs when loading from file - only define what you need,
    defaults fill the rest.
    """

    parse: ParseConfig = ParseConfig()
    ingest: IngestConfig = IngestConfig()
    embed: EmbeddingsConfig = EmbeddingsConfig()
    output: OutputConfig = OutputConfig()
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()
    execution: ExecutionConfig = ExecutionConfig()
    retry: RetryConfig = RetryConfig()

    @classmethod
    def from_file(cls, path: str) -> "FilePipelineConfig":
        """Load JSON config file and convert to FilePipelineConfig.

        Supports partial configs - only define what you need, defaults fill the rest.
        All fields in the JSON are optional to allow minimal config files.

        Parameters
        ----------
        path : str
            Path to JSON config file

        Returns
        -------
        FilePipelineConfig
            Validated and populated FilePipelineConfig instance

        Raises
        ------
        FileNotFoundError
            If config file doesn't exist
        ValueError
            If JSON is invalid or doesn't match schema
        """
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}") from e

        # Helper class for validating JSON file structure (all fields optional)
        class _FilePipelineConfigFile(BaseModel):
            """Temporary model for validating JSON file structure."""

            parse: Optional[Dict[str, Any]] = None
            ingest: Optional[Dict[str, Any]] = None
            embed: Optional[Dict[str, Any]] = None
            output: Optional[Dict[str, Any]] = None
            diagnostics: Optional[Dict[str, Any]] = None
            execution: Optional[Dict[str, Any]] = None
            retry: Optional[Dict[str, Any]] = None

        # Validate JSON structure
        config_file = _FilePipelineConfigFile.model_validate(data)

        # Build FilePipelineConfig with defaults, then override with provided values
        cfg = cls()

        # Parse config
        if config_file.parse:
            if "max_concurrent_parses" in config_file.parse:
                cfg.parse.max_concurrent_parses = int(
                    config_file.parse["max_concurrent_parses"],
                )
            elif "batch_size" in config_file.parse:
                # Back-compat alias: batch_size historically controlled parse concurrency.
                cfg.parse.max_concurrent_parses = int(config_file.parse["batch_size"])
            if "backend_class_paths_by_format" in config_file.parse:
                m = config_file.parse["backend_class_paths_by_format"]
                if isinstance(m, dict):
                    cfg.parse.backend_class_paths_by_format.update(
                        {str(k): str(v) for k, v in m.items() if v},
                    )

        # Ingest config
        if config_file.ingest:
            ingest_data = config_file.ingest
            for key, value in ingest_data.items():
                if key == "business_contexts":
                    # New structure: business_contexts is a dict with global_rules and file_contexts
                    if isinstance(value, dict):
                        # Parse new structure: {global_rules: [...], file_contexts: [...]}
                        global_rules = value.get("global_rules", [])
                        file_contexts_data = value.get("file_contexts", [])
                        file_contexts = []
                        for fc_dict in file_contexts_data:
                            # Extract table_contexts and convert each to TableBusinessContextSpec
                            table_contexts_data = fc_dict.get("table_contexts", [])
                            table_specs = [
                                TableBusinessContextSpec(**tc_dict)
                                for tc_dict in table_contexts_data
                            ]
                            # Create FileBusinessContextSpec with file_path, file_rules, and table_contexts
                            file_contexts.append(
                                FileBusinessContextSpec(
                                    file_path=fc_dict["file_path"],
                                    file_rules=fc_dict.get("file_rules", []),
                                    table_contexts=table_specs,
                                ),
                            )
                        cfg.ingest.business_contexts = BusinessContextsConfig(
                            global_rules=global_rules,
                            file_contexts=file_contexts,
                        )
                    else:
                        # Legacy support: business_contexts was a list (deprecated)
                        # Convert to new structure with empty global_rules
                        file_contexts = []
                        for bc_dict in value:
                            # Support both old "tables" and new "table_contexts" keys
                            table_contexts_data = bc_dict.get(
                                "table_contexts",
                                bc_dict.get("tables", []),
                            )
                            table_specs = [
                                TableBusinessContextSpec(**tc_dict)
                                for tc_dict in table_contexts_data
                            ]
                            file_contexts.append(
                                FileBusinessContextSpec(
                                    file_path=bc_dict["file_path"],
                                    file_rules=bc_dict.get("file_rules", []),
                                    table_contexts=table_specs,
                                ),
                            )
                        cfg.ingest.business_contexts = BusinessContextsConfig(
                            global_rules=[],
                            file_contexts=file_contexts,
                        )
                elif hasattr(cfg.ingest, key):
                    setattr(cfg.ingest, key, value)

        # Embed config
        if config_file.embed:
            embed_data = config_file.embed
            if "strategy" in embed_data:
                cfg.embed.strategy = embed_data["strategy"]
            if "file_specs" in embed_data:
                # Convert dicts to FileEmbeddingSpec instances with nested TableEmbeddingSpec
                file_specs = []
                for fs_dict in embed_data["file_specs"]:
                    # Extract tables list and convert each to TableEmbeddingSpec
                    tables_data = fs_dict.get("tables", [])
                    table_specs = [
                        TableEmbeddingSpec(**table_dict) for table_dict in tables_data
                    ]
                    # Create FileEmbeddingSpec with file_path, context, and tables
                    file_specs.append(
                        FileEmbeddingSpec(
                            file_path=fs_dict["file_path"],
                            context=fs_dict["context"],
                            tables=table_specs,
                        ),
                    )
                cfg.embed.file_specs = file_specs

        # Output config
        if config_file.output:
            if "return_mode" in config_file.output:
                cfg.output.return_mode = config_file.output["return_mode"]

        # Diagnostics config
        if config_file.diagnostics:
            if "enable_progress" in config_file.diagnostics:
                cfg.diagnostics.enable_progress = config_file.diagnostics[
                    "enable_progress"
                ]
            if "progress_mode" in config_file.diagnostics:
                cfg.diagnostics.progress_mode = config_file.diagnostics["progress_mode"]
            if "progress_file" in config_file.diagnostics:
                cfg.diagnostics.progress_file = config_file.diagnostics["progress_file"]
            if "verbosity" in config_file.diagnostics:
                cfg.diagnostics.verbosity = config_file.diagnostics["verbosity"]

        # Execution config
        if config_file.execution:
            if "parallel_files" in config_file.execution:
                cfg.execution.parallel_files = config_file.execution["parallel_files"]
            if "max_file_workers" in config_file.execution:
                cfg.execution.max_file_workers = config_file.execution[
                    "max_file_workers"
                ]
            if "max_embed_workers" in config_file.execution:
                cfg.execution.max_embed_workers = config_file.execution[
                    "max_embed_workers"
                ]

        # Retry config
        if config_file.retry:
            if "max_retries" in config_file.retry:
                cfg.retry.max_retries = config_file.retry["max_retries"]
            if "retry_delay_seconds" in config_file.retry:
                cfg.retry.retry_delay_seconds = config_file.retry["retry_delay_seconds"]
            if "fail_fast" in config_file.retry:
                cfg.retry.fail_fast = config_file.retry["fail_fast"]

        return cfg
