from __future__ import annotations

"""Config models for the FileManager parsing → ingestion → embedding pipeline.

This module intentionally keeps the number of models small and grouped by concern
to remain approachable while still being extensible. The design goals are:

- Single, obvious entry point (`FilePipelineConfig`) that callers provide.
- Clear separation of concerns: parser knobs, ingest/storage layout, embeddings,
  and optional plugin hooks for dependency injection.
- Backward compatibility: existing ad-hoc kwargs in manager methods can be
  coerced into this config without breaking callers.
- Extensibility: future features (e.g., images ingestion, custom schemas,
  alternate layouts, pre/post step plugins) can be added with minimal churn.

Tip for future contributors:
- Prefer adding options to these grouped models rather than creating many small
  models. Keep defaults sensible so the default behavior matches the current
  FileManager pipeline without configuration.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Literal
from pydantic import BaseModel, Field, model_validator
import importlib
import json
from pathlib import Path


# ------------------------------ Parser ------------------------------------ #


class ParseConfig(BaseModel):
    """Options forwarded to the underlying parser.

    - batch_size controls parse_batch_async parallelism when available.
    - parser_kwargs are forwarded verbatim to the parser's parse/parse_batch
      methods. Keep this small and declarative; the parser owns validation.
    """

    batch_size: int = 3
    parser_kwargs: Dict[str, Any] = Field(default_factory=dict)


# --------------------------- Ingest / Storage ----------------------------- #


class ContentIngestPolicy(BaseModel):
    """Per-file-format policy controlling what goes into the per-file Content context.

    mode:
        - "default": keep rows as produced by the parser (current behavior)
        - "none": do not create any per-file Content rows
        - "document_only": keep only a single document-level row. If the parser
          did not produce one, a synthetic minimal document row is created.
    omit_fields:
        Field names to drop from rows (e.g., ["content_text", "summary"]).
    """

    mode: Literal["default", "none", "document_only"] = "default"
    omit_fields: List[str] = Field(default_factory=list)


class IngestConfig(BaseModel):
    """Controls how parsed rows are stored in Unify contexts.

    Layout modes:
    - "per_file": current default. Each file has `Files/<alias>/<safe_file>/Content`
      and tables under `.../Tables/<table>`.
    - "unified": all rows for this job go to a single logical bucket by reusing
      per-file ops with a fixed `unified_label` as the "filename". This avoids
      new ops while enabling a global sink.
    """

    mode: Literal["per_file", "unified"] = "per_file"

    # Unified layout target. When None, a sensible default is chosen by the caller.
    unified_label: Optional[str] = None

    # Chunking controls for ingestion
    # - content_rows_batch_size controls batching for per-file Content rows
    #   (used by along/auto strategies to incrementally ingest+embed).
    content_rows_batch_size: int = 1000

    # ID layout and hierarchy configuration
    # - id_layout selects how hierarchical identifiers are represented in Content rows:
    #   • "map" (default): a single dict field `content_id` like
    #       {"document": 0, "section": 2, "paragraph": 1, "sentence": 3}
    #   • "columns": legacy per-level id columns are included
    #       (document_id, section_id, paragraph_id, sentence_id, image_id, table_id)
    #     and auto_counting controls apply. Use this only when required.
    #   • "string": future-friendly hook for a string encoding (e.g. "doc:0>sec:2>…").
    id_layout: Literal["map", "columns", "string"] = "map"
    # When id_layout == "columns", this hierarchy dictates parentage for auto-counting.
    # Defaults mirror the previous behavior. Ignored for id_layout == "map".
    id_hierarchy: Optional[Dict[str, Optional[str]]] = {
        "document_id": None,
        "section_id": "document_id",
        "image_id": "section_id",
        "table_id": "section_id",
        "paragraph_id": "section_id",
        "sentence_id": "paragraph_id",
    }
    # Optional string format template when id_layout == "string" (reserved for future use).
    id_string_format: Optional[str] = None

    # Row management
    replace_existing: bool = True
    allowed_columns: Optional[List[str]] = None

    # Auto-counting configuration fed into Document.to_schema_rows when provided.
    # When None, manager defaults are used. These only apply when id_layout == "columns".
    auto_counting_per_file: Optional[Dict[str, Optional[str]]] = {
        "document_id": None,
        "section_id": "document_id",
        "image_id": "section_id",
        "table_id": "section_id",
        "paragraph_id": "section_id",
        "sentence_id": "paragraph_id",
    }
    auto_counting_unified: Optional[Dict[str, Optional[str]]] = {
        "document_id": None,
        "section_id": "document_id",
        "image_id": "section_id",
        "table_id": "section_id",
        "paragraph_id": "section_id",
        "sentence_id": "paragraph_id",
    }

    # Table ingestion control
    table_ingest: bool = True
    table_label_strategy: Literal["sheet_name", "section_path", "index"] = "sheet_name"
    table_rows_batch_size: int = 100

    # Per-format policy for Content ingestion (document/section/paragraph rows)
    # Keys are normalized file formats (e.g., "pdf", "docx", "xlsx", "csv").
    # Defaults aim to avoid redundant full-text/summary storage for pure tables:
    # - For spreadsheets (xlsx/csv), keep a single document row and drop text fields.
    content_policy_by_format: Dict[str, ContentIngestPolicy] = Field(
        default_factory=lambda: {
            "xlsx": ContentIngestPolicy(
                mode="document_only",
                omit_fields=["content_text", "summary"],
            ),
            "csv": ContentIngestPolicy(
                mode="document_only",
                omit_fields=["content_text", "summary"],
            ),
        },
    )

    # Business context specifications for enriching table contexts with descriptions
    business_contexts: List["BusinessContextSpec"] = Field(default_factory=list)


# --------------------------- Business Context ----------------------------- #


class TableBusinessContextSpec(BaseModel):
    """Table-level business context specification for enriching a single table context with descriptions.

    - table: exact table label (required for matching)
    - column_descriptions: mapping of column name → description
    - table_description: optional description for the table context itself
    """

    table: str
    column_descriptions: Dict[str, str] = Field(default_factory=dict)
    table_description: Optional[str] = None


class BusinessContextSpec(BaseModel):
    """Business context specification for enriching table contexts with descriptions.

    - file_path: exact file path (required for matching)
    - tables: list of table specs for this file (at least one required)
    """

    file_path: str
    tables: List[TableBusinessContextSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tables_not_empty(self) -> "BusinessContextSpec":
        """Ensure tables list is not empty."""
        if not self.tables:
            raise ValueError(
                "BusinessContextSpec must have at least one table specification",
            )
        return self


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
        - "along": ingest and embed in chunks within a single file loop
        - "auto": choose between "after" and "along" using `large_threshold`
    large_threshold:
        Size heuristic (total_records or total table rows) used when strategy == "auto".
    hooks_per_chunk:
        When True, run pre/post embed hooks for each chunk in along mode.
    file_specs:
        File-level embedding specifications describing where and what to embed, organized by file_path.
        Each FileEmbeddingSpec can target multiple tables within a file.
    """

    strategy: Literal["off", "after", "along", "auto"] = "auto"
    large_threshold: int = 2000
    hooks_per_chunk: bool = True
    file_specs: List[FileEmbeddingSpec] = Field(default_factory=list)


# ------------------------------- Plugins ---------------------------------- #


class PluginsConfig(BaseModel):
    """Optional plugin hooks for dependency injection.

    Hooks receive: (manager, filename, result, document, config) and run at the
    designated stage. List entries can be dotted paths (module.func) or direct
    callables injected by the caller.
    """

    pre_parse: List[Any] = Field(default_factory=list)
    post_parse: List[Any] = Field(default_factory=list)
    pre_ingest: List[Any] = Field(default_factory=list)
    post_ingest: List[Any] = Field(default_factory=list)
    pre_embed: List[Any] = Field(default_factory=list)
    post_embed: List[Any] = Field(default_factory=list)

    # Optional kwargs per plugin name (key: dotted name or identifier)
    plugin_kwargs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


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
    """Controls optional pipeline diagnostics output (stdout).

    - enable_progress: when True, emit human-friendly progress prints for
      ingest/embed steps, including chunk counters where applicable.
    """

    enable_progress: bool = False


def resolve_callables(names_or_callables: Iterable[Any]) -> List[Callable]:
    """Resolve a sequence of dotted names or callables into callables.

    - Callables are passed through.
    - Strings are resolved by importing the module then attribute. Non-callables
      are ignored silently to keep pipeline robust.
    """

    out: List[Callable] = []
    for item in names_or_callables:
        if callable(item):
            out.append(item)  # type: ignore[arg-type]
            continue
        if isinstance(item, str) and item:
            mod, _, attr = item.rpartition(".")
            if mod and attr:
                try:
                    fn = getattr(importlib.import_module(mod), attr, None)
                    if callable(fn):
                        out.append(fn)  # type: ignore[arg-type]
                except Exception:
                    continue
    return out


# ------------------------------ Entry point ------------------------------- #


class FilePipelineConfig(BaseModel):
    """Top-level configuration for the FileManager pipeline.

    Keep this as the single entry point. Extend the grouped sub-configs as
    needed instead of growing many small models. Defaults preserve current
    behavior (per-file layout, no embeddings, modest parser batch size).

    Can be instantiated with defaults or loaded from a JSON file using `from_file()`.
    Supports partial configs when loading from file - only define what you need,
    defaults fill the rest.
    """

    parse: ParseConfig = ParseConfig()
    ingest: IngestConfig = IngestConfig()
    embed: EmbeddingsConfig = EmbeddingsConfig()
    plugins: PluginsConfig = PluginsConfig()
    output: OutputConfig = OutputConfig()
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()

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
            plugins: Optional[Dict[str, Any]] = None
            output: Optional[Dict[str, Any]] = None
            diagnostics: Optional[Dict[str, Any]] = None

        # Validate JSON structure
        config_file = _FilePipelineConfigFile.model_validate(data)

        # Build FilePipelineConfig with defaults, then override with provided values
        cfg = cls()

        # Parse config
        if config_file.parse:
            if "batch_size" in config_file.parse:
                cfg.parse.batch_size = config_file.parse["batch_size"]
            if "parser_kwargs" in config_file.parse:
                cfg.parse.parser_kwargs.update(config_file.parse["parser_kwargs"])

        # Ingest config
        if config_file.ingest:
            ingest_data = config_file.ingest
            for key, value in ingest_data.items():
                if key == "business_contexts":
                    # Convert dicts to BusinessContextSpec instances with nested TableBusinessContextSpec
                    business_contexts = []
                    for bc_dict in value:
                        # Extract tables list and convert each to TableBusinessContextSpec
                        tables_data = bc_dict.get("tables", [])
                        table_specs = [
                            TableBusinessContextSpec(**table_dict)
                            for table_dict in tables_data
                        ]
                        # Create BusinessContextSpec with file_path and tables
                        business_contexts.append(
                            BusinessContextSpec(
                                file_path=bc_dict["file_path"],
                                tables=table_specs,
                            ),
                        )
                    cfg.ingest.business_contexts = business_contexts
                elif hasattr(cfg.ingest, key):
                    setattr(cfg.ingest, key, value)

        # Embed config
        if config_file.embed:
            embed_data = config_file.embed
            if "strategy" in embed_data:
                cfg.embed.strategy = embed_data["strategy"]
            if "large_threshold" in embed_data:
                cfg.embed.large_threshold = embed_data["large_threshold"]
            if "hooks_per_chunk" in embed_data:
                cfg.embed.hooks_per_chunk = embed_data["hooks_per_chunk"]
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

        # Plugins config
        if config_file.plugins:
            plugins_data = config_file.plugins
            for key in [
                "pre_parse",
                "post_parse",
                "pre_ingest",
                "post_ingest",
                "pre_embed",
                "post_embed",
            ]:
                if key in plugins_data:
                    setattr(cfg.plugins, key, plugins_data[key])
            if "plugin_kwargs" in plugins_data:
                cfg.plugins.plugin_kwargs.update(plugins_data["plugin_kwargs"])

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

        return cfg
