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
from pydantic import BaseModel, Field
import importlib


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


# ------------------------------ Embeddings -------------------------------- #


class EmbeddingSpec(BaseModel):
    """Embedding target specification.

    - context selects where to embed.
    - table allows narrowing to a specific table label in per_file_table mode;
      use "*" or None to apply broadly.
    """

    context: Literal["per_file", "per_file_table", "unified"]
    table: Optional[str] = None
    source_column: str
    target_column: str


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
    specs:
        Target specifications describing where and what to embed.
    """

    strategy: Literal["off", "after", "along", "auto"] = "auto"
    large_threshold: int = 2000
    hooks_per_chunk: bool = True
    specs: List[EmbeddingSpec] = Field(default_factory=list)


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
    """

    parse: ParseConfig = ParseConfig()
    ingest: IngestConfig = IngestConfig()
    embed: EmbeddingsConfig = EmbeddingsConfig()
    plugins: PluginsConfig = PluginsConfig()
    output: OutputConfig = OutputConfig()
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()
