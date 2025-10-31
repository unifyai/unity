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


class IngestConfig(BaseModel):
    """Controls how parsed rows are stored in Unify contexts.

    Layout modes:
    - "per_file": current default. Each file has `File/<alias>/<safe_file>/Content`
      and tables under `.../Tables/<table>`.
    - "unified": all rows for this job go to a single logical bucket by reusing
      per-file ops with a fixed `unified_label` as the "filename". This avoids
      new ops while enabling a global sink.
    """

    mode: Literal["per_file", "unified"] = "per_file"

    # Unified layout target. When None, a sensible default is chosen by the caller.
    unified_label: Optional[str] = None

    # Row management
    replace_existing: bool = True
    allowed_columns: Optional[List[str]] = None

    # Auto-counting configuration fed into Document.to_schema_rows when provided.
    # When None, manager defaults are used.
    auto_counting_per_file: Optional[Dict[str, Optional[str]]] = None
    auto_counting_unified: Optional[Dict[str, Optional[str]]] = Field(
        default_factory=lambda: {"row_id": None},
    )

    # Table ingestion control
    table_ingest: bool = True
    table_label_strategy: Literal["sheet_name", "section_path", "index"] = "sheet_name"
    table_rows_batch_size: int = 100


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
    """Embedding behavior applied after insertions.

    When embed_along is True, embeddings are created after inserts according to
    the provided specs. If row ids are available from inserts, they can be used
    for targeted embedding; otherwise, embedding runs over the whole column.
    """

    embed_along: bool = False
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
