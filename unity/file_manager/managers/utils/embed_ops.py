"""Pure batch embedding operations for the FileManager pipeline.

This module contains the core embedding logic extracted from ops.py,
providing clean, stateless functions for embedding content and table columns.

These functions:
- Are pure batch operations without async orchestration
- Take explicit parameters instead of relying on manager state
- Return results directly without side effects on progress tracking
- Are designed to be called by the PipelineExecutor's task functions

The orchestration of these operations (parallel execution, progress reporting,
retries) is handled by the executor layer.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from unity.common.embed_utils import ensure_vector_column
from unity.file_manager.types.config import (
    FilePipelineConfig,
    FileEmbeddingSpec,
    TableEmbeddingSpec,
)

logger = logging.getLogger(__name__)


def embed_column(
    *,
    context: str,
    source_column: str,
    target_column: str,
    from_ids: Optional[List[int]] = None,
    async_embeddings: bool = True,
) -> bool:
    """
    Embed a single column in a context.

    Parameters
    ----------
    context : str
        The Unify context containing the source column.
    source_column : str
        Name of the column to embed.
    target_column : str
        Name of the vector column to create/update.
    from_ids : list[int] | None
        If provided, only embed rows with these log IDs.
    async_embeddings : bool
        Whether to generate embeddings asynchronously.

    Returns
    -------
    bool
        True if embedding succeeded, False otherwise.
    """
    try:
        ensure_vector_column(
            context,
            embed_column=target_column,
            source_column=source_column,
            async_embeddings=async_embeddings,
            from_ids=list(from_ids or []),
        )
        return True
    except Exception as e:
        logger.error(f"Failed to embed column {source_column} -> {target_column}: {e}")
        return False


def embed_content_batch(
    *,
    context: str,
    specs: List[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]],
    inserted_ids: Optional[List[int]] = None,
) -> Dict[str, bool]:
    """
    Embed content columns for a batch of inserted rows.

    Parameters
    ----------
    context : str
        The content context to embed.
    specs : list[tuple[FileEmbeddingSpec, TableEmbeddingSpec]]
        Embedding specifications to apply.
    inserted_ids : list[int] | None
        Log IDs to scope the embedding to.

    Returns
    -------
    dict[str, bool]
        Mapping of target_column -> success status.
    """
    results: Dict[str, bool] = {}

    # Filter applicable specs (content contexts only)
    applicable_specs = [
        (file_spec, table_spec)
        for file_spec, table_spec in specs
        if file_spec.context in ("per_file", "unified")
    ]

    if not applicable_specs:
        return results

    for file_spec, table_spec in applicable_specs:
        for source_col, target_col in zip(
            table_spec.source_columns,
            table_spec.target_columns,
        ):
            success = embed_column(
                context=context,
                source_column=source_col,
                target_column=target_col,
                async_embeddings=True,
                from_ids=inserted_ids,
            )
            results[target_col] = success

    return results


def embed_table_batch(
    *,
    context: str,
    table_label: str,
    specs: List[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]],
    inserted_ids: Optional[List[int]] = None,
    safe_fn: Optional[callable] = None,
) -> Dict[str, bool]:
    """
    Embed table columns for a batch of inserted rows.

    Parameters
    ----------
    context : str
        The table context to embed.
    table_label : str
        The table label (used to match specs).
    specs : list[tuple[FileEmbeddingSpec, TableEmbeddingSpec]]
        Embedding specifications to apply.
    inserted_ids : list[int] | None
        Log IDs to scope the embedding to.
    safe_fn : callable | None
        Optional function to sanitize table names for matching.

    Returns
    -------
    dict[str, bool]
        Mapping of target_column -> success status.
    """
    results: Dict[str, bool] = {}

    # Extract tail label from context for matching
    tail = None
    try:
        if "/Tables/" in context:
            tail = context.split("/Tables/", 1)[-1]
    except Exception:
        tail = None

    if safe_fn is None:
        safe_fn = lambda x: x

    # Filter applicable specs (per_file_table contexts only)
    applicable_specs: List[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]] = []
    for file_spec, table_spec in specs:
        if file_spec.context != "per_file_table":
            continue

        table_filter = table_spec.table
        if table_filter in (None, "*"):
            applicable_specs.append((file_spec, table_spec))
            continue

        # Match sanitized names
        try:
            safe_target = safe_fn(str(table_filter))
        except Exception:
            safe_target = str(table_filter)

        if tail and (tail == safe_target or tail == str(table_filter)):
            applicable_specs.append((file_spec, table_spec))

    if not applicable_specs:
        return results

    for file_spec, table_spec in applicable_specs:
        for source_col, target_col in zip(
            table_spec.source_columns,
            table_spec.target_columns,
        ):
            success = embed_column(
                context=context,
                source_column=source_col,
                target_column=target_col,
                async_embeddings=True,
                from_ids=inserted_ids,
            )
            results[target_col] = success

    return results


def get_embedding_specs_for_file(
    file_path: str,
    config: FilePipelineConfig,
) -> List[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]]:
    """
    Get all embedding specs that apply to a file.

    Parameters
    ----------
    file_path : str
        The file path to match against.
    config : FilePipelineConfig
        Pipeline configuration containing embed specs.

    Returns
    -------
    list[tuple[FileEmbeddingSpec, TableEmbeddingSpec]]
        Matching (file_spec, table_spec) pairs.
    """
    specs: List[Tuple[FileEmbeddingSpec, TableEmbeddingSpec]] = []

    file_specs = getattr(getattr(config, "embed", None), "file_specs", []) or []

    for file_spec in file_specs:
        # Match file path ("*" matches all)
        if file_spec.file_path != "*" and file_spec.file_path != file_path:
            continue

        # Collect all table specs
        for table_spec in file_spec.tables:
            specs.append((file_spec, table_spec))

    return specs


def has_embedding_work(
    config: FilePipelineConfig,
    file_path: str,
) -> bool:
    """
    Check if there is any embedding work to do for a file.

    Parameters
    ----------
    config : FilePipelineConfig
        Pipeline configuration.
    file_path : str
        The file path to check.

    Returns
    -------
    bool
        True if embedding is enabled and there are matching specs.
    """
    strategy = getattr(getattr(config, "embed", None), "strategy", "off")
    if strategy == "off":
        return False

    specs = get_embedding_specs_for_file(file_path, config)
    return len(specs) > 0
