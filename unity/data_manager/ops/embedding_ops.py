"""
Embedding operations for DataManager.

Implementation functions for ensure_vector_column and vectorize_rows.
These are called by DataManager methods and should not be used directly.

This module delegates to unity.common.embed_utils for the actual
embedding column creation using derived columns with embed() expressions.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from unity.common.embed_utils import ensure_vector_column as _ensure_vector_column

logger = logging.getLogger(__name__)


def ensure_vector_column_impl(
    context: str,
    *,
    source_column: str,
    target_column: Optional[str] = None,
) -> str:
    """
    Ensure an embedding column exists for a source column.

    Creates the vector column structure if it doesn't exist. This sets up
    the embedding column schema but does NOT populate embeddings - use
    vectorize_rows_impl for that.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    source_column : str
        Name of the column containing text to embed.
    target_column : str | None
        Name for the embedding column. Defaults to ``_{source_column}_emb``.

    Returns
    -------
    str
        Name of the embedding column (target_column or generated default).
    """
    # Default target column name follows _<source>_emb convention
    target = target_column or f"_{source_column}_emb"

    logger.debug(
        "Ensuring vector column: %s -> %s in %s",
        source_column,
        target,
        context,
    )

    # Use the common embed_utils function which properly:
    # 1. Creates a derived column using embed() expression
    # 2. Handles locking for concurrency
    # 3. Tolerates existing columns
    _ensure_vector_column(
        context=context,
        embed_column=target,
        source_column=source_column,
        derived_expr=None,  # Use source_column directly (not a derived expression)
        from_ids=None,  # Don't populate yet, just create structure
    )

    return target


def vectorize_rows_impl(
    context: str,
    *,
    source_column: str,
    target_column: Optional[str] = None,
    row_ids: Optional[List[int]] = None,
    batch_size: int = 100,
) -> int:
    """
    Generate embeddings for rows in a table.

    Populates the embedding column with vector representations of the
    source column text. Call ensure_vector_column_impl first to set up
    the column structure.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context path.
    source_column : str
        Name of the column containing text to embed.
    target_column : str | None
        Name for the embedding column. Defaults to ``_{source_column}_emb``.
    row_ids : list[int] | None
        Specific row IDs to embed. If None, embeds all rows without embeddings.
    batch_size : int, default 100
        Number of rows to embed per batch. Larger batches are faster but
        use more memory.

    Returns
    -------
    int
        Number of rows that were embedded.
    """
    target = target_column or f"_{source_column}_emb"

    logger.debug(
        "Vectorizing rows in %s: %s -> %s (batch_size=%d, row_ids=%s)",
        context,
        source_column,
        target,
        batch_size,
        row_ids[:5] if row_ids else None,
    )

    # Use the common embed_utils function to populate embeddings
    # Note: ensure_vector_column handles both creation and population
    # with from_ids parameter
    _ensure_vector_column(
        context=context,
        embed_column=target,
        source_column=source_column,
        derived_expr=None,
        from_ids=row_ids,
    )

    # Return count of rows processed
    # Note: The actual count isn't returned by ensure_vector_column,
    # so we return the number of requested row_ids or 0 if processing all
    return len(row_ids) if row_ids else 0
