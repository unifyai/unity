"""
Embedding operations for DataManager.

Implementation functions for ensure_vector_column, vectorize_rows.
These are called by DataManager methods and should not be used directly.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import unify

logger = logging.getLogger(__name__)


def ensure_vector_column_impl(
    context: str,
    *,
    source_column: str,
    target_column: Optional[str] = None,
) -> str:
    """
    Implementation of ensure_vector_column operation.

    Creates the embedding column structure if it doesn't exist.
    """
    # Default target column name follows _<source>_emb convention
    target = target_column or f"_{source_column}_emb"

    logger.debug(
        "Ensuring vector column: %s -> %s in %s",
        source_column,
        target,
        context,
    )

    try:
        # Check if the embedding column already exists
        fields = unify.get_fields(context=context) or {}
        if target in fields:
            logger.debug("Vector column %s already exists", target)
            return target
    except Exception as e:
        logger.debug("Could not check existing fields: %s", e)

    # Create/add the embedding column
    try:
        if hasattr(unify, "add_embedding_column"):
            unify.add_embedding_column(
                source_column=source_column,
                target_column=target,
                context=context,
            )
        else:
            # Fallback: create as a regular field (embeddings may need separate setup)
            logger.warning(
                "add_embedding_column not available; "
                "embedding column may need manual setup",
            )
            unify.create_fields({target: "list"}, context=context)
    except Exception as e:
        logger.warning("Could not create embedding column: %s", e)

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
    Implementation of vectorize_rows operation.

    Generates embeddings for rows in a table.
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

    try:
        if hasattr(unify, "vectorize"):
            result = unify.vectorize(
                source_column=source_column,
                target_column=target,
                context=context,
                row_ids=row_ids,
                batch_size=batch_size,
            )
            if isinstance(result, int):
                return result
            return 0
        else:
            logger.warning(
                "unify.vectorize not available; "
                "embedding generation may need alternative approach",
            )
            return 0
    except Exception as e:
        logger.warning("Vectorization failed: %s", e)
        return 0
