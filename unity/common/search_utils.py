from __future__ import annotations

from typing import Dict, List, Optional, Any

# Centralised wrappers for semantic search across tables/contexts.
# These helpers provide a single import point so managers can share
# the same behaviour and we can later harden vector provisioning
# against concurrency races in one place.

from .semantic_search import (
    fetch_top_k_by_references as _fetch_top_k_by_references,
    backfill_rows as _backfill_rows,
    # Re-exported helpers for callers that classify/ensure terms themselves
    is_plain_identifier as is_plain_identifier_expr,
    ensure_vector_for_source as ensure_vector_for_source_expr,
    fetch_top_k_by_terms as fetch_top_k_by_terms_fn,
    fetch_top_k_by_terms_with_score as fetch_top_k_by_terms_with_score_fn,
    fetch_scores_for_ids as fetch_scores_for_ids_fn,
)


def table_search_top_k(
    context: str,
    references: Optional[Dict[str, str]],
    *,
    k: int = 10,
    allowed_fields: Optional[List[str]] = None,
    row_filter: Optional[str] = None,
    unique_id_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return up to k rows from a Unify context using semantic search with backfill.

    Parameters
    ----------
    context : str
        Unify context (table) to search.
    references : dict[str, str] | None
        Mapping of source_expr → reference_text. When None/empty, backfill-only
        logic will return most recent rows (handled by the underlying helpers).
    k : int, default 10
        Maximum number of rows to return.
    allowed_fields : list[str] | None, default None
        Whitelist of fields to include in payloads. Private/vector columns are
        typically omitted by callers.
    row_filter : str | None, default None
        Additional boolean expression to restrict candidate rows (e.g., exclude
        system rows). Applied both to the primary search and the backfill step.
    unique_id_field : str | None, default None
        Name of the unique identifier column for de-duplication in backfill. When None,
        attempts to infer from the context's `unique_keys`.

    Returns
    -------
    list[dict]
        List of row dictionaries suitable for constructing Pydantic models or
        writing through to local caches.
    """
    rows = _fetch_top_k_by_references(
        context,
        references,
        k=k,
        allowed_fields=allowed_fields,
        row_filter=row_filter,
    )
    filled = _backfill_rows(
        context,
        rows,
        k,
        row_filter=row_filter,
        unique_id_field=unique_id_field,
        allowed_fields=allowed_fields,
    )
    return filled


# Lightweight re-exports so call-sites can migrate to this module without
# changing behaviour; centralises the import path for future hardening.
is_plain_identifier = is_plain_identifier_expr
ensure_vector_for_source = ensure_vector_for_source_expr
fetch_top_k_by_terms = fetch_top_k_by_terms_fn
fetch_top_k_by_terms_with_score = fetch_top_k_by_terms_with_score_fn
fetch_scores_for_ids = fetch_scores_for_ids_fn
fetch_top_k_by_references = _fetch_top_k_by_references
backfill_rows = _backfill_rows
