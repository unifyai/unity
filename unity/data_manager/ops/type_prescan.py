"""Pre-scan type inference and value coercion for bulk data ingestion.

This module implements the stratified-sampling type determination and
per-cell coercion pipeline that runs **before** rows are chunked and
sent to Orchestra. The goal is to:

1. Determine the "best" column type from a representative sample
   (avoiding a full scan on very large datasets).
2. Coerce non-conforming cell values to ``None`` so that Orchestra's
   strict type enforcement never rejects individual rows.
3. Always coerce empty strings ``""`` to ``None`` (universal rule).

For streaming ingestion, :class:`TypeMap` and :func:`prescan_from_rows`
provide a forward-only variant that works on any row iterable (no random
access). :func:`coerce_batch` applies a pre-computed :class:`TypeMap` to
a single chunk of rows.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, FrozenSet, Iterable, List, Tuple

from unity.common.type_utils import (
    _is_date_string,
    _is_time_string,
    _is_timedelta_string,
    infer_type_from_value,
)

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_SIZE = 500


@dataclass
class CoercionStats:
    """Aggregated coercion statistics for a set of rows."""

    total_cells: int = 0
    empty_strings_coerced: int = 0
    type_coerced: int = 0
    coerced_by_column: Dict[str, int] = field(default_factory=dict)


# =========================================================================
# Stratified sampling + majority-vote type determination
# =========================================================================


def _stratified_indices(n: int, k: int) -> List[int]:
    """Return up to *k* indices drawn from head/middle/tail of range(n).

    Distribution: 20 % head, 20 % tail, 60 % random from middle.
    """
    if n <= k:
        return list(range(n))

    head_count = max(1, k // 5)
    tail_count = max(1, k // 5)
    mid_count = k - head_count - tail_count

    head = list(range(min(head_count, n)))
    tail = list(range(max(0, n - tail_count), n))

    mid_start = head_count
    mid_end = max(mid_start, n - tail_count)
    mid_pool = range(mid_start, mid_end)
    if len(mid_pool) <= mid_count:
        mid = list(mid_pool)
    else:
        mid = sorted(random.sample(mid_pool, mid_count))

    combined = sorted(set(head + mid + tail))
    return combined[:k]


def prescan_column_types(
    rows: List[Dict[str, Any]],
    *,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
) -> Dict[str, str]:
    """Determine the dominant type for each column via stratified sampling.

    Parameters
    ----------
    rows : list[dict]
        The full dataset (already in memory).
    sample_size : int
        Maximum number of rows to sample per column.

    Returns
    -------
    dict[str, str]
        Mapping of column name to canonical type string.
    """
    if not rows:
        return {}

    indices = _stratified_indices(len(rows), sample_size)

    col_votes: Dict[str, Dict[str, int]] = {}

    for idx in indices:
        row = rows[idx]
        for col, value in row.items():
            if value is None or value == "":
                continue
            inferred = infer_type_from_value(value)
            if inferred == "NoneType":
                continue
            votes = col_votes.setdefault(col, {})
            votes[inferred] = votes.get(inferred, 0) + 1

    result: Dict[str, str] = {}
    all_cols = {col for row in rows for col in row}
    for col in sorted(all_cols):
        votes = col_votes.get(col, {})
        if not votes:
            result[col] = "str"
            continue
        max_count = max(votes.values())
        winners = [t for t, c in votes.items() if c == max_count]
        result[col] = winners[0] if len(winners) == 1 else "str"
    return result


# =========================================================================
# Per-value coercion
# =========================================================================

_VALIDATORS: Dict[str, Any] = {}


def _build_validators() -> Dict[str, Any]:
    """Lazily build type-specific validator functions.

    Each validator returns the value unchanged if valid, or ``None``.
    """
    from datetime import date as _date
    from datetime import time as _time
    from datetime import timedelta as _timedelta

    def _check_datetime(v: Any) -> Any:
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                datetime.fromisoformat(v)
                return v
            except Exception:
                return None
        return None

    def _check_date(v: Any) -> Any:
        if isinstance(v, _date) and not isinstance(v, datetime):
            return v
        if isinstance(v, str) and _is_date_string(v):
            return v
        return None

    def _check_time(v: Any) -> Any:
        if isinstance(v, _time):
            return v
        if isinstance(v, str) and _is_time_string(v):
            return v
        return None

    def _check_timedelta(v: Any) -> Any:
        if isinstance(v, _timedelta):
            return v
        if isinstance(v, str) and _is_timedelta_string(v):
            return v
        return None

    def _check_int(v: Any) -> Any:
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            stripped = v.lstrip("-")
            if stripped.isdigit():
                return v
        return None

    def _check_float(v: Any) -> Any:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            try:
                float(v)
                return v
            except (ValueError, TypeError):
                return None
        return None

    def _check_bool(v: Any) -> Any:
        if isinstance(v, bool):
            return v
        return None

    def _check_str(_v: Any) -> Any:
        return _v

    return {
        "datetime": _check_datetime,
        "date": _check_date,
        "time": _check_time,
        "timedelta": _check_timedelta,
        "int": _check_int,
        "float": _check_float,
        "bool": _check_bool,
        "str": _check_str,
    }


def coerce_value(value: Any, target_type: str) -> Any:
    """Return *value* unchanged if it conforms to *target_type*, else ``None``.

    ``None`` input always passes through (NoneType is weak in Orchestra).
    """
    if value is None:
        return None

    global _VALIDATORS
    if not _VALIDATORS:
        _VALIDATORS = _build_validators()

    validator = _VALIDATORS.get(target_type)
    if validator is None:
        return value
    return validator(value)


# =========================================================================
# Full-row coercion pass
# =========================================================================


def coerce_rows(
    rows: List[Dict[str, Any]],
    column_types: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], CoercionStats]:
    """Coerce all rows using column-oriented processing.

    Iterates **by column** rather than by row so that each column's
    validator is resolved once and then swept across all rows.  This
    avoids a per-cell validator lookup and is more CPU-cache-friendly
    for large datasets.

    For every cell:
    1. If ``value == ""`` → replace with ``None`` (universal rule).
    2. If ``value is not None`` and does not conform to the column's
       determined type → replace with ``None``.

    Returns the (possibly mutated) rows and aggregated statistics.
    """
    global _VALIDATORS
    if not _VALIDATORS:
        _VALIDATORS = _build_validators()

    all_cols = {col for row in rows for col in row}
    total_cells = 0
    empty_coerced = 0
    type_coerced = 0
    coerced_by_col: Dict[str, int] = {}

    for col in all_cols:
        target = column_types.get(col)
        validator = _VALIDATORS.get(target) if target else None
        col_coerced = 0

        for row in rows:
            if col not in row:
                continue
            total_cells += 1
            value = row[col]

            if value == "":
                row[col] = None
                empty_coerced += 1
                col_coerced += 1
                continue

            if value is None or validator is None:
                continue

            if validator(value) is None:
                row[col] = None
                type_coerced += 1
                col_coerced += 1

        if col_coerced:
            coerced_by_col[col] = col_coerced

    stats = CoercionStats(
        total_cells=total_cells,
        empty_strings_coerced=empty_coerced,
        type_coerced=type_coerced,
        coerced_by_column=coerced_by_col,
    )
    return rows, stats


def coerce_empty_strings(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Coerce only empty strings to ``None`` (used when ``coerce_types=False``).

    Returns the rows and a count of coerced cells.
    """
    count = 0
    for row in rows:
        for col in list(row.keys()):
            if row[col] == "":
                row[col] = None
                count += 1
    return rows, count


# =========================================================================
# Streaming-compatible prescan (forward-only, no random access)
# =========================================================================


@dataclass(frozen=True)
class TypeMap:
    """Immutable result of a type prescan.

    Produced once from a row sample and reused across all subsequent
    chunks during streaming ingestion so every chunk is coerced with
    the same consistent type decisions.
    """

    column_types: Dict[str, str]
    columns: FrozenSet[str]
    sample_size: int


def prescan_from_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
) -> TypeMap:
    """Forward-only prescan that works on any row iterable.

    Unlike :func:`prescan_column_types`, this does **not** require
    random access.  It reads up to *sample_size* rows sequentially,
    accumulates type votes, and returns a frozen :class:`TypeMap`.

    For the materialised-list path the caller can pass the full list;
    for streaming the caller passes the first N rows drained from the
    handle.
    """
    col_votes: Dict[str, Dict[str, int]] = {}
    all_cols: set[str] = set()
    count = 0

    for row in rows:
        if count >= sample_size:
            break
        all_cols.update(row.keys())
        for col, value in row.items():
            if value is None or value == "":
                continue
            inferred = infer_type_from_value(value)
            if inferred == "NoneType":
                continue
            votes = col_votes.setdefault(col, {})
            votes[inferred] = votes.get(inferred, 0) + 1
        count += 1

    column_types: Dict[str, str] = {}
    for col in sorted(all_cols):
        votes = col_votes.get(col, {})
        if not votes:
            column_types[col] = "str"
            continue
        max_count = max(votes.values())
        winners = [t for t, c in votes.items() if c == max_count]
        column_types[col] = winners[0] if len(winners) == 1 else "str"

    return TypeMap(
        column_types=column_types,
        columns=frozenset(all_cols),
        sample_size=count,
    )


def coerce_batch(
    batch: List[Dict[str, Any]],
    type_map: TypeMap,
) -> Tuple[List[Dict[str, Any]], CoercionStats]:
    """Apply coercion to a single chunk using a pre-computed :class:`TypeMap`.

    Semantics are identical to :func:`coerce_rows` but the type map is
    passed in rather than computed on the fly, enabling consistent
    coercion across multiple streaming chunks.
    """
    global _VALIDATORS
    if not _VALIDATORS:
        _VALIDATORS = _build_validators()

    total_cells = 0
    empty_coerced = 0
    type_coerced = 0
    coerced_by_col: Dict[str, int] = {}

    batch_cols = {col for row in batch for col in row}
    for col in batch_cols:
        target = type_map.column_types.get(col)
        validator = _VALIDATORS.get(target) if target else None
        col_coerced = 0

        for row in batch:
            if col not in row:
                continue
            total_cells += 1
            value = row[col]

            if value == "":
                row[col] = None
                empty_coerced += 1
                col_coerced += 1
                continue

            if value is None or validator is None:
                continue

            if validator(value) is None:
                row[col] = None
                type_coerced += 1
                col_coerced += 1

        if col_coerced:
            coerced_by_col[col] = col_coerced

    stats = CoercionStats(
        total_cells=total_cells,
        empty_strings_coerced=empty_coerced,
        type_coerced=type_coerced,
        coerced_by_column=coerced_by_col,
    )
    return batch, stats
