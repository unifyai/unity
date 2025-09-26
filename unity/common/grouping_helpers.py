from typing import Any, Dict, Iterable, List, Union
import json


def _freeze_key(value: Any) -> Any:
    """
    Return a stable, hashable key for arbitrary JSON-like values to use in grouping.
    """
    try:
        if value is None or isinstance(value, (int, float, str, bool)):
            return value
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except Exception:
        return repr(value)


def _compute_column_order(
    *,
    rows: List[Dict[str, Any]],
    exclude_fields: Iterable[str] | None,
) -> List[str]:
    """
    Compute a deterministic ordering of columns by ascending unique value counts,
    excluding any provided in `exclude_fields`. Ties are broken lexicographically.
    """
    if not rows:
        return []

    excluded = set(exclude_fields or [])

    # Gather candidate columns across rows
    candidate_columns: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in excluded:
                candidate_columns.add(k)

    # Count unique values per column
    unique_counts: Dict[str, int] = {}
    for col in candidate_columns:
        seen: set[Any] = set()
        for r in rows:
            seen.add(_freeze_key(r.get(col)))
        unique_counts[col] = len(seen)

    ordered = sorted(candidate_columns, key=lambda c: (unique_counts[c], c))
    return ordered


def _group_rows_recursive(
    *,
    rows: List[Dict[str, Any]],
    ordered_columns: List[str],
    start_index: int,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Recursively group rows by ordered columns starting at `start_index`.
    Stop when grouping would produce one group per row.
    Returns the grouped dict shape compatible with Unify nested-groups.
    """
    if not rows or start_index >= len(ordered_columns):
        return rows

    col = ordered_columns[start_index]

    # Partition rows by the current column, preserving first-seen key order
    key_to_rows: Dict[Any, List[Dict[str, Any]]] = {}
    order_of_keys: List[Any] = []
    for r in rows:
        kf = _freeze_key(r.get(col))
        if kf not in key_to_rows:
            key_to_rows[kf] = []
            order_of_keys.append(kf)
        key_to_rows[kf].append(r)

    unique_count = len(order_of_keys)
    if unique_count == len(rows):
        return rows

    groups: List[Dict[str, Any]] = []
    for kf in order_of_keys:
        subset = key_to_rows[kf]

        next_index = start_index + 1
        if next_index < len(ordered_columns):
            next_col = ordered_columns[next_index]
            seen_next: set[Any] = set(_freeze_key(r.get(next_col)) for r in subset)
            if len(seen_next) == len(subset):
                nested = subset
            else:
                nested = _group_rows_recursive(
                    rows=subset,
                    ordered_columns=ordered_columns,
                    start_index=next_index,
                )
        else:
            nested = subset

        representative_value = subset[0].get(col) if subset else None
        groups.append({"key": representative_value, "value": nested})

    return {col: {"group": groups, "group_count": unique_count, "count": len(rows)}}


def maybe_group_rows(
    *,
    rows: List[Dict[str, Any]],
    exclude_fields: Iterable[str] | None,
    enabled: bool,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    When `enabled` is True, group `rows` client-side using all non-excluded columns,
    ordered by ascending unique counts, with early stopping when grouping becomes
    one-to-one. Otherwise return `rows` unchanged.
    """
    if not enabled:
        return rows

    ordered_columns = _compute_column_order(rows=rows, exclude_fields=exclude_fields)
    if not ordered_columns:
        return rows

    first_col = ordered_columns[0]
    seen_first: set[Any] = set(_freeze_key(r.get(first_col)) for r in rows)
    if len(seen_first) == len(rows):
        return rows

    return _group_rows_recursive(
        rows=rows,
        ordered_columns=ordered_columns,
        start_index=0,
    )
