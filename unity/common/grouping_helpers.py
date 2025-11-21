from typing import Any, Dict, Iterable, List, Union, Tuple
import json
import time
import logging
import unify
from .embed_utils import list_private_fields


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

    _t0 = time.perf_counter()
    ordered_columns = _compute_column_order(rows=rows, exclude_fields=exclude_fields)
    if not ordered_columns:
        _t1 = time.perf_counter()
        try:
            logging.getLogger(__name__).info(
                "maybe_group_rows: no-op | rows=%d | excluded=%d | ms=%.2f",
                len(rows),
                len(list(exclude_fields or [])),
                (_t1 - _t0) * 1000.0,
            )
        except Exception:
            pass
        return rows

    first_col = ordered_columns[0]
    seen_first: set[Any] = set(_freeze_key(r.get(first_col)) for r in rows)
    if len(seen_first) == len(rows):
        _t1 = time.perf_counter()
        try:
            logging.getLogger(__name__).info(
                "maybe_group_rows: early-stop one-to-one | rows=%d | col=%s | ms=%.2f",
                len(rows),
                first_col,
                (_t1 - _t0) * 1000.0,
            )
        except Exception:
            pass
        return rows

    result = _group_rows_recursive(
        rows=rows,
        ordered_columns=ordered_columns,
        start_index=0,
    )
    _t1 = time.perf_counter()
    try:
        logging.getLogger(__name__).info(
            "maybe_group_rows: grouped | rows=%d | depth=%d | ms=%.2f",
            len(rows),
            len(ordered_columns),
            (_t1 - _t0) * 1000.0,
        )
    except Exception:
        pass
    return result


# --------------- Dump helpers relocated from token_helpers ------------------


def iter_unique_values_via_groups(context: str, column: str) -> List[Any]:
    """Return a list of unique values for `column` using unify.get_groups.

    Accept diverse backend return shapes (dict or list)."""
    try:
        groups = unify.get_groups(context=context, key=column)
    except Exception:
        # Fallback: try alternate param name
        try:
            groups = unify.get_groups(context=context, field=column)
        except Exception:
            groups = None
    vals: List[Any] = []
    if isinstance(groups, dict):
        try:
            vals = list(groups.values())
        except Exception:
            vals = []
    elif isinstance(groups, list):
        # Could be list of unique values or list of dicts
        if groups and not isinstance(groups[0], dict):
            vals = list(groups)
        else:
            try:
                vals = [g.get(column) for g in groups if isinstance(g, dict)]
            except Exception:
                vals = []
    return [v for v in vals if v is not None]


def read_all_rows(context: str, *, limit: int = 1000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    limit = max(50, int(limit))
    _t0 = time.perf_counter()
    while True:
        try:
            batch = unify.get_logs(
                context=context,
                offset=offset,
                limit=limit,
                exclude_fields=list_private_fields(context),
            )
        except Exception:
            break
        if not batch:
            break
        for lg in batch:
            try:
                rows.append(getattr(lg, "entries", {}))
            except Exception:
                continue
        offset += len(batch)
    _t1 = time.perf_counter()
    try:
        logging.getLogger(__name__).info(
            "read_all_rows: context=%s | rows=%d | ms=%.2f",
            context,
            len(rows),
            (_t1 - _t0) * 1000.0,
        )
    except Exception:
        pass
    return rows


def build_grouped_dump_payload(
    table_to_ctx: Dict[str, str],
    selected_tables: List[str],
    *,
    limit: int = 1000,
) -> Tuple[str, Dict[str, int]]:
    from unity.common.token_utils import count_tokens_per_utf_byte

    _t0 = time.perf_counter()
    dump: Dict[str, Any] = {}
    per_table_tokens: Dict[str, int] = {}
    total_rows = 0
    for t in selected_tables:
        ctx = table_to_ctx[t]
        rows = read_all_rows(ctx, limit=limit)
        total_rows += len(rows)
        _g0 = time.perf_counter()
        grouped_rows = maybe_group_rows(
            rows=rows,
            exclude_fields=list_private_fields(ctx),
            enabled=True,
        )
        _g1 = time.perf_counter()
        dump[t] = grouped_rows
        try:
            tbl_json = json.dumps(grouped_rows, ensure_ascii=False)
        except Exception:
            tbl_json = json.dumps(grouped_rows, default=str, ensure_ascii=False)
        per_table_tokens[t] = count_tokens_per_utf_byte(tbl_json)
        try:
            logging.getLogger(__name__).info(
                "build_grouped_dump_payload: table=%s | rows=%d | group_ms=%.2f | tokens=%d",
                t,
                len(rows),
                (_g1 - _g0) * 1000.0,
                per_table_tokens[t],
            )
        except Exception:
            pass

    try:
        payload = json.dumps(dump, ensure_ascii=False)
    except Exception:
        payload = json.dumps(dump, default=str, ensure_ascii=False)
    _t1 = time.perf_counter()
    try:
        logging.getLogger(__name__).info(
            "build_grouped_dump_payload: tables=%d | total_rows=%d | ms=%.2f",
            len(selected_tables),
            total_rows,
            (_t1 - _t0) * 1000.0,
        )
    except Exception:
        pass
    return payload, per_table_tokens
