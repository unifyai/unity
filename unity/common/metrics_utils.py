from __future__ import annotations

from typing import Any, Dict, List, Union, Optional

import unify

from .filter_utils import normalize_filter_expr

SUPPORTED_REDUCTION_METRICS: set[str] = {
    "count",
    "sum",
    "mean",
    "var",
    "std",
    "min",
    "max",
    "median",
    "mode",
}


def _normalize_filter(
    flt: Optional[Union[str, Dict[str, str]]],
) -> Optional[Union[str, Dict[str, str]]]:
    """
    Normalise a string or per-key filter mapping using the shared helper.

    This keeps reduction tools consistent with other filter-capable tools that
    rely on :func:`normalize_filter_expr`.
    """
    if flt is None:
        return None
    if isinstance(flt, str):
        return normalize_filter_expr(flt)
    # Dict form: {key_name: filter_expr_for_that_key}
    out: Dict[str, str] = {}
    for k, expr in flt.items():
        out[str(k)] = normalize_filter_expr(expr)
    return out


def reduce_logs(
    *,
    context: str,
    metric: str,
    keys: Union[str, List[str]],
    filter: Optional[Union[str, Dict[str, str]]] = None,
    group_by: Optional[Union[str, List[str]]] = None,
) -> Any:
    """
    Compute one or more reduction metrics over a Unify context.

    This is a thin convenience wrapper around :func:`unify.get_logs_metric`
    that enforces a common contract for manager-level ``reduce`` tools.

    Parameters
    ----------
    context : str
        Fully-qualified Unify context to aggregate over.
    metric : str
        Reduction metric to compute. Supported values (case-insensitive) are:
        ``\"sum\"``, ``\"mean\"``, ``\"var\"``, ``\"std\"``, ``\"min\"``,
        ``\"max\"``, ``\"median\"``, ``\"mode\"``, and ``\"count\"``.
    keys : str | list[str]
        Field name(s) to compute the metric for. A single column name (string)
        produces a scalar result (when ``group_by`` is not used); a list of
        column names computes the same metric independently for each key and
        returns a ``{key -> value}`` mapping.
    filter : str | dict[str, str] | None, default None
        Optional filter expression(s) to restrict which rows contribute to the
        metric. Mirrors the behaviour of :func:`unify.get_logs_metric`:

        * When a string, the same expression is applied for all keys.
        * When a dict, each key maps to its own filter expression.

        In both cases, expressions are normalised via
        :func:`normalize_filter_expr` for consistency with other tools.
    group_by : str | list[str] | None, default None
        Optional field(s) to group by. Use a single column name for a single
        grouping level, or a list like ``[\"status\", \"queue_id\"]`` to group
        hierarchically in the given order. When provided, the return value
        follows the grouped forms described in the Unify ``get_logs_metric``
        docs (nested ``dict`` structures keyed by group values).

    Returns
    -------
    Any
        The metric value(s) produced by :func:`unify.get_logs_metric`:

        * Single key, no grouping  → scalar (float/int/str/bool).
        * Multiple keys, no grouping → ``dict[key -> scalar]``.
        * With grouping             → nested ``dict`` keyed by group values.

    Raises
    ------
    ValueError
        If ``metric`` is not one of the supported reduction metrics.
    """
    metric_norm = metric.strip().lower()
    if metric_norm not in SUPPORTED_REDUCTION_METRICS:
        raise ValueError(
            f"Unsupported reduction metric {metric!r}. "
            f"Supported metrics are: {sorted(SUPPORTED_REDUCTION_METRICS)}",
        )

    normalized_filter = _normalize_filter(filter)

    return unify.get_logs_metric(
        metric=metric_norm,
        key=keys,
        filter=normalized_filter,
        context=context,
        group_by=group_by,
    )
