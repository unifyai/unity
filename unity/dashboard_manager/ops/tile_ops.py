"""Tile operations for DashboardManager.

Helper functions for building tile records and validating data bindings.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import unify

from unity.common.context_registry import ContextRegistry
from unity.common.join_utils import rewrite_join_paths
from unity.dashboard_manager.types.tile import (
    FilterBinding,
    JoinBinding,
    JoinReduceBinding,
    ReduceBinding,
    TileRecordRow,
)

if TYPE_CHECKING:
    from unity.data_manager.base import BaseDataManager

logger = logging.getLogger(__name__)

AnyBinding = Union[FilterBinding, ReduceBinding, JoinBinding, JoinReduceBinding]

_JS_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_$][a-zA-Z0-9_$]*$")


# ---------------------------------------------------------------------------
# Context resolution for data bindings
# ---------------------------------------------------------------------------


def _match_context(path: str, base: str, known: set[str]) -> str:
    """Match an actor-provided context path to its fully qualified form.

    Resolution cascade:

    1. **Exact match** -- path is already fully qualified.
    2. **Base-prefixed** -- ``{base}/{path}`` exists (handles root-relative
       forms like ``Data/Sales`` or ``Contacts``).
    3. **Suffix match** -- exactly one known context ends with ``/{path}``
       (handles manager-relative forms like ``MidlandHeart/Repairs/...``).
    4. **No match** -- raises ``ValueError``.

    Raises ``ValueError`` on ambiguous suffix matches (more than one
    candidate) so the actor must provide a more specific path.
    """
    path = path.strip().lstrip("/")
    if not path:
        raise ValueError("Empty context path")

    if path in known:
        return path

    candidate = f"{base}/{path}"
    if candidate in known:
        return candidate

    suffix = f"/{path}"
    matches = [c for c in known if c.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous context '{path}' matches multiple contexts: {matches}",
        )
    raise ValueError(f"No context found matching '{path}'")


def resolve_binding_contexts(
    bindings: List[AnyBinding],
) -> List[AnyBinding]:
    """Resolve all context paths in *bindings* to fully qualified form.

    Fetches all contexts scoped to the current user/assistant prefix via
    ``unify.get_contexts(prefix=base)`` and resolves each binding's context
    path(s) through :func:`_match_context`.  For join bindings, table
    references embedded in ``join_expr`` and ``select`` keys are rewritten
    to match the resolved table names.

    Falls through gracefully (returning *bindings* unchanged) when no base
    context is available (offline / test scenarios).
    """
    base = ContextRegistry._base_context
    if not base:
        active = unify.get_active_context()
        base = (active or {}).get("read", "")
    if not base:
        return bindings

    known = set(unify.get_contexts(prefix=base).keys())
    if not known:
        return bindings

    result: List[AnyBinding] = []
    for b in bindings:
        if isinstance(b, (FilterBinding, ReduceBinding)):
            resolved = _match_context(b.context, base, known)
            b = b.model_copy(update={"context": resolved})
        elif isinstance(b, (JoinBinding, JoinReduceBinding)):
            resolved_tables = [_match_context(t, base, known) for t in b.tables]
            new_expr, new_sel = rewrite_join_paths(
                list(b.tables),
                resolved_tables,
                b.join_expr,
                dict(b.select),
            )
            b = b.model_copy(
                update={
                    "tables": resolved_tables,
                    "join_expr": new_expr,
                    "select": new_sel,
                },
            )
        result.append(b)
    return result


def _contexts_for_binding(binding: AnyBinding) -> List[str]:
    """Extract context paths from a binding (single context or join tables)."""
    if isinstance(binding, (FilterBinding, ReduceBinding)):
        return [binding.context]
    return list(binding.tables)


def _alias_from_context(context: str) -> str:
    """Derive a JS-safe alias from a context path (last segment, sanitised)."""
    segment = context.rsplit("/", 1)[-1]
    alias = re.sub(r"[^a-zA-Z0-9_$]", "_", segment).lower()
    if alias and alias[0].isdigit():
        alias = f"_{alias}"
    return alias or "binding"


def serialize_bindings(bindings: Sequence[AnyBinding]) -> str:
    """Serialize a list of data bindings to a JSON string."""
    return json.dumps(
        [b.model_dump(mode="json") for b in bindings],
        separators=(",", ":"),
    )


def ensure_binding_aliases(
    bindings: Sequence[AnyBinding],
) -> List[AnyBinding]:
    """Return a copy of *bindings* where every binding has an alias.

    Auto-generates aliases from the context path (last segment) for
    single-context bindings, or ``binding_{i}`` for joins.  Raises
    ``ValueError`` on duplicate aliases.
    """
    result: List[AnyBinding] = []
    seen: dict[str, int] = {}
    for i, b in enumerate(bindings):
        alias = b.alias
        if not alias:
            if isinstance(b, (FilterBinding, ReduceBinding)):
                alias = _alias_from_context(b.context)
            else:
                alias = f"binding_{i}"
            b = b.model_copy(update={"alias": alias})
        if not _JS_IDENTIFIER_RE.match(alias):
            raise ValueError(
                f"alias '{alias}' is not a valid JS identifier "
                "(must match [a-zA-Z_$][a-zA-Z0-9_$]*)",
            )
        if alias in seen:
            raise ValueError(
                f"duplicate alias '{alias}' on bindings {seen[alias]} and {i}",
            )
        seen[alias] = i
        result.append(b)
    return result


def validate_on_data(
    on_data: Optional[str],
    data_bindings: Optional[Sequence[AnyBinding]],
) -> None:
    """Enforce runtime constraints on the ``on_data`` / ``data_bindings`` pair.

    Raises ``ValueError`` when the combination is invalid.
    """
    if on_data is None:
        return

    if not on_data.strip():
        raise ValueError("on_data must be non-empty JS code or None")

    if not data_bindings:
        raise ValueError(
            "on_data requires data_bindings -- there is no data to feed "
            "the callback without declared bindings",
        )

    if "UnifyData." in on_data:
        logger.warning(
            "on_data contains 'UnifyData.' -- this is likely a mistake. "
            "on_data receives already-fetched data; bridge calls inside "
            "on_data would be redundant.",
        )


def build_tile_record_row(
    token: str,
    html: str,
    title: str,
    description: Optional[str] = None,
    data_bindings: Optional[Sequence[AnyBinding]] = None,
    on_data: Optional[str] = None,
) -> TileRecordRow:
    """Build a TileRecordRow ready for insertion into the Unify context."""
    has_bindings = bool(data_bindings)
    binding_contexts: Optional[str] = None
    bindings_json: Optional[str] = None
    if data_bindings:
        all_ctxs: List[str] = []
        for b in data_bindings:
            all_ctxs.extend(_contexts_for_binding(b))
        binding_contexts = ",".join(dict.fromkeys(all_ctxs))
        bindings_json = serialize_bindings(data_bindings)
    now = datetime.now(timezone.utc).isoformat()

    return TileRecordRow(
        token=token,
        title=title,
        description=description,
        html_content=html,
        has_data_bindings=has_bindings,
        data_binding_contexts=binding_contexts,
        on_data_script=on_data,
        data_bindings_json=bindings_json,
        created_at=now,
        updated_at=now,
    )


def validate_data_bindings(
    data_bindings: Optional[List[AnyBinding]],
) -> Optional[List[AnyBinding]]:
    """Validate and clean data bindings. Returns typed list or None."""
    if not data_bindings:
        return None
    cleaned: List[AnyBinding] = []
    for b in data_bindings:
        if isinstance(b, (FilterBinding, ReduceBinding)):
            ctx = b.context.strip()
            if not ctx:
                continue
        elif isinstance(b, (JoinBinding, JoinReduceBinding)):
            if not b.tables or not any(t.strip() for t in b.tables):
                continue
        else:
            continue
        cleaned.append(b)
    return cleaned or None


def _filter_binding_has_query_params(binding: FilterBinding) -> bool:
    """True if a filter binding declares params beyond just ``context``."""
    return (
        binding.filter is not None
        or binding.columns is not None
        or binding.exclude_columns is not None
        or binding.order_by is not None
        or binding.descending is not False
        or binding.limit is not None
        or binding.offset is not None
        or binding.group_by is not None
    )


def verify_data_bindings(
    data_bindings: List[AnyBinding],
    dm: BaseDataManager,
) -> None:
    """Dry-run each binding through the corresponding DataManager method.

    Dispatches to the appropriate DM primitive per binding type:

    - ``FilterBinding``  -> ``dm.filter(limit=5)``
    - ``ReduceBinding``  -> ``dm.reduce(...)``
    - ``JoinBinding``    -> ``dm.filter_join(result_limit=5)``
    - ``JoinReduceBinding`` -> ``dm.reduce_join(...)``

    Called inside ``create_tile`` before the tile is stored.  If any
    binding references an invalid context, column, or expression,
    this raises and ``create_tile`` returns ``TileResult(error=...)``.
    """
    for binding in data_bindings:
        if isinstance(binding, FilterBinding):
            if not _filter_binding_has_query_params(binding):
                continue
            dm.filter(
                binding.context,
                filter=binding.filter,
                columns=binding.columns,
                exclude_columns=binding.exclude_columns,
                order_by=binding.order_by,
                descending=binding.descending,
                limit=5,
            )

        elif isinstance(binding, ReduceBinding):
            dm.reduce(
                binding.context,
                metric=binding.metric,
                columns=binding.columns,
                filter=binding.filter,
                group_by=binding.group_by,
            )

        elif isinstance(binding, JoinBinding):
            dm.filter_join(
                tables=binding.tables,
                join_expr=binding.join_expr,
                select=binding.select,
                mode=binding.mode,
                left_where=binding.left_where,
                right_where=binding.right_where,
                result_where=binding.result_where,
                result_limit=5,
                result_offset=0,
            )

        elif isinstance(binding, JoinReduceBinding):
            dm.reduce_join(
                tables=binding.tables,
                join_expr=binding.join_expr,
                select=binding.select,
                metric=binding.metric,
                columns=binding.columns,
                mode=binding.mode,
                left_where=binding.left_where,
                right_where=binding.right_where,
                result_where=binding.result_where,
                group_by=binding.group_by,
            )
