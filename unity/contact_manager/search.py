from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import unify

from ..common.filter_utils import normalize_filter_expr
from ..common.search_utils import table_search_top_k
from .types.contact import Contact

# A body mints at most two system overlays — one for ``self`` and one for
# ``boss`` — so the row filter only ever needs to pull back that handful.
# A small ceiling keeps the per-search hop cheap and still leaves headroom
# for a future third system relationship.
_SYSTEM_OVERLAY_LIMIT: int = 8


def _pack_contacts_result(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    contacts_list = [Contact(**r) for r in rows]
    if not contacts_list:
        return {"contacts": []}
    fwd = Contact.shorthand_map()
    inv = Contact.shorthand_inverse_map()
    return {
        "contact_keys_to_shorthand": fwd,
        "contacts": contacts_list,
        "shorthand_to_contact_keys": inv,
    }


def filter_contacts(
    self,
    *,
    filter: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict[str, Any]:
    eff_limit = limit
    if isinstance(filter, str):
        if re.fullmatch(r"\s*contact_id\s*==\s*\d+\s*", filter):
            eff_limit = min(eff_limit, 1)
        else:
            unique_eq_patterns = (
                r"\s*email_address\s*==\s*(['\"])\S.*?\1\s*",
                r"\s*phone_number\s*==\s*(['\"])\S.*?\1\s*",
                r"\s*whatsapp_number\s*==\s*(['\"])\S.*?\1\s*",
                r"\s*discord_id\s*==\s*(['\"])\S.*?\1\s*",
            )
            if any(re.fullmatch(p, filter) for p in unique_eq_patterns):
                eff_limit = min(eff_limit, 1)
            else:
                m = re.fullmatch(
                    r"\s*contact_id\s*in\s*\[\s*([0-9,\s]+)\s*\]\s*",
                    filter,
                )
                if m:
                    count_ids = len(re.findall(r"\d+", m.group(1)))
                    if count_ids > 0:
                        eff_limit = min(eff_limit, count_ids)

    from_fields = list(self._BUILTIN_FIELDS)
    if getattr(self, "_known_custom_fields", None):  # type: ignore[attr-defined]
        from_fields.extend(sorted(self._known_custom_fields))  # type: ignore[attr-defined]
    normalized = normalize_filter_expr(filter)
    logs = unify.get_logs(
        context=self._ctx,
        filter=normalized,
        offset=offset,
        limit=eff_limit,
        from_fields=from_fields,
    )
    try:
        for lg in logs:
            self._data_store.put(lg.entries)
    except Exception:
        pass
    rows = [lg.entries for lg in logs]
    return _pack_contacts_result(rows)


def _system_contact_row_filter(self) -> str:
    """Build the row filter that hides system contacts from semantic search.

    System contacts (this body's ``self`` and ``boss`` rows) are
    identified from two independent signals, combined with AND so a
    positive overlay signal is authoritative even if the shared-row
    ``is_system`` flag drifts:

    1. Overlay signal — every ``{user}/{assistant}/ContactMembership``
       row whose ``relationship`` is ``"self"`` or ``"boss"`` contributes
       its ``contact_id`` to the excluded set. This is the single source
       of truth for which contacts this body considers "system".
    2. Shared-row signal — ``is_system != True`` on the ``Contacts``
       row. This covers bodies that have not yet materialized their
       ``self`` / ``boss`` overlay.
    """
    excluded_ids: list[int] = []
    try:
        from .settings import RELATIONSHIP_BOSS, RELATIONSHIP_SELF

        overlay_rows = unify.get_logs(
            context=self._membership_ctx,
            filter=(
                f"relationship == '{RELATIONSHIP_SELF}' or "
                f"relationship == '{RELATIONSHIP_BOSS}'"
            ),
            limit=_SYSTEM_OVERLAY_LIMIT,
            from_fields=["contact_id", "relationship"],
        )
    except Exception:
        overlay_rows = []

    for lg in overlay_rows:
        try:
            excluded_ids.append(int(lg.entries.get("contact_id")))
        except Exception:
            continue

    clauses: list[str] = ["is_system != True"]
    if excluded_ids:
        ids_expr = ", ".join(str(cid) for cid in excluded_ids)
        clauses.append(f"contact_id not in [{ids_expr}]")
    return " and ".join(clauses)


def search_contacts(
    self,
    *,
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
) -> Dict[str, Any]:
    allowed_fields = list(self._BUILTIN_FIELDS)
    if getattr(self, "_known_custom_fields", None):  # type: ignore[attr-defined]
        allowed_fields.extend(sorted(self._known_custom_fields))  # type: ignore[attr-defined]

    system_filter = _system_contact_row_filter(self)
    filled = table_search_top_k(
        self._ctx,
        references,
        k=k,
        allowed_fields=allowed_fields,
        row_filter=system_filter,
        unique_id_field="contact_id",
    )
    try:
        for r in filled:
            self._data_store.put(r)
    except Exception:
        pass
    return _pack_contacts_result(filled)
