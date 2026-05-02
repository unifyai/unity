from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import unify

from ..common.filter_utils import normalize_filter_expr
from ..common.search_utils import table_search_top_k
from .types.contact import Contact


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
    context: Optional[str] = None,
    data_store: Any = None,
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
        context=context or self._ctx,
        filter=normalized,
        offset=offset,
        limit=eff_limit,
        from_fields=from_fields,
    )
    store = data_store or self._data_store
    try:
        for lg in logs:
            store.put(lg.entries)
    except Exception:
        pass
    rows = [lg.entries for lg in logs]
    return _pack_contacts_result(rows)


def search_contacts(
    self,
    *,
    references: Optional[Dict[str, str]] = None,
    k: int = 10,
    context: Optional[str] = None,
    data_store: Any = None,
) -> Dict[str, Any]:
    allowed_fields = list(self._BUILTIN_FIELDS)
    if getattr(self, "_known_custom_fields", None):  # type: ignore[attr-defined]
        allowed_fields.extend(sorted(self._known_custom_fields))  # type: ignore[attr-defined]

    from ..session_details import SESSION_DETAILS

    system_filter = (
        f"contact_id != {int(SESSION_DETAILS.self_contact_id)} "
        f"and contact_id != {int(SESSION_DETAILS.boss_contact_id)}"
    )
    filled = table_search_top_k(
        context or self._ctx,
        references,
        k=k,
        allowed_fields=allowed_fields,
        row_filter=system_filter,
        unique_id_field="contact_id",
    )
    store = data_store or self._data_store
    try:
        for r in filled:
            store.put(r)
    except Exception:
        pass
    return _pack_contacts_result(filled)
