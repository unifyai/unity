"""
Generic hash-based seed data sync utility.

Syncs code-defined records to DB-backed state manager contexts.
Each manager provides a thin adapter describing its natural key,
ID field, and CRUD operations.  The generic ``sync_seed_data()``
handles hash comparison, diffing, and create/update/delete.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable, TYPE_CHECKING

import unify

if TYPE_CHECKING:
    from unity.customization.clients import ResolvedCustomization

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def _record_hash(record: dict, exclude_fields: set[str]) -> str:
    payload = {k: v for k, v in sorted(record.items()) if k not in exclude_fields}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode(),
    ).hexdigest()[:16]


def _aggregate_hash(
    records: list[dict],
    natural_key_fn: Callable[[dict], str],
    exclude_fields: set[str],
) -> str:
    if not records:
        return ""
    pairs = sorted(
        (natural_key_fn(r), _record_hash(r, exclude_fields)) for r in records
    )
    combined = "|".join(f"{k}:{h}" for k, h in pairs)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Meta store (persists per-manager hashes to DB)
# ---------------------------------------------------------------------------


class SeedMetaStore:
    """Read/write per-manager seed hashes in a ``SeedData/Meta`` context."""

    def __init__(self) -> None:
        self._ctx: str | None = None

    def _ensure_ctx(self) -> str:
        if self._ctx is not None:
            return self._ctx

        active = unify.get_active_context()["read"]
        self._ctx = f"{active}/SeedData/Meta"
        try:
            unify.create_context(self._ctx)
        except Exception:
            pass
        return self._ctx

    def get_hash(self, manager_key: str) -> str:
        ctx = self._ensure_ctx()
        logs = unify.get_logs(
            context=ctx,
            filter=f"manager_key == '{manager_key}'",
            limit=1,
        )
        if not logs:
            return ""
        return logs[0].entries.get("seed_hash", "")

    def set_hash(self, manager_key: str, seed_hash: str) -> None:
        ctx = self._ensure_ctx()
        logs = unify.get_logs(
            context=ctx,
            filter=f"manager_key == '{manager_key}'",
            limit=1,
        )
        if logs:
            unify.update_logs(
                logs=[logs[0].id],
                context=ctx,
                entries=[{"seed_hash": seed_hash}],
                overwrite=True,
            )
        else:
            unify.log(context=ctx, manager_key=manager_key, seed_hash=seed_hash)


# ---------------------------------------------------------------------------
# Generic sync
# ---------------------------------------------------------------------------


def sync_seed_data(
    *,
    manager_key: str,
    source_records: list[dict],
    natural_key_fn: Callable[[dict], str],
    get_existing_fn: Callable[[], list[dict]],
    create_fn: Callable[[dict], Any],
    update_fn: Callable[[int, dict], Any] | None,
    delete_fn: Callable[[int], Any] | None,
    id_field: str,
    meta_store: SeedMetaStore,
) -> bool:
    """Sync code-defined records to a DB-backed state manager.

    Returns True if any changes were made.
    """
    exclude_fields = {id_field}
    expected_hash = _aggregate_hash(source_records, natural_key_fn, exclude_fields)
    current_hash = meta_store.get_hash(manager_key)

    if current_hash == expected_hash:
        logger.debug("Seed data for %s unchanged, skipping sync", manager_key)
        return False

    logger.info(
        "Seed data for %s changed (current=%s, expected=%s), syncing...",
        manager_key,
        current_hash,
        expected_hash,
    )

    existing = get_existing_fn()
    existing_by_key: dict[str, dict] = {}
    for rec in existing:
        try:
            existing_by_key[natural_key_fn(rec)] = rec
        except Exception:
            pass

    source_by_key = {natural_key_fn(r): r for r in source_records}
    processed_keys: set[str] = set()

    for key, src in source_by_key.items():
        processed_keys.add(key)
        if key in existing_by_key:
            db_rec = existing_by_key[key]
            src_hash = _record_hash(src, exclude_fields)
            db_hash = _record_hash(
                {k: v for k, v in db_rec.items() if k not in exclude_fields},
                exclude_fields,
            )
            if src_hash != db_hash and update_fn is not None:
                db_id = db_rec.get(id_field)
                if db_id is not None:
                    logger.info("Updating %s record: %s", manager_key, key)
                    update_fn(db_id, src)
        else:
            logger.info("Creating %s record: %s", manager_key, key)
            create_fn(src)

    if delete_fn is not None:
        for key, db_rec in existing_by_key.items():
            if key not in processed_keys:
                db_id = db_rec.get(id_field)
                if db_id is not None:
                    logger.info("Deleting %s record: %s", manager_key, key)
                    delete_fn(db_id)

    meta_store.set_hash(manager_key, expected_hash)
    return True


# ---------------------------------------------------------------------------
# Per-manager adapters
# ---------------------------------------------------------------------------


def _sync_contacts(records: list[dict], meta: SeedMetaStore) -> bool:
    if not records:
        return False
    from unity.manager_registry import ManagerRegistry

    cm = ManagerRegistry.get_contact_manager()

    def natural_key(r: dict) -> str:
        return f"{r.get('first_name') or ''}|{r.get('surname') or ''}".lower()

    def get_existing() -> list[dict]:
        result = cm.filter_contacts(limit=1000)
        contacts = result.get("contacts", [])
        return [c.model_dump() if hasattr(c, "model_dump") else c for c in contacts]

    def create(rec: dict) -> Any:
        return cm.create_contact(**{k: v for k, v in rec.items() if k != "contact_id"})

    def update(contact_id: int, rec: dict) -> Any:
        fields = {k: v for k, v in rec.items() if k != "contact_id"}
        return cm.update_contact(contact_id=contact_id, **fields)

    return sync_seed_data(
        manager_key="contacts",
        source_records=records,
        natural_key_fn=natural_key,
        get_existing_fn=get_existing,
        create_fn=create,
        update_fn=update,
        delete_fn=None,
        id_field="contact_id",
        meta_store=meta,
    )


def _sync_guidance(records: list[dict], meta: SeedMetaStore) -> bool:
    if not records:
        return False
    from unity.manager_registry import ManagerRegistry

    gm = ManagerRegistry.get_guidance_manager()

    def natural_key(r: dict) -> str:
        return str(r.get("title", ""))

    def get_existing() -> list[dict]:
        entries = gm.filter(limit=1000)
        return [g.model_dump() if hasattr(g, "model_dump") else g for g in entries]

    def create(rec: dict) -> Any:
        return gm.add_guidance(**{k: v for k, v in rec.items() if k != "guidance_id"})

    def update(guidance_id: int, rec: dict) -> Any:
        fields = {k: v for k, v in rec.items() if k != "guidance_id"}
        return gm.update_guidance(guidance_id=guidance_id, **fields)

    def delete(guidance_id: int) -> Any:
        return gm.delete_guidance(guidance_id=guidance_id)

    return sync_seed_data(
        manager_key="guidance",
        source_records=records,
        natural_key_fn=natural_key,
        get_existing_fn=get_existing,
        create_fn=create,
        update_fn=update,
        delete_fn=delete,
        id_field="guidance_id",
        meta_store=meta,
    )


def _sync_secrets(records: list[dict], meta: SeedMetaStore) -> bool:
    if not records:
        return False
    from unity.manager_registry import ManagerRegistry

    sm = ManagerRegistry.get_secret_manager()

    def natural_key(r: dict) -> str:
        return str(r.get("name", ""))

    def get_existing() -> list[dict]:
        keys = sm._list_secret_keys()
        result = []
        for name in keys:
            logs = unify.get_logs(
                context=sm._ctx,
                filter=f"name == '{name}'",
                limit=1,
                from_fields=["secret_id", "name", "value", "description"],
            )
            if logs:
                result.append(logs[0].entries)
        return result

    def create(rec: dict) -> Any:
        return sm._create_secret(
            name=rec["name"],
            value=rec["value"],
            description=rec.get("description"),
        )

    def update(_secret_id: int, rec: dict) -> Any:
        return sm._update_secret(
            name=rec["name"],
            value=rec.get("value"),
            description=rec.get("description"),
        )

    def delete(_secret_id: int) -> Any:
        logs = unify.get_logs(
            context=sm._ctx,
            filter=f"secret_id == {_secret_id}",
            limit=1,
            from_fields=["name"],
        )
        if logs:
            return sm._delete_secret(name=logs[0].entries["name"])

    return sync_seed_data(
        manager_key="secrets",
        source_records=records,
        natural_key_fn=natural_key,
        get_existing_fn=get_existing,
        create_fn=create,
        update_fn=update,
        delete_fn=delete,
        id_field="secret_id",
        meta_store=meta,
    )


def _sync_blacklist(records: list[dict], meta: SeedMetaStore) -> bool:
    if not records:
        return False
    from unity.manager_registry import ManagerRegistry

    bm = ManagerRegistry.get_blacklist_manager()

    def natural_key(r: dict) -> str:
        return f"{r.get('medium', '')}|{r.get('contact_detail', '')}"

    def get_existing() -> list[dict]:
        result = bm.filter_blacklist(limit=1000)
        entries = result.get("entries", [])
        return [e.model_dump() if hasattr(e, "model_dump") else e for e in entries]

    def create(rec: dict) -> Any:
        return bm.create_blacklist_entry(
            medium=rec["medium"],
            contact_detail=rec["contact_detail"],
            reason=rec.get("reason", ""),
        )

    def update(blacklist_id: int, rec: dict) -> Any:
        return bm.update_blacklist_entry(
            blacklist_id=blacklist_id,
            medium=rec.get("medium"),
            contact_detail=rec.get("contact_detail"),
            reason=rec.get("reason"),
        )

    def delete(blacklist_id: int) -> Any:
        return bm.delete_blacklist_entry(blacklist_id=blacklist_id)

    return sync_seed_data(
        manager_key="blacklist",
        source_records=records,
        natural_key_fn=natural_key,
        get_existing_fn=get_existing,
        create_fn=create,
        update_fn=update,
        delete_fn=delete,
        id_field="blacklist_id",
        meta_store=meta,
    )


def _sync_knowledge(tables: dict[str, dict], meta: SeedMetaStore) -> bool:
    """Sync knowledge seed data.

    ``tables`` is a dict like::

        {
            "Companies": {
                "description": "Known companies",
                "columns": {"company_name": "str", "industry": "str"},
                "seed_key": "company_name",
                "rows": [{"company_name": "Colliers", "industry": "Real Estate"}],
            },
        }
    """
    if not tables:
        return False
    from unity.manager_registry import ManagerRegistry

    km = ManagerRegistry.get_knowledge_manager()

    any_changed = False
    for table_name, table_spec in tables.items():
        rows = table_spec.get("rows", [])
        if not rows:
            continue
        seed_key = table_spec.get("seed_key")
        if not seed_key:
            logger.warning("Knowledge table %s has no seed_key, skipping", table_name)
            continue

        existing_tables = km._tables_overview()
        if table_name not in existing_tables:
            km._create_table(
                name=table_name,
                description=table_spec.get("description"),
                columns=table_spec.get("columns"),
            )

        def make_natural_key(r: dict, _sk: str = seed_key) -> str:
            return str(r.get(_sk, ""))

        def get_existing(_tn: str = table_name) -> list[dict]:
            result = km._filter(tables=[_tn], limit=1000)
            return result.get(_tn, [])

        unique_key = "row_id"
        if table_name in existing_tables:
            tbl_info = existing_tables[table_name]
            if isinstance(tbl_info, dict) and "unique_key" in tbl_info:
                unique_key = tbl_info["unique_key"]

        def create(rec: dict, _tn: str = table_name, _uk: str = unique_key) -> Any:
            clean = {k: v for k, v in rec.items() if k != _uk}
            return km._add_rows(table=_tn, rows=[clean])

        def update(
            row_id: int,
            rec: dict,
            _tn: str = table_name,
            _uk: str = unique_key,
        ) -> Any:
            clean = {k: v for k, v in rec.items() if k != _uk}
            return km._update_rows(table=_tn, updates={row_id: clean})

        def delete(row_id: int, _tn: str = table_name, _uk: str = unique_key) -> Any:
            return km._delete_rows(filter=f"{_uk} == {row_id}", tables=[_tn])

        changed = sync_seed_data(
            manager_key=f"knowledge/{table_name}",
            source_records=rows,
            natural_key_fn=make_natural_key,
            get_existing_fn=get_existing,
            create_fn=create,
            update_fn=update,
            delete_fn=delete,
            id_field=unique_key,
            meta_store=meta,
        )
        if changed:
            any_changed = True

    return any_changed


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def sync_all_seed_data(resolved: ResolvedCustomization) -> bool:
    """Sync all seed data from a resolved customization to the DB.

    Called from ``_init_managers()`` as an explicit cross-cutting step
    after all managers are constructed but before the Actor is initialized.
    Returns True if any manager was updated.
    """
    has_data = (
        resolved.contacts
        or resolved.guidance
        or resolved.knowledge
        or resolved.secrets
        or resolved.blacklist
    )
    if not has_data:
        return False

    meta = SeedMetaStore()
    changed = False

    if resolved.contacts:
        try:
            changed |= _sync_contacts(resolved.contacts, meta)
        except Exception:
            logger.exception("Failed to sync seed contacts")

    if resolved.guidance:
        try:
            changed |= _sync_guidance(resolved.guidance, meta)
        except Exception:
            logger.exception("Failed to sync seed guidance")

    if resolved.secrets:
        try:
            changed |= _sync_secrets(resolved.secrets, meta)
        except Exception:
            logger.exception("Failed to sync seed secrets")

    if resolved.blacklist:
        try:
            changed |= _sync_blacklist(resolved.blacklist, meta)
        except Exception:
            logger.exception("Failed to sync seed blacklist")

    if resolved.knowledge:
        try:
            changed |= _sync_knowledge(resolved.knowledge, meta)
        except Exception:
            logger.exception("Failed to sync seed knowledge")

    return changed
