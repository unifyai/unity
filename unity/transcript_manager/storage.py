from __future__ import annotations

from typing import Dict, Optional, Set

import unify

from ..common.context_store import TableStore
from ..common.model_to_fields import model_to_fields
from .types.message import Message


def provision_storage(self) -> None:
    """Ensure contexts, fields, helper columns and local caches exist (idempotent)."""
    # Ensure transcripts context and fields deterministically
    self._store = TableStore(
        self._transcripts_ctx,
        unique_keys={"message_id": "int"},
        auto_counting={"message_id": None, "exchange_id": None},
        description=(
            "List of *all* timestamped messages sent between *all* contacts across *all* mediums."
        ),
        fields=model_to_fields(Message),
    )
    self._store.ensure_context()

    # Exchanges context: one row per exchange_id with optional metadata
    self._exchanges_store = TableStore(
        self._exchanges_ctx,
        unique_keys={"exchange_id": "int"},
        description="One row per conversation exchange/thread with optional metadata.",
        fields={
            "exchange_id": {
                "type": "int",
                "description": "Unique identifier for the exchange/thread",
            },
            "metadata": {
                "type": "dict",
                "description": "Arbitrary exchange-level metadata (e.g., URLs, external refs)",
            },
            "medium": {
                "type": "string",
                "description": "Communication medium for the exchange (same semantics as Message.medium)",
            },
        },
    )
    self._exchanges_store.ensure_context()

    # Ensure a private `_metadata` column exists (dict, mutable)
    try:
        existing_fields = unify.get_fields(context=self._transcripts_ctx)
        if "_metadata" not in existing_fields:
            unify.create_fields(
                {
                    "_metadata": {
                        "type": "dict",
                        "mutable": True,
                        "description": "Internal, non user-facing metadata for infrastructure.",
                    },
                },
                context=self._transcripts_ctx,
            )
    except Exception:
        # Non-fatal; logging will still work without the helper if backend creates implicitly
        pass

    # Update columns cache best-effort
    try:
        self._columns_cache_all = dict(self._store.get_columns())
    except Exception:
        self._columns_cache_all = {}


def get_columns(self) -> Dict[str, str]:
    """Return {column_name: column_type} for the transcripts table."""
    # Serve from the in-process cache when available; otherwise fetch once
    # and remember for subsequent reads within this manager's lifetime.
    if getattr(self, "_columns_cache_all", None):
        return dict(self._columns_cache_all)
    cols = self._store.get_columns()
    try:
        self._columns_cache_all = dict(cols)
    except Exception:
        pass
    return cols


def list_columns(
    self,
    *,
    include_types: bool = True,
    include_private: bool = False,
) -> Dict[str, str] | list[str]:
    """Return the list of available columns in the transcripts table."""
    cols = get_columns(self)
    if not include_private:
        cols = {k: v for k, v in cols.items() if not str(k).startswith("_")}
    return cols if include_types else list(cols)


def num_messages(self) -> int:
    """Return the total number of messages in transcripts."""
    ret = unify.get_logs_metric(
        metric="count",
        key="message_id",
        context=self._transcripts_ctx,
    )
    if ret is None:
        return 0
    return int(ret)


def clear(self) -> None:
    """Best-effort deletion of both contexts and re-provision storage."""
    # Best-effort deletion of both contexts
    try:
        unify.delete_context(self._transcripts_ctx)
    except Exception:
        pass
    try:
        unify.delete_context(self._exchanges_ctx)
    except Exception:
        pass

    # Reset local cached state
    try:
        self._columns_cache_all = {}
    except Exception:
        pass

    # Drop ensure memo then re-provision via shared helper
    try:
        from ..common.context_store import TableStore as _TS  # local import

        try:
            _TS._ENSURED.discard((unify.active_project(), self._transcripts_ctx))
        except Exception:
            pass
        try:
            _TS._ENSURED.discard((unify.active_project(), self._exchanges_ctx))
        except Exception:
            pass
    except Exception:
        pass

    # Recreate contexts and required columns via shared helper
    provision_storage(self)

    # Verify both contexts become visible before returning
    try:
        import time as _time  # local import

        for _ in range(3):
            try:
                unify.get_fields(context=self._transcripts_ctx)
                break
            except Exception:
                _time.sleep(0.05)
    except Exception:
        pass

    try:
        import time as _time  # local import

        for _ in range(3):
            try:
                unify.get_fields(context=self._exchanges_ctx)
                break
            except Exception:
                _time.sleep(0.05)
    except Exception:
        pass


def ensure_exchanges_records(
    self,
    exchange_ids: Set[int],
    *,
    eid_to_medium: Optional[Dict[int, str]] = None,
) -> None:
    """Idempotently create rows in the Exchanges context for given ids."""
    if not exchange_ids:
        return
    try:
        ids_expr = ", ".join(str(i) for i in sorted(exchange_ids))
        existing: set[int] = set()
        try:
            rows = unify.get_logs(
                context=self._exchanges_ctx,
                filter=f"exchange_id in [{ids_expr}]",
                from_fields=["exchange_id"],
                limit=len(exchange_ids),
            )
            for lg in rows or []:
                try:
                    existing.add(int(lg.entries.get("exchange_id")))
                except Exception:
                    continue
        except Exception:
            existing = set()

        missing = [eid for eid in exchange_ids if eid not in existing]
        for eid in missing:
            try:
                unify.log(
                    context=self._exchanges_ctx,
                    exchange_id=int(eid),
                    metadata={},
                    medium=(eid_to_medium or {}).get(int(eid), ""),
                    new=True,
                    mutable=True,
                    params={},
                )
            except Exception:
                # Ignore duplicates or backend races
                pass
    except Exception:
        # Defensive: never propagate to caller
        pass
