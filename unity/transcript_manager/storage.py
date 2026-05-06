from __future__ import annotations

from typing import Dict, Optional, Set

import unify

from ..common.log_utils import log as unity_log
from ..common.context_registry import ContextRegistry
from ..common.context_store import TableStore
from ..common.model_to_fields import model_to_fields
from .types.message import Message


def provision_storage(self) -> None:
    """Ensure contexts, fields, helper columns and local caches exist (idempotent)."""
    # Ensure transcripts context and fields deterministically
    self._store = TableStore(
        self._transcripts_ctx,
        unique_keys={"message_id": "int"},
        auto_counting={"message_id": None},
        description=(
            "List of *all* timestamped messages sent between *all* contacts across *all* mediums."
        ),
        fields=model_to_fields(Message),
    )

    # No local columns cache; always read from TableStore when needed


def get_columns(self) -> Dict[str, str]:
    """Return {column_name: column_type} for the transcripts table."""
    return self._store.get_columns()


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
    total = 0
    for context in self._read_transcript_contexts():
        ret = unify.get_logs_metric(
            metric="count",
            key="message_id",
            context=context,
        )
        if ret is not None:
            total += int(ret)
    return total


def clear(self) -> None:
    """Delete both contexts and re-provision storage."""
    unify.delete_context(self._transcripts_ctx)
    unify.delete_context(self._exchanges_ctx)

    # No local cache to reset

    # Drop ensure memo then re-provision via shared helper
    ContextRegistry.refresh(self, "Transcripts")
    ContextRegistry.refresh(self, "Exchanges")

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
    context: str | None = None,
) -> None:
    """Idempotently create rows in the Exchanges context for given ids."""
    if not exchange_ids:
        return
    exchanges_context = context or self._exchanges_ctx
    try:
        ids_expr = ", ".join(str(i) for i in sorted(exchange_ids))
        existing: set[int] = set()
        try:
            rows = unify.get_logs(
                context=exchanges_context,
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
                unity_log(
                    context=exchanges_context,
                    exchange_id=int(eid),
                    metadata={},
                    medium=(eid_to_medium or {}).get(int(eid), ""),
                    new=True,
                    mutable=True,
                    stamp_authoring=True,
                    add_to_all_context=self._should_add_to_all_context(
                        exchanges_context,
                    ),
                )
            except Exception:
                # Ignore duplicates or backend races
                pass
    except Exception:
        # Defensive: never propagate to caller
        pass
