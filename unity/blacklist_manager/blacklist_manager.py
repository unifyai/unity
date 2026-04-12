from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import functools

import unify

from ..common.log_utils import log as unity_log
from ..common.data_store import DataStore
from ..common.model_to_fields import model_to_fields
from ..common.filter_utils import normalize_filter_expr
from ..blacklist_manager.types.blacklist import BlackList
from unity.conversation_manager.cm_types import Medium
from .base import BaseBlackListManager
from ..common.context_registry import ContextRegistry, TableContext


class BlackListManager(BaseBlackListManager):
    """
    Manages a minimal catalogue of blacklisted contact details, keyed by ``blacklist_id``.
    """

    class Config:
        required_contexts = [
            TableContext(
                name="BlackList",
                description="List of blacklisted contact details (per medium).",
                fields=model_to_fields(BlackList),
                unique_keys={"blacklist_id": "int"},
                auto_counting={"blacklist_id": None},
            ),
        ]

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        super().__init__()
        self.include_in_multi_assistant_table = True
        self._ctx = ContextRegistry.get_context(self, "BlackList")

        # Local DataStore mirror (write-through only; never read from it)
        self._data_store = DataStore.for_context(
            self._ctx,
            key_fields=("blacklist_id",),
        )

        # Immutable built-in columns derived directly from the model
        self._BUILTIN_FIELDS: Tuple[str, ...] = tuple(BlackList.model_fields.keys())

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseBlackListManager.clear, updated=())
    def clear(self) -> None:
        unify.delete_context(self._ctx)

        # Force re-provisioning by clearing TableStore ensure memo for this context
        ContextRegistry.refresh(self, "BlackList")

        # Verify visibility before proceeding
        try:
            import time as _time  # local import

            for _ in range(3):
                try:
                    unify.get_fields(context=self._ctx)
                    break
                except Exception:
                    _time.sleep(0.05)
        except Exception:
            pass

    @functools.wraps(BaseBlackListManager.filter_blacklist, updated=())
    def filter_blacklist(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        normalized = normalize_filter_expr(filter)
        logs = unify.get_logs(
            context=self._ctx,
            filter=normalized,
            offset=offset,
            limit=limit,
            from_fields=list(self._BUILTIN_FIELDS),
        )
        rows = [lg.entries for lg in logs]

        # Write-through to local DataStore
        for r in rows:
            self._data_store.put(r)

        entries = [BlackList(**r) for r in rows]
        return {
            "blacklist_keys_to_shorthand": BlackList.shorthand_map(),
            "entries": entries,
            "shorthand_to_blacklist_keys": BlackList.shorthand_inverse_map(),
        }

    @functools.wraps(BaseBlackListManager.create_blacklist_entry, updated=())
    def create_blacklist_entry(
        self,
        *,
        medium: Medium,
        contact_detail: str,
        reason: str,
    ) -> Dict[str, Any]:
        payload = BlackList(
            medium=medium,
            contact_detail=contact_detail,
            reason=reason,
        ).to_post_json()
        log = unity_log(
            context=self._ctx,
            new=True,
            mutable=True,
            add_to_all_context=self.include_in_multi_assistant_table,
            **payload,
        )
        self._data_store.put(log.entries)
        return {
            "outcome": "blacklist entry created",
            "details": {"blacklist_id": log.entries["blacklist_id"]},
        }

    @functools.wraps(BaseBlackListManager.update_blacklist_entry, updated=())
    def update_blacklist_entry(
        self,
        *,
        blacklist_id: int,
        medium: Optional[Medium] = None,
        contact_detail: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        if medium is not None:
            updates["medium"] = medium
        if contact_detail is not None:
            updates["contact_detail"] = contact_detail
        if reason is not None:
            updates["reason"] = reason
        if not updates:
            raise ValueError(
                "At least one field must be provided to update a blacklist entry.",
            )

        # Resolve target log id
        target_ids = unify.get_logs(
            context=self._ctx,
            filter=f"blacklist_id == {int(blacklist_id)}",
            return_ids_only=True,
        )
        if not target_ids:
            raise ValueError(
                f"No blacklist entry found with blacklist_id {blacklist_id} to update.",
            )
        if len(target_ids) > 1:
            raise ValueError(
                f"Multiple blacklist rows found with blacklist_id {blacklist_id}. Data integrity issue.",
            )
        log_id = target_ids[0]

        unify.update_logs(
            logs=[log_id],
            context=self._ctx,
            entries=updates,
            overwrite=True,
        )

        # Refresh local cache from backend
        row = unify.get_logs(
            context=self._ctx,
            filter=f"blacklist_id == {int(blacklist_id)}",
            limit=1,
            from_fields=list(self._BUILTIN_FIELDS),
        )
        if row:
            self._data_store.put(row[0].entries)

        return {
            "outcome": "blacklist entry updated",
            "details": {"blacklist_id": int(blacklist_id)},
        }

    @functools.wraps(BaseBlackListManager.delete_blacklist_entry, updated=())
    def delete_blacklist_entry(
        self,
        *,
        blacklist_id: int,
    ) -> Dict[str, Any]:
        # Resolve target log id
        target_ids = unify.get_logs(
            context=self._ctx,
            filter=f"blacklist_id == {int(blacklist_id)}",
            limit=2,
            return_ids_only=True,
        )
        if not target_ids:
            raise ValueError(
                f"No blacklist entry found with blacklist_id {blacklist_id} to delete.",
            )
        if len(target_ids) > 1:
            raise RuntimeError(
                f"Multiple blacklist rows found with blacklist_id {blacklist_id}. Data integrity issue.",
            )
        unify.delete_logs(context=self._ctx, logs=target_ids[0])
        try:
            self._data_store.delete(blacklist_id)
        except KeyError:
            # If cache did not contain the row, proceed without error
            pass
        return {
            "outcome": "blacklist entry deleted",
            "details": {"blacklist_id": int(blacklist_id)},
        }
