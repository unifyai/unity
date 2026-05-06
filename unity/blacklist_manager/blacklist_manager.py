from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import functools

import unify

from ..common.log_utils import log as unity_log
from ..common.data_store import DataStore
from ..common.model_to_fields import model_to_fields
from ..common.filter_utils import normalize_filter_expr
from ..common.tool_outcome import ToolErrorException
from ..blacklist_manager.types.blacklist import BlackList
from unity.conversation_manager.cm_types import Medium
from .base import BaseBlackListManager
from ..common.context_registry import (
    SPACE_CONTEXT_PREFIX,
    ContextRegistry,
    TableContext,
)


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
        self._BUILTIN_FIELDS: Tuple[str, ...] = tuple(
            field for field in BlackList.model_fields.keys() if field != "destination"
        )

    def _blacklist_context_from_root(self, root_context: str) -> str:
        """Return the concrete BlackList context under one registry root."""

        return f"{root_context.strip('/')}/BlackList"

    def _blacklist_context_for_destination(self, destination: str | None) -> str:
        """Resolve a public write destination into a concrete BlackList context."""

        root_context = ContextRegistry.write_root(
            self,
            "BlackList",
            destination=destination,
        )
        return self._blacklist_context_from_root(root_context)

    def _read_blacklist_contexts(self) -> list[str]:
        """Return ordered concrete BlackList contexts visible to this assistant."""

        try:
            root_contexts = ContextRegistry.read_roots(self, "BlackList")
            contexts = [
                self._blacklist_context_from_root(root) for root in root_contexts
            ]
        except RuntimeError as exc:
            if "no base context available" not in str(exc):
                raise
            from ..session_details import SESSION_DETAILS

            contexts = [self._ctx]
            contexts.extend(
                f"{SPACE_CONTEXT_PREFIX}{space_id}/BlackList"
                for space_id in sorted(set(SESSION_DETAILS.space_ids))
            )
        return list(dict.fromkeys(contexts))

    def _data_store_for_context(self, context: str) -> DataStore:
        """Return the per-root local cache for a concrete BlackList context."""

        if context == self._ctx:
            return self._data_store
        return DataStore.for_context(context, key_fields=("blacklist_id",))

    def _destination_for_context(self, context: str) -> str:
        """Return the public destination label for a concrete BlackList context."""

        if context.startswith(SPACE_CONTEXT_PREFIX):
            parts = context.split("/")
            if len(parts) >= 2:
                return f"space:{parts[1]}"
        return "personal"

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @functools.wraps(BaseBlackListManager.clear, updated=())
    def clear(self, *, destination: str | None = None) -> None:
        try:
            context = self._blacklist_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]
        unify.delete_context(context)

        # Force re-provisioning by clearing TableStore ensure memo for this context
        ContextRegistry.forget(self, "BlackList")
        try:
            context = self._blacklist_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]

        # Verify visibility before proceeding
        try:
            import time as _time  # local import

            for _ in range(3):
                try:
                    unify.get_fields(context=context)
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
        rows: list[dict[str, Any]] = []
        target_count = offset + limit
        last_error: Exception | None = None
        for context in self._read_blacklist_contexts():
            destination = self._destination_for_context(context)
            context_offset = 0
            context_rows: list[dict[str, Any]] = []
            try:
                while len(context_rows) < target_count:
                    page_limit = min(1000, target_count - len(context_rows))
                    logs = unify.get_logs(
                        context=context,
                        filter=normalized,
                        offset=context_offset,
                        limit=page_limit,
                        from_fields=list(self._BUILTIN_FIELDS),
                    )
                    if not logs:
                        break
                    context_rows.extend(log.entries for log in logs)
                    if len(logs) < page_limit:
                        break
                    context_offset += page_limit
            except Exception as exc:
                last_error = exc
                continue
            store = self._data_store_for_context(context)
            for row in context_rows:
                row["destination"] = destination
                rows.append(row)
                store.put(row)
        if not rows and last_error is not None:
            raise last_error
        rows = rows[offset:target_count]

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
        destination: str | None = None,
    ) -> Dict[str, Any]:
        try:
            context = self._blacklist_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload
        payload = BlackList(
            medium=medium,
            contact_detail=contact_detail,
            reason=reason,
        ).to_post_json()
        log = unity_log(
            context=context,
            new=True,
            mutable=True,
            stamp_authoring=True,
            add_to_all_context=(
                self.include_in_multi_assistant_table
                and not context.startswith(SPACE_CONTEXT_PREFIX)
            ),
            **payload,
        )
        self._data_store_for_context(context).put(log.entries)
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
        destination: str | None = None,
    ) -> Dict[str, Any]:
        try:
            context = self._blacklist_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload
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
            context=context,
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
            context=context,
            entries=updates,
            overwrite=True,
        )

        # Refresh local cache from backend
        row = unify.get_logs(
            context=context,
            filter=f"blacklist_id == {int(blacklist_id)}",
            limit=1,
            from_fields=list(self._BUILTIN_FIELDS),
        )
        if row:
            self._data_store_for_context(context).put(row[0].entries)

        return {
            "outcome": "blacklist entry updated",
            "details": {"blacklist_id": int(blacklist_id)},
        }

    @functools.wraps(BaseBlackListManager.delete_blacklist_entry, updated=())
    def delete_blacklist_entry(
        self,
        *,
        blacklist_id: int,
        destination: str | None = None,
    ) -> Dict[str, Any]:
        try:
            context = self._blacklist_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload
        # Resolve target log id
        target_ids = unify.get_logs(
            context=context,
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
        unify.delete_logs(context=context, logs=target_ids[0])
        try:
            self._data_store_for_context(context).delete(blacklist_id)
        except KeyError:
            # If cache did not contain the row, proceed without error
            pass
        return {
            "outcome": "blacklist entry deleted",
            "details": {"blacklist_id": int(blacklist_id)},
        }
