from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import FrozenSet, List, Dict, Optional, Any, Tuple
import functools
import inspect
import json
import logging
import threading

import unisdk

from ..common.log_utils import (
    assigned_row_id,
    create_logs as unity_create_logs,
    log as unity_log,
)
from ..common.authorship import strip_authoring_assistant_id
from ..common.stale_reason import (
    StaleReason,
    coerce_stale_reasons,
    merge_stale_reasons,
    stale_reason_key,
)
from ..common.tool_outcome import ToolErrorException, ToolOutcome
from ..common.model_to_fields import model_to_fields
from ..common.context_store import TableStore
from ..common.federated_search import (
    FederatedSearchContext,
    federated_count,
    federated_filter,
    federated_ranked_search,
)
from .base import BaseKnowledgeManager
from .types.knowledge import Knowledge, KnowledgeKind, KnowledgeStatus
from .types.meta import KnowledgeMeta
from .types.source_ref import SourceKind, SourceRef, coerce_source_refs
from .custom_knowledge import (
    compute_custom_knowledge_hash,
)
from ..common.embed_utils import ensure_vector_column, list_private_fields
from ..common.filter_utils import normalize_filter_expr
from ..common.context_registry import TableContext, ContextRegistry
from ..common.custom_sync import (
    MANAGED_BY_DEPLOYMENT,
    CustomSyncAdapter,
    managed_rows_filter,
    run_custom_sync,
    stored_hash_field,
)
from ..common.sync_lease import exclusive_sync_lease

KNOWLEDGE_TABLE = "Knowledge"
KNOWLEDGE_META_TABLE = "Knowledge/Meta"
FILE_RECORDS_TABLE = "FileRecords"
CONTACTS_TABLE = "Contacts"

logger = logging.getLogger(__name__)

# Content cap for search/filter result payloads. Long claims flood the
# caller's context window; list reads return a preview and the full text is
# fetched per entry via ``get_knowledge``.
KNOWLEDGE_PREVIEW_CHARS = 2000
KNOWLEDGE_DESTINATION_GUIDANCE = """destination : str | None, default None
    Where this knowledge claim lives. Pass ``"personal"`` (the default) for
    private working knowledge. Pass ``"team:<id>"`` for team-level claims
    every member of the team should share. See the *Accessible shared teams*
    block in your system prompt for available teams and descriptions. Pick
    personal when in doubt; call ``request_clarification`` when the right
    audience is unclear."""

_ACTIVE_STATUS_FILTER = "status == 'active'"


class KnowledgeManager(BaseKnowledgeManager):
    """
    Concrete Knowledge manager backed by Unify contexts and fields.

    Stores typed claims in a single Knowledge ledger (not dynamic tables).
    Passive store: no EventBus callbacks.
    """

    class Config:
        required_contexts = [
            TableContext(
                name=KNOWLEDGE_TABLE,
                description="Typed claim ledger for durable domain knowledge.",
                fields=model_to_fields(Knowledge),
                unique_keys={"knowledge_id": "int"},
                auto_counting={"knowledge_id": None},
                foreign_keys=[
                    {
                        "name": "source_refs[*].file_id",
                        "references": f"{FILE_RECORDS_TABLE}.file_id",
                        "on_delete": "CASCADE",
                        "on_update": "CASCADE",
                    },
                    {
                        "name": "source_refs[*].contact_id",
                        "references": f"{CONTACTS_TABLE}.contact_id",
                        "on_delete": "CASCADE",
                        "on_update": "CASCADE",
                    },
                    {
                        "name": "source_refs[*].knowledge_id",
                        "references": f"{KNOWLEDGE_TABLE}.knowledge_id",
                        "on_delete": "CASCADE",
                        "on_update": "CASCADE",
                    },
                ],
            ),
            TableContext(
                name=KNOWLEDGE_META_TABLE,
                description="Metadata for source-defined custom knowledge sync state.",
                fields=model_to_fields(KnowledgeMeta),
                unique_keys={"meta_id": "int"},
            ),
        ]

    def __init__(
        self,
        *,
        rolling_summary_in_prompts: bool = True,
        filter_scope: Optional[str] = None,
        exclude_ids: Optional[FrozenSet[int]] = None,
    ) -> None:
        super().__init__()
        self._ctx = ContextRegistry.get_context(self, KNOWLEDGE_TABLE)
        self._meta_ctx = ContextRegistry.get_context(self, KNOWLEDGE_META_TABLE)

        self._filter_scope = filter_scope
        self._exclude_ids = frozenset(exclude_ids) if exclude_ids else None
        self._custom_knowledge_synced_sources: set[tuple[str, str]] = set()
        self._destination_context_lock = threading.RLock()
        self._destination_write_scoped = False

        self._BUILTIN_FIELDS: Tuple[str, ...] = tuple(Knowledge.model_fields.keys())
        self._REQUIRED_COLUMNS: set[str] = set(self._BUILTIN_FIELDS)

        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        self._provision_storage()

    def _knowledge_context_for_root(self, root_context: str) -> str:
        return f"{root_context.strip('/')}/{KNOWLEDGE_TABLE}"

    def _knowledge_context_for_destination(self, destination: str | None) -> str:
        root_context = ContextRegistry.write_root(
            self,
            KNOWLEDGE_TABLE,
            destination=destination,
        )
        return self._knowledge_context_for_root(root_context)

    def _meta_context_for_destination(self, destination: str | None) -> str:
        root_context = ContextRegistry.write_root(
            self,
            KNOWLEDGE_META_TABLE,
            destination=destination,
        )
        return f"{root_context.strip('/')}/{KNOWLEDGE_META_TABLE}"

    @contextmanager
    def _temporary_knowledge_context(self, attr_name: str, context: str):
        with self._destination_context_lock:
            original = getattr(self, attr_name)
            was_write_scoped = self._destination_write_scoped
            setattr(self, attr_name, context)
            self._destination_write_scoped = True
            try:
                yield
            finally:
                setattr(self, attr_name, original)
                self._destination_write_scoped = was_write_scoped

    def _sync_destination_contexts(
        self,
        destination: str | None,
    ) -> tuple[str, str, bool]:
        data_context = self._knowledge_context_for_destination(destination)
        meta_context = self._meta_context_for_destination(destination)
        return data_context, meta_context, destination in (None, "personal")

    def _read_knowledge_contexts(self) -> list[str]:
        return list(
            dict.fromkeys(
                self._knowledge_context_for_root(root)
                for root in ContextRegistry.read_roots(self, KNOWLEDGE_TABLE)
            ),
        )

    # -- Scope / exclusion properties ----------------------------------------

    @property
    def filter_scope(self) -> Optional[str]:
        return self._filter_scope

    @filter_scope.setter
    def filter_scope(self, value: Optional[str]) -> None:
        self._filter_scope = value

    @property
    def exclude_ids(self) -> Optional[FrozenSet[int]]:
        return self._exclude_ids

    @exclude_ids.setter
    def exclude_ids(self, value: Optional[FrozenSet[int]]) -> None:
        self._exclude_ids = frozenset(value) if value else None

    @staticmethod
    def _build_id_exclusion(ids: Optional[FrozenSet[int]]) -> Optional[str]:
        if not ids:
            return None
        sorted_ids = sorted(ids)
        if len(sorted_ids) == 1:
            return f"knowledge_id != {sorted_ids[0]}"
        joined_ids = ", ".join(str(kid) for kid in sorted_ids)
        return f"knowledge_id not in [{joined_ids}]"

    @staticmethod
    def _filter_mentions_status(caller_filter: Optional[str]) -> bool:
        if not caller_filter:
            return False
        return "status" in caller_filter

    def _scoped_filter(
        self,
        caller_filter: Optional[str],
        *,
        default_active: bool = True,
    ) -> Optional[str]:
        """Compose caller filter with scope, id exclusions, and active default."""
        parts: list[str] = []
        if caller_filter:
            parts.append(caller_filter)
            if default_active and not self._filter_mentions_status(caller_filter):
                parts.append(_ACTIVE_STATUS_FILTER)
        elif default_active:
            parts.append(_ACTIVE_STATUS_FILTER)
        if self._filter_scope:
            parts.append(self._filter_scope)
        id_excl = self._build_id_exclusion(self._exclude_ids)
        if id_excl:
            parts.append(id_excl)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return " and ".join(f"({p})" for p in parts)

    def _raise_if_builtin(self, knowledge_id: int, action: str) -> None:
        rows = federated_filter(
            [
                FederatedSearchContext(context=context, source=context)
                for context in self._read_knowledge_contexts()
            ],
            filter=f"knowledge_id == {int(knowledge_id)} and is_builtin == True",
            limit=1,
            annotate=False,
        )
        if rows:
            raise ValueError(
                f"knowledge_id {knowledge_id} is a built-in platform knowledge "
                f"entry and cannot be {action}. Built-in knowledge is "
                "read-only. Create your own claim with add_knowledge instead.",
            )

    @staticmethod
    def _with_content_preview(row: Knowledge) -> Knowledge:
        if len(row.content) <= KNOWLEDGE_PREVIEW_CHARS:
            return row
        preview = (
            row.content[:KNOWLEDGE_PREVIEW_CHARS]
            + f"\n\n… [content preview truncated at {KNOWLEDGE_PREVIEW_CHARS:,} "
            f"of {len(row.content):,} chars — fetch the full entry with "
            f"get_knowledge(knowledge_id={row.knowledge_id})]"
        )
        return row.model_copy(update={"content": preview})

    def _num_items(self) -> int:
        return federated_count(
            [
                FederatedSearchContext(context=context, source=context)
                for context in self._read_knowledge_contexts()
            ],
            key="knowledge_id",
            filter=self._scoped_filter(None),
        )

    @functools.wraps(BaseKnowledgeManager.clear, updated=())
    def clear(self) -> None:
        unisdk.delete_context(self._ctx)
        unisdk.delete_context(self._meta_ctx)

        try:
            self._custom_knowledge_synced_sources.clear()
        except Exception:
            pass

        self._ctx = ContextRegistry.refresh(self, KNOWLEDGE_TABLE) or self._ctx
        self._meta_ctx = (
            ContextRegistry.refresh(self, KNOWLEDGE_META_TABLE) or self._meta_ctx
        )
        self._provision_storage()

        try:
            import time as _time

            for _ in range(3):
                try:
                    unisdk.get_fields(context=self._ctx)
                    break
                except Exception:
                    _time.sleep(0.05)
        except Exception:
            pass

    def warm_embeddings(self) -> None:
        try:
            ensure_vector_column(
                self._ctx,
                embed_column="_content_emb",
                source_column="content",
            )
        except Exception:
            pass

    def _provision_storage(self) -> None:
        self._store = TableStore(
            self._ctx,
            unique_keys={"knowledge_id": "int"},
            auto_counting={"knowledge_id": None},
            description="Typed claim ledger for durable domain knowledge.",
            fields=model_to_fields(Knowledge),
        )

    def _build_knowledge(
        self,
        *,
        title: str,
        content: str,
        kind: KnowledgeKind | str = KnowledgeKind.fact,
        topics: Optional[List[str]] = None,
        source_refs: Optional[List[SourceRef | dict]] = None,
        confidence: Optional[float] = None,
        observed_at: Optional[datetime] = None,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        status: KnowledgeStatus = KnowledgeStatus.active,
        supersedes_ids: Optional[List[int]] = None,
        superseded_by_id: Optional[int] = None,
        custom_key: Optional[str] = None,
        custom_hash: Optional[str] = None,
        knowledge_id: Optional[int] = None,
    ) -> Knowledge:
        kwargs: Dict[str, Any] = {
            "title": title,
            "content": content,
            "kind": kind,
            "topics": topics or [],
            "source_refs": coerce_source_refs(source_refs),
            "confidence": confidence,
            "observed_at": observed_at,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "status": status,
            "supersedes_ids": supersedes_ids or [],
            "superseded_by_id": superseded_by_id,
            "custom_key": custom_key,
            "custom_hash": custom_hash,
        }
        if knowledge_id is not None:
            kwargs["knowledge_id"] = knowledge_id
        return Knowledge(**kwargs)

    def _resolve_log_id(
        self,
        *,
        knowledge_id: int,
        context: str,
    ) -> int:
        ids = unisdk.get_logs(
            context=context,
            filter=f"knowledge_id == {int(knowledge_id)}",
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            raise ValueError(
                f"No knowledge found with knowledge_id {knowledge_id}.",
            )
        if len(ids) > 1:
            raise RuntimeError(
                f"Multiple rows found with knowledge_id {knowledge_id}. "
                "Data integrity issue.",
            )
        return ids[0]

    @functools.wraps(BaseKnowledgeManager.add_knowledge, updated=())
    def add_knowledge(
        self,
        *,
        title: str,
        content: str,
        kind: KnowledgeKind | str = KnowledgeKind.fact,
        topics: Optional[List[str]] = None,
        source_refs: Optional[List[SourceRef | dict]] = None,
        confidence: Optional[float] = None,
        observed_at: Optional[datetime] = None,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        destination: str | None = None,
    ) -> ToolOutcome:
        if not title or not content:
            raise ValueError("Both title and content are required.")
        claim = self._build_knowledge(
            title=title,
            content=content,
            kind=kind,
            topics=topics,
            source_refs=source_refs,
            confidence=confidence,
            observed_at=observed_at,
            valid_from=valid_from,
            valid_until=valid_until,
        )
        try:
            context = self._knowledge_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]
        payload = claim.to_post_json()
        log = unity_log(
            context=context,
            **payload,
            new=True,
            mutable=True,
            stamp_authoring=True,
        )
        knowledge_id = assigned_row_id(log, "knowledge_id", context=context)
        self.reconcile_sources(
            knowledge_ids=[knowledge_id],
            destination=destination,
        )
        return {
            "outcome": "knowledge created successfully",
            "details": {"knowledge_id": knowledge_id},
        }

    @functools.wraps(BaseKnowledgeManager.update_knowledge, updated=())
    def update_knowledge(
        self,
        *,
        knowledge_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        kind: Optional[KnowledgeKind | str] = None,
        topics: Optional[List[str]] = None,
        source_refs: Optional[List[SourceRef | dict]] = None,
        confidence: Optional[float] = None,
        observed_at: Optional[datetime] = None,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        destination: str | None = None,
    ) -> ToolOutcome:
        updates: Dict[str, Any] = {}
        if title is not None:
            updates["title"] = title
        if content is not None:
            updates["content"] = content
        if kind is not None:
            updates["kind"] = KnowledgeKind(str(kind)).value
        if topics is not None:
            updates["topics"] = [str(t) for t in topics]
        if source_refs is not None:
            refs = coerce_source_refs(source_refs)
            updates["source_refs"] = [r.model_dump(mode="json") for r in refs]
        if confidence is not None:
            updates["confidence"] = confidence
        if observed_at is not None:
            updates["observed_at"] = observed_at
        if valid_from is not None:
            updates["valid_from"] = valid_from
        if valid_until is not None:
            updates["valid_until"] = valid_until
        if source_refs is not None:
            updates["stale_reasons"] = []
        if not updates:
            raise ValueError("At least one field must be provided for an update.")

        # Validate via model when enough fields are present
        _ = Knowledge(
            title=title or "tmp",
            content=content or "tmp",
            kind=kind or KnowledgeKind.fact,
            topics=topics or [],
            source_refs=(
                coerce_source_refs(source_refs) if source_refs is not None else []
            ),
            confidence=confidence,
        )

        try:
            context = self._knowledge_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]
        self._raise_if_builtin(knowledge_id, "updated")
        log_id = self._resolve_log_id(knowledge_id=knowledge_id, context=context)
        unisdk.update_logs(
            logs=[log_id],
            context=context,
            entries=updates,
            overwrite=True,
        )
        self.reconcile_sources(
            knowledge_ids=[knowledge_id],
            destination=destination,
        )
        return {
            "outcome": "knowledge updated",
            "details": {"knowledge_id": knowledge_id},
        }

    @functools.wraps(BaseKnowledgeManager.delete_knowledge, updated=())
    def delete_knowledge(
        self,
        *,
        knowledge_id: int,
        destination: str | None = None,
    ) -> ToolOutcome:
        try:
            context = self._knowledge_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]
        self._raise_if_builtin(knowledge_id, "deleted")
        mark_knowledge_stale_for_deleted_sources(
            reasons=[
                StaleReason(
                    dep_kind="knowledge",
                    id=int(knowledge_id),
                    message=f"missing knowledge_id={int(knowledge_id)}",
                ),
            ],
        )
        log_id = self._resolve_log_id(knowledge_id=knowledge_id, context=context)
        unisdk.delete_logs(context=context, logs=log_id)
        return {
            "outcome": "knowledge deleted",
            "details": {"knowledge_id": knowledge_id},
        }

    @functools.wraps(BaseKnowledgeManager.invalidate_knowledge, updated=())
    def invalidate_knowledge(
        self,
        *,
        knowledge_id: int,
        destination: str | None = None,
    ) -> ToolOutcome:
        try:
            context = self._knowledge_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]
        self._raise_if_builtin(knowledge_id, "invalidated")
        log_id = self._resolve_log_id(knowledge_id=knowledge_id, context=context)
        unisdk.update_logs(
            logs=[log_id],
            context=context,
            entries={"status": KnowledgeStatus.invalidated.value},
            overwrite=True,
        )
        return {
            "outcome": "knowledge invalidated",
            "details": {"knowledge_id": knowledge_id},
        }

    @functools.wraps(BaseKnowledgeManager.supersede_knowledge, updated=())
    def supersede_knowledge(
        self,
        *,
        old_knowledge_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        kind: Optional[KnowledgeKind | str] = None,
        topics: Optional[List[str]] = None,
        source_refs: Optional[List[SourceRef | dict]] = None,
        confidence: Optional[float] = None,
        observed_at: Optional[datetime] = None,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        new_knowledge_id: Optional[int] = None,
        destination: str | None = None,
    ) -> ToolOutcome:
        try:
            context = self._knowledge_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]

        self._raise_if_builtin(old_knowledge_id, "superseded")
        old_log_id = self._resolve_log_id(
            knowledge_id=old_knowledge_id,
            context=context,
        )

        if new_knowledge_id is None:
            if not title or not content:
                raise ValueError(
                    "title and content are required when creating a replacement "
                    "claim (or pass new_knowledge_id).",
                )
            claim = self._build_knowledge(
                title=title,
                content=content,
                kind=kind or KnowledgeKind.fact,
                topics=topics,
                source_refs=source_refs,
                confidence=confidence,
                observed_at=observed_at,
                valid_from=valid_from,
                valid_until=valid_until,
                supersedes_ids=[int(old_knowledge_id)],
            )
            payload = claim.to_post_json()
            log = unity_log(
                context=context,
                **payload,
                new=True,
                mutable=True,
                stamp_authoring=True,
            )
            new_knowledge_id = assigned_row_id(log, "knowledge_id", context=context)
        else:
            new_log_id = self._resolve_log_id(
                knowledge_id=new_knowledge_id,
                context=context,
            )
            existing = unisdk.get_logs(
                context=context,
                filter=f"knowledge_id == {int(new_knowledge_id)}",
                limit=1,
                exclude_fields=list_private_fields(context),
            )
            supersedes = list(existing[0].entries.get("supersedes_ids") or [])
            if int(old_knowledge_id) not in supersedes:
                supersedes.append(int(old_knowledge_id))
            unisdk.update_logs(
                logs=[new_log_id],
                context=context,
                entries={"supersedes_ids": supersedes},
                overwrite=True,
            )

        unisdk.update_logs(
            logs=[old_log_id],
            context=context,
            entries={
                "status": KnowledgeStatus.superseded.value,
                "superseded_by_id": int(new_knowledge_id),
            },
            overwrite=True,
        )
        self.reconcile_sources(
            knowledge_ids=[int(new_knowledge_id)],
            destination=destination,
        )
        return {
            "outcome": "knowledge superseded",
            "details": {
                "old_knowledge_id": old_knowledge_id,
                "new_knowledge_id": new_knowledge_id,
            },
        }

    def _file_id_exists(self, file_id: int) -> bool:
        """Best-effort check that a FileRecords row still exists."""
        try:
            from ..file_manager.managers.file_manager import FileManager

            roots = list(ContextRegistry.read_roots(FileManager, FILE_RECORDS_TABLE))
        except Exception:
            return True
        for root in roots:
            prefix = f"{root.strip('/')}/{FILE_RECORDS_TABLE}"
            try:
                children = unisdk.get_contexts(prefix=prefix)
                contexts = (
                    list(children.keys())
                    if isinstance(children, dict)
                    else list(children or [])
                )
            except Exception:
                contexts = []
            if not contexts:
                contexts = [prefix]
            for child_ctx in contexts:
                try:
                    rows = unisdk.get_logs(
                        context=child_ctx,
                        filter=f"file_id == {int(file_id)}",
                        limit=1,
                        return_ids_only=True,
                    )
                    if rows:
                        return True
                except Exception:
                    continue
        return False

    def _contact_id_exists(self, contact_id: int) -> bool:
        """Best-effort check that a Contacts row still exists."""
        try:
            from ..contact_manager.contact_manager import ContactManager

            roots = ContextRegistry.read_roots(ContactManager, CONTACTS_TABLE)
        except Exception:
            return True
        for root in roots:
            context = f"{root.strip('/')}/{CONTACTS_TABLE}"
            try:
                rows = unisdk.get_logs(
                    context=context,
                    filter=f"contact_id == {int(contact_id)}",
                    limit=1,
                    return_ids_only=True,
                )
            except Exception:
                continue
            if rows:
                return True
        return False

    @staticmethod
    def _data_context_exists(context: str) -> bool:
        """Best-effort check that an exact Unify context path exists."""
        normalized = context.strip("/")
        try:
            raw_contexts = unisdk.get_contexts(prefix=normalized)
        except Exception:
            return True
        if isinstance(raw_contexts, dict):
            names = raw_contexts.keys()
        else:
            names = (
                item.get("name", "") if isinstance(item, dict) else str(item)
                for item in (raw_contexts or [])
            )
        return normalized in {str(name).strip("/") for name in names}

    def _knowledge_id_exists(self, knowledge_id: int) -> bool:
        rows = federated_filter(
            [
                FederatedSearchContext(context=context, source=context)
                for context in self._read_knowledge_contexts()
            ],
            filter=f"knowledge_id == {int(knowledge_id)}",
            limit=1,
            annotate=False,
        )
        return bool(rows)

    @staticmethod
    def _reason_identity(reason: StaleReason) -> tuple[str, object] | None:
        if (
            reason.dep_kind in {"file", "contact", "knowledge"}
            and reason.id is not None
        ):
            return reason.dep_kind, int(reason.id)
        if reason.dep_kind == "data" and reason.context:
            return "data", reason.context.strip("/")
        return None

    @staticmethod
    def _ref_identity(ref: SourceRef) -> tuple[str, object] | None:
        if ref.kind == SourceKind.file and ref.file_id is not None:
            return "file", int(ref.file_id)
        if ref.kind == SourceKind.contact:
            return "contact", int(ref.contact_id)
        if ref.kind == SourceKind.derived_from_knowledge:
            return "knowledge", int(ref.knowledge_id)
        if ref.kind == SourceKind.data:
            return "data", ref.context.strip("/")
        return None

    def _missing_source_reasons(self, claim: Knowledge) -> list[StaleReason]:
        reasons: list[StaleReason] = []
        for ref in claim.source_refs:
            if (
                ref.kind == SourceKind.file
                and ref.file_id is not None
                and not self._file_id_exists(ref.file_id)
            ):
                reasons.append(
                    StaleReason(
                        dep_kind="file",
                        id=ref.file_id,
                        path=ref.filepath,
                        message=f"missing file_id={ref.file_id}",
                    ),
                )
            elif ref.kind == SourceKind.contact and not self._contact_id_exists(
                ref.contact_id,
            ):
                reasons.append(
                    StaleReason(
                        dep_kind="contact",
                        id=ref.contact_id,
                        message=f"missing contact_id={ref.contact_id}",
                    ),
                )
            elif ref.kind == SourceKind.data and not self._data_context_exists(
                ref.context,
            ):
                reasons.append(
                    StaleReason(
                        dep_kind="data",
                        context=ref.context,
                        message=f"missing data context={ref.context}",
                    ),
                )
            elif (
                ref.kind == SourceKind.derived_from_knowledge
                and not self._knowledge_id_exists(ref.knowledge_id)
            ):
                reasons.append(
                    StaleReason(
                        dep_kind="knowledge",
                        id=ref.knowledge_id,
                        message=f"missing knowledge_id={ref.knowledge_id}",
                    ),
                )
        return reasons

    def _claims_for_reconcile(
        self,
        *,
        knowledge_ids: Optional[List[int]],
        context: str,
    ) -> list[Knowledge]:
        if knowledge_ids:
            filt = " or ".join(f"knowledge_id == {int(kid)}" for kid in knowledge_ids)
        else:
            filt = _ACTIVE_STATUS_FILTER
        rows = unisdk.get_logs(
            context=context,
            filter=filt,
            limit=1000,
            exclude_fields=list_private_fields(context),
        )
        return [Knowledge(**row.entries) for row in rows]

    def _append_stale_reasons(
        self,
        *,
        knowledge_ids: List[int],
        reasons: List[StaleReason],
        destination: str | None = None,
    ) -> None:
        """Append deduplicated link debt to claims before dependency deletion."""
        context = self._knowledge_context_for_destination(destination)
        for claim in self._claims_for_reconcile(
            knowledge_ids=knowledge_ids,
            context=context,
        ):
            merged = merge_stale_reasons(claim.stale_reasons, *reasons)
            if [stale_reason_key(r) for r in merged] == [
                stale_reason_key(r) for r in claim.stale_reasons
            ]:
                continue
            log_id = self._resolve_log_id(
                knowledge_id=claim.knowledge_id,
                context=context,
            )
            unisdk.update_logs(
                logs=[log_id],
                context=context,
                entries={
                    "stale_reasons": [r.model_dump(mode="json") for r in merged],
                },
                overwrite=True,
            )

    def mark_stale_for_missing_source(
        self,
        *,
        knowledge_ids: List[int],
        reason: StaleReason | dict,
        destination: str | None = None,
    ) -> None:
        """Snapshot source-link debt before an external dependency is deleted."""
        stale_reason = (
            reason
            if isinstance(reason, StaleReason)
            else StaleReason.model_validate(reason)
        )
        self._append_stale_reasons(
            knowledge_ids=knowledge_ids,
            reasons=[stale_reason],
            destination=destination,
        )

    @functools.wraps(BaseKnowledgeManager.reconcile_sources, updated=())
    def reconcile_sources(
        self,
        *,
        knowledge_ids: Optional[List[int]] = None,
        destination: str | None = None,
    ) -> ToolOutcome:
        try:
            context = self._knowledge_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]
        claims = self._claims_for_reconcile(
            knowledge_ids=knowledge_ids,
            context=context,
        )
        stale_knowledge_ids: list[int] = []
        for claim in claims:
            declared_identities = {
                identity
                for ref in claim.source_refs
                if (identity := self._ref_identity(ref)) is not None
            }
            preserved = [
                reason
                for reason in claim.stale_reasons
                if self._reason_identity(reason) not in declared_identities
            ]
            refreshed = merge_stale_reasons(
                preserved,
                *self._missing_source_reasons(claim),
            )
            if refreshed:
                stale_knowledge_ids.append(int(claim.knowledge_id))
            if [r.model_dump(mode="json") for r in refreshed] == [
                r.model_dump(mode="json") for r in claim.stale_reasons
            ]:
                continue
            log_id = self._resolve_log_id(
                knowledge_id=claim.knowledge_id,
                context=context,
            )
            unisdk.update_logs(
                logs=[log_id],
                context=context,
                entries={
                    "stale_reasons": [r.model_dump(mode="json") for r in refreshed],
                },
                overwrite=True,
            )

        return {
            "outcome": "sources reconciled",
            "details": {
                "checked": len(claims),
                "stale_knowledge_ids": stale_knowledge_ids,
                "stale_count": len(stale_knowledge_ids),
            },
        }

    @functools.wraps(BaseKnowledgeManager.search, updated=())
    def search(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Knowledge]:
        allowed_fields = list(self._BUILTIN_FIELDS)
        rows = federated_ranked_search(
            [
                FederatedSearchContext(
                    context=context,
                    source=context,
                    row_filter=self._scoped_filter(None),
                    allowed_fields=allowed_fields,
                )
                for context in self._read_knowledge_contexts()
            ],
            references,
            limit=k,
            backfill=True,
            annotate=False,
        )
        return [self._with_content_preview(Knowledge(**r)) for r in rows]

    @functools.wraps(BaseKnowledgeManager.filter, updated=())
    def filter(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Knowledge]:
        from_fields = list(self._BUILTIN_FIELDS)
        try:
            rows = federated_filter(
                [
                    FederatedSearchContext(
                        context=context,
                        source=context,
                        allowed_fields=from_fields,
                    )
                    for context in self._read_knowledge_contexts()
                ],
                filter=self._scoped_filter(normalize_filter_expr(filter)),
                offset=offset,
                limit=limit,
                annotate=False,
            )
        except ToolErrorException as exc:
            return exc.payload
        return [self._with_content_preview(Knowledge(**row)) for row in rows]

    @functools.wraps(BaseKnowledgeManager.get_knowledge, updated=())
    def get_knowledge(
        self,
        *,
        knowledge_id: int,
    ) -> Knowledge:
        from_fields = list(self._BUILTIN_FIELDS)
        rows = federated_filter(
            [
                FederatedSearchContext(
                    context=context,
                    source=context,
                    allowed_fields=from_fields,
                )
                for context in self._read_knowledge_contexts()
            ],
            filter=self._scoped_filter(
                f"knowledge_id == {int(knowledge_id)}",
                default_active=False,
            ),
            limit=1,
            annotate=False,
        )
        if not rows:
            raise ValueError(f"No knowledge found with knowledge_id {knowledge_id}.")
        return Knowledge(**rows[0])

    # ------------------------------------------------------------------ #
    #  Custom Knowledge Sync                                             #
    # ------------------------------------------------------------------ #

    def _get_stored_custom_knowledge_hash(
        self,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> str:
        field = stored_hash_field("custom_knowledge_hash", managed_by)
        try:
            logs = unisdk.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                return logs[0].entries.get(field, "") or ""
        except Exception as exc:
            logger.warning("Failed to retrieve custom knowledge hash: %s", exc)
        return ""

    def _store_custom_knowledge_hash(
        self,
        hash_value: str,
        *,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> None:
        field = stored_hash_field("custom_knowledge_hash", managed_by)
        try:
            logs = unisdk.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                unisdk.update_logs(
                    context=self._meta_ctx,
                    logs=[logs[0].id],
                    entries={field: hash_value},
                    overwrite=True,
                )
            else:
                unity_create_logs(
                    context=self._meta_ctx,
                    entries=[{"meta_id": 1, field: hash_value}],
                    stamp_authoring=True,
                )
        except Exception as exc:
            logger.warning("Failed to store custom knowledge hash: %s", exc)

    def _delete_custom_knowledge_by_key(
        self,
        custom_key: str,
        *,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> bool:
        logs = unisdk.get_logs(
            context=self._ctx,
            filter=(
                f"custom_key == '{custom_key}' and "
                f"{managed_rows_filter(managed_by)}"
            ),
            limit=1,
        )
        if not logs:
            return False
        unisdk.delete_logs(context=self._ctx, logs=[logs[0].id])
        return True

    def _update_custom_knowledge(
        self,
        knowledge_id: int,
        data: Dict[str, Any],
    ) -> None:
        log_ids = unisdk.get_logs(
            context=self._ctx,
            filter=f"knowledge_id == {int(knowledge_id)}",
            limit=1,
            return_ids_only=True,
        )
        if not log_ids:
            raise ValueError(
                f"No knowledge found with knowledge_id {knowledge_id} to update.",
            )
        update_data = strip_authoring_assistant_id(
            {k: v for k, v in data.items() if k != "knowledge_id"},
        )
        unisdk.update_logs(
            context=self._ctx,
            logs=[log_ids[0]],
            entries=update_data,
            overwrite=True,
        )

    def _insert_custom_knowledge(self, data: Dict[str, Any]) -> int:
        insert_data = {k: v for k, v in data.items() if k != "knowledge_id"}
        result = unity_create_logs(
            context=self._ctx,
            entries=[insert_data],
            stamp_authoring=True,
            recompute_derived=True,
        )
        if isinstance(result, list) and len(result) > 0:
            log = result[0]
            if hasattr(log, "entries"):
                return log.entries.get("knowledge_id", -1)
        elif isinstance(result, dict):
            log_ids = result.get("log_event_ids", [])
            if log_ids:
                logs = unisdk.get_logs(
                    context=self._ctx,
                    filter=f"id == {log_ids[0]}",
                    limit=1,
                )
                if logs and hasattr(logs[0], "entries"):
                    return logs[0].entries.get("knowledge_id")
        return -1

    def sync_custom_knowledge(
        self,
        *,
        source_claims: Optional[Dict[str, Dict[str, Any]]] = None,
        destination: str | None = None,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> bool:
        """Ensure custom knowledge claims match source definitions.

        Reconciles only the rows *managed_by* owns; rows planted in the same
        context by other sources are neither read nor pruned.
        """
        try:
            knowledge_context, meta_context, _is_personal = (
                self._sync_destination_contexts(destination)
            )
        except ToolErrorException as exc:
            logger.warning(
                "Skipping custom knowledge sync for destination %r: %s",
                destination,
                exc.payload,
            )
            return False

        with (
            exclusive_sync_lease(f"{meta_context}:custom_sync"),
            self._temporary_knowledge_context("_ctx", knowledge_context),
            self._temporary_knowledge_context("_meta_ctx", meta_context),
        ):
            source_claims = source_claims or {}
            synced_key = (knowledge_context, managed_by)

            return run_custom_sync(
                adapter=_KnowledgeSyncAdapter(self, managed_by=managed_by),
                source=source_claims,
                expected_hash=compute_custom_knowledge_hash(
                    source_claims=source_claims,
                ),
                stored_hash=self._get_stored_custom_knowledge_hash(managed_by),
                already_synced=synced_key in self._custom_knowledge_synced_sources,
                mark_synced=lambda: self._custom_knowledge_synced_sources.add(
                    synced_key,
                ),
                store_hash=lambda value: self._store_custom_knowledge_hash(
                    value,
                    managed_by=managed_by,
                ),
            )

    def sync_custom(
        self,
        *,
        source_claims: Optional[Dict[str, Dict[str, Any]]] = None,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> bool:
        """Sync custom knowledge claims from pre-collected sources."""
        if source_claims is None:
            source_claims = {}

        by_destination: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for custom_key, source_data in source_claims.items():
            destination = source_data.get("destination") or "personal"
            by_destination.setdefault(destination, {})[custom_key] = source_data

        changed = False
        for destination, group in by_destination.items():
            destination_arg = None if destination == "personal" else destination
            changed |= self.sync_custom_knowledge(
                source_claims=group,
                destination=destination_arg,
                managed_by=managed_by,
            )
        return changed


def _append_destination_guidance(method_name: str) -> None:
    method = getattr(KnowledgeManager, method_name)
    method.__doc__ = f"{method.__doc__ or ''}\n\n{KNOWLEDGE_DESTINATION_GUIDANCE}"
    signature = inspect.signature(method)
    if "destination" not in signature.parameters:
        parameters = list(signature.parameters.values())
        parameters.append(
            inspect.Parameter(
                "destination",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=str | None,
            ),
        )
        method.__signature__ = signature.replace(parameters=parameters)  # type: ignore[attr-defined]


for _destination_method in (
    "add_knowledge",
    "update_knowledge",
    "delete_knowledge",
    "invalidate_knowledge",
    "supersede_knowledge",
    "reconcile_sources",
):
    _append_destination_guidance(_destination_method)


def _source_ref_matches_reason(ref: SourceRef, reason: StaleReason) -> bool:
    if reason.dep_kind == "file" and ref.kind == SourceKind.file:
        return reason.id is not None and ref.file_id == int(reason.id)
    if reason.dep_kind == "contact" and ref.kind == SourceKind.contact:
        return reason.id is not None and int(ref.contact_id) == int(reason.id)
    if reason.dep_kind == "knowledge" and ref.kind == SourceKind.derived_from_knowledge:
        return reason.id is not None and int(ref.knowledge_id) == int(reason.id)
    if reason.dep_kind == "data" and ref.kind == SourceKind.data:
        return reason.context is not None and ref.context.strip(
            "/",
        ) == reason.context.strip("/")
    return False


def _orchestra_filter_for_reason(reason: StaleReason) -> str | None:
    if reason.dep_kind == "file" and reason.id is not None:
        return f"source_refs[*].file_id == {int(reason.id)}"
    if reason.dep_kind == "contact" and reason.id is not None:
        return f"source_refs[*].contact_id == {int(reason.id)}"
    if reason.dep_kind == "knowledge" and reason.id is not None:
        return f"source_refs[*].knowledge_id == {int(reason.id)}"
    if reason.dep_kind == "data" and reason.context:
        return f"source_refs[*].context == {reason.context.strip('/')!r}"
    return None


def mark_knowledge_stale_for_deleted_sources(
    *,
    reasons: List[StaleReason | dict],
) -> None:
    """Snapshot Knowledge link debt before a cited dependency is deleted.

    Callers must invoke this *before* Orchestra FK CASCADE (or context
    deletion) removes identity fields from ``source_refs``. Scans every
    Knowledge root and appends structured ``stale_reasons`` on matching
    claims. Never invents new provenance links.
    """
    if not reasons:
        return
    stale_reasons = [
        (
            reason
            if isinstance(reason, StaleReason)
            else StaleReason.model_validate(reason)
        )
        for reason in reasons
    ]
    for root in ContextRegistry.read_roots(KnowledgeManager, KNOWLEDGE_TABLE):
        context = f"{root.strip('/')}/{KNOWLEDGE_TABLE}"
        private = list_private_fields(context)
        for reason in stale_reasons:
            filt = _orchestra_filter_for_reason(reason)
            logs = []
            if filt is not None:
                try:
                    logs = unisdk.get_logs(
                        context=context,
                        filter=filt,
                        exclude_fields=private,
                    )
                except Exception:
                    logs = []
            if not logs:
                try:
                    logs = unisdk.get_logs(
                        context=context,
                        filter=_ACTIVE_STATUS_FILTER,
                        limit=1000,
                        exclude_fields=private,
                    )
                except Exception:
                    continue
            for log in logs:
                refs = coerce_source_refs(log.entries.get("source_refs") or [])
                if not any(_source_ref_matches_reason(ref, reason) for ref in refs):
                    continue
                existing = coerce_stale_reasons(log.entries.get("stale_reasons"))
                merged = merge_stale_reasons(existing, reason)
                if [stale_reason_key(r) for r in merged] == [
                    stale_reason_key(r) for r in existing
                ]:
                    continue
                unisdk.update_logs(
                    context=context,
                    logs=[log.id],
                    entries={
                        "stale_reasons": [
                            item.model_dump(mode="json") for item in merged
                        ],
                    },
                    overwrite=True,
                )


class _KnowledgeSyncAdapter(CustomSyncAdapter):
    """Storage mechanics for the custom knowledge reconcile."""

    kind = "knowledge"

    def __init__(
        self,
        manager: KnowledgeManager,
        *,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> None:
        self._manager = manager
        self.managed_by = managed_by

    def live_rows(self) -> List[Dict[str, Any]]:
        logs = unisdk.get_logs(
            context=self._manager._ctx,
            filter=managed_rows_filter(self.managed_by),
            exclude_fields=list_private_fields(self._manager._ctx),
        )
        return [dict(lg.entries or {}) for lg in logs]

    def transform(self, key: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        fields.pop("destination", None)
        return fields

    def insert(self, key: str, fields: Dict[str, Any]) -> None:
        self._manager._insert_custom_knowledge(fields)

    def update(
        self,
        key: str,
        live_row: Dict[str, Any],
        fields: Dict[str, Any],
    ) -> None:
        self._manager._update_custom_knowledge(
            knowledge_id=live_row["knowledge_id"],
            data=fields,
        )

    def delete(self, key: str, live_row: Dict[str, Any]) -> None:
        self._manager._delete_custom_knowledge_by_key(key, managed_by=self.managed_by)

    def find_adoptable(
        self,
        key: str,
        fields: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Claim a provenance-less row carrying this entry's title rather
        than inserting a second copy beside it.

        Rows planted before ``custom_key``/``custom_hash`` existed carry
        neither, so the managed index and the collision probe (both keyed
        on ``custom_key``) cannot see them and every reconcile would
        duplicate the whole seed set. Title is the only identity such a
        row still shares with its source entry; adoption updates it in
        place via ``knowledge_id`` and the write stamps the provenance.

        Deployment-only: another source adopting by bare title would
        claim rows it never planted."""
        if self.managed_by != MANAGED_BY_DEPLOYMENT:
            return None
        title = json.dumps(str(fields.get("title", "")))
        existing = unisdk.get_logs(
            context=self._manager._ctx,
            filter=f"title == {title} and custom_hash == None",
            limit=1,
        )
        if not existing:
            return None
        return dict(existing[0].entries or {})

    def find_collision(
        self,
        key: str,
        fields: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        existing = unisdk.get_logs(
            context=self._manager._ctx,
            filter=f"custom_key == '{key}' and custom_hash == None",
            limit=1,
        )
        if not existing:
            return None
        return {"_log_id": existing[0].id, **dict(existing[0].entries or {})}

    def remove_collision(self, key: str, live_row: Dict[str, Any]) -> None:
        unisdk.delete_logs(
            context=self._manager._ctx,
            logs=[live_row["_log_id"]],
        )
