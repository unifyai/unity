from __future__ import annotations

from contextlib import contextmanager
from typing import FrozenSet, List, Dict, Optional, Any, Tuple
import base64
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
from ..common.tool_outcome import ToolErrorException, ToolOutcome
from ..common.model_to_fields import model_to_fields
from ..common.context_store import TableStore
from ..common.federated_search import (
    FederatedSearchContext,
    federated_count,
    federated_filter,
    federated_ranked_search,
    is_missing_context_error,
)
from ..common.builtins import builtins_project
from .base import BaseGuidanceManager
from .builtins_catalog import BUILTINS_GUIDANCE_CONTEXT
from .types.guidance import Guidance
from .types.meta import GuidanceMeta
from .custom_guidance import (
    compute_custom_guidance_hash,
)
from ..manager_registry import ManagerRegistry
from ..image_manager.types import AnnotatedImageRefs, AnnotatedImageRef
from ..common.embed_utils import ensure_vector_column, list_private_fields
from ..common.filter_utils import normalize_filter_expr
from ..common.context_registry import TableContext, ContextRegistry
from ..common.stale_reason import StaleReason, merge_stale_reasons
from ..common.sync_lease import exclusive_sync_lease
from ..common.custom_sync import (
    MANAGED_BY_DEPLOYMENT,
    CustomSyncAdapter,
    managed_rows_filter,
    run_custom_sync,
    stored_hash_field,
)

GUIDANCE_TABLE = "Guidance"
GUIDANCE_META_TABLE = "Guidance/Meta"
FUNCTIONS_COMPOSITIONAL_TABLE = "Functions/Compositional"

logger = logging.getLogger(__name__)

# Content cap for search/filter result payloads. Entries (notably imported
# builtin skills) can run to 100KB+; returning them wholesale from list-style
# reads floods the caller's context window. Reads above this cap return a
# preview and the full text is fetched per entry via ``get_guidance``.
GUIDANCE_PREVIEW_CHARS = 2000
GUIDANCE_DESTINATION_GUIDANCE = """destination : str | None, default None
    Where this guidance lives. Pass ``"personal"`` (the default) for private
    working preferences and individual reminders. Pass ``"team:<id>"`` for
    team-level guidance every member of the team should follow: shared
    response style, team-wide do/don't rules, and operational SOPs the team
    agrees on. See the *Accessible shared teams* block in your system prompt
    for available teams and descriptions. Pick personal when in doubt; call
    ``request_clarification`` when the right audience is unclear."""


class GuidanceManager(BaseGuidanceManager):
    """
    Concrete Guidance manager backed by Unify contexts and fields.
    """

    class Config:
        required_contexts = [
            TableContext(
                name=GUIDANCE_TABLE,
                description="Table of distilled guidance entries from transcripts and images.",
                fields=model_to_fields(Guidance),
                unique_keys={"guidance_id": "int"},
                auto_counting={"guidance_id": None},
                foreign_keys=[
                    {
                        "name": "images[*].raw_image_ref.image_id",
                        "references": "Images.image_id",
                        "on_delete": "SET NULL",
                        "on_update": "CASCADE",
                    },
                    {
                        "name": "function_ids[*]",
                        "references": f"{FUNCTIONS_COMPOSITIONAL_TABLE}.function_id",
                        "on_delete": "CASCADE",  # pop on function deletion
                        "on_update": "CASCADE",
                    },
                ],
            ),
            TableContext(
                name=GUIDANCE_META_TABLE,
                description="Metadata for source-defined custom guidance sync state.",
                fields=model_to_fields(GuidanceMeta),
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
        self._ctx = ContextRegistry.get_context(self, GUIDANCE_TABLE)
        self._meta_ctx = ContextRegistry.get_context(self, GUIDANCE_META_TABLE)

        self._filter_scope = filter_scope
        self._exclude_ids = frozenset(exclude_ids) if exclude_ids else None
        self._custom_guidance_synced_sources: set[tuple[str, str]] = set()
        self._destination_context_lock = threading.RLock()
        self._destination_write_scoped = False

        # Built-in fields derived from Guidance model
        self._BUILTIN_FIELDS: Tuple[str, ...] = tuple(Guidance.model_fields.keys())
        self._REQUIRED_COLUMNS: set[str] = set(self._BUILTIN_FIELDS)

        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # Get ImageManager via registry for resolving and attaching images
        self._image_manager = ManagerRegistry.get_image_manager()

        # Ensure context/schema exist
        self._provision_storage()

    def _guidance_context_for_root(self, root_context: str) -> str:
        """Return the concrete Guidance context under a registry root."""
        return f"{root_context.strip('/')}/{GUIDANCE_TABLE}"

    def _guidance_context_for_destination(self, destination: str | None) -> str:
        """Resolve a public destination into one concrete Guidance context."""
        root_context = ContextRegistry.write_root(
            self,
            GUIDANCE_TABLE,
            destination=destination,
        )
        return self._guidance_context_for_root(root_context)

    def _meta_context_for_destination(self, destination: str | None) -> str:
        """Resolve a public destination into one concrete Guidance/Meta context."""
        root_context = ContextRegistry.write_root(
            self,
            GUIDANCE_META_TABLE,
            destination=destination,
        )
        return f"{root_context.strip('/')}/{GUIDANCE_META_TABLE}"

    @contextmanager
    def _temporary_guidance_context(self, attr_name: str, context: str):
        """Temporarily bind an existing storage method to a resolved context."""
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
        """Return destination-scoped guidance context, meta context, and personal flag."""
        data_context = self._guidance_context_for_destination(destination)
        meta_context = self._meta_context_for_destination(destination)
        return data_context, meta_context, destination in (None, "personal")

    def _read_guidance_contexts(self) -> list[str]:
        """Return personal-first Guidance contexts visible to this assistant."""
        return list(
            dict.fromkeys(
                self._guidance_context_for_root(root)
                for root in ContextRegistry.read_roots(self, GUIDANCE_TABLE)
            ),
        )

    def _builtins_read_spec(
        self,
        *,
        row_filter: Optional[str] = None,
        allowed_fields: Optional[List[str]] = None,
    ) -> FederatedSearchContext:
        """Return the federated source for the global builtins guidance catalogue."""
        return FederatedSearchContext(
            context=BUILTINS_GUIDANCE_CONTEXT,
            source="builtins",
            row_filter=row_filter,
            allowed_fields=allowed_fields,
            project=builtins_project(),
        )

    def _function_contexts_for_read(self) -> list[str]:
        """Return compositional function contexts visible to guidance reads."""
        from ..function_manager.function_manager import (
            FUNCTIONS_COMPOSITIONAL_TABLE as FUNCTION_MANAGER_COMPOSITIONAL_TABLE,
            FunctionManager,
        )

        return [
            f"{root.strip('/')}/{FUNCTION_MANAGER_COMPOSITIONAL_TABLE}"
            for root in ContextRegistry.read_roots(
                FunctionManager,
                FUNCTION_MANAGER_COMPOSITIONAL_TABLE,
            )
        ]

    def _available_functions_by_id(self) -> dict[int, str]:
        """Return currently resolvable compositional functions keyed by id."""
        available: dict[int, str] = {}
        for context in self._function_contexts_for_read():
            for log in unisdk.get_logs(
                context=context,
                from_fields=["function_id", "name"],
            ):
                function_id = log.entries.get("function_id")
                if function_id is not None:
                    available[int(function_id)] = str(log.entries.get("name") or "")
        return available

    @staticmethod
    def _missing_function_reasons(
        guidance: Guidance,
        *,
        available: dict[int, str],
        preserve_historical: bool,
    ) -> list[StaleReason]:
        preserved = [
            reason for reason in guidance.stale_reasons if reason.dep_kind != "function"
        ]
        candidates: dict[int, str | None] = {
            int(function_id): None for function_id in guidance.function_ids
        }
        if preserve_historical:
            for reason in guidance.stale_reasons:
                if reason.dep_kind == "function" and reason.id is not None:
                    candidates.setdefault(int(reason.id), reason.name)
        missing = [
            StaleReason(
                dep_kind="function",
                id=function_id,
                name=name,
                message=(
                    f"missing function_id={function_id}"
                    + (f" name={name}" if name else "")
                ),
            )
            for function_id, name in candidates.items()
            if function_id not in available
        ]
        return merge_stale_reasons(preserved, *missing)

    # ------------------------------- Helpers ---------------------------------

    # -- Scope / exclusion properties ----------------------------------------

    @property
    def filter_scope(self) -> Optional[str]:
        """A boolean expression permanently applied to all read queries."""
        return self._filter_scope

    @filter_scope.setter
    def filter_scope(self, value: Optional[str]) -> None:
        self._filter_scope = value

    @property
    def exclude_ids(self) -> Optional[FrozenSet[int]]:
        """Guidance IDs excluded from all read queries."""
        return self._exclude_ids

    @exclude_ids.setter
    def exclude_ids(self, value: Optional[FrozenSet[int]]) -> None:
        self._exclude_ids = frozenset(value) if value else None

    @staticmethod
    def _build_id_exclusion(ids: Optional[FrozenSet[int]]) -> Optional[str]:
        """Build a filter clause excluding a set of guidance IDs."""
        if not ids:
            return None
        sorted_ids = sorted(ids)
        if len(sorted_ids) == 1:
            return f"guidance_id != {sorted_ids[0]}"
        joined_ids = ", ".join(str(gid) for gid in sorted_ids)
        return f"guidance_id not in [{joined_ids}]"

    def _scoped_filter(self, caller_filter: Optional[str]) -> Optional[str]:
        """Compose *caller_filter* with ``_filter_scope`` and id exclusions.

        Returns ``None`` when all parts are absent, meaning "no filter".
        """
        parts = [
            p
            for p in [
                caller_filter,
                self._filter_scope,
                self._build_id_exclusion(self._exclude_ids),
            ]
            if p
        ]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return " and ".join(f"({p})" for p in parts)

    def _is_builtin_guidance(self, guidance_id: int) -> bool:
        """Return whether an id refers to a row in the builtins catalogue."""
        try:
            rows = unisdk.get_logs(
                context=BUILTINS_GUIDANCE_CONTEXT,
                project=builtins_project(),
                filter=f"guidance_id == {int(guidance_id)}",
                limit=1,
                from_fields=["guidance_id"],
            )
        except Exception as exc:
            if is_missing_context_error(exc):
                return False
            raise
        return bool(rows)

    def _raise_if_builtin(self, guidance_id: int, action: str) -> None:
        """Refuse mutations of builtins entries with an actionable error."""
        if self._is_builtin_guidance(guidance_id):
            raise ValueError(
                f"guidance_id {guidance_id} is a built-in platform guidance "
                f"entry and cannot be {action}. Built-in guidance is "
                "read-only for everyone. To tailor it, create your own "
                "entry with add_guidance (optionally adapting the built-in "
                "content) and the personal copy can then be updated or "
                "deleted freely.",
            )

    @staticmethod
    def _with_content_preview(row: Guidance) -> Guidance:
        """Return *row* with content truncated to the list-read preview cap."""
        if len(row.content) <= GUIDANCE_PREVIEW_CHARS:
            return row
        preview = (
            row.content[:GUIDANCE_PREVIEW_CHARS]
            + f"\n\n… [content preview truncated at {GUIDANCE_PREVIEW_CHARS:,} "
            f"of {len(row.content):,} chars — fetch the full entry with "
            f"get_guidance(guidance_id={row.guidance_id})]"
        )
        return row.model_copy(update={"content": preview})

    def _num_items(self) -> int:
        return federated_count(
            [
                *(
                    FederatedSearchContext(context=context, source=context)
                    for context in self._read_guidance_contexts()
                ),
                self._builtins_read_spec(),
            ],
            key="guidance_id",
            filter=self._scoped_filter(None),
        )

    @functools.wraps(BaseGuidanceManager.clear, updated=())
    def clear(self) -> None:
        unisdk.delete_context(self._ctx)
        unisdk.delete_context(self._meta_ctx)

        # Reset sync bookkeeping for this manager instance
        try:
            self._custom_guidance_synced_sources.clear()
        except Exception:
            pass

        # Ensure the schema exists again via shared provisioning helper
        self._ctx = ContextRegistry.refresh(self, GUIDANCE_TABLE) or self._ctx
        self._meta_ctx = (
            ContextRegistry.refresh(self, GUIDANCE_META_TABLE) or self._meta_ctx
        )
        self._provision_storage()

        # Verify the context is visible before attempting reads
        try:
            import time as _time  # local import to avoid polluting module namespace

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
        """Ensure Guidance context and schema exist (idempotent)."""
        self._store = TableStore(
            self._ctx,
            unique_keys={"guidance_id": "int"},
            auto_counting={"guidance_id": None},
            description=(
                "Table of distilled guidance entries from transcripts and images."
            ),
            fields=model_to_fields(Guidance),
        )

    def _get_columns(self) -> Dict[str, str]:
        return self._store.get_columns()

    def _list_columns(
        self,
        *,
        include_types: bool = True,
    ) -> Dict[str, Any] | List[str]:
        """List available columns in the Guidance table.

        Parameters
        ----------
        include_types : bool, default True
            When True, return a mapping of column_name → type information as
            stored in the backing context. When False, return a simple list of
            column names. This is useful for building prompts or validating
            filter expressions without exposing the full schema payload.

        Returns
        -------
        Dict[str, Any] | List[str]
            Either a dict of column metadata or a list of column names,
            depending on ``include_types``.
        """
        cols = self._get_columns()
        return cols if include_types else list(cols)

    # ------------------------------- Private tools ----------------------------
    def _get_images_for_guidance(
        self,
        *,
        guidance_id: int,
    ) -> List[Dict[str, Any]]:
        """Return image metadata (no raw/base64) for images referenced by a guidance row.

        Output schema (list of objects):
        - image_id: int
        - caption: str | None
        - timestamp: str (ISO8601)
        - annotation: str | None  → freeform explanation describing how the image relates to the text

        Notes
        -----
        This tool is read-only and returns metadata only. It never exposes raw
        image bytes.
        """
        rows = self.filter(filter=f"guidance_id == {int(guidance_id)}", limit=1)
        if not rows:
            return []
        guidance_row = rows[0]
        refs: AnnotatedImageRefs = (
            guidance_row.images or AnnotatedImageRefs.model_validate([])
        )
        items = list(getattr(refs, "root", refs))
        if not items:
            return []
        # Resolve handles for all referenced ids
        image_ids: List[int] = []
        annotations_by_id: Dict[int, List[str]] = {}
        for r in items:
            if not isinstance(r, AnnotatedImageRef):
                continue
            # Skip deleted images (SET NULL from FK policy)
            if r.raw_image_ref.image_id is None:
                continue
            iid = int(r.raw_image_ref.image_id)
            image_ids.append(iid)
            annotations_by_id.setdefault(iid, []).append(str(r.annotation))
        # Preserve order while de-duplicating
        image_ids = list(dict.fromkeys(image_ids))
        handles = self._image_manager.get_images(image_ids)
        by_id = {h.image_id: h for h in handles}
        out: List[Dict[str, Any]] = []
        for iid in image_ids:
            h = by_id.get(int(iid))
            if h is None:
                continue
            try:
                ts_str = h.timestamp.isoformat()
            except Exception:
                ts_str = ""
            annotation_list = annotations_by_id.get(int(h.image_id), [])
            annotation = annotation_list[0] if annotation_list else None
            out.append(
                {
                    "image_id": int(h.image_id),
                    "caption": h.caption,
                    "timestamp": ts_str,
                    "annotation": annotation,
                },
            )
        return out

    async def _ask_image(self, *, image_id: int, question: str) -> str:
        """Ask a one‑off question about a specific stored image.

        Mirrors :pyfunc:`ImageHandle.ask` behaviour but requires an explicit
        ``image_id`` so the correct image is resolved first. Sends the image to
        a vision‑capable model as an image block and returns a textual answer only.

        Parameters
        ----------
        image_id : int
            Identifier of the image to analyse. If the underlying ``data`` is a
            Google Cloud Storage URL, a short‑lived signed URL is generated to
            grant access to the model; otherwise base64 is delivered via a
            ``data:image/...;base64,`` URL.
        question : str
            Natural‑language question to ask about the image.

        Returns
        -------
        str
            Text answer from the LLM. This does not persist visual
            context across turns.
        """
        handles = self._image_manager.get_images([int(image_id)])
        if not handles:
            raise ValueError(f"No image found with image_id {image_id}")
        handle = handles[0]
        answer = await handle.ask(question)
        if not isinstance(answer, str):
            answer = str(answer)
        return answer

    def _attach_image_to_context(
        self,
        *,
        image_id: int,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attach a single image (by id) as raw base64 for persistent context.

        Behaviour mirrors :pyfunc:`ImageHandle.raw` for source resolution:
        - If the stored ``data`` is a GCS URL (``gs://`` or
          ``https://storage.googleapis.com/...``), bytes are downloaded
          (raising if inaccessible).
        - Otherwise, ``data`` is expected to be base64 and is decoded to bytes.

        Parameters
        ----------
        image_id : int
            Identifier of the image to attach.
        note : str | None
            Optional note describing why the image is attached.

        Returns
        -------
        dict
            {"note": str, "image": base64_string} where ``image`` contains the
            raw image bytes encoded as base64 (PNG or JPEG).
        """
        handles = self._image_manager.get_images([int(image_id)])
        if not handles:
            raise ValueError(f"No image found with image_id {image_id}")
        h = handles[0]
        try:
            raw_bytes = h.raw()
        except Exception as exc:
            raise ValueError("Failed to load raw image bytes") from exc
        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        payload: Dict[str, Any] = {
            "note": note
            or f"Attached image {h.image_id} for persistent context (caption={h.caption!r}).",
            "image": b64,
        }
        return payload

    def _attach_guidance_images_to_context(
        self,
        *,
        guidance_id: int,
        limit: int = 3,
    ) -> Dict[str, Any]:
        """Attach multiple images referenced by a guidance row to the loop context.

        Characteristics
        ---------------
        - Batches attachment of several images linked via the guidance's image references.
        - Returns metadata (including collected annotations) alongside the base64 for each image.
        - Useful for multi‑image tasks where the loop should retain visual context.

        Parameters
        ----------
        limit : int
            Cap on how many images are attached (order preserved by first appearance).

        Returns
        -------
        dict
            { "attached_count": int, "images": [ { "meta": {...}, "image": base64 }, ... ] }
            Each ``meta`` includes ``image_id``, ``caption``, ``timestamp``, and an ``annotations`` list.
        """
        rows = self.filter(filter=f"guidance_id == {int(guidance_id)}", limit=1)
        if not rows:
            return {"attached_count": 0, "images": []}
        guidance_row = rows[0]
        refs: AnnotatedImageRefs = (
            guidance_row.images or AnnotatedImageRefs.model_validate([])
        )
        items = list(getattr(refs, "root", refs))
        if not items:
            return {"attached_count": 0, "images": []}
        unique_ids: List[int] = []
        annotations_by_id: Dict[int, List[str]] = {}
        for r in items:
            if not isinstance(r, AnnotatedImageRef):
                continue
            # Skip deleted images (SET NULL from FK policy)
            if r.raw_image_ref.image_id is None:
                continue
            iid = int(r.raw_image_ref.image_id)
            unique_ids.append(iid)
            annotations_by_id.setdefault(iid, []).append(str(r.annotation))
        # Preserve original appearance order while de-duplicating
        unique_ids = list(dict.fromkeys(unique_ids))
        if limit is not None:
            try:
                limit = int(limit)
            except Exception:
                limit = 3
            if limit >= 0:
                unique_ids = unique_ids[:limit]

        handles = self._image_manager.get_images(unique_ids)
        images: List[Dict[str, Any]] = []
        for h in handles:
            try:
                raw_bytes = h.raw()
                b64 = base64.b64encode(raw_bytes).decode("utf-8")
            except Exception:
                continue
            annotations = annotations_by_id.get(int(h.image_id), [])
            images.append(
                {
                    "meta": {
                        "image_id": int(h.image_id),
                        "caption": h.caption,
                        "timestamp": getattr(h.timestamp, "isoformat", lambda: "")(),
                        "annotations": annotations,
                    },
                    "image": b64,
                },
            )
        return {"attached_count": len(images), "images": images}

    def _resolve_image_refs(
        self,
        images: AnnotatedImageRefs | list | str,
    ) -> AnnotatedImageRefs:
        """Ensure every ref in *images* has a concrete ``image_id``."""
        if isinstance(images, str):
            import json

            images = json.loads(images)
        if not isinstance(images, AnnotatedImageRefs):
            images = AnnotatedImageRefs.model_validate(images)
        for ref in images.root:
            ref.resolve_image_id(self._image_manager)
        return images

    @functools.wraps(BaseGuidanceManager.add_guidance, updated=())
    def add_guidance(
        self,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        images: AnnotatedImageRefs | None = None,
        function_ids: Optional[List[int]] = None,
        destination: str | None = None,
    ) -> ToolOutcome:
        if not title and not content and not images:
            raise ValueError(
                "At least one field (title/content/images) must be provided.",
            )
        if images is not None:
            images = self._resolve_image_refs(images)
        g = Guidance(
            title=title or "",
            content=content or "",
            images=(
                images if images is not None else AnnotatedImageRefs.model_validate([])
            ),
            function_ids=function_ids or [],
        )
        try:
            context = self._guidance_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]
        payload = g.to_post_json()
        log = unity_log(
            context=context,
            **payload,
            new=True,
            mutable=True,
            stamp_authoring=True,
        )
        return {
            "outcome": "guidance created successfully",
            "details": {
                "guidance_id": assigned_row_id(log, "guidance_id", context=context),
            },
        }

    @functools.wraps(BaseGuidanceManager.update_guidance, updated=())
    def update_guidance(
        self,
        *,
        guidance_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        images: AnnotatedImageRefs | None = None,
        function_ids: Optional[List[int]] = None,
        destination: str | None = None,
    ) -> ToolOutcome:
        updates: Dict[str, Any] = {}
        if title is not None:
            updates["title"] = title
        if content is not None:
            updates["content"] = content
        if images is not None:
            images = self._resolve_image_refs(images)
            _ = Guidance(
                title=title or "tmp",
                content=content or "tmp",
                images=images,
            )
            updates["images"] = _.model_dump(mode="json")["images"]
        if function_ids is not None:
            # Validate via model validator
            _g = Guidance(
                title=title or "tmp",
                content=content or "tmp",
                images=updates.get("images") or AnnotatedImageRefs.model_validate([]),
                function_ids=function_ids,
            )
            updates["function_ids"] = _g.function_ids
        if not updates:
            raise ValueError("At least one field must be provided for an update.")

        try:
            context = self._guidance_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]
        self._raise_if_builtin(guidance_id, "updated")
        logs = unisdk.get_logs(
            context=context,
            filter=f"guidance_id == {int(guidance_id)}",
            limit=2,
            exclude_fields=list_private_fields(context),
        )
        if not logs:
            raise ValueError(
                f"No guidance found with guidance_id {guidance_id} to update.",
            )
        if len(logs) > 1:
            raise RuntimeError(
                f"Multiple rows found with guidance_id {guidance_id}. Data integrity issue.",
            )
        if function_ids is not None:
            candidate = Guidance(**{**logs[0].entries, **updates})
            updates["stale_reasons"] = [
                reason.model_dump(mode="json")
                for reason in self._missing_function_reasons(
                    candidate,
                    available=self._available_functions_by_id(),
                    preserve_historical=False,
                )
            ]
        unisdk.update_logs(
            logs=[logs[0].id],
            context=context,
            entries=updates,
            overwrite=True,
        )
        self._invalidate_linked_function_trust(guidance_id)
        return {"outcome": "guidance updated", "details": {"guidance_id": guidance_id}}

    def _invalidate_linked_function_trust(self, guidance_id: int) -> None:
        """A guidance entry a function is linked to changed: the function re-earns trust.

        The linked function baked this entry's rule into its implementation,
        so an edit or removal is a content change for the function even
        though its source did not move.
        """
        from unify.function_manager.function_manager import FunctionManager

        fm = ManagerRegistry.get_function_manager()
        if isinstance(fm, FunctionManager):
            fm.invalidate_trust_for_guidance(int(guidance_id))

    # ─────────────────────────── Functions helpers ───────────────────────────
    def _functions_context(self) -> str:
        """Return the personal compositional functions context."""
        return self._function_contexts_for_read()[0]

    def _get_functions_for_guidance(
        self,
        *,
        guidance_id: int,
        include_implementations: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return metadata for functions linked to a guidance entry.

        Parameters
        ----------
        guidance_id : int
            Identifier of the guidance row whose related functions to fetch.
        include_implementations : bool, default False
            When True, include the function implementation source in the
            payload; otherwise only surface metadata useful for selection.

        Returns
        -------
        list[dict]
            One item per related function, including ``function_id``, ``name``,
            ``argspec``, ``docstring``, ``calls``, and ``precondition`` fields.
        """
        rows = self.filter(filter=f"guidance_id == {int(guidance_id)}", limit=1)
        if not rows:
            return []
        fids = list(dict.fromkeys(int(fid) for fid in (rows[0].function_ids or [])))
        if not fids:
            return []

        # Build a safe filter like: (function_id == 1) or (function_id == 2)
        filt = " or ".join(f"function_id == {int(fid)}" for fid in fids)
        funcs = []
        for context in self._function_contexts_for_read():
            funcs.extend(
                unisdk.get_logs(
                    context=context,
                    filter=filt or "False",
                    exclude_fields=list_private_fields(context),
                ),
            )

        out: List[Dict[str, Any]] = []
        for lg in funcs:
            ent = lg.entries
            item: Dict[str, Any] = {
                "function_id": ent.get("function_id"),
                "name": ent.get("name"),
                "argspec": ent.get("argspec"),
                "docstring": ent.get("docstring"),
                "calls": ent.get("calls"),
                "precondition": ent.get("precondition"),
            }
            if include_implementations:
                item["implementation"] = ent.get("implementation")
            out.append(item)
        return out

    def _attach_functions_for_guidance_to_context(
        self,
        *,
        guidance_id: int,
        include_implementations: bool = False,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Attach related functions into the loop context as structured data.

        Returns a dict with keys:
            attached_count: int
            functions: list of function dicts (see _get_functions_for_guidance)
        """
        funcs = self._get_functions_for_guidance(
            guidance_id=guidance_id,
            include_implementations=include_implementations,
        )
        if limit is not None:
            try:
                limit = int(limit)
            except Exception:
                limit = None
            if isinstance(limit, int) and limit >= 0:
                funcs = funcs[:limit]
        return {"attached_count": len(funcs), "functions": funcs}

    @functools.wraps(BaseGuidanceManager.delete_guidance, updated=())
    def delete_guidance(
        self,
        *,
        guidance_id: int,
        destination: str | None = None,
    ) -> ToolOutcome:
        try:
            context = self._guidance_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]
        self._raise_if_builtin(guidance_id, "deleted")
        ids = unisdk.get_logs(
            context=context,
            filter=f"guidance_id == {int(guidance_id)}",
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            raise ValueError(
                f"No guidance found with guidance_id {guidance_id} to delete.",
            )
        if len(ids) > 1:
            raise RuntimeError(
                f"Multiple rows found with guidance_id {guidance_id}. Data integrity issue.",
            )
        # Invalidate while the inverse links still exist; the delete cascades
        # guidance_ids off the function rows.
        self._invalidate_linked_function_trust(guidance_id)
        unisdk.delete_logs(context=context, logs=ids[0])
        return {"outcome": "guidance deleted", "details": {"guidance_id": guidance_id}}

    @functools.wraps(BaseGuidanceManager.reconcile_dependencies, updated=())
    def reconcile_dependencies(
        self,
        *,
        guidance_ids: Optional[List[int]] = None,
        destination: str | None = None,
    ) -> ToolOutcome:
        try:
            context = self._guidance_context_for_destination(destination)
        except ToolErrorException as exc:
            return exc.payload  # type: ignore[return-value]
        filter_expr = (
            " or ".join(
                f"guidance_id == {int(guidance_id)}" for guidance_id in guidance_ids
            )
            if guidance_ids
            else None
        )
        logs = unisdk.get_logs(
            context=context,
            filter=filter_expr,
            limit=1000,
            exclude_fields=list_private_fields(context),
        )
        available = self._available_functions_by_id()
        stale_guidance_ids: list[int] = []
        for log in logs:
            guidance = Guidance(**log.entries)
            refreshed = self._missing_function_reasons(
                guidance,
                available=available,
                preserve_historical=True,
            )
            if refreshed:
                stale_guidance_ids.append(int(guidance.guidance_id))
            serialized = [reason.model_dump(mode="json") for reason in refreshed]
            if serialized == [
                reason.model_dump(mode="json") for reason in guidance.stale_reasons
            ]:
                continue
            unisdk.update_logs(
                logs=[log.id],
                context=context,
                entries={"stale_reasons": serialized},
                overwrite=True,
            )
        return {
            "outcome": "dependencies reconciled",
            "details": {
                "checked": len(logs),
                "stale_guidance_ids": stale_guidance_ids,
                "stale_count": len(stale_guidance_ids),
            },
        }

    @functools.wraps(BaseGuidanceManager.search, updated=())
    def search(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Guidance]:
        allowed_fields = list(self._BUILTIN_FIELDS)
        rows = federated_ranked_search(
            [
                *(
                    FederatedSearchContext(
                        context=context,
                        source=context,
                        row_filter=self._scoped_filter(None),
                        allowed_fields=allowed_fields,
                    )
                    for context in self._read_guidance_contexts()
                ),
                self._builtins_read_spec(
                    row_filter=self._scoped_filter(None),
                    allowed_fields=allowed_fields,
                ),
            ],
            references,
            limit=k,
            backfill=True,
            annotate=False,
        )
        return [self._with_content_preview(Guidance(**r)) for r in rows]

    @functools.wraps(BaseGuidanceManager.filter, updated=())
    def filter(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Guidance]:
        from_fields = list(self._BUILTIN_FIELDS)
        try:
            rows = federated_filter(
                [
                    *(
                        FederatedSearchContext(
                            context=context,
                            source=context,
                            allowed_fields=from_fields,
                        )
                        for context in self._read_guidance_contexts()
                    ),
                    self._builtins_read_spec(allowed_fields=from_fields),
                ],
                filter=self._scoped_filter(normalize_filter_expr(filter)),
                offset=offset,
                limit=limit,
                annotate=False,
            )
        except ToolErrorException as exc:
            return exc.payload
        return [self._with_content_preview(Guidance(**row)) for row in rows]

    @functools.wraps(BaseGuidanceManager.get_guidance, updated=())
    def get_guidance(
        self,
        *,
        guidance_id: int,
    ) -> Guidance:
        from_fields = list(self._BUILTIN_FIELDS)
        rows = federated_filter(
            [
                *(
                    FederatedSearchContext(
                        context=context,
                        source=context,
                        allowed_fields=from_fields,
                    )
                    for context in self._read_guidance_contexts()
                ),
                self._builtins_read_spec(allowed_fields=from_fields),
            ],
            filter=self._scoped_filter(f"guidance_id == {int(guidance_id)}"),
            limit=1,
            annotate=False,
        )
        if not rows:
            raise ValueError(f"No guidance found with guidance_id {guidance_id}.")
        return Guidance(**rows[0])

    # ------------------------------------------------------------------ #
    #  Custom Guidance Sync                                              #
    # ------------------------------------------------------------------ #

    def _get_stored_custom_guidance_hash(
        self,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> str:
        field = stored_hash_field("custom_guidance_hash", managed_by)
        try:
            logs = unisdk.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                return logs[0].entries.get(field, "")
        except Exception as exc:
            logger.warning("Failed to retrieve custom guidance hash: %s", exc)
        return ""

    def _store_custom_guidance_hash(
        self,
        hash_value: str,
        *,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> None:
        field = stored_hash_field("custom_guidance_hash", managed_by)
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
            logger.warning("Failed to store custom guidance hash: %s", exc)

    def _get_custom_guidance_from_db(self) -> Dict[str, Dict[str, Any]]:
        logs = unisdk.get_logs(
            context=self._ctx,
            filter="custom_hash != None",
            exclude_fields=list_private_fields(self._ctx),
        )
        return {
            lg.entries.get("custom_key"): lg.entries
            for lg in logs
            if lg.entries.get("custom_key")
        }

    def _delete_custom_guidance_by_key(
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

    def _update_custom_guidance(
        self,
        guidance_id: int,
        data: Dict[str, Any],
    ) -> None:
        log_ids = unisdk.get_logs(
            context=self._ctx,
            filter=f"guidance_id == {int(guidance_id)}",
            limit=1,
            return_ids_only=True,
        )
        if not log_ids:
            raise ValueError(
                f"No guidance found with guidance_id {guidance_id} to update.",
            )
        update_data = strip_authoring_assistant_id(
            {k: v for k, v in data.items() if k != "guidance_id"},
        )
        unisdk.update_logs(
            context=self._ctx,
            logs=[log_ids[0]],
            entries=update_data,
            overwrite=True,
        )

    def _insert_custom_guidance(self, data: Dict[str, Any]) -> int:
        insert_data = {k: v for k, v in data.items() if k != "guidance_id"}
        result = unity_create_logs(
            context=self._ctx,
            entries=[insert_data],
            stamp_authoring=True,
            recompute_derived=True,
        )
        if isinstance(result, list) and len(result) > 0:
            log = result[0]
            if hasattr(log, "entries"):
                return log.entries.get("guidance_id", -1)
        elif isinstance(result, dict):
            log_ids = result.get("log_event_ids", [])
            if log_ids:
                logs = unisdk.get_logs(
                    context=self._ctx,
                    filter=f"id == {log_ids[0]}",
                    limit=1,
                )
                if logs and hasattr(logs[0], "entries"):
                    return logs[0].entries.get("guidance_id")
        return -1

    def sync_custom_guidance(
        self,
        *,
        source_guidance: Optional[Dict[str, Dict[str, Any]]] = None,
        function_name_to_id: Optional[Dict[str, int]] = None,
        destination: str | None = None,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> bool:
        """Ensure custom guidance rows match source ``guidance.jsonl`` definitions.

        Reconciles only the rows *managed_by* owns; rows planted in the same
        context by other sources are neither read nor pruned.
        """
        try:
            guidance_context, meta_context, _is_personal = (
                self._sync_destination_contexts(destination)
            )
        except ToolErrorException as exc:
            logger.warning(
                "Skipping custom guidance sync for destination %r: %s",
                destination,
                exc.payload,
            )
            return False

        with (
            exclusive_sync_lease(f"{meta_context}:custom_sync"),
            self._temporary_guidance_context("_ctx", guidance_context),
            self._temporary_guidance_context("_meta_ctx", meta_context),
        ):
            source_guidance = source_guidance or {}
            synced_key = (guidance_context, managed_by)

            # A caller that pre-built the name map (the deployment
            # reconcile) stays authoritative, including an empty map. A
            # caller that didn't — a workflow install fanning out through
            # the surface registry — gets names resolved against the
            # functions actually readable right now, so a guidance row's
            # ``function_names`` links survive regardless of who drove the
            # sync. Without this, workflow guidance planted with every
            # link silently dropped.
            if function_name_to_id is None and any(
                (row or {}).get("function_names") for row in source_guidance.values()
            ):
                function_name_to_id = {
                    name: fid
                    for fid, name in self._available_functions_by_id().items()
                    if name
                }

            return run_custom_sync(
                adapter=_GuidanceSyncAdapter(
                    self,
                    function_name_to_id=function_name_to_id or {},
                    managed_by=managed_by,
                ),
                source=source_guidance,
                expected_hash=compute_custom_guidance_hash(
                    source_guidance=source_guidance,
                ),
                stored_hash=self._get_stored_custom_guidance_hash(managed_by),
                already_synced=synced_key in self._custom_guidance_synced_sources,
                mark_synced=lambda: self._custom_guidance_synced_sources.add(
                    synced_key,
                ),
                store_hash=lambda value: self._store_custom_guidance_hash(
                    value,
                    managed_by=managed_by,
                ),
            )

    def sync_custom(
        self,
        *,
        source_guidance: Optional[Dict[str, Dict[str, Any]]] = None,
        function_name_to_id: Optional[Dict[str, int]] = None,
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> bool:
        """Sync custom guidance from pre-collected sources across destinations."""
        if source_guidance is None:
            source_guidance = {}

        by_destination: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for custom_key, source_data in source_guidance.items():
            destination = source_data.get("destination") or "personal"
            by_destination.setdefault(destination, {})[custom_key] = source_data

        changed = False
        for destination, group in by_destination.items():
            destination_arg = None if destination == "personal" else destination
            result = self.sync_custom_guidance(
                source_guidance=group,
                function_name_to_id=function_name_to_id,
                destination=destination_arg,
                managed_by=managed_by,
            )
            # Destination helpers may return a tool-error payload dict; skip
            # that destination and continue syncing the remaining groups.
            if isinstance(result, bool):
                changed |= result
        return changed


class _GuidanceSyncAdapter(CustomSyncAdapter):
    """Storage mechanics for the custom guidance reconcile."""

    kind = "guidance"

    def __init__(
        self,
        manager: GuidanceManager,
        *,
        function_name_to_id: Dict[str, int],
        managed_by: str = MANAGED_BY_DEPLOYMENT,
    ) -> None:
        self._manager = manager
        self._function_name_to_id = function_name_to_id
        self.managed_by = managed_by

    def live_rows(self) -> List[Dict[str, Any]]:
        logs = unisdk.get_logs(
            context=self._manager._ctx,
            filter=managed_rows_filter(self.managed_by),
            exclude_fields=list_private_fields(self._manager._ctx),
        )
        return [dict(lg.entries or {}) for lg in logs]

    def transform(self, key: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        function_names = fields.pop("function_names", None) or []
        fields.pop("destination", None)
        resolved_ids: List[int] = []
        for function_name in function_names:
            function_id = self._function_name_to_id.get(function_name)
            if function_id is None:
                logger.warning(
                    "Could not resolve function_name=%s for guidance key=%s",
                    function_name,
                    key,
                )
                continue
            resolved_ids.append(function_id)
        fields["function_ids"] = resolved_ids
        return fields

    def insert(self, key: str, fields: Dict[str, Any]) -> None:
        self._manager._insert_custom_guidance(fields)

    def update(
        self,
        key: str,
        live_row: Dict[str, Any],
        fields: Dict[str, Any],
    ) -> None:
        self._manager._update_custom_guidance(
            guidance_id=live_row["guidance_id"],
            data=fields,
        )

    def delete(self, key: str, live_row: Dict[str, Any]) -> None:
        self._manager._delete_custom_guidance_by_key(key, managed_by=self.managed_by)

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
        place via ``guidance_id`` and the write stamps the provenance.

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


def _append_destination_guidance(method_name: str) -> None:
    method = getattr(GuidanceManager, method_name)
    method.__doc__ = f"{method.__doc__ or ''}\n\n{GUIDANCE_DESTINATION_GUIDANCE}"
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


for _destination_method in ("add_guidance", "update_guidance", "delete_guidance"):
    _append_destination_guidance(_destination_method)
