from __future__ import annotations

from typing import FrozenSet, List, Dict, Optional, Any, Tuple
import base64
import functools
import re

import unify

from ..common.log_utils import log as unity_log
from ..common.tool_outcome import ToolOutcome
from ..common.model_to_fields import model_to_fields
from ..common.context_store import TableStore
from ..common.search_utils import table_search_top_k
from .base import BaseGuidanceManager
from .types.guidance import Guidance
from ..manager_registry import ManagerRegistry
from ..image_manager.types import AnnotatedImageRefs, AnnotatedImageRef
from ..common.embed_utils import list_private_fields
from ..common.filter_utils import normalize_filter_expr
from ..common.context_registry import TableContext, ContextRegistry


class GuidanceManager(BaseGuidanceManager):
    """
    Concrete Guidance manager backed by Unify contexts and fields.
    """

    class Config:
        required_contexts = [
            TableContext(
                name="Guidance",
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
                        "references": "Functions/Compositional.function_id",
                        "on_delete": "CASCADE",  # pop on function deletion
                        "on_update": "CASCADE",
                    },
                ],
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
        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
        if not read_ctx:
            try:
                from .. import ensure_initialised as _ensure_initialised  # local

                _ensure_initialised()
                ctxs = unify.get_active_context()
                read_ctx, write_ctx = ctxs.get("read"), ctxs.get("write")
            except Exception:
                pass

        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a GuidanceManager."

        self.include_in_multi_assistant_table = True
        self._ctx = ContextRegistry.get_context(self, "Guidance")

        self._filter_scope = filter_scope
        self._exclude_ids = frozenset(exclude_ids) if exclude_ids else None

        # Built-in fields derived from Guidance model
        self._BUILTIN_FIELDS: Tuple[str, ...] = tuple(Guidance.model_fields.keys())
        self._REQUIRED_COLUMNS: set[str] = set(self._BUILTIN_FIELDS)

        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # Get ImageManager via registry for resolving and attaching images
        self._image_manager = ManagerRegistry.get_image_manager()

        # Track custom fields seen/created during lifetime
        self._known_custom_fields: set[str] = set()

        # Ensure context/schema and prefill known custom fields
        self._provision_storage()

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
        """Build a ``guidance_id != X and ...`` filter clause from a set of IDs."""
        if not ids:
            return None
        clauses = [f"guidance_id != {gid}" for gid in sorted(ids)]
        return " and ".join(clauses)

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

    def _num_items(self) -> int:
        ret = unify.get_logs_metric(
            metric="count",
            key="guidance_id",
            filter=self._scoped_filter(None),
            context=self._ctx,
        )
        if ret is None:
            return 0
        return int(ret)

    @functools.wraps(BaseGuidanceManager.clear, updated=())
    def clear(self) -> None:
        unify.delete_context(self._ctx)

        # Reset observed custom fields for this manager instance
        try:
            self._known_custom_fields = set()
        except Exception:
            pass

        # Ensure the schema exists again via shared provisioning helper
        ContextRegistry.refresh(self, "Guidance")
        self._provision_storage()

        # Verify the context is visible before attempting reads
        try:
            import time as _time  # local import to avoid polluting module namespace

            for _ in range(3):
                try:
                    unify.get_fields(context=self._ctx)
                    break
                except Exception:
                    _time.sleep(0.05)
        except Exception:
            pass

    def _provision_storage(self) -> None:
        """Ensure Guidance context, schema, and custom-field bookkeeping exist."""
        # Ensure context/fields exist deterministically (idempotent)
        self._store = TableStore(
            self._ctx,
            unique_keys={"guidance_id": "int"},
            auto_counting={"guidance_id": None},
            description=(
                "Table of distilled guidance entries from transcripts and images."
            ),
            fields=model_to_fields(Guidance),
        )

        # Prefill known custom fields once to include any preexisting non-private columns
        try:
            existing_cols = self._get_columns()
            for col in existing_cols:
                if col not in self._REQUIRED_COLUMNS and not str(col).startswith("_"):
                    self._known_custom_fields.add(col)
        except Exception:
            # Best-effort only; tools fall back safely
            pass

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

    def _create_custom_column(
        self,
        *,
        column_name: str,
        column_type: str,
        column_description: Optional[str] = None,
    ) -> Dict[str, str]:
        """Create a new mutable custom column on the Guidance table.

        Notes
        -----
        - Required/built-in columns cannot be recreated.
        - ``column_name`` must be snake_case: start with a letter, followed by
          letters, digits or underscores.

        Parameters
        ----------
        column_name : str
            Name of the column to create (snake_case).
        column_type : str
            Logical type label recorded in the context metadata.
        column_description : str | None, default None
            Optional human-friendly description to attach to the column.

        Returns
        -------
        Dict[str, str]
            Service response confirming the created field metadata.
        """
        if column_name in self._REQUIRED_COLUMNS:
            raise ValueError(
                f"'{column_name}' is a required column and cannot be recreated.",
            )
        if not re.fullmatch(r"[a-z][a-z0-9_]*", column_name):
            raise ValueError(
                "column_name must be snake_case: start with a letter, then letters/digits/underscores",
            )
        if (
            getattr(self, "_known_custom_fields", None)
            and column_name in self._known_custom_fields
        ):
            raise ValueError(f"Column '{column_name}' already exists.")
        info: Dict[str, Any] = {"type": str(column_type), "mutable": True}
        if column_description is not None:
            info["description"] = column_description
        resp = unify.create_fields(fields={column_name: info}, context=self._ctx)
        try:
            self._known_custom_fields.add(column_name)
        except Exception:
            pass
        return resp

    def _delete_custom_column(self, *, column_name: str) -> Dict[str, str]:
        """Delete a custom column previously added to the Guidance table.

        Parameters
        ----------
        column_name : str
            Name of the column to remove. Required columns cannot be deleted.

        Returns
        -------
        Dict[str, str]
            Service response confirming deletion.
        """
        if column_name in self._REQUIRED_COLUMNS:
            raise ValueError(f"Cannot delete required column '{column_name}'.")
        resp = unify.delete_fields(fields=[column_name], context=self._ctx)
        try:
            if column_name in getattr(self, "_known_custom_fields", set()):
                self._known_custom_fields.discard(column_name)
        except Exception:
            pass
        return resp

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
            Text answer from the vision model. This does not persist visual
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

    def add_guidance(
        self,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        images: AnnotatedImageRefs | None = None,
        function_ids: Optional[List[int]] = None,
    ) -> ToolOutcome:
        """Create a new guidance entry for procedural or operational how-to
        information: step-by-step instructions, standard operating procedures,
        software usage walkthroughs, composition strategies for combining
        functions, or any other actionable "how to do X" content.

        At least one of ``title``, ``content`` or ``images`` must be provided.

        Parameters
        ----------
        title : str | None
            Short human-readable title for the guidance entry.
        content : str | None
            Longer freeform guidance text.
        images : AnnotatedImageRefs | None
            Annotated image references to attach to this guidance entry.
        function_ids : list[int] | None
            Optional ids of related functions to surface in read flows.

        Returns
        -------
        ToolOutcome
            Outcome string and details containing the newly assigned
            ``guidance_id``.
        """
        if not title and not content and not images:
            raise ValueError(
                "At least one field (title/content/images) must be provided.",
            )
        g = Guidance(
            title=title or "",
            content=content or "",
            images=(
                images if images is not None else AnnotatedImageRefs.model_validate([])
            ),
            function_ids=function_ids or [],
        )
        payload = g.to_post_json()
        log = unity_log(
            context=self._ctx,
            **payload,
            new=True,
            mutable=True,
            add_to_all_context=self.include_in_multi_assistant_table,
        )
        return {
            "outcome": "guidance created successfully",
            "details": {"guidance_id": log.entries["guidance_id"]},
        }

    def update_guidance(
        self,
        *,
        guidance_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        images: AnnotatedImageRefs | None = None,
        function_ids: Optional[List[int]] = None,
    ) -> ToolOutcome:
        """Update fields of an existing guidance entry by id.

        Use this to revise procedural instructions, operating procedures,
        or compositional strategies that are already stored.

        Parameters
        ----------
        guidance_id : int
            Identifier of the row to update.
        title : str | None
            New title (omit to keep existing value).
        content : str | None
            New content (omit to keep existing value).
        images : Any | None
            Replacement image references; validated to the model format.
        function_ids : list[int] | None
            Replacement list of related function ids.

        Returns
        -------
        ToolOutcome
            Outcome string and details with the ``guidance_id``.
        """
        updates: Dict[str, Any] = {}
        if title is not None:
            updates["title"] = title
        if content is not None:
            updates["content"] = content
        if images is not None:
            _ = Guidance(
                title=title or "tmp",
                content=content or "tmp",
                images=(
                    images
                    if images is not None
                    else AnnotatedImageRefs.model_validate([])
                ),
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

        ids = unify.get_logs(
            context=self._ctx,
            filter=f"guidance_id == {int(guidance_id)}",
            limit=2,
            return_ids_only=True,
        )
        if not ids:
            raise ValueError(
                f"No guidance found with guidance_id {guidance_id} to update.",
            )
        if len(ids) > 1:
            raise RuntimeError(
                f"Multiple rows found with guidance_id {guidance_id}. Data integrity issue.",
            )
        unify.update_logs(
            logs=[ids[0]],
            context=self._ctx,
            entries=updates,
            overwrite=True,
        )
        return {"outcome": "guidance updated", "details": {"guidance_id": guidance_id}}

    # ─────────────────────────── Functions helpers ───────────────────────────
    def _functions_context(self) -> str:
        ctxs = unify.get_active_context()
        read_ctx = ctxs.get("read")
        return f"{read_ctx}/Functions" if read_ctx else "Functions"

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
        funcs = unify.get_logs(
            context=self._functions_context(),
            filter=filt or "False",
            exclude_fields=list_private_fields(self._functions_context()),
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

    def delete_guidance(self, *, guidance_id: int) -> ToolOutcome:
        """Delete a guidance entry by id.

        Parameters
        ----------
        guidance_id : int
            Identifier of the row to delete.

        Returns
        -------
        ToolOutcome
            Outcome string and details with the removed ``guidance_id``.
        """
        ids = unify.get_logs(
            context=self._ctx,
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
        unify.delete_logs(context=self._ctx, logs=ids[0])
        return {"outcome": "guidance deleted", "details": {"guidance_id": guidance_id}}

    def search(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Guidance]:
        """Search for guidance entries by semantic similarity to reference content.

        Guidance entries contain procedural how-to information: step-by-step
        instructions, operating procedures, software walkthroughs, and
        strategies for composing functions together.

        Parameters
        ----------
        references : Dict[str, str] | None, default None
            Mapping of source expressions to reference text for semantic search.
        k : int, default 10
            Maximum number of results to return. Must be <= 1000.

        Returns
        -------
        List[Guidance]
            Up to k rows ranked by similarity, backfilled to k when
            similarity yields fewer rows. Payload is restricted to built‑in
            fields for efficiency.
        """
        allowed_fields = list(self._BUILTIN_FIELDS)
        rows = table_search_top_k(
            context=self._ctx,
            references=references,
            k=k,
            allowed_fields=allowed_fields,
            unique_id_field="guidance_id",
            row_filter=self._scoped_filter(None),
        )
        return [Guidance(**r) for r in rows]

    def filter(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Guidance]:
        """Filter guidance entries using a Python filter expression.

        Guidance entries contain procedural how-to information: step-by-step
        instructions, operating procedures, software walkthroughs, and
        strategies for composing functions together.

        Parameters
        ----------
        filter : str | None, default None
            A Python boolean expression evaluated with column names in scope.
            When None, returns all guidance records.
        offset : int, default 0
            Zero-based index of the first result to include.
        limit : int, default 100
            Maximum number of records to return. Must be <= 1000.

        Returns
        -------
        List[Guidance]
            Matching guidance records as Guidance models.
        """
        from_fields = list(self._BUILTIN_FIELDS)
        normalized = self._scoped_filter(normalize_filter_expr(filter))
        logs = unify.get_logs(
            context=self._ctx,
            filter=normalized,
            offset=offset,
            limit=limit,
            from_fields=from_fields,
        )
        return [Guidance(**lg.entries) for lg in logs]
