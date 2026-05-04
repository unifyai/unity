from __future__ import annotations

from typing import FrozenSet, List, Dict, Optional, Any, Tuple
import base64
import functools
import inspect
import re

import unify

from ..common.log_utils import log as unity_log
from ..common.tool_outcome import ToolErrorException, ToolOutcome
from ..common.model_to_fields import model_to_fields
from ..common.context_store import TableStore
from ..common.search_utils import table_search_top_k
from .base import BaseGuidanceManager
from .types.guidance import Guidance
from ..manager_registry import ManagerRegistry
from ..image_manager.types import AnnotatedImageRefs, AnnotatedImageRef
from ..common.embed_utils import ensure_vector_column, list_private_fields
from ..common.filter_utils import normalize_filter_expr
from ..common.context_registry import TableContext, ContextRegistry

GUIDANCE_TABLE = "Guidance"
FUNCTIONS_COMPOSITIONAL_TABLE = "Functions/Compositional"
GUIDANCE_DESTINATION_GUIDANCE = """destination : str | None, default None
    Where this guidance lives. Pass ``"personal"`` (the default) for private
    working preferences and individual reminders. Pass ``"space:<id>"`` for
    team-level guidance every member of the space should follow: shared
    response style, team-wide do/don't rules, and operational SOPs the team
    agrees on. See the *Accessible shared spaces* block in your system prompt
    for available spaces and descriptions. Pick personal when in doubt; call
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
        ]

    def __init__(
        self,
        *,
        rolling_summary_in_prompts: bool = True,
        filter_scope: Optional[str] = None,
        exclude_ids: Optional[FrozenSet[int]] = None,
    ) -> None:
        super().__init__()
        self.include_in_multi_assistant_table = True
        self._ctx = ContextRegistry.get_context(self, GUIDANCE_TABLE)

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

    def _read_guidance_contexts(self) -> list[str]:
        """Return personal-first Guidance contexts visible to this assistant."""
        return list(
            dict.fromkeys(
                self._guidance_context_for_root(root)
                for root in ContextRegistry.read_roots(self, GUIDANCE_TABLE)
            ),
        )

    def _function_contexts_for_read(self) -> list[str]:
        """Return compositional function contexts visible to guidance reads."""
        return [
            f"{root.strip('/')}/{FUNCTIONS_COMPOSITIONAL_TABLE}"
            for root in ContextRegistry.read_roots(self, FUNCTIONS_COMPOSITIONAL_TABLE)
        ]

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
        total = 0
        for context in self._read_guidance_contexts():
            ret = unify.get_logs_metric(
                metric="count",
                key="guidance_id",
                filter=self._scoped_filter(None),
                context=context,
            )
            if ret is not None:
                total += int(ret)
        return total

    @functools.wraps(BaseGuidanceManager.clear, updated=())
    def clear(self) -> None:
        unify.delete_context(self._ctx)

        # Reset observed custom fields for this manager instance
        try:
            self._known_custom_fields = set()
        except Exception:
            pass

        # Ensure the schema exists again via shared provisioning helper
        self._ctx = ContextRegistry.refresh(self, GUIDANCE_TABLE) or self._ctx
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
            add_to_all_context=self.include_in_multi_assistant_table,
        )
        return {
            "outcome": "guidance created successfully",
            "details": {"guidance_id": log.entries["guidance_id"]},
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
        ids = unify.get_logs(
            context=context,
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
            context=context,
            entries=updates,
            overwrite=True,
        )
        return {"outcome": "guidance updated", "details": {"guidance_id": guidance_id}}

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
                unify.get_logs(
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
        ids = unify.get_logs(
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
        unify.delete_logs(context=context, logs=ids[0])
        return {"outcome": "guidance deleted", "details": {"guidance_id": guidance_id}}

    @functools.wraps(BaseGuidanceManager.search, updated=())
    def search(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Guidance]:
        allowed_fields = list(self._BUILTIN_FIELDS)
        rows: list[dict[str, Any]] = []
        for context in self._read_guidance_contexts():
            rows.extend(
                table_search_top_k(
                    context=context,
                    references=references,
                    k=k,
                    allowed_fields=allowed_fields,
                    unique_id_field="guidance_id",
                    row_filter=self._scoped_filter(None),
                ),
            )
        sort_key = next(
            (key for row in rows for key in row if key.startswith("_")),
            None,
        )
        if sort_key:
            rows.sort(key=lambda row: row.get(sort_key, float("inf")))
        rows = rows[:k]
        return [Guidance(**r) for r in rows]

    @functools.wraps(BaseGuidanceManager.filter, updated=())
    def filter(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Guidance]:
        from_fields = list(self._BUILTIN_FIELDS)
        normalized = self._scoped_filter(normalize_filter_expr(filter))
        logs = []
        for context in self._read_guidance_contexts():
            logs.extend(
                unify.get_logs(
                    context=context,
                    filter=normalized,
                    offset=0,
                    limit=offset + limit,
                    from_fields=from_fields,
                ),
            )
        return [Guidance(**lg.entries) for lg in logs[offset : offset + limit]]


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
