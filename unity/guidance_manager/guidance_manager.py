from __future__ import annotations

from typing import List, Dict, Optional, Callable, Any, Tuple
import base64
import asyncio
import json
import functools
import os
import re

import unify

from .prompt_builders import build_ask_prompt, build_update_prompt
from ..common.tool_outcome import ToolOutcome
from ..common.model_to_fields import model_to_fields
from ..common.context_store import TableStore
from ..common.llm_helpers import (
    methods_to_tool_dict,
    inject_broader_context,
    make_request_clarification_tool,
)
from ..common.async_tool_loop import (
    start_async_tool_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..constants import is_readonly_ask_guard_enabled
from ..common.read_only_ask_guard import ReadOnlyAskGuardHandle
from ..events.event_bus import EVENT_BUS, Event
from ..events.manager_event_logging import log_manager_call
from ..common.search_utils import table_search_top_k
from .base import BaseGuidanceManager
from .types.guidance import Guidance
from ..image_manager.image_manager import ImageManager
from ..image_manager.types import ImageRefs, RawImageRef, AnnotatedImageRef
from ..common.embed_utils import list_private_fields
from ..common.filter_utils import normalize_filter_expr


class GuidanceManager(BaseGuidanceManager):
    """
    Concrete Guidance manager backed by Unify contexts and fields.
    """

    def __init__(self, *, rolling_summary_in_prompts: bool = True) -> None:
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

        self._ctx = f"{read_ctx}/Guidance" if read_ctx else "Guidance"

        # Built-in fields derived from Guidance model
        self._BUILTIN_FIELDS: Tuple[str, ...] = tuple(Guidance.model_fields.keys())
        self._REQUIRED_COLUMNS: set[str] = set(self._BUILTIN_FIELDS)

        # Public tools
        self._ask_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self._list_columns,
                self._filter,
                self._search,
                # Image-aware tools (read-only / persistent context helpers)
                self._get_images_for_guidance,
                self._ask_image,
                self._attach_image_to_context,
                self._attach_guidance_images_to_context,
                # Function-aware helpers (read-only / context helpers)
                self._get_functions_for_guidance,
                self._attach_functions_for_guidance_to_context,
                include_class_name=False,
            ),
        }
        self._update_tools: Dict[str, Callable] = {
            **methods_to_tool_dict(
                self.ask,
                self._add_guidance,
                self._update_guidance,
                self._delete_guidance,
                self._create_custom_column,
                self._delete_custom_column,
                include_class_name=False,
            ),
        }

        self._rolling_summary_in_prompts = rolling_summary_in_prompts

        # Lazy-safe image manager for resolving and attaching images
        self._image_manager: ImageManager = ImageManager()

        # Track custom fields seen/created during lifetime
        self._known_custom_fields: set[str] = set()

        # Ensure context/schema and prefill known custom fields
        self._provision_storage()

    # ------------------------------- Public API -------------------------------
    @functools.wraps(BaseGuidanceManager.ask, updated=())
    @log_manager_call("GuidanceManager", "ask", payload_key="question")
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        client = self._new_llm_client("gpt-5@openai")

        tools = dict(self._ask_tools)
        if clarification_up_q is not None and clarification_down_q is not None:

            async def _on_request(q: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "GuidanceManager",
                            "method": "ask",
                            "action": "clarification_request",
                            "question": q,
                        },
                    ),
                )

            async def _on_answer(ans: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "GuidanceManager",
                            "method": "ask",
                            "action": "clarification_answer",
                            "answer": ans,
                        },
                    ),
                )

            tools["request_clarification"] = make_request_clarification_tool(
                clarification_up_q,
                clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_ask_prompt(
                tools=tools,
                num_items=self._num_items(),
                columns=self._list_columns(),
                include_activity=include_activity,
            ),
        )
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.ask.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            tool_policy=self._default_ask_tool_policy,
            preprocess_msgs=inject_broader_context,
            handle_cls=(
                ReadOnlyAskGuardHandle if is_readonly_ask_guard_enabled() else None
            ),
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    @functools.wraps(BaseGuidanceManager.update, updated=())
    @log_manager_call("GuidanceManager", "update", payload_key="request")
    async def update(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
        rolling_summary_in_prompts: Optional[bool] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:
        client = self._new_llm_client("gpt-5@openai")

        tools = dict(self._update_tools)
        if clarification_up_q is not None and clarification_down_q is not None:

            async def _on_request(q: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "GuidanceManager",
                            "method": "update",
                            "action": "clarification_request",
                            "question": q,
                        },
                    ),
                )

            async def _on_answer(ans: str):
                await EVENT_BUS.publish(
                    Event(
                        type="ManagerMethod",
                        calling_id=_call_id,
                        payload={
                            "manager": "GuidanceManager",
                            "method": "update",
                            "action": "clarification_answer",
                            "answer": ans,
                        },
                    ),
                )

            tools["request_clarification"] = make_request_clarification_tool(
                clarification_up_q,
                clarification_down_q,
                on_request=_on_request,
                on_answer=_on_answer,
            )

        include_activity = (
            self._rolling_summary_in_prompts
            if rolling_summary_in_prompts is None
            else rolling_summary_in_prompts
        )

        client.set_system_message(
            build_update_prompt(
                tools,
                num_items=self._num_items(),
                columns=self._list_columns(),
                include_activity=include_activity,
            ),
        )
        handle = start_async_tool_loop(
            client,
            text,
            tools,
            loop_id=f"{self.__class__.__name__}.{self.update.__name__}",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            parent_chat_context=parent_chat_context,
            tool_policy=self._default_update_tool_policy,
            preprocess_msgs=inject_broader_context,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore

        return handle

    # ------------------------------- Helpers ---------------------------------
    def _num_items(self) -> int:
        ret = unify.get_logs_metric(
            metric="count",
            key="guidance_id",
            context=self._ctx,
        )
        if ret is None:
            return 0
        return int(ret)

    @functools.wraps(BaseGuidanceManager.clear, updated=())
    def clear(self) -> None:
        try:
            # Drop the entire guidance table for this active assistant context
            unify.delete_context(self._ctx)
        except Exception:
            # Proceed even if deletion fails (context may already be absent)
            pass

        # Reset observed custom fields for this manager instance
        try:
            self._known_custom_fields = set()
        except Exception:
            pass

        # Ensure the schema exists again via shared provisioning helper
        try:
            # Remove any previous ensure memo and force re-provisioning
            from ..common.context_store import TableStore as _TS  # local import

            try:
                _TS._ENSURED.discard((unify.active_project(), self._ctx))
            except Exception:
                pass
        except Exception:
            pass

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

    def _new_llm_client(self, model: str) -> "unify.AsyncUnify":
        return unify.AsyncUnify(
            model,
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "false")),
            reasoning_effort="high",
            service_tier="priority",
        )

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
        self._store.ensure_context()

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
        cols = self._get_columns()
        return cols if include_types else list(cols)

    def _create_custom_column(
        self,
        *,
        column_name: str,
        column_type: str,
        column_description: Optional[str] = None,
    ) -> Dict[str, str]:
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
        image bytes. Use `attach_image_to_context` or
        `attach_guidance_images_to_context` when persistent visual context is
        required inside the loop.
        """
        rows = self._filter(filter=f"guidance_id == {int(guidance_id)}", limit=1)
        if not rows:
            return []
        guidance_row = rows[0]
        refs: ImageRefs = guidance_row.images or ImageRefs([])
        items = list(getattr(refs, "root", refs))
        if not items:
            return []
        # Resolve handles for all referenced ids
        image_ids: List[int] = []
        annotations_by_id: Dict[int, List[str]] = {}
        for r in items:
            if isinstance(r, AnnotatedImageRef):
                iid = int(r.raw_image_ref.image_id)
                image_ids.append(iid)
                annotations_by_id.setdefault(iid, []).append(str(r.annotation))
            elif isinstance(r, RawImageRef):
                iid = int(r.image_id)
                image_ids.append(iid)
            elif isinstance(r, dict):
                # Best-effort parsing if entries came from raw dicts
                if "raw_image_ref" in r and isinstance(r["raw_image_ref"], dict):
                    iid = int(r["raw_image_ref"].get("image_id"))
                    image_ids.append(iid)
                    ann = r.get("annotation")
                    if ann is not None:
                        annotations_by_id.setdefault(iid, []).append(str(ann))
                elif "image_id" in r:
                    image_ids.append(int(r.get("image_id")))
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
            context across turns; use ``attach_image_to_context`` or
            ``attach_guidance_images_to_context`` when follow‑ups should keep
            seeing the image(s).
        """
        handles = self._image_manager.get_images([int(image_id)])
        if not handles:
            raise ValueError(f"No image found with image_id {image_id}")
        handle = handles[0]
        sub = await handle.ask(question)
        answer = await sub.result()
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
        rows = self._filter(filter=f"guidance_id == {int(guidance_id)}", limit=1)
        if not rows:
            return {"attached_count": 0, "images": []}
        guidance_row = rows[0]
        refs: ImageRefs = guidance_row.images or ImageRefs([])
        items = list(getattr(refs, "root", refs))
        if not items:
            return {"attached_count": 0, "images": []}
        unique_ids: List[int] = []
        annotations_by_id: Dict[int, List[str]] = {}
        for r in items:
            if isinstance(r, AnnotatedImageRef):
                iid = int(r.raw_image_ref.image_id)
                unique_ids.append(iid)
                annotations_by_id.setdefault(iid, []).append(str(r.annotation))
            elif isinstance(r, RawImageRef):
                unique_ids.append(int(r.image_id))
            elif isinstance(r, dict):
                if "raw_image_ref" in r and isinstance(r["raw_image_ref"], dict):
                    iid = int(r["raw_image_ref"].get("image_id"))
                    unique_ids.append(iid)
                    ann = r.get("annotation")
                    if ann is not None:
                        annotations_by_id.setdefault(iid, []).append(str(ann))
                elif "image_id" in r:
                    unique_ids.append(int(r.get("image_id")))
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

    def _add_guidance(
        self,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        images: Optional[Any] = None,
        function_ids: Optional[List[int]] = None,
    ) -> ToolOutcome:
        if not title and not content and not images:
            raise ValueError(
                "At least one field (title/content/images) must be provided.",
            )
        # Accept ImageRefs or a raw list of refs/dicts
        refs: ImageRefs
        try:
            if isinstance(images, ImageRefs):
                refs = images
            elif images is None:
                refs = ImageRefs([])
            else:
                # Expect a list of RawImageRef/AnnotatedImageRef/dicts
                refs = ImageRefs(images)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(
                "Invalid images payload; expected ImageRefs-compatible list",
            ) from exc

        g = Guidance(
            title=title or "",
            content=content or "",
            images=refs,
            function_ids=function_ids or [],
        )
        payload = g.to_post_json()
        # Ensure images is plain JSON (list) not a Pydantic object
        try:
            if isinstance(payload.get("images"), ImageRefs):
                payload["images"] = payload["images"].model_dump(mode="json")
        except Exception:
            pass
        log = unify.log(
            context=self._ctx,
            **payload,
            new=True,
            mutable=True,
        )
        return {
            "outcome": "guidance created successfully",
            "details": {"guidance_id": log.entries["guidance_id"]},
        }

    def _update_guidance(
        self,
        *,
        guidance_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        images: Optional[Any] = None,
        function_ids: Optional[List[int]] = None,
    ) -> ToolOutcome:
        updates: Dict[str, Any] = {}
        if title is not None:
            updates["title"] = title
        if content is not None:
            updates["content"] = content
        if images is not None:
            # Validate via model by constructing minimal model (accepts ImageRefs or list)
            try:
                refs = images if isinstance(images, ImageRefs) else ImageRefs(images)  # type: ignore[arg-type]
            except Exception as exc:
                raise ValueError(
                    "Invalid images payload; expected ImageRefs-compatible list",
                ) from exc
            _ = Guidance(title=title or "tmp", content=content or "tmp", images=refs)
            # Store as plain JSON-serialisable value (list of refs), not a Pydantic object
            updates["images"] = _.model_dump(mode="json")["images"]
        if function_ids is not None:
            # Validate via model validator
            _g = Guidance(
                title=title or "tmp",
                content=content or "tmp",
                images=updates.get("images") or ImageRefs([]),
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
        rows = self._filter(filter=f"guidance_id == {int(guidance_id)}", limit=1)
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

    def _delete_guidance(self, *, guidance_id: int) -> ToolOutcome:
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

    def _search(
        self,
        *,
        references: Optional[Dict[str, str]] = None,
        k: int = 10,
    ) -> List[Guidance]:
        """Semantic search over guidance rows using shared table helper.

        Returns up to k rows ranked by similarity, backfilled to k when
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
        )
        return [Guidance(**r) for r in rows]

    def _filter(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Guidance]:
        from_fields = list(self._BUILTIN_FIELDS)
        normalized = normalize_filter_expr(filter)
        logs = unify.get_logs(
            context=self._ctx,
            filter=normalized,
            offset=offset,
            limit=limit,
            from_fields=from_fields,
        )
        return [Guidance(**lg.entries) for lg in logs]

    # ------------------------------- Policies ---------------------------------
    @staticmethod
    def _default_ask_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        if step_index < 1 and "search" in current_tools:
            return ("required", {"search": current_tools["search"]})
        return ("auto", current_tools)

    @staticmethod
    def _default_update_tool_policy(
        step_index: int,
        current_tools: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        if step_index < 1 and "ask" in current_tools:
            return ("required", {"ask": current_tools["ask"]})
        return ("auto", current_tools)
