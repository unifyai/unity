from __future__ import annotations

import base64
import json
import functools
from datetime import datetime
import asyncio
import concurrent.futures
import threading
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from ..common.llm_client import new_llm_client
from ..common.log_utils import log as unity_log, create_logs as unity_create_logs
from ..common.context_dump import make_messages_safe_for_context_dump
import unify


from ..common.model_to_fields import model_to_fields
from ..common.embed_utils import ensure_vector_column
from ..common.semantic_search import backfill_rows, fetch_top_k_by_references
from unity.conversation_manager.gcs_images import signed_url as _gcs_signed_url

from .base import BaseImageManager
from .prompt_builders import build_image_ask_prompt
from .types.image import Image
from ..common.filter_utils import normalize_filter_expr
from ..common.data_store import DataStore
from ..common.context_registry import ContextRegistry, TableContext
import itertools


class ImageHandle:
    """A lightweight handle around a single stored image."""

    def __init__(
        self,
        *,
        manager: "ImageManager",
        image: Image,
        annotation: Optional[str] = None,
        auto_caption: bool = True,
    ) -> None:
        self._manager = manager
        self._image = image
        # Handle-local, non-persistent annotation. This is NOT written to the
        # backend Images table or the local DataStore; it is specific to this
        # handle instance only.
        self._annotation: Optional[str] = None
        self._annotation_event = threading.Event()
        # Caption-ready event, set if caption already exists
        self._caption_event = threading.Event()
        try:
            if self._image.caption is not None:
                self._caption_event.set()
        except Exception:
            pass
        # If an initial annotation is provided, set it now (and trigger the event)
        try:
            if annotation is not None:
                self.annotation = annotation
        except Exception:
            # Best-effort; keep initialization robust even if setter fails
            try:
                self._annotation = annotation
                self._annotation_event.set()
            except Exception:
                pass
        # Deferred persistence state for updates made while pending
        self._deferred_lock = threading.Lock()
        self._deferred_updates: Dict[str, Any] = {}
        self._deferred_task: Any = (
            None  # asyncio.Task | concurrent.futures.Future | None
        )
        # Track any background task launched for auto-captioning
        self._auto_caption_task: Any = (
            None  # asyncio.Task | concurrent.futures.Future | None
        )

        # Optionally auto-generate a caption if requested and none exists yet
        if auto_caption:
            try:
                if self._image.caption is None:

                    async def _auto_caption_worker() -> None:
                        try:
                            answer = await self.ask(
                                "Please describe the contents of the image",
                            )
                            if isinstance(answer, str):
                                answer_str = answer.strip()
                                if answer_str:
                                    # Only apply if caption still missing to avoid overriding later edits
                                    if self.caption is None:
                                        # Update locally (and persist immediately or defer if pending)
                                        self.update_metadata(caption=answer_str)
                        except Exception:
                            # Best-effort; ignore failures
                            pass

                    try:
                        loop = asyncio.get_running_loop()
                        self._auto_caption_task = loop.create_task(
                            _auto_caption_worker(),
                        )
                    except RuntimeError:
                        # No running loop; execute in a background thread with its own loop
                        self._auto_caption_task = self._manager._executor.submit(
                            lambda: asyncio.run(_auto_caption_worker()),
                        )
            except Exception:
                # Defensive coding: auto-captioning is optional and must not break construction
                pass

    @property
    def image_id(self) -> int:
        return int(self._image.image_id)

    @property
    def is_pending(self) -> bool:
        return self._manager.is_pending_id(self.image_id)

    @property
    def caption(self) -> Optional[str]:
        return self._image.caption

    @property
    def timestamp(self) -> datetime:
        return self._image.timestamp

    @property
    def filepath(self) -> Optional[str]:
        return self._image.filepath

    # ------------------------------ Local-only fields ----------------------
    @property
    def annotation(self) -> Optional[str]:
        """Return/assign a handle-local annotation (never persisted)."""
        return getattr(self, "_annotation", None)

    @annotation.setter
    def annotation(self, value: Optional[str]) -> None:
        self._annotation = value
        if value is not None:
            try:
                self._annotation_event.set()
            except Exception:
                pass

    def resolve(self, real_image_id: int) -> None:
        """
        Rebind this handle to a resolved backend image id.

        Assumes the caller has already flushed the pending row and ensured the
        DataStore now contains a row under the resolved id.
        """
        try:
            self._image.image_id = int(real_image_id)
        except Exception:
            # Best-effort; if mutation fails, leave as-is
            pass

    def update_metadata(
        self,
        *,
        caption: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        data: Optional[Union[bytes, bytearray, str]] = None,
        filepath: Optional[str] = None,
    ) -> None:
        """
        Update metadata for this image in-place.

        - Always updates the local DataStore so pending handles remain consistent
          and the background upload (or subsequent resolution) includes the changes.
        - If the image is resolved (not pending), also persists to the backend
          via ImageManager.update_images.
        """
        _PERSIST_KEYS = ("caption", "timestamp", "data", "filepath")
        updates: Dict[str, Any] = {}
        if caption is not None:
            updates["caption"] = caption
            try:
                self._image.caption = caption
            except Exception:
                pass
            try:
                if caption is not None:
                    self._caption_event.set()
            except Exception:
                pass
        if timestamp is not None:
            updates["timestamp"] = timestamp
            try:
                self._image.timestamp = timestamp
            except Exception:
                pass
        if data is not None:
            if isinstance(data, (bytes, bytearray)):
                data_b64 = base64.b64encode(data).decode("utf-8")
            else:
                data_b64 = data
            updates["data"] = data_b64
            try:
                self._image.data = data_b64
            except Exception:
                pass
        if filepath is not None:
            updates["filepath"] = filepath
            try:
                self._image.filepath = filepath
            except Exception:
                pass

        if not updates:
            return

        # Update local DataStore (create row if missing)
        try:
            try:
                self._manager._data_store.update(self.image_id, updates)
            except KeyError:
                row = {"image_id": self.image_id, **updates}
                self._manager._data_store.put(row)
        except Exception:
            pass

        # Persist to backend
        if not self.is_pending:
            payload: Dict[str, Any] = {"image_id": self.image_id}
            for k in _PERSIST_KEYS:
                if k in updates:
                    payload[k] = updates[k]
            try:
                self._manager.update_images([payload])
            except Exception:
                pass
            return

        # If pending, coalesce updates and schedule deferred persistence after resolution
        try:
            with self._deferred_lock:
                for k in _PERSIST_KEYS:
                    if k in updates:
                        self._deferred_updates[k] = updates[k]
                if (
                    self._deferred_task is None
                    or getattr(self._deferred_task, "done", lambda: True)()
                ):
                    self._deferred_task = self._schedule_deferred_persist()
        except Exception:
            # Best-effort; if scheduling fails we still have local cache updated
            pass

    def raw(self) -> bytes:
        """
        Return the decoded image bytes.

        If the data is a GCS URL, it downloads the content via unify.download_object().
        Otherwise, it assumes the data is a base64 string and decodes it.
        """
        # Prefer locally cached base64 data from the DataStore to avoid re-downloading
        try:
            cached = self._manager._data_store.get(self.image_id)
            data_str = cached.get("data") if cached is not None else self._image.data
        except Exception:
            data_str = self._image.data

        # Convert HTTPS GCS URLs to gs:// format for unify.download_object
        gcs_uri = None
        if data_str.startswith("gs://"):
            gcs_uri = data_str
        elif data_str.startswith("https://storage.googleapis.com/"):
            parsed_url = urlparse(data_str)
            path_parts = parsed_url.path.lstrip("/").split("/", 1)
            if len(path_parts) == 2:
                bucket_name, object_path = path_parts
                gcs_uri = f"gs://{bucket_name}/{object_path}"

        if gcs_uri:
            try:
                content = unify.download_object(gcs_uri)
                # Cache the downloaded bytes as base64 in the DataStore to prevent future downloads
                try:
                    import base64 as _b64

                    try:
                        self._manager._data_store.update(
                            self.image_id,
                            {"data": _b64.b64encode(content).decode("utf-8")},
                        )
                    except KeyError:
                        # If the row isn't present yet, insert a minimal row
                        self._manager._data_store.put(
                            {
                                "image_id": self.image_id,
                                "data": _b64.b64encode(content).decode("utf-8"),
                            },
                        )
                except Exception:
                    pass
                return content
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download image from GCS: {data_str}",
                ) from exc
        else:
            # Fallback to assuming it's base64
            try:
                return base64.b64decode(data_str)
            except Exception as exc:
                raise ValueError("Invalid base64 image data") from exc

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Ask a high-level question about this image with a single LLM call.

        Sends the underlying image to the model as an image block alongside the
        `question`, and returns the model's textual answer directly (no nested
        tool-use loop).
        If the image is stored as a GCS URL, a temporary signed URL is generated
        to make it accessible to the vision model.

        Parameters
        ----------
        question : str
            The natural-language question to ask about the image.
        parent_chat_context : list[dict] | None, optional
            Optional parent chat context. When provided, a single synthetic system
            message is inserted at the start of the chat that summarises the broader
            context as JSON (read-only), helping the model understand why the question
            is being asked.
        """
        # Single-call client
        client = new_llm_client()

        # Build a succinct system message tailored to image Q&A
        client.set_system_message(
            build_image_ask_prompt(
                caption=self._image.caption,
                timestamp=self._image.timestamp,
            ).to_list(),
        )

        # Provide the image as a user content block (vision input).
        # Prefer cached base64 from the DataStore when available to avoid signing/downloading again
        try:
            cached = self._manager._data_store.get(self.image_id)
            data_str = cached.get("data") if cached is not None else self._image.data
        except Exception:
            data_str = self._image.data
        content_block: dict

        # Check if the data string is a GCS URL and convert to gs:// format
        gcs_uri = None
        if isinstance(data_str, str):
            if data_str.startswith("gs://"):
                gcs_uri = data_str
            elif data_str.startswith("https://storage.googleapis.com/"):
                parsed_url = urlparse(data_str)
                path_parts = parsed_url.path.lstrip("/").split("/", 1)
                if len(path_parts) == 2:
                    bucket_name, object_path = path_parts
                    gcs_uri = f"gs://{bucket_name}/{object_path}"

        if gcs_uri:
            try:
                # Generate a signed URL valid for 1 hour via unify
                signed_url = unify.get_signed_url(gcs_uri, expiration_minutes=60)

                content_block = {
                    "type": "image_url",
                    "image_url": {"url": signed_url},
                }

            except Exception as e:
                raise RuntimeError(
                    f"Failed to generate signed URL for GCS image: {e}",
                ) from e

        elif isinstance(data_str, str) and (
            data_str.startswith("http://") or data_str.startswith("https://")
        ):
            # Pass the URL through directly; upstream must ensure it is fetchable
            content_block = {
                "type": "image_url",
                "image_url": {"url": data_str},
            }
        elif isinstance(data_str, str) and data_str.startswith("data:image/"):
            # Full data URL provided; pass as-is
            content_block = {
                "type": "image_url",
                "image_url": {"url": data_str},
            }
        else:
            # Expect a raw base64 payload – validate and infer mime from header
            try:
                decoded = base64.b64decode(data_str, validate=True)
            except Exception as exc:
                raise ValueError("Invalid base64 image data") from exc

            head = decoded[:10]
            if head.startswith(b"\xff\xd8"):
                mime = "image/jpeg"
            elif head.startswith(b"\x89PNG\r\n\x1a\n"):
                mime = "image/png"
            else:
                raise ValueError(
                    "Unsupported image format; only PNG and JPEG are supported.",
                )
            content_block = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data_str}"},
            }

        # Build the messages list
        messages = []

        # Optional: inject broader parent chat context as a system header
        if _parent_chat_context:
            parent_ctx_safe = make_messages_safe_for_context_dump(_parent_chat_context)
            messages.append(
                {
                    "role": "system",
                    "_ctx_header": True,
                    "content": (
                        "You are handling an image analysis request.\n\n"
                        "## Parent Chat Context\n"
                        "This is the broader conversation context from which this image question "
                        "originated. It may help explain why this question is being asked.\n\n"
                        f"{json.dumps(parent_ctx_safe, indent=2)}\n\n"
                        "Your task: Analyze the provided image and answer the question. "
                        "Respond with plain text only, do not attempt to call other tools."
                    ),
                },
            )

        # Add the user message with image
        messages.append(
            {
                "role": "user",
                "content": [content_block, {"type": "text", "text": question}],
            },
        )

        # Single shot – no nested tool loop
        answer = await client.generate(messages=messages)
        return answer

    async def wait_until_resolved(self, timeout: Optional[float] = None) -> int:
        """
        Await until this handle's pending id is resolved to a real backend id.

        Returns the resolved image id. If already resolved, returns immediately.
        """
        if not self.is_pending:
            return self.image_id
        # Defer to manager's await_pending so we share the same scheduling/cache
        if timeout is None:
            mapping = await self._manager.await_pending([self.image_id])
        else:
            mapping = await asyncio.wait_for(
                self._manager.await_pending([self.image_id]),
                timeout=timeout,
            )
        rid = mapping.get(self.image_id)
        if isinstance(rid, int):
            self.resolve(int(rid))
            return int(rid)
        return self.image_id

    def __await__(self):  # convenience alias
        return self.wait_until_resolved().__await__()

    async def wait_for_annotation(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[str]:
        """
        Await until a non-None annotation is set on this handle, then return it.

        If already set, returns immediately.
        """
        if self.annotation is not None:
            return self.annotation
        # Use polling instead of asyncio.to_thread(_annotation_event.wait).
        # asyncio.to_thread creates executor threads that block indefinitely,
        # preventing clean event loop shutdown. Polling allows cancellation.
        start = asyncio.get_event_loop().time() if timeout else None
        while not self._annotation_event.is_set():
            await asyncio.sleep(0.1)
            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start  # type: ignore
                if elapsed >= timeout:
                    raise asyncio.TimeoutError()
        return self.annotation

    async def wait_for_caption(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[str]:
        """
        Await until a non-None caption (label) is set for this image, then return it.

        Returns immediately if already present.
        """
        if self.caption is not None:
            return self.caption
        # Use polling instead of asyncio.to_thread(_caption_event.wait).
        # asyncio.to_thread creates executor threads that block indefinitely,
        # preventing clean event loop shutdown. Polling allows cancellation.
        start = asyncio.get_event_loop().time() if timeout else None
        while not self._caption_event.is_set():
            await asyncio.sleep(0.1)
            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start  # type: ignore
                if elapsed >= timeout:
                    raise asyncio.TimeoutError()
        return self.caption

    # ------------------------------ Deferred persistence ------------------
    def _schedule_deferred_persist(self):
        async def _async_worker() -> None:
            # Await resolution; if already resolved this returns immediately
            try:
                rid = await self.wait_until_resolved()
            except Exception:
                return

            # Drain any accumulated updates and persist; loop to catch races
            while True:
                try:
                    with self._deferred_lock:
                        pending_updates = dict(self._deferred_updates)
                        self._deferred_updates.clear()
                except Exception:
                    pending_updates = {}

                # Filter to supported keys
                payload_body: Dict[str, Any] = {}
                for k in ("caption", "timestamp", "data", "filepath"):
                    if k in pending_updates:
                        payload_body[k] = pending_updates[k]

                if not payload_body:
                    break

                payload: Dict[str, Any] = {"image_id": int(rid), **payload_body}
                try:
                    self._manager.update_images([payload])
                except Exception:
                    # Tolerate backend failure; local cache already updated
                    pass

                # If more updates arrived during the write, loop again
                try:
                    with self._deferred_lock:
                        has_more = bool(self._deferred_updates)
                except Exception:
                    has_more = False
                if not has_more:
                    break

        try:
            loop = asyncio.get_running_loop()
            return loop.create_task(_async_worker())
        except RuntimeError:
            # No running loop: execute in background thread with its own loop
            return self._manager._executor.submit(lambda: asyncio.run(_async_worker()))


class ImageManager(BaseImageManager):
    """Concrete implementation backed by Unify contexts and fields."""

    class Config:
        required_contexts = [
            TableContext(
                name="Images",
                description="Collection of images with timestamps, captions, and raw base64 data.",
                fields=model_to_fields(Image),
                unique_keys={"image_id": "int"},
                auto_counting={"image_id": None},
            ),
        ]

    def __init__(self) -> None:
        self.include_in_multi_assistant_table = True
        self._ctx = ContextRegistry.get_context(self, "Images")

        # Local DataStore mirror for Images (write-through on reads/writes)
        self._data_store = DataStore.for_context(self._ctx, key_fields=("image_id",))

        # Cache built-in fields for fast whitelisting
        self._BUILTIN_FIELDS: tuple[str, ...] = tuple(Image.model_fields.keys())

        # Pending id generation (process-local)
        self._PENDING_BASE: int = 10**12
        # Single counter per manager instance; uniqueness is sufficient per-process
        self._pending_counter = itertools.count(self._PENDING_BASE)

        # Cache of known resolutions for fast, race-tolerant lookups
        self._resolved_pid_map: Dict[int, int] = {}

        # Executor for background uploads when no event loop is running
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        # Map of pending_id -> concurrent future that resolves to real_id
        self._pending_uploads: Dict[int, concurrent.futures.Future[int]] = {}

        # Internal helper ensures we preserve any local-only columns such as
        # temp_image_id when writing backend-fetched rows into the DataStore.
        def _put_preserve_temp(row: Dict[str, Any]) -> None:
            try:
                iid = int(row.get("image_id"))
            except Exception:
                # Fallback to raw put if image_id missing/unparseable
                try:
                    self._data_store.put(row)
                except Exception:
                    pass
                return

            try:
                existing = self._data_store.get(iid)
            except Exception:
                existing = None
            merged: Dict[str, Any] = {}
            if isinstance(existing, dict):
                merged.update(existing)
            merged.update(row)
            if (
                isinstance(existing, dict)
                and ("temp_image_id" in existing)
                and ("temp_image_id" not in merged)
            ):
                merged["temp_image_id"] = existing["temp_image_id"]
            try:
                self._data_store.put(merged)
            except Exception:
                pass

        # Bind helper for reuse
        self._put_preserve_temp = _put_preserve_temp  # type: ignore[attr-defined]

    # ------------------------------ Reads ---------------------------------
    def filter_images(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Image]:
        normalized = normalize_filter_expr(filter)
        logs = unify.get_logs(
            context=self._ctx,
            filter=normalized,
            offset=offset,
            limit=limit,
            from_fields=list(self._BUILTIN_FIELDS),
        )
        # Write-through to local DataStore mirror (preserve local-only columns)
        try:
            for lg in logs:
                self._put_preserve_temp(getattr(lg, "entries", {}) or {})
        except Exception:
            pass
        return [Image(**lg.entries) for lg in logs]

    def search_images(
        self,
        *,
        reference_text: str,
        k: int = 10,
    ) -> List[Image]:
        # Only captions participate in semantic search for images
        initial = fetch_top_k_by_references(
            self._ctx,
            references={"caption": reference_text},
            k=k,
            allowed_fields=list(self._BUILTIN_FIELDS),
        )
        filled = backfill_rows(
            self._ctx,
            initial,
            k,
            unique_id_field="image_id",
            allowed_fields=list(self._BUILTIN_FIELDS),
        )
        # Write-through to local DataStore mirror (preserve local-only columns)
        try:
            for r in filled:
                self._put_preserve_temp(r)
        except Exception:
            pass
        return [Image(**r) for r in filled]

    def get_images(self, image_ids: List[int]) -> List[ImageHandle]:
        """Return handles for the given image ids (missing ids are skipped)."""
        if not image_ids:
            return []
        # 1) Try local DataStore first
        by_id: Dict[int, Image] = {}
        misses: List[int] = []
        for iid in image_ids:
            try:
                row = self._data_store.get(int(iid))
                if row is not None:
                    by_id[int(iid)] = Image(**row)
                else:
                    misses.append(int(iid))
            except Exception:
                misses.append(int(iid))

        # 2) Fetch any misses from backend and write-through to DataStore
        if misses:
            id_list = ", ".join(str(int(i)) for i in misses)
            logs = unify.get_logs(
                context=self._ctx,
                filter=f"image_id in [{id_list}]",
                limit=len(misses),
                from_fields=list(self._BUILTIN_FIELDS),
            )
            for lg in logs:
                try:
                    self._put_preserve_temp(getattr(lg, "entries", {}) or {})
                    img = Image(**lg.entries)
                    by_id[int(img.image_id)] = img
                except Exception:
                    continue

        # Preserve requested order
        handles: List[ImageHandle] = []
        for req_id in image_ids:
            img = by_id.get(int(req_id))
            if img is not None:
                handles.append(ImageHandle(manager=self, image=img))
        return handles

    # ------------------------------ Writes --------------------------------
    def is_pending_id(self, image_id: Union[int, str]) -> bool:
        try:
            iid = int(image_id) if not isinstance(image_id, int) else image_id
        except Exception:
            return False
        return iid >= self._PENDING_BASE

    # Non-blocking create functionality merged into add_images

    async def await_pending(self, pending_ids: List[int]) -> Dict[int, int]:
        """
        Await resolution for the given pending ids.

        - Does not trigger duplicate uploads – each pending id is uploaded at most once,
          scheduled when add_images(..., synchronous=False) is called (or lazily here if missing).
        - Returns a mapping {pending_id -> real_id} for all requested ids that can be
          resolved in this session (either already resolved or after the awaited upload).
        """
        if not pending_ids:
            return {}

        # 0) First, resolve any pids that were already uploaded earlier in
        # this session by scanning the local DataStore snapshot (using the
        # persisted temp_image_id) or using the cached _resolved_pid_map.
        mapping: Dict[int, int] = {}
        snapshot: Dict[str, Dict[str, Any]]
        try:
            snapshot = self._data_store.snapshot()
        except Exception:
            snapshot = {}
        for pid in list(pending_ids):
            # Known from cache?
            rid_cached = self._resolved_pid_map.get(int(pid))
            if isinstance(rid_cached, int):
                mapping[int(pid)] = int(rid_cached)
                continue
            # Scan snapshot for a row with matching temp_image_id and a
            # different (resolved) image_id
            try:
                for _k, row in snapshot.items():
                    try:
                        if int(row.get("temp_image_id", -1)) != int(pid):
                            continue
                        rid = int(row.get("image_id", -1))
                        if rid != int(pid) and rid >= 0:
                            mapping[int(pid)] = rid
                            self._resolved_pid_map[int(pid)] = rid
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        # 1) For any pids not yet resolved, ensure an upload is scheduled
        pending_unresolved: List[int] = [
            int(pid) for pid in pending_ids if int(pid) not in mapping
        ]
        for pid in pending_unresolved:
            self._ensure_upload_started(pid)

        # 2) Await completion for any still-unresolved pids
        to_await: List[asyncio.Future] = []
        pid_for_future: List[int] = []
        for pid in pending_unresolved:
            if int(pid) in mapping:
                continue
            fut = self._pending_uploads.get(int(pid))
            if fut is None:
                continue
            try:
                # Wrap a concurrent future so we can await it in asyncio
                wrapped = asyncio.wrap_future(fut)
            except Exception:
                # Fallback – await in a thread
                async def _wait_in_thread(cf: concurrent.futures.Future[int]) -> int:
                    return await asyncio.to_thread(cf.result)

                wrapped = asyncio.ensure_future(_wait_in_thread(fut))
            to_await.append(wrapped)
            pid_for_future.append(int(pid))

        if to_await:
            results = await asyncio.gather(*to_await, return_exceptions=True)
            for i, res in enumerate(results):
                pid = pid_for_future[i]
                if isinstance(res, Exception):
                    continue
                try:
                    rid = int(res)
                except Exception:
                    continue
                if rid < 0:
                    # Missing/failed upload – do not include in mapping
                    continue
                mapping[int(pid)] = rid
                self._resolved_pid_map[int(pid)] = rid

        return mapping

    # ------------------------------ Upload scheduling ---------------------
    def _ensure_upload_started(self, pending_id: int) -> None:
        if int(pending_id) in self._pending_uploads:
            return
        # Submit a background upload job that returns the real_id
        try:
            fut = self._executor.submit(self._upload_one_sync, int(pending_id))
            self._pending_uploads[int(pending_id)] = fut
        except Exception:
            pass

    def _upload_one_sync(self, pending_id: int) -> int:
        """Blocking upload of a single pending image; returns real_id."""
        try:
            row = self._data_store.get(int(pending_id))
        except Exception:
            row = None
        if not isinstance(row, dict):
            # Nothing to upload; try to find resolved id via snapshot
            try:
                snap = self._data_store.snapshot()
                for _k, r in snap.items():
                    try:
                        if int(r.get("temp_image_id", -1)) == int(pending_id):
                            rid = int(r.get("image_id", -1))
                            if rid >= 0 and rid != int(pending_id):
                                self._resolved_pid_map[int(pending_id)] = rid
                                return rid
                    except Exception:
                        continue
            except Exception:
                pass
            return -1

        payload = {
            "timestamp": row.get("timestamp") or datetime.utcnow(),
            "caption": row.get("caption"),
            "data": row.get("data"),
            "filepath": row.get("filepath"),
        }
        [real_id] = self.add_images([payload])  # reuse existing robust path
        try:
            rid = int(real_id)
        except Exception:
            return -1

        # Re-key local DataStore to the resolved id and preserve temp_image_id
        try:
            src = self._data_store.get(int(pending_id))
        except Exception:
            src = None
        if isinstance(src, dict):
            new_row = dict(src)
            new_row["image_id"] = rid
            if "temp_image_id" not in new_row:
                new_row["temp_image_id"] = int(pending_id)
            try:
                self._data_store.put(new_row)
                try:
                    self._data_store.delete(int(pending_id))
                except Exception:
                    pass
            except Exception:
                pass

        self._resolved_pid_map[int(pending_id)] = rid
        return rid

    def add_images(
        self,
        items: List[Dict[str, Any]],
        *,
        synchronous: bool = True,
        return_handles: bool = False,
    ) -> Union[List[int], List[Optional[ImageHandle]]]:
        """
        Add new images. Each item may include ``timestamp``, ``caption``, ``data``,
        and ``filepath``.

        Extended support
        ----------------
        - ``annotation`` (str, optional) may also be provided per item. It is applied
          to returned ``ImageHandle`` instances only (when ``return_handles=True``) and
          is never persisted to the backend or the local ``DataStore``.

        Modes
        -----
        - synchronous=True,  return_handles=False (default): return list[int] ids.
        - synchronous=True,  return_handles=True:  return List[ImageHandle] for created rows.
        - synchronous=False, return_handles=True:  enqueue local pending rows, schedule uploads, return pending List[ImageHandle].
        - synchronous=False, return_handles=False: INVALID → raises ValueError explaining why.
        """
        if (not synchronous) and (not return_handles):
            # Invalid pairing per spec: non-blocking enqueue requires handles for tracking
            raise ValueError(
                "Invalid argument combination: synchronous=False with return_handles=False. "
                "Non-blocking mode must return ImageHandle instances so callers can await resolution.",
            )
        if not items:
            return []

        # If asynchronous enqueue is requested, create pending rows locally and schedule uploads
        if not synchronous:
            handles: List[Optional[ImageHandle]] = []
            pending_ids: List[int] = []
            for raw in items or []:
                payload = dict(raw or {})
                ts = payload.get("timestamp") or datetime.utcnow()
                d = payload.get("data")
                ann = payload.get("annotation")
                ac_flag = bool(payload.get("auto_caption", True))
                if d is None:
                    handles.append(None)
                    continue
                if isinstance(d, (bytes, bytearray)):
                    d_b64 = base64.b64encode(d).decode("utf-8")
                else:
                    d_b64 = d

                temp_id = next(self._pending_counter)
                row_local: Dict[str, Any] = {
                    "image_id": int(temp_id),
                    "temp_image_id": int(temp_id),
                    "timestamp": ts,
                    "caption": payload.get("caption"),
                    "data": d_b64,
                    "filepath": payload.get("filepath"),
                }
                try:
                    self._data_store.put(row_local)
                except Exception:
                    pass
                handles.append(
                    ImageHandle(
                        manager=self,
                        image=Image(**row_local),
                        annotation=ann,
                        auto_caption=ac_flag,
                    ),
                )
                pending_ids.append(int(temp_id))

            for pid in pending_ids:
                try:
                    self._ensure_upload_started(int(pid))
                except Exception:
                    pass

            # In async mode, only return handles (ids are unknown yet)
            return handles

        # Prepare payloads (convert bytes → base64, or resolve gcs_uri) – sync path only
        prepared: List[Dict[str, Any]] = []
        annotations: List[Optional[str]] = []
        auto_caption_flags: List[bool] = []
        for raw in items or []:
            payload = dict(raw or {})
            ann = payload.pop("annotation", None)
            ac_flag = bool(payload.pop("auto_caption", True))

            gcs_uri = payload.pop("gcs_uri", None)
            data_val = payload.get("data")
            if data_val is None and gcs_uri:
                payload["data"] = _gcs_signed_url(gcs_uri)
            elif data_val is None:
                raise ValueError("'data' or 'gcs_uri' is required for add_images")
            elif isinstance(data_val, (bytes, bytearray)):
                payload["data"] = base64.b64encode(data_val).decode("utf-8")
            img = Image(**payload)
            prepared.append(img.to_post_json())
            annotations.append(ann)
            auto_caption_flags.append(ac_flag)

        # Synchronous create path: Result list aligned to input order; None when a per-item create fails
        out_ids: List[Optional[int]] = [None] * len(prepared)

        # Fast path: batch create to avoid O(N) round trips and allow parallelism upstream
        try:
            resp = unity_create_logs(
                context=self._ctx,
                entries=prepared,
                batched=True,
                add_to_all_context=self.include_in_multi_assistant_table,
            )

            # Helper: write-through to DataStore with a given row payload
            def _put_row(row: Dict[str, Any]) -> None:
                try:
                    self._put_preserve_temp(row)
                except Exception:
                    pass

            handled = False

            # Case 1: list of Log objects
            if isinstance(resp, list):
                for i, lg in enumerate(resp):
                    try:
                        entries = getattr(lg, "entries", {}) or {}
                        iid = entries.get("image_id")
                        if iid is not None:
                            out_ids[i] = int(iid)
                            _put_row(entries)
                    except Exception:
                        continue
                handled = True

            # Case 2: dict response – handle common shapes
            elif isinstance(resp, dict):
                # 2a) logs field present → treat as list-of-logs
                logs_list = resp.get("logs")
                if isinstance(logs_list, list):
                    for i, lg in enumerate(logs_list):
                        try:
                            entries = getattr(lg, "entries", {}) or {}
                            iid = entries.get("image_id")
                            if iid is not None:
                                out_ids[i] = int(iid)
                                _put_row(entries)
                        except Exception:
                            continue
                    handled = True

                # 2b) row_ids present (commonly {"image_id": [..]} or a plain list)
                if not handled and ("row_ids" in resp):
                    row_ids_obj = resp.get("row_ids")
                    ids_list: Optional[List[Any]] = None
                    if isinstance(row_ids_obj, list):
                        ids_list = row_ids_obj
                    elif isinstance(row_ids_obj, dict):
                        ids_list = row_ids_obj.get("image_id")
                        if ids_list is None and row_ids_obj:
                            try:
                                ids_list = next(iter(row_ids_obj.values()))
                            except Exception:
                                ids_list = None
                    if isinstance(ids_list, list):
                        for i, iid in enumerate(ids_list):
                            try:
                                if iid is None:
                                    continue
                                iid_int = int(iid)
                                out_ids[i] = iid_int
                                # Compose a best-effort row for the local DataStore mirror
                                row = dict(prepared[i])
                                row["image_id"] = iid_int
                                _put_row(row)
                            except Exception:
                                continue
                        handled = True

                # 2c) log_event_ids present → fetch logs to resolve image_id values
                if not handled:
                    log_ids = resp.get("log_event_ids") or resp.get("log_ids")
                    if isinstance(log_ids, list) and log_ids:
                        fetched = unify.get_logs(
                            context=self._ctx,
                            from_ids=log_ids,
                            return_ids_only=False,
                        )
                        try:
                            logs_list2 = (
                                fetched.get("logs")
                                if isinstance(fetched, dict)
                                else fetched
                            )
                        except Exception:
                            logs_list2 = fetched
                        if isinstance(logs_list2, list):
                            for i, lg in enumerate(logs_list2):
                                try:
                                    entries = getattr(lg, "entries", {}) or {}
                                    iid = entries.get("image_id")
                                    if iid is not None:
                                        out_ids[i] = int(iid)
                                        _put_row(entries)
                                except Exception:
                                    continue
                            handled = True

            # If none of the above matched, leave out_ids as None entries (fallback below does not run)
        except Exception:
            # Fallback: per-item create; on failure return None for that entry
            for i, payload in enumerate(prepared):
                try:
                    lg = unity_log(
                        context=self._ctx,
                        **payload,
                        new=True,
                        mutable=False,
                        add_to_all_context=self.include_in_multi_assistant_table,
                    )
                    try:
                        self._put_preserve_temp(lg.entries)
                    except Exception:
                        pass
                    try:
                        out_ids[i] = int(lg.entries.get("image_id"))
                    except Exception:
                        out_ids[i] = None
                except Exception:
                    out_ids[i] = None

        if not return_handles:
            # Coerce to Python ints where available; keep None where creation failed
            return [x if isinstance(x, int) else None for x in out_ids]  # type: ignore[return-value]

        # Build handles aligned to input order; None where creation failed
        handles_out: List[Optional[ImageHandle]] = []
        for i, maybe_id in enumerate(out_ids):
            if isinstance(maybe_id, int):
                try:
                    # Prefer the row from the local DataStore mirror
                    row = self._data_store.get(int(maybe_id))
                except Exception:
                    row = None
                if isinstance(row, dict):
                    try:
                        handles_out.append(
                            ImageHandle(
                                manager=self,
                                image=Image(**row),
                                annotation=(
                                    annotations[i] if i < len(annotations) else None
                                ),
                                auto_caption=(
                                    auto_caption_flags[i]
                                    if i < len(auto_caption_flags)
                                    else False
                                ),
                            ),
                        )
                        continue
                    except Exception:
                        pass
                # Fallback: reconstruct from prepared payload + resolved id
                try:
                    row_guess = dict(prepared[i])
                    row_guess["image_id"] = int(maybe_id)
                    handles_out.append(
                        ImageHandle(
                            manager=self,
                            image=Image(**row_guess),
                            annotation=annotations[i] if i < len(annotations) else None,
                            auto_caption=(
                                auto_caption_flags[i]
                                if i < len(auto_caption_flags)
                                else False
                            ),
                        ),
                    )
                except Exception:
                    handles_out.append(None)
            else:
                handles_out.append(None)

        return handles_out

    def update_images(self, updates: List[Dict[str, Any]]) -> List[int]:
        """
        Update existing images. Each update dict must include ``image_id`` and may
        set ``timestamp``, ``caption``, ``data``, and/or ``filepath``.
        Returns updated ids.
        """
        updated: List[int] = []
        for change in updates or []:
            if not isinstance(change, dict):
                continue
            if "image_id" not in change:
                raise ValueError("Each update must include 'image_id'.")
            image_id = int(change["image_id"])
            entries: Dict[str, Any] = {}
            if "timestamp" in change and change["timestamp"] is not None:
                entries["timestamp"] = change["timestamp"]
            if "caption" in change:
                entries["caption"] = change["caption"]
            if "filepath" in change:
                entries["filepath"] = change["filepath"]
            if "data" in change and change["data"] is not None:
                d = change["data"]
                if isinstance(d, (bytes, bytearray)):
                    d = base64.b64encode(d).decode("utf-8")
                entries["data"] = d
                # No per-log explicit_types needed; field is strongly typed in schema
            if not entries:
                continue
            ids = unify.get_logs(
                context=self._ctx,
                filter=f"image_id == {image_id}",
                limit=2,
                return_ids_only=True,
            )
            if not ids:
                continue
            if len(ids) > 1:
                raise RuntimeError(
                    f"Multiple rows found with image_id {image_id}. Data integrity issue.",
                )
            unify.update_logs(
                logs=[ids[0]],
                context=self._ctx,
                entries=entries,
                overwrite=True,
            )
            # Refresh from backend and write-through to DataStore (preserve temp id)
            try:
                rows = unify.get_logs(
                    context=self._ctx,
                    filter=f"image_id == {image_id}",
                    limit=1,
                    from_fields=list(self._BUILTIN_FIELDS),
                )
                if rows:
                    self._put_preserve_temp(rows[0].entries)
            except Exception:
                pass
            updated.append(image_id)
        return updated

    # ------------------------------ Resolution -----------------------------
    @functools.wraps(BaseImageManager.resolve_filepath, updated=())
    def resolve_filepath(self, filepath: str) -> int:
        from pathlib import Path
        from datetime import timezone

        existing = self.filter_images(
            filter=f"filepath == '{filepath}'",
            limit=1,
        )
        if existing:
            return existing[0].image_id

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(
                f"No image with filepath '{filepath}' in the Images context "
                f"and the file does not exist on disk",
            )

        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        ids = self.add_images(
            [
                {
                    "data": b64,
                    "filepath": filepath,
                    "timestamp": datetime.now(timezone.utc),
                },
            ],
            synchronous=True,
        )
        image_id = ids[0] if ids else None
        if image_id is None:
            raise RuntimeError(
                f"Backend rejected upload for filepath '{filepath}'",
            )
        return image_id

    def warm_embeddings(self) -> None:
        try:
            ensure_vector_column(
                self._ctx,
                embed_column="_caption_emb",
                source_column="caption",
            )
        except Exception:
            pass

    # ------------------------------ Maintenance ---------------------------
    @functools.wraps(BaseImageManager.clear, updated=())
    def clear(self) -> None:
        unify.delete_context(self._ctx)

        # Ensure the schema exists again via shared provisioning helper
        ContextRegistry.refresh(self, "Images")

        # Clear local DataStore cache for this context
        try:
            self._data_store.clear()
        except Exception:
            pass

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
