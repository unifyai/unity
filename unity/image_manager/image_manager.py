from __future__ import annotations

import base64
import os
import json
import functools
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import unify
from google.cloud import storage
from google.oauth2 import service_account
from google.cloud.exceptions import NotFound


from ..common.context_store import TableStore
from ..common.model_to_fields import model_to_fields
from ..common.semantic_search import backfill_rows, fetch_top_k_by_references
from .base import BaseImageManager
from .prompt_builders import build_image_ask_prompt
from .types.image import Image
from ..common.filter_utils import normalize_filter_expr
from ..common.data_store import DataStore


class ImageHandle:
    """A lightweight handle around a single stored image."""

    def __init__(self, *, manager: "ImageManager", image: Image) -> None:
        self._manager = manager
        self._image = image

    @property
    def image_id(self) -> int:
        return int(self._image.image_id)

    @property
    def caption(self) -> Optional[str]:
        return self._image.caption

    @property
    def timestamp(self) -> datetime:
        return self._image.timestamp

    def raw(self) -> bytes:
        """
        Return the decoded image bytes.

        If the data is a GCS URL, it downloads the content. Otherwise, it assumes
        the data is a base64 string and decodes it.
        """
        # Prefer locally cached base64 data from the DataStore to avoid re-downloading
        try:
            cached = self._manager._data_store.get(self.image_id)
            data_str = cached.get("data") if cached is not None else self._image.data
        except Exception:
            data_str = self._image.data
        is_gcs_url = data_str.startswith("gs://") or data_str.startswith(
            "https://storage.googleapis.com/",
        )

        if is_gcs_url:
            try:
                parsed_url = urlparse(data_str)
                bucket_name = ""
                object_path = ""

                if parsed_url.scheme == "gs":
                    bucket_name = parsed_url.netloc
                    object_path = parsed_url.path.lstrip("/")
                elif parsed_url.hostname == "storage.googleapis.com":
                    path_parts = parsed_url.path.lstrip("/").split("/", 1)
                    if len(path_parts) == 2:
                        bucket_name, object_path = path_parts
                    else:
                        raise ValueError("Invalid GCS HTTPS URL format.")

                if not bucket_name or not object_path:
                    raise ValueError("Could not parse bucket or path from GCS URL.")

                storage_client = self._manager.storage_client
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(object_path)

                if not blob.exists():
                    raise FileNotFoundError(f"Image not found at GCS URL: {data_str}")

                content = blob.download_as_bytes()
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
    ) -> str:
        """
        Ask a high-level question about this image with a single LLM call.

        Sends the underlying image to the model as an image block alongside the
        `question`, and returns the model's textual answer directly (no nested
        tool-use loop).
        If the image is stored as a GCS URL, a temporary signed URL is generated
        to make it accessible to the vision model.
        """
        # Single-call client
        client = unify.AsyncUnify(
            "gpt-5@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "false")),
            reasoning_effort="high",
            service_tier="priority",
        )

        # Build a succinct system message tailored to image Q&A
        client.set_system_message(
            build_image_ask_prompt(
                caption=self._image.caption,
                timestamp=self._image.timestamp,
            ),
        )

        # Provide the image as a user content block (vision input).
        # Prefer cached base64 from the DataStore when available to avoid signing/downloading again
        try:
            cached = self._manager._data_store.get(self.image_id)
            data_str = cached.get("data") if cached is not None else self._image.data
        except Exception:
            data_str = self._image.data
        content_block: dict

        # Check if the data string is a GCS URL
        is_gcs_url = isinstance(data_str, str) and (
            data_str.startswith("gs://")
            or data_str.startswith("https://storage.googleapis.com/")
        )

        if is_gcs_url:
            try:
                parsed_url = urlparse(data_str)
                bucket_name = ""
                object_path = ""

                if parsed_url.scheme == "gs":
                    bucket_name = parsed_url.netloc
                    object_path = parsed_url.path.lstrip("/")
                elif parsed_url.hostname == "storage.googleapis.com":
                    path_parts = parsed_url.path.lstrip("/").split("/", 1)
                    if len(path_parts) == 2:
                        bucket_name, object_path = path_parts
                    else:
                        raise ValueError("Invalid GCS HTTPS URL format.")

                if not bucket_name or not object_path:
                    raise ValueError("Could not parse bucket or path from GCS URL.")

                storage_client = self._manager.storage_client
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(object_path)

                if not blob.exists():
                    raise NotFound(f"File not found at GCS URL: {data_str}")

                # Generate a URL valid for 1 hour
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(hours=1),
                    method="GET",
                )

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

        client.append_messages(
            [
                {
                    "role": "user",
                    "content": [content_block],
                },
            ],
        )

        # Single shot – no nested tool loop
        answer = await client.generate(user_message=question)
        return answer


class ImageManager(BaseImageManager):
    """Concrete implementation backed by Unify contexts and fields."""

    def __init__(self) -> None:
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
        ), "read and write contexts must be the same when instantiating an ImageManager."

        self._ctx = f"{read_ctx}/Images" if read_ctx else "Images"

        # Local DataStore mirror for Images (write-through on reads/writes)
        self._data_store = DataStore.for_context(self._ctx, key_fields=("image_id",))

        # Initialize the storage client
        try:
            # Assumes the credentials file is at the root of the project
            credentials_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "application_default_credentials.json",
            )
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
            )
            self.storage_client = storage.Client(credentials=credentials)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Google Cloud Storage client: {e}",
            ) from e

        # Ensure context/fields exist deterministically
        self._provision_storage()

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
        # Write-through to local DataStore mirror
        try:
            for lg in logs:
                self._data_store.put(lg.entries)
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
        # Write-through to local DataStore mirror
        try:
            for r in filled:
                self._data_store.put(r)
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
                    self._data_store.put(lg.entries)
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
    def add_images(self, items: List[Dict[str, Any]]) -> List[int]:
        """
        Add new images. Each item may include ``timestamp``, ``caption``, ``data``.
        Returns the allocated ``image_id`` values in insertion order.
        """
        if not items:
            return []

        # Prepare payloads (preserve explicit_types; convert bytes → base64)
        prepared: List[Dict[str, Any]] = []
        for raw in items or []:
            payload = dict(raw or {})
            data_val = payload.get("data")
            if data_val is None:
                raise ValueError("'data' is required for add_images")
            if isinstance(data_val, (bytes, bytearray)):
                payload["data"] = base64.b64encode(data_val).decode("utf-8")
            img = Image(**payload)
            prepared.append(img.to_post_json())

        # Result list aligned to input order; None when a per-item create fails
        out_ids: List[Optional[int]] = [None] * len(prepared)

        # Fast path: batch create to avoid O(N) round trips and allow parallelism upstream
        try:
            resp = unify.create_logs(context=self._ctx, entries=prepared, batched=True)

            # Helper: write-through to DataStore with a given row payload
            def _put_row(row: Dict[str, Any]) -> None:
                try:
                    self._data_store.put(row)
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
                    lg = unify.log(context=self._ctx, **payload, new=True, mutable=None)
                    try:
                        self._data_store.put(lg.entries)
                    except Exception:
                        pass
                    try:
                        out_ids[i] = int(lg.entries.get("image_id"))
                    except Exception:
                        out_ids[i] = None
                except Exception:
                    out_ids[i] = None

        # Coerce to Python ints where available; keep None where creation failed
        return [x if isinstance(x, int) else None for x in out_ids]  # type: ignore[return-value]

    def update_images(self, updates: List[Dict[str, Any]]) -> List[int]:
        """
        Update existing images. Each update dict must include ``image_id`` and may
        set ``timestamp``, ``caption``, and/or ``data``. Returns updated ids.
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
            if "data" in change and change["data"] is not None:
                d = change["data"]
                if isinstance(d, (bytes, bytearray)):
                    d = base64.b64encode(d).decode("utf-8")
                entries["data"] = d
                # Ensure backend keeps the data column typed as an image
                existing_et = entries.get("explicit_types") or {}
                et_for_data = dict(existing_et.get("data") or {})
                et_for_data["type"] = "image"
                existing_et["data"] = et_for_data
                entries["explicit_types"] = existing_et
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
            # Refresh from backend and write-through to DataStore
            try:
                rows = unify.get_logs(
                    context=self._ctx,
                    filter=f"image_id == {image_id}",
                    limit=1,
                    from_fields=list(self._BUILTIN_FIELDS),
                )
                if rows:
                    self._data_store.put(rows[0].entries)
            except Exception:
                pass
            updated.append(image_id)
        return updated

    # ------------------------------ Maintenance ---------------------------
    @functools.wraps(BaseImageManager.clear, updated=())
    def clear(self) -> None:
        try:
            # Drop the entire images table for this active assistant context
            unify.delete_context(self._ctx)
        except Exception:
            # Proceed even if deletion fails (context may already be absent)
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

    # ------------------------------ Internals -----------------------------
    def _provision_storage(self) -> None:
        """Ensure Images context and schema exist deterministically."""
        self._store = TableStore(
            self._ctx,
            unique_keys={"image_id": "int"},
            auto_counting={"image_id": None},
            description="Collection of images with timestamps, captions, and raw base64 data.",
            fields=model_to_fields(Image),
        )
        self._store.ensure_context()

        # Cache built-in fields for fast whitelisting
        self._BUILTIN_FIELDS: tuple[str, ...] = tuple(Image.model_fields.keys())
