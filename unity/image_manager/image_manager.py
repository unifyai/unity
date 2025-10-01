from __future__ import annotations

import base64
import os
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import unify
from google.cloud import storage
from google.oauth2 import service_account
from google.cloud.exceptions import NotFound


from ..common.async_tool_loop import (
    start_async_tool_use_loop,
    SteerableToolHandle,
    TOOL_LOOP_LINEAGE,
)
from ..common.context_store import TableStore
from ..common.model_to_fields import model_to_fields
from ..common.semantic_search import backfill_rows, fetch_top_k_by_references
from .base import BaseImageManager
from .prompt_builders import build_image_ask_prompt
from .types.image import Image


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
        """Return the decoded image bytes from the stored base64 string."""
        try:
            return base64.b64decode(self._image.data)
        except Exception as exc:
            raise ValueError("Invalid base64 image data") from exc

    async def ask(
        self,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
    ) -> SteerableToolHandle:
        """
        Ask a high-level question about this image using a small tool loop.

        The loop sends the underlying image to the model as an image block.
        If the image is stored as a GCS URL, it generates a temporary signed URL
        to make it accessible to the vision model.
        """

        # Use a vision-capable default
        client = unify.AsyncUnify(
            "gpt-4o@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

        # Build a succinct system message tailored to image Q&A
        client.set_system_message(
            build_image_ask_prompt(
                caption=self._image.caption,
                timestamp=self._image.timestamp,
            ),
        )

        # Provide the image as a user content block (vision input).
        data_str = self._image.data
        content_block: dict

        # Check if the data string is a GCS URL
        is_gcs_url = data_str.startswith("gs://") or data_str.startswith("https://storage.googleapis.com/")

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
                # If signing fails, raise an error as the image is inaccessible
                raise RuntimeError(f"Failed to generate signed URL for GCS image: {e}") from e

        elif isinstance(data_str, str) and (data_str.startswith("http://") or data_str.startswith("https://")):
            # Pass the URL through directly; upstream must ensure it is fetchable
            content_block = {
                "type": "image_url",
                "image_url": {"url": data_str},                                                                           }
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

        handle = start_async_tool_use_loop(
            client=client,
            message=question,
            tools={},
            loop_id=f"ImageHandle.ask({self.image_id})",
            parent_lineage=TOOL_LOOP_LINEAGE.get([]),
            max_consecutive_failures=1,
            timeout=90,
        )

        if _return_reasoning_steps:
            original_result = handle.result

            async def wrapped_result():
                answer = await original_result()
                return answer, client.messages

            handle.result = wrapped_result  # type: ignore[assignment]

        return handle


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

        # Initialize the storage client
        try:
            # Assumes the credentials file is at the root of the project
            credentials_path = os.path.join(os.path.dirname(__file__), "..", "..", "application_default_credentials.json")
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.storage_client = storage.Client(credentials=credentials)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Cloud Storage client: {e}") from e

        # Ensure context/fields exist deterministically
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

    # ------------------------------ Reads ---------------------------------
    def filter_images(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Image]:
        logs = unify.get_logs(
            context=self._ctx,
            filter=filter,
            offset=offset,
            limit=limit,
            from_fields=list(self._BUILTIN_FIELDS),
        )
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
        return [Image(**r) for r in filled]

    def get_images(self, image_ids: List[int]) -> List[ImageHandle]:
        """Return handles for the given image ids (missing ids are skipped)."""
        if not image_ids:
            return []
        id_list = ", ".join(str(int(i)) for i in image_ids)
        logs = unify.get_logs(
            context=self._ctx,
            filter=f"image_id in [{id_list}]",
            limit=len(image_ids),
            from_fields=list(self._BUILTIN_FIELDS),
        )
        by_id: Dict[int, Image] = {}
        for lg in logs:
            try:
                img = Image(**lg.entries)
                by_id[int(img.image_id)] = img
            except Exception:
                continue
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
        out_ids: List[int] = []
        for raw in items or []:
            payload = dict(raw or {})
            data_val = payload.get("data")
            if data_val is None:
                raise ValueError("'data' is required for add_images")
            if isinstance(data_val, (bytes, bytearray)):
                payload["data"] = base64.b64encode(data_val).decode("utf-8")
            img = Image(**payload)
            # Preserve explicit_types from the model (marks data as type=image)
            log = unify.log(
                context=self._ctx,
                **img.to_post_json(),
                new=True,
                mutable=None,
            )
            try:
                out_ids.append(int(log.entries["image_id"]))
            except Exception:
                try:
                    last = unify.get_logs(
                        context=self._ctx,
                        sorting={"image_id": "descending"},
                        limit=1,
                    )
                    if last:
                        out_ids.append(int(last[0].entries.get("image_id")))
                except Exception:
                    out_ids.append(-1)
        return out_ids

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
            updated.append(image_id)
        return updated
