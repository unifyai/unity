from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import unify

from ..manager_registry import ManagerRegistry
from .types.message import Message


def _image_destination_for_transcript_context(context: str) -> str:
    """Return the Images destination that matches a concrete Transcripts context."""

    if context.startswith("Spaces/"):
        return f"space:{context.split('/')[1]}"
    return "personal"


def get_images_for_message(self, *, message_id: int) -> List[Dict[str, Any]]:
    """Return image metadata (no raw data) for a message's image references."""
    logs = []
    transcript_context = self._transcripts_ctx
    for context in self._read_transcript_contexts():
        logs = unify.get_logs(
            context=context,
            filter=f"message_id == {int(message_id)}",
            limit=1,
            from_fields=list(Message.model_fields.keys()),
        )
        if logs:
            transcript_context = context
            break
    if not logs:
        return []
    try:
        msg = Message(**logs[0].entries)
    except Exception:
        return []
    refs = getattr(msg.images, "root", None) or []
    if not refs:
        return []
    image_ids: List[int] = []
    for ref in refs:
        try:
            if ref.raw_image_ref.image_id is not None:
                image_ids.append(int(ref.raw_image_ref.image_id))
        except Exception:
            continue
    handles = self._image_manager.get_images(
        image_ids,
        destination=_image_destination_for_transcript_context(transcript_context),
    )
    image_destination = _image_destination_for_transcript_context(transcript_context)
    for image_id in image_ids:
        self._image_destinations_by_id[int(image_id)] = image_destination
    by_id = {h.image_id: h for h in handles}
    out: List[Dict[str, Any]] = []
    for ref in refs:
        try:
            if ref.raw_image_ref.image_id is None:
                continue
            iid = int(ref.raw_image_ref.image_id)
        except Exception:
            continue
        h = by_id.get(iid)
        if h is None:
            continue
        try:
            ts_str = h.timestamp.isoformat()
        except Exception:
            ts_str = ""
        out.append(
            {
                "image_id": int(h.image_id),
                "caption": h.caption,
                "timestamp": ts_str,
                "annotation": ref.annotation,
            },
        )
    return out


async def ask_image(
    self,
    *,
    image_id: int,
    question: str,
    destination: str | None = None,
) -> str:
    """Ask a one‑off question about a specific stored image and return text."""
    resolved_destination = destination or self._image_destinations_by_id.get(
        int(image_id),
    )
    handles = self._image_manager.get_images(
        [int(image_id)],
        destination=resolved_destination,
    )
    if not handles:
        raise ValueError(f"No image found with image_id {image_id}")
    handle = handles[0]
    answer = await handle.ask(question)
    if not isinstance(answer, str):
        answer = str(answer)
    return answer


def attach_image_to_context(
    self,
    *,
    image_id: int,
    note: Optional[str] = None,
    destination: str | None = None,
) -> Dict[str, Any]:
    """Attach a single image (raw base64) as persistent context payload."""
    resolved_destination = destination or self._image_destinations_by_id.get(
        int(image_id),
    )
    handles = self._image_manager.get_images(
        [int(image_id)],
        destination=resolved_destination,
    )
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


def attach_message_images_to_context(
    self,
    *,
    message_id: int,
    limit: int = 3,
) -> Dict[str, Any]:
    """Attach multiple images referenced by a message into the loop context."""
    logs = []
    transcript_context = self._transcripts_ctx
    for context in self._read_transcript_contexts():
        logs = unify.get_logs(
            context=context,
            filter=f"message_id == {int(message_id)}",
            limit=1,
            from_fields=list(Message.model_fields.keys()),
        )
        if logs:
            transcript_context = context
            break
    if not logs:
        return {"attached_count": 0, "images": []}
    try:
        msg = Message(**logs[0].entries)
    except Exception:
        return {"attached_count": 0, "images": []}
    refs = getattr(msg.images, "root", None) or []
    if not refs:
        return {"attached_count": 0, "images": []}

    ids_to_attach: List[int] = []
    annotations_by_index: List[str] = []
    for ref in refs:
        try:
            if ref.raw_image_ref.image_id is not None:
                ids_to_attach.append(int(ref.raw_image_ref.image_id))
                annotations_by_index.append(ref.annotation)
        except Exception:
            continue

    if limit is not None:
        try:
            limit = int(limit)
        except Exception:
            limit = 3
        if limit >= 0:
            ids_to_attach = ids_to_attach[:limit]
            annotations_by_index = annotations_by_index[:limit]

    handles = self._image_manager.get_images(
        ids_to_attach,
        destination=_image_destination_for_transcript_context(transcript_context),
    )
    images: List[Dict[str, Any]] = []
    for idx, h in enumerate(handles):
        try:
            raw_bytes = h.raw()
            b64 = base64.b64encode(raw_bytes).decode("utf-8")
        except Exception:
            continue
        annotation_val = (
            annotations_by_index[idx] if idx < len(annotations_by_index) else ""
        )
        images.append(
            {
                "meta": {
                    "image_id": int(h.image_id),
                    "caption": h.caption,
                    "timestamp": getattr(h.timestamp, "isoformat", lambda: "")(),
                    "annotation": annotation_val,
                },
                "image": b64,
            },
        )
    return {"attached_count": len(images), "images": images}


def ensure_image_manager(self) -> None:
    """Ensure a lazy ImageManager exists on the manager instance."""
    if not hasattr(self, "_image_manager") or self._image_manager is None:
        self._image_manager = ManagerRegistry.get_image_manager()
