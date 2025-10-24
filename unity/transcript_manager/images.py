from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import unify

from ..image_manager.image_manager import ImageManager
from .types.message import Message


def get_images_for_message(self, *, message_id: int) -> List[Dict[str, Any]]:
    """Return image metadata (no raw data) for a message's image references."""
    logs = unify.get_logs(
        context=self._transcripts_ctx,
        filter=f"message_id == {int(message_id)}",
        limit=1,
        from_fields=list(Message.model_fields.keys()),
    )
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
            image_ids.append(int(ref.raw_image_ref.image_id))
        except Exception:
            continue
    handles = self._image_manager.get_images(image_ids)
    by_id = {h.image_id: h for h in handles}
    out: List[Dict[str, Any]] = []
    for ref in refs:
        try:
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


async def ask_image(self, *, image_id: int, question: str) -> str:
    """Ask a one‑off question about a specific stored image and return text."""
    handles = self._image_manager.get_images([int(image_id)])
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
) -> Dict[str, Any]:
    """Attach a single image (raw base64) as persistent context payload."""
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


def attach_message_images_to_context(
    self,
    *,
    message_id: int,
    limit: int = 3,
) -> Dict[str, Any]:
    """Attach multiple images referenced by a message into the loop context."""
    logs = unify.get_logs(
        context=self._transcripts_ctx,
        filter=f"message_id == {int(message_id)}",
        limit=1,
        from_fields=list(Message.model_fields.keys()),
    )
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

    handles = self._image_manager.get_images(ids_to_attach)
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
        self._image_manager = ImageManager()
