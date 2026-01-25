from __future__ import annotations

from contextvars import ContextVar
from contextlib import suppress
from typing import Any, List, Optional
import inspect
from .tools_utils import create_tool_call_message
from ..llm_helpers import _dumps, short_id

# New typed container for image references
from unity.image_manager.types.image_refs import ImageRefs, AnnotatedImageRefs
from unity.image_manager.types.raw_image_ref import RawImageRef
from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef

# from unity.image_manager.image_manager import ImageManager  # avoid import-time cycles; use local imports instead


# Loop-scoped registry of live images (id -> ImageHandle) for validation/lookup.
LIVE_IMAGES_REGISTRY: ContextVar[dict[int, Any]] = ContextVar(
    "LIVE_IMAGES_REGISTRY",
    default={},
)

# Loop-scoped log entries for image overview.
# Each entry: {"image_id": int, "annotation": str | None}
LIVE_IMAGES_LOG: ContextVar[list[dict]] = ContextVar(
    "LIVE_IMAGES_LOG",
    default=[],
)


# NOTE: All legacy character-index alignment and span parsing has been removed.
parse_arg_scoped_span = None  # maintained for import compatibility; not used


def extract_alignment_text_from_value(value: Any) -> str:
    """
    Legacy helper retained for compatibility elsewhere; returns best-effort text.
    Alignment indices are no longer used; this is only used for logging context.
    """
    try:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return str(value.get("content", ""))
        if isinstance(value, list):
            return str(value[0]) if value else ""
        return str(value)
    except Exception:
        return ""


# Legacy parse function removed; keep a placeholder for imports.
parse_source_scoped_span = None


def append_images_with_source(
    images: ImageRefs | List[RawImageRef | AnnotatedImageRef] | None,
) -> None:
    """
    Append a batch of ImageRefs into the loop context.

    - Registers handles for known image_ids (idempotent) in LIVE_IMAGES_REGISTRY.
    - Records a log entry per image with its annotation (if any).
    """
    try:
        if images is None:
            return
        # Support ImageRefs, AnnotatedImageRefs, RawImageRefs, or plain list via duck typing on `root`
        items = getattr(images, "root", images) or []
        refs: List[RawImageRef | AnnotatedImageRef] = list(items)

        reg = LIVE_IMAGES_REGISTRY.get()
        log = LIVE_IMAGES_LOG.get()
        # Resolve references to handles for any ids not already present
        try:
            missing_ids: List[int] = []
            ids_in_refs: List[int] = []
            for ref in list(items):
                if isinstance(ref, AnnotatedImageRef):
                    iid = int(ref.raw_image_ref.image_id)
                elif isinstance(ref, RawImageRef):
                    iid = int(ref.image_id)
                else:
                    continue
                ids_in_refs.append(iid)
                if int(iid) not in reg:
                    missing_ids.append(int(iid))
            if missing_ids:
                from unity.manager_registry import (
                    ManagerRegistry,
                )  # local import to avoid cycles

                manager = ManagerRegistry.get_image_manager()
                handles = manager.get_images(missing_ids)
                for h in handles:
                    try:
                        reg[int(getattr(h, "image_id", -1))] = h
                    except Exception:
                        continue
        except Exception:
            pass

        # Append log entry (no source)
        with suppress(Exception):
            for ref in refs:
                try:
                    if isinstance(ref, AnnotatedImageRef):
                        image_id = int(ref.raw_image_ref.image_id)
                        annotation = str(ref.annotation)
                    elif isinstance(ref, RawImageRef):
                        image_id = int(ref.image_id)
                        annotation = None
                    else:
                        continue
                    log.append(
                        {
                            "image_id": image_id,
                            "annotation": annotation,
                        },
                    )
                except Exception:
                    continue
    except Exception:
        return


# NOTE: Compatibility wrappers for source-labelled images have been removed.


pass  # module-level placeholder to preserve import ordering


# Removed: next_source_index (source labels no longer used)


# Removed: default_source_label (source labels no longer used)


def normalize_arg_scoped_images(
    *args,
    **kwargs,
):  # removed – retained for import compatibility
    return args[0] if args else {}


def set_live_images_context(
    images: ImageRefs | List[RawImageRef | AnnotatedImageRef] | None,
    reference_message: Any | None = None,
) -> tuple[Any, Any]:
    """
    Seed LIVE_IMAGES_REGISTRY and LIVE_IMAGES_LOG for the current loop scope using ImageRefs.

    Returns (registry_token, log_token) to allow resetting later.
    """
    try:
        # Seed registry from any existing registry and resolve referenced ids
        reg_current = LIVE_IMAGES_REGISTRY.get()
        id_map: dict[int, Any] = {}
        with suppress(Exception):
            id_map.update(reg_current if isinstance(reg_current, dict) else {})
        # Resolve referenced ids to handles where missing
        # Resolve referenced ids lazily with a local import to avoid circular dependencies at module import
        if images:
            try:
                from unity.manager_registry import ManagerRegistry  # local import

                ids_to_fetch: List[int] = []
                # Support ImageRefs, AnnotatedImageRefs, RawImageRefs, or plain list via duck typing on `root`
                refs_list = list(getattr(images, "root", images) or [])
                for ref in refs_list:
                    if isinstance(ref, AnnotatedImageRef):
                        iid = int(ref.raw_image_ref.image_id)
                    elif isinstance(ref, RawImageRef):
                        iid = int(ref.image_id)
                    else:
                        continue
                    if iid not in id_map:
                        ids_to_fetch.append(iid)
                if ids_to_fetch:
                    manager = ManagerRegistry.get_image_manager()
                    handles = manager.get_images(ids_to_fetch)
                    for h in handles:
                        try:
                            id_map[int(getattr(h, "image_id", -1))] = h
                        except Exception:
                            continue
            except Exception:
                pass
        reg_token = LIVE_IMAGES_REGISTRY.set(id_map)

        # Seed log from refs under source 'user_message'
        logs: list[dict] = []
        if images is not None:
            items = list(getattr(images, "root", images) or [])
            for ref in items:
                if isinstance(ref, AnnotatedImageRef):
                    logs.append(
                        {
                            "image_id": int(ref.raw_image_ref.image_id),
                            "annotation": str(ref.annotation),
                        },
                    )
                elif isinstance(ref, RawImageRef):
                    logs.append(
                        {
                            "image_id": int(ref.image_id),
                            "annotation": None,
                        },
                    )
        log_token = LIVE_IMAGES_LOG.set(logs)
        return reg_token, log_token
    except Exception:
        return None, None


def reset_live_images_context(registry_token: Any, log_token: Any) -> None:
    with suppress(Exception):
        if registry_token is not None:
            LIVE_IMAGES_REGISTRY.reset(registry_token)
    with suppress(Exception):
        if log_token is not None:
            LIVE_IMAGES_LOG.reset(log_token)


# ── Helper tools for live images (overview, ask, attach) ─────────────────────
def build_live_image_tools(
    reference_message: Any,
    *,
    append_user_messages,
    client: Any = None,
    parent_chat_context: Optional[list[dict]] = None,
    propagate_chat_context: bool = True,
) -> dict[str, Any]:
    """
    Construct helper tools for working with live images within a loop.
    - ask_image
    - attach_image_raw
    """
    from contextlib import suppress as _suppress
    from datetime import timedelta as _timedelta

    id_to_handle: dict[int, Any] = {}
    listings: list[str] = []

    # Build id → handle map and enriched listings from registry and logs (initial snapshot for docs only)
    with _suppress(Exception):
        reg = LIVE_IMAGES_REGISTRY.get() or {}
        logs = LIVE_IMAGES_LOG.get() or []
    for iid, ih in getattr(reg, "items", lambda: [])():
        try:
            id_to_handle[int(iid)] = ih
        except Exception:
            continue
    # Generate listing lines using logs (uniform format, no source; include caption and timestamp)
    for rec in logs:
        try:
            _iid = int(rec.get("image_id"))
            _annotation = rec.get("annotation")
            _caption = None
            _ts = ""
            with _suppress(Exception):
                _h = id_to_handle.get(_iid)
                _caption = getattr(_h, "caption", None)
                _ts = getattr(getattr(_h, "timestamp", None), "isoformat", lambda: "")()
            listings.append(
                f"- id={_iid}, caption={_caption!r}, timestamp={_ts!r}, annotation={_annotation!r}",
            )
        except Exception:
            continue

    # Synthetic image overview is injected directly into the transcript elsewhere.

    # Merge previously appended images (if any)
    with _suppress(Exception):
        prior = LIVE_IMAGES_LOG.get() or []
    if prior:
        pass

    # Keep a set of already-attached ids (idempotent attach)
    attached_ids: set[int] = set()

    async def ask_image(
        *,
        image_id: int,
        question: str,
    ) -> Any:
        # Resolve the handle from the current registry; if absent, best-effort fetch from the manager
        iid = int(image_id)
        ih = None
        try:
            cur_reg = LIVE_IMAGES_REGISTRY.get() or {}
            ih = cur_reg.get(iid)
            if ih is None:
                try:
                    from unity.manager_registry import ManagerRegistry  # local import

                    _handles = ManagerRegistry.get_image_manager().get_images([iid])
                    ih = next(
                        (h for h in _handles if int(getattr(h, "image_id", -1)) == iid),
                        None,
                    )
                    if ih is not None:
                        try:
                            cur_reg[iid] = ih
                        except Exception:
                            pass
                except Exception:
                    ih = None
        except Exception:
            ih = None

        if ih is None:
            return {"error": f"image_id {iid} not found"}
        try:
            # Automatically include parent chat context, mirroring nested tool loops
            if propagate_chat_context:
                try:
                    # Avoid duplicating the synthetic header; use current messages only
                    cur_msgs = [
                        m
                        for m in getattr(client, "messages", [])
                        if not m.get("_ctx_header")
                    ]
                except Exception:
                    cur_msgs = []

                ctx_repr = None
                try:
                    from .messages import (
                        chat_context_repr as _chat_ctx_repr,
                    )  # local import

                    ctx_repr = _chat_ctx_repr(parent_chat_context, cur_msgs)
                except Exception:
                    ctx_repr = parent_chat_context or []

                # Pass parent context only when the handle supports it
                fn = getattr(ih, "ask")
                params = inspect.signature(fn).parameters
                if "parent_chat_context_cont" in params:
                    return await ih.ask(
                        question,
                        parent_chat_context_cont=ctx_repr,
                    )
                return await ih.ask(question)

            # Fallback: no propagation
            return await ih.ask(question)
        except Exception as _exc:  # noqa: BLE001
            return {"error": str(_exc)}

    async def attach_image_raw(*, image_id: int, note: str | None = None) -> dict:
        iid = int(image_id)
        if iid in attached_ids:
            return {"status": "already_attached", "image_id": iid}

        # Resolve the handle dynamically from the current registry, with a best-effort fallback to the manager
        ih = None
        try:
            cur_reg = LIVE_IMAGES_REGISTRY.get() or {}
            ih = cur_reg.get(iid)
            if ih is None:
                try:
                    from unity.manager_registry import ManagerRegistry  # local import

                    _handles = ManagerRegistry.get_image_manager().get_images([iid])
                    ih = next(
                        (h for h in _handles if int(getattr(h, "image_id", -1)) == iid),
                        None,
                    )
                    if ih is not None:
                        try:
                            cur_reg[iid] = ih
                        except Exception:
                            pass
                except Exception:
                    ih = None
        except Exception:
            ih = None

        if ih is None:
            return {"error": f"image_id {iid} not found"}

        try:
            data_str = ih._image.data  # type: ignore[attr-defined]
            # GCS signed URL path mirrors ImageHandle.ask
            is_gcs_url = isinstance(data_str, str) and (
                data_str.startswith("gs://")
                or data_str.startswith("https://storage.googleapis.com/")
            )
            content_block: dict
            if is_gcs_url:
                try:
                    from urllib.parse import urlparse as _urlparse

                    parsed_url = _urlparse(data_str)
                    bucket_name = ""
                    object_path = ""
                    if parsed_url.scheme == "gs":
                        bucket_name = parsed_url.netloc
                        object_path = parsed_url.path.lstrip("/")
                    elif parsed_url.hostname == "storage.googleapis.com":
                        parts = parsed_url.path.lstrip("/").split("/", 1)
                        if len(parts) == 2:
                            bucket_name, object_path = parts
                    storage_client = ih._manager.storage_client  # type: ignore[attr-defined]
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(object_path)
                    signed_url = blob.generate_signed_url(version="v4", expiration=_timedelta(hours=1), method="GET")  # type: ignore[name-defined]
                    content_block = {
                        "type": "image_url",
                        "image_url": {"url": signed_url},
                    }
                except Exception:
                    # fallback: try raw bytes
                    raw = ih.raw()
                    import base64 as _b64  # local import

                    head = (
                        bytes(raw[:10]) if isinstance(raw, (bytes, bytearray)) else b""
                    )
                    if head.startswith(b"\xff\xd8"):
                        mime = "image/jpeg"
                    elif head.startswith(b"\x89PNG\r\n\x1a\n"):
                        mime = "image/png"
                    else:
                        mime = "image/png"
                    b64 = _b64.b64encode(raw).decode("ascii")
                    content_block = {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
            elif isinstance(data_str, str) and (
                data_str.startswith("http://")
                or data_str.startswith("https://")
                or data_str.startswith("data:image/")
            ):
                content_block = {
                    "type": "image_url",
                    "image_url": {"url": data_str},
                }
            else:
                raw = ih.raw()
                import base64 as _b64  # local import

                head = bytes(raw[:10]) if isinstance(raw, (bytes, bytearray)) else b""
                if head.startswith(b"\xff\xd8"):
                    mime = "image/jpeg"
                elif head.startswith(b"\x89PNG\r\n\x1a\n"):
                    mime = "image/png"
                else:
                    mime = "image/png"
                b64 = _b64.b64encode(raw).decode("ascii")
                content_block = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }

            await append_user_messages(
                [
                    {
                        "role": "user",
                        "content": (
                            [content_block]
                            if note is None
                            else [
                                {"type": "text", "text": note},
                                content_block,
                            ]
                        ),
                    },
                ],
            )
            attached_ids.add(iid)
            return {"status": "attached", "image_id": iid}
        except Exception as _exc:  # noqa: BLE001
            return {"error": str(_exc)}

    # Docstrings
    ask_image.__doc__ = (
        "Ask a question about one of the live images by its numeric id.\n\n"
        "Parameters\n"
        "----------\n"
        "image_id : int\n"
        "    The unique id of the image (see overview).\n"
        "question : str\n"
        "    The question to ask about the image.\n\n"
        "Behaviour\n"
        "---------\n"
        "- When the caption is vague or context is unknown, START BROAD: first ask a descriptive question such as\n"
        "  'What is shown in this image? What activity appears to be in progress? Which app/page is visible?'\n"
        "  Extract salient, observable details (apps, headings, steps, key text) rather than database fields.\n"
        "- If the caption already clearly describes the scene and intent, you may skip the broad question and ask a targeted\n"
        "  question that reads a specific on-screen element your context suggests exists.\n"
        "- Avoid system-specific identifiers or structured record fields (e.g., ids, names, statuses, queue/thread references,\n"
        "  timestamps) in the first question unless they are clearly visible on-screen.\n\n"
        "Notes\n"
        "-----\n"
        "- Favour this tool for surgical single-image questions.\n"
        "- Prevents 'polluting' the outer context with full image bytes."
    )

    attach_image_raw.__doc__ = (
        "Attach an image (by id) into the current chat as vision context.\n\n"
        "Parameters\n"
        "----------\n"
        "image_id : int\n"
        "    The unique id of the image (see overview).\n"
        "note : str | None\n"
        "    Optional text note; if provided the note and image will be appended in one user message.\n\n"
        "Behaviour\n"
        "---------\n"
        "- Idempotent per image_id (re-attaching the same id is a no-op).\n"
        "- Resolves ids to handles and surfaces them in the overview."
        "Notes\n"
        "-----\n"
        "- Favour this tool for multi-image comparative and/or queries related to the broader task in a more open-end manner."
        "- Ensures all future reasoning in this loop has *full access* to the image data for *maximual context*, good if this data is relevant beyond the scope of a single question."
    )

    return {
        "ask_image": ask_image,
        "attach_image_raw": attach_image_raw,
    }


async def align_images_for(
    *,
    args: dict,
    hints: list[dict],
) -> dict:  # deprecated helper retained for compatibility
    return {"images": []}


def refresh_overview_doc_if_present(normalized_tools: dict) -> None:
    return


# ── Lightweight helpers for logging image attachments ───────────────────────
def get_image_log_entries() -> list[tuple[int, Optional[str]]]:
    """Return all image log entries as (image_id, annotation)."""
    entries: list[tuple[int, Optional[str]]] = []
    try:
        for rec in LIVE_IMAGES_LOG.get() or []:
            try:
                entries.append((int(rec.get("image_id")), rec.get("annotation")))
            except Exception:
                continue
    except Exception:
        return []
    return entries


def has_live_images_context() -> bool:
    try:
        reg = LIVE_IMAGES_REGISTRY.get()
        logs = LIVE_IMAGES_LOG.get()
        return bool(reg) or bool(logs)
    except Exception:
        return False


# Helper: build a synthetic assistant→tool pair for the live images overview
def build_live_images_overview_msgs(reason: str = "") -> tuple[dict, dict]:
    """Return (assistant_msg, tool_msg) representing a live images overview call.

    The assistant message contains a single tool_call to "live_images_overview" and
    the tool message carries a structured payload with AnnotatedImageRefs and
    lightweight per-image metadata (caption, timestamp).
    """
    try:
        reg = LIVE_IMAGES_REGISTRY.get() or {}
    except Exception:
        reg = {}
    try:
        logs = LIVE_IMAGES_LOG.get() or []
    except Exception:
        logs = []

    # Compute the last annotation seen per image id
    last_ann: dict[int, str] = {}
    for rec in logs:
        try:
            _iid = int(rec.get("image_id"))
        except Exception:
            continue
        ann = rec.get("annotation")
        last_ann[_iid] = str(ann) if ann is not None else ""

    annotated_list: list[AnnotatedImageRef] = []
    images_meta: list[dict] = []
    for _iid, _h in getattr(reg, "items", lambda: [])():
        try:
            iid = int(_iid)
        except Exception:
            continue
        ann_txt = last_ann.get(iid) or str(getattr(_h, "annotation", "") or "")
        try:
            annotated_list.append(
                AnnotatedImageRef(
                    raw_image_ref=RawImageRef(image_id=iid),
                    annotation=ann_txt or "",
                ),
            )
        except Exception:
            # Best-effort: skip malformed entries
            continue
        # Enrich with optional metadata
        try:
            images_meta.append(
                {
                    "image_id": iid,
                    "caption": getattr(_h, "caption", None),
                    "timestamp": getattr(
                        getattr(_h, "timestamp", None),
                        "isoformat",
                        lambda: "",
                    )(),
                },
            )
        except Exception:
            pass

    payload = {
        "status": "ok",
        "reason": reason,
        "images": AnnotatedImageRefs.model_validate(annotated_list),
        "images_meta": images_meta,
        "hint": (
            "Forward these images into future tools that declare an 'images' argument (prefer AnnotatedImageRefs). "
            "Rewrite or augment annotations so they align with the delegated question/action (not the original phrasing), "
            "and preserve user-referenced ordering when it matters."
        ),
    }

    call_id = short_id(8)
    asst_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "live_images_overview",
                    "arguments": "{}",
                },
            },
        ],
    }
    tool_msg = create_tool_call_message(
        name="live_images_overview",
        call_id=call_id,
        content=_dumps(payload, indent=4),
    )
    return asst_msg, tool_msg
