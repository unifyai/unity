from __future__ import annotations

from contextvars import ContextVar
from contextlib import suppress
from typing import Any, List, Optional

# New typed container for image references
from unity.image_manager.types.image_refs import ImageRefs
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


def append_image_refs_with_source(
    image_refs: ImageRefs | List[RawImageRef | AnnotatedImageRef] | None,
) -> None:
    """
    Append a batch of ImageRefs into the loop context.

    - Registers handles for known image_ids (idempotent) in LIVE_IMAGES_REGISTRY.
    - Records a log entry per image with its annotation (if any).
    """
    try:
        if image_refs is None:
            return
        refs: List[RawImageRef | AnnotatedImageRef]
        if isinstance(image_refs, ImageRefs):
            refs = list(image_refs.root)
        else:
            refs = list(image_refs or [])

        reg = LIVE_IMAGES_REGISTRY.get()
        log = LIVE_IMAGES_LOG.get()
        # Resolve references to handles for any ids not already present
        try:
            missing_ids: List[int] = []
            ids_in_refs: List[int] = []
            for ref in (
                list(image_refs.root)
                if isinstance(image_refs, ImageRefs)
                else list(image_refs)
            ):
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
                from unity.image_manager.image_manager import (
                    ImageManager as _ImageManager,
                )  # local import to avoid cycles

                manager = _ImageManager()
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
                from unity.image_manager.image_manager import (
                    ImageManager as _ImageManager,
                )  # local import

                ids_to_fetch: List[int] = []
                refs_list = (
                    list(images.root) if isinstance(images, ImageRefs) else list(images)
                )
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
                    manager = _ImageManager()
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
            refs = list(images.root) if isinstance(images, ImageRefs) else list(images)
            for ref in refs:
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
) -> dict[str, Any]:
    """
    Construct helper tools for working with live images within a loop.
    - live_images_overview
    - ask_image
    - attach_image_raw
    """
    from contextlib import suppress as _suppress
    from datetime import timedelta as _timedelta

    id_to_handle: dict[int, Any] = {}
    listings: list[str] = []

    # Build id → handle map and enriched listings from registry and logs
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

    overview_doc = (
        "Live images available in the current session (calling this overview is optional).\n"
        + "\n".join(listings or ["(none)"])
        + "\n\n"
        + "Notes:\n"
        + "- `ask_image` accepts only two arguments: `image_id` and `question`.\n"
        + "- Some dynamic helpers (e.g. `interject_…`, `clarify_…`, `stop_…`) may accept `image_refs` using the `ImageRefs` model:\n"
        + "  a list of `RawImageRef` (just an id) or `AnnotatedImageRef` (id + freeform annotation).\n"
        + "  The annotation should briefly explain how the image relates to the current request.\n"
        + "  Example: [{ 'raw_image_ref': { 'image_id': 42 }, 'annotation': 'Jenny\u2019s paint' }]\n"
    )

    # Merge previously appended images (if any)
    with _suppress(Exception):
        prior = LIVE_IMAGES_LOG.get() or []
    if prior:
        prior_lines = []
        for rec in prior:
            try:
                _iid = int(rec.get("image_id"))
                _annotation = rec.get("annotation")
                _caption = None
                _ts = ""
                with _suppress(Exception):
                    _h = id_to_handle.get(_iid)
                    _caption = getattr(_h, "caption", None)
                    _ts = getattr(
                        getattr(_h, "timestamp", None),
                        "isoformat",
                        lambda: "",
                    )()
                prior_lines.append(
                    f"- id={_iid}, caption={_caption!r}, timestamp={_ts!r}, annotation={_annotation!r}",
                )
            except Exception:
                continue
        if prior_lines:
            overview_doc = (
                overview_doc
                + "\n\nAppended images (this session):\n"
                + "\n".join(prior_lines)
            )

    async def live_images_overview() -> dict:
        return {"status": "ok"}

    live_images_overview.__doc__ = overview_doc

    # Keep a set of already-attached ids (idempotent attach)
    attached_ids: set[int] = set()

    async def ask_image(
        *,
        image_id: int,
        question: str,
    ) -> Any:
        ih = id_to_handle.get(int(image_id))
        if ih is None:
            return {"error": f"image_id {int(image_id)} not found"}
        try:
            return await ih.ask(question)
        except Exception as _exc:  # noqa: BLE001
            return {"error": str(_exc)}

    async def attach_image_raw(*, image_id: int, note: str | None = None) -> dict:
        iid = int(image_id)
        if iid in attached_ids:
            return {"status": "already_attached", "image_id": iid}
        ih = id_to_handle.get(iid)
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
        "- Returns the answer from the image handle."
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
    )

    return {
        "live_images_overview": live_images_overview,
        "ask_image": ask_image,
        "attach_image_raw": attach_image_raw,
    }


async def align_images_for(
    *,
    args: dict,
    hints: list[dict],
) -> dict:  # deprecated helper retained for compatibility
    return {"image_refs": []}


def refresh_overview_doc_if_present(normalized_tools: dict) -> None:
    """Refresh live_images_overview docstring to include current appended images log."""
    try:
        if "live_images_overview" not in normalized_tools:
            return
        fn = normalized_tools["live_images_overview"].fn
        base_doc = getattr(fn, "__doc__", "") or ""
        sep = "\n\nAppended images (this session):\n"
        if sep in base_doc:
            base_doc = base_doc.split(sep, 1)[0]

        prior_lines = []
        with suppress(Exception):
            for rec in LIVE_IMAGES_LOG.get() or []:
                try:
                    prior_lines.append(
                        f"- source={rec.get('source')}, id={int(rec.get('image_id'))}, annotation={rec.get('annotation')!r}",
                    )
                except Exception:
                    continue
        fn.__doc__ = base_doc + (sep + "\n".join(prior_lines) if prior_lines else "")
    except Exception:
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
