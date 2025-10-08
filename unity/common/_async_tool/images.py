from __future__ import annotations

from contextvars import ContextVar
from contextlib import suppress
from typing import Any
import re
from unity.image_manager.utils import substring_from_span


# Loop-scoped registry of live images (id -> ImageHandle) for validation/lookup.
# This module is the single source of truth for image-related context.
LIVE_IMAGES_REGISTRY: ContextVar[dict[int, Any]] = ContextVar(
    "LIVE_IMAGES_REGISTRY",
    default={},
)

# Loop-scoped log lines for image overview (source-tagged entries)
LIVE_IMAGES_LOG: ContextVar[list[str]] = ContextVar(
    "LIVE_IMAGES_LOG",
    default=[],
)

# Loop-scoped mapping of source label (e.g. "user_message", "interjection0")
# to the base text used for span alignment so we can display substrings.
LIVE_IMAGES_SOURCE_TEXTS: ContextVar[dict[str, str]] = ContextVar(
    "LIVE_IMAGES_SOURCE_TEXTS",
    default={},
)


# ── Helpers for arg-scoped span keys (e.g. "question[2:9]") ─────────────———
_ARG_SPAN_RX = re.compile(
    r"^(?P<arg>[A-Za-z_]\w*)\[(?P<start>-?\d+)?\:(?P<end>-?\d+)?\]$",
)


def parse_arg_scoped_span(key: str) -> tuple[str, str] | None:
    """
    Parse a key of the form "<arg_name>[start:end]" and return
    (arg_name, "[start:end]") when valid; else None.

    The bracket portion preserves the original indices; downstream helpers can
    compute concrete ranges or substrings with Python-slice semantics.
    """
    try:
        m = _ARG_SPAN_RX.fullmatch(str(key))
        if not m:
            return None
        arg = m.group("arg")
        span = key[key.find("[") :]
        return arg, span
    except Exception:
        return None


def extract_alignment_text_from_value(value: Any) -> str:
    """
    Return a best-effort string to align spans against using the shared rules:
      - str: use as-is
      - dict: use str(value.get("content", ""))
      - list: if chat messages → first with role=="user"; else first text block
    """
    try:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return str(value.get("content", ""))
        if isinstance(value, list):
            # Case 1: chat messages
            for m in value:
                if isinstance(m, dict) and m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, list):
                        parts: list[str] = []
                        for it in c:
                            if isinstance(it, dict) and it.get("type") == "text":
                                parts.append(str(it.get("text", "")))
                            else:
                                parts.append(str(it))
                        return "".join(parts)
                    return str(c)
            # Case 2: content blocks (no roles)
            for it in value:
                if isinstance(it, dict) and it.get("type") == "text":
                    return str(it.get("text", ""))
            return str(value[0]) if value else ""
        # Fallback best-effort stringification
        return str(value)
    except Exception:
        return ""


# ── Helpers for source-scoped keys (e.g. "user_message[0:10]", "this[:]") ───
_SRC_SPAN_RX = re.compile(
    r"^(?P<src>(this|user_message|interjection\d+|ask\d+|clar_request\d+|clar_answer\d+|notification\d+))\[(?P<start>-?\d+)?\:(?P<end>-?\d+)?\]$",
)


def parse_source_scoped_span(key: str) -> tuple[str, str] | None:
    """
    Parse a key of the form "<source>[start:end]" where <source> is one of:
    this, user_message, interjectionN, askN, clar_requestN, clar_answerN, notificationN.
    Return (source, "[start:end]") when valid; else None.
    """
    try:
        m = _SRC_SPAN_RX.fullmatch(str(key))
        if not m:
            return None
        source = m.group("src")
        span = key[key.find("[") :]
        return source, span
    except Exception:
        return None


def append_source_scoped_images(images: dict | None, default_source_label: str) -> None:
    """
    Append `images` (source-scoped mapping) into the loop's live image registry and log.

    Behaviour
    ---------
    - Accepts mapping of key → value where key is either `<source>[start:end]` or omitted
      (treated as `this[:]`), and value is an image id or an ImageHandle.
    - Resolves ids using LIVE_IMAGES_REGISTRY, appends handles idempotently.
    - Records a compact log entry "<source>:<id>:[start:end]" for overview display.
    - If `<source>` is literally `this`, it is mapped to `default_source_label`.
    """
    try:
        if not isinstance(images, dict) or not images:
            return
        reg = LIVE_IMAGES_REGISTRY.get()
        log = LIVE_IMAGES_LOG.get()
        for k, v in images.items():
            parsed = parse_source_scoped_span(str(k))
            if parsed:
                src, span = parsed
                if src == "this":
                    src = default_source_label
            else:
                src, span = default_source_label, "[:]"

            handle = None
            with suppress(Exception):
                if isinstance(v, int):
                    handle = reg.get(int(v)) if isinstance(reg, dict) else None
                elif hasattr(v, "image_id"):
                    handle = v
            if handle is None:
                continue
            with suppress(Exception):
                reg[int(getattr(handle, "image_id", -1))] = handle
            with suppress(Exception):
                log.append(f"{src}:{int(getattr(handle, 'image_id', -1))}:{span}")
    except Exception:
        return


def append_source_scoped_images_with_text(
    images: dict | None,
    prefix: str,
    text: Any,
) -> str | None:
    """
    Convenience wrapper: generate a new source label for the given prefix (e.g.,
    "interjection" → "interjectionN"), record the base text for substring display,
    and append the provided images under that source.
    Returns the computed source label, or None on failure.
    """
    try:
        label = default_source_label(prefix)
        record_source_text(label, text)
        append_source_scoped_images(images, label)
        return label
    except Exception:
        return None


def record_source_text(source_label: str, text: Any) -> None:
    """
    Record a human-readable base text for a given dynamic image source label.

    The text is later used to render the extracted substring alongside indices
    (e.g., for entries like "interjection0[5:11]") in the live overview.
    """
    try:
        if not source_label:
            return
        base_text = extract_alignment_text_from_value(text)
        if base_text is None:
            return
        mapping = LIVE_IMAGES_SOURCE_TEXTS.get()
        if not isinstance(mapping, dict):
            mapping = {}
        mapping = dict(mapping)
        mapping[str(source_label)] = str(base_text)
        LIVE_IMAGES_SOURCE_TEXTS.set(mapping)
    except Exception:
        return


def next_source_index(prefix: str) -> int:
    """Return the next numeric index for a given source prefix based on LIVE_IMAGES_LOG."""
    try:
        log = LIVE_IMAGES_LOG.get()
        if not isinstance(log, list):
            return 0
        return sum(1 for e in log if isinstance(e, str) and e.startswith(prefix))
    except Exception:
        return 0


def default_source_label(prefix: str) -> str:
    """Return a default `<prefix>N` label using the next available index."""
    return f"{prefix}{next_source_index(prefix)}"


def normalize_arg_scoped_images(
    merged_kwargs: dict,
    *,
    tool_name: str | None = None,
    param_names: set[str] | None = None,
) -> dict:
    """
    Normalize arg-scoped images mapping in `merged_kwargs` for inner tool calls.

    Behaviour mirrors prior inline implementation in tools_data:
    - Skip entirely for the helper tool `ask_image` (expects source-scoped keys).
    - Accept `images` mapping with keys of the form `<arg>[start:end]`.
    - Validate that `<arg>` exists in the call arguments/parameters.
    - Resolve value to a live image handle via LIVE_IMAGES_REGISTRY or accept handle objects.
    - Validate spans against the referenced argument text; keep only non-empty matches.
    - Returns the (possibly) updated `merged_kwargs` with a filtered `images` dict.
    """
    try:
        if tool_name == "ask_image":
            return merged_kwargs
        images_val = merged_kwargs.get("images")
        if not isinstance(images_val, dict):
            return merged_kwargs

        raw_images = dict(images_val or {})
        registry = LIVE_IMAGES_REGISTRY.get()
        norm_images: dict[str, Any] = {}
        params = set(param_names or [])

        for key, val in raw_images.items():
            parsed = parse_arg_scoped_span(str(key))
            if not parsed:
                continue
            arg_name, span = parsed

            # Only accept if referenced arg is available in the call
            if arg_name not in params and arg_name not in merged_kwargs:
                continue

            # Resolve id → handle or accept provided handle
            handle = None
            with suppress(Exception):
                if isinstance(val, int):
                    handle = (
                        registry.get(int(val)) if isinstance(registry, dict) else None
                    )
                elif hasattr(val, "image_id"):
                    handle = val
                elif isinstance(val, dict):
                    # Accept explicit id fields inside the dict
                    _id_field = None
                    for _k in ("image_id", "imageId", "id"):
                        if _k in val:
                            _id_field = val[_k]
                            break
                    if _id_field is not None:
                        try:
                            handle = (
                                registry.get(int(_id_field))
                                if isinstance(registry, dict)
                                else None
                            )
                        except Exception:
                            handle = None
                    elif bool(val.get("__handle__")):
                        # Fallback: when a single live image exists, use it
                        if isinstance(registry, dict) and len(registry) == 1:
                            try:
                                handle = next(iter(registry.values()))
                            except Exception:
                                handle = None
            if handle is None:
                continue

            # Validate the span against the referenced argument's text; drop if invalid/empty.
            try:
                align_txt = extract_alignment_text_from_value(
                    merged_kwargs.get(arg_name),
                )
                if align_txt is not None:
                    matched = substring_from_span(str(align_txt), span)
                    if isinstance(matched, str) and matched != "":
                        norm_images[str(key)] = handle
            except Exception:
                continue

        merged_kwargs["images"] = norm_images
        return merged_kwargs
    except Exception:
        # Best-effort; if anything goes wrong, return original
        return merged_kwargs


# ── Context management for live images (registry/log) ─────────────────────────
def set_live_images_context(
    images: dict[str, Any],
    reference_message: Any,
) -> tuple[Any, Any]:
    """
    Seed LIVE_IMAGES_REGISTRY and LIVE_IMAGES_LOG for the current loop scope.

    Returns (registry_token, log_token) to allow resetting later.
    """
    try:
        if not images:
            return None, None
        id_map: dict[int, Any] = {}
        for _k, _ih in images.items():
            with suppress(Exception):
                _iid = int(getattr(_ih, "image_id", -1))
                if _iid >= 0:
                    id_map[_iid] = _ih
        reg_token = LIVE_IMAGES_REGISTRY.set(id_map)

        seed_log: list[str] = []
        with suppress(Exception):
            for _k, _ih in images.items():
                with suppress(Exception):
                    _iid = int(getattr(_ih, "image_id", -1))
                seed_log.append(f"user_message:{_iid}:{_k}")
        log_token = LIVE_IMAGES_LOG.set(seed_log)

        # Also seed source→text mapping so substrings can be shown for user_message
        try:
            base_text = extract_alignment_text_from_value(reference_message)
            mapping = LIVE_IMAGES_SOURCE_TEXTS.get()
            if not isinstance(mapping, dict):
                mapping = {}
            mapping = dict(mapping)
            mapping["user_message"] = str(base_text)
            LIVE_IMAGES_SOURCE_TEXTS.set(mapping)
        except Exception:
            pass
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
    images: dict[str, Any],
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

    # Build id → handle map and enriched listings
    reference_text = extract_alignment_text_from_value(reference_message)
    for span_key, ih in list(images.items()):
        with _suppress(Exception):
            img_id = int(getattr(ih, "image_id", -1))
        if "img_id" not in locals():
            img_id = -1
        id_to_handle[img_id] = ih
        substr = ""
        with _suppress(Exception):
            substr = substring_from_span(str(reference_text), str(span_key))
        if "substr" not in locals():
            substr = ""
        with _suppress(Exception):
            caption = getattr(ih, "caption", None)
        if "caption" not in locals():
            caption = None
        listings.append(
            f"- id={img_id}, span={span_key}, substring={substr!r}, caption={caption!r}",
        )

    overview_doc = (
        "Live images aligned to the current user_message (visible in this description; calling is optional).\n"
        + "\n".join(listings or ["(none)"])
        + "\n\n"
        + "Arg-scoped image keys for inner tools\n"
        + "-----------------------------------\n"
        + "When calling an inner tool that accepts `images`, reference each image with an arg-scoped span key: `<arg>[start:end]`.\n"
        + "- `<arg>` is the name of the tool's string parameter (e.g., `question`, `text`, `prompt`).\n"
        + "- `[start:end]` uses Python-slice semantics over that parameter's text.\n"
        + "- Values are image ids (or handles) that should align to that substring.\n\n"
        + "Example (manual):\n"
        + "  images = { 'question[10:23]': 42 }\n\n"
        + "Example (recommended with helper):\n"
        + "  align_images_for(\n"
        + "    args={ 'question': 'Please compare the Cairo skyline images for clarity' },\n"
        + "    hints=[ { 'arg': 'question', 'substring': 'Cairo skyline', 'image_id': 42 } ]\n"
        + "  )  →  { 'images': { 'question[15:28]': 42 } }\n"
        + "\n"
        + "Source-scoped image keys for dynamic methods\n"
        + "-------------------------------------------\n"
        + "When sending images with dynamic methods (ask, interject, stop, clarify, notifications), use `<source>[start:end]` keys:\n"
        + "- Supported sources: `this`, `user_message`, `interjectionN`, `askN`, `clar_requestN`, `clar_answerN`, `notificationN`, `stopN`.\n"
        + "- `this[:]` is a shorthand that refers to the current outgoing payload (e.g., the very text of this interjection/ask/clarify/notify).\n"
        + "- Values are image ids (or handles). These images are appended to the loop’s live registry and reflected below.\n"
    )

    # Merge previously appended images (if any)
    with _suppress(Exception):
        prior = LIVE_IMAGES_LOG.get()
    if "prior" not in locals():
        prior = []
    if prior:
        prior_lines = []
        for rec in prior:
            with _suppress(Exception):
                src, iid_s, span_key = rec.split(":", 2)
                base_text = ""
                try:
                    base_text = (LIVE_IMAGES_SOURCE_TEXTS.get() or {}).get(src, "")
                except Exception:  # pragma: no cover - defensive
                    base_text = ""
                _substr = ""
                with _suppress(Exception):
                    _substr = substring_from_span(str(base_text), str(span_key))
                prior_lines.append(
                    f"- source={src}, id={int(iid_s)}, span={span_key}, substring={_substr!r}",
                )
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
        images: dict | None = None,
    ) -> Any:
        ih = id_to_handle.get(int(image_id))
        if ih is None:
            return {"error": f"image_id {int(image_id)} not found"}
        # Record source text for this ask-turn so appended images can show substrings
        with _suppress(Exception):
            _label = default_source_label("ask")
            record_source_text(_label, question)
            append_source_scoped_images(images, _label)
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
        "    The question to ask about the image.\n"
        "images : dict | None\n"
        "    Optional source-scoped images mapping to append at the time of this call.\n"
        "    Keys use `<source>[start:end]` (see overview). Use `this[:]` to associate images with the `question` text.\n\n"
        "Behaviour\n"
        "---------\n"
        "- Resolves ids to handles, appends them to this loop’s live registry, and surfaces them in the overview.\n"
        "- Returns a nested handle; await its result for the answer."
    )

    attach_image_raw.__doc__ = (
        "Attach an image (by id) into the current chat as vision context.\n\n"
        "Parameters\n"
        "----------\n"
        "image_id : int\n"
        "    The unique id of the image (see overview).\n"
        "note : str | None\n"
        "    Optional text note; if provided the note and image will be appended in one user message.\n"
        "images : dict | None\n"
        "    Optional source-scoped images mapping to append at the time of this call.\n"
        "    Keys use `<source>[start:end]` (see overview). Use `this[:]` to associate images with `note` (or with an empty string when `note` is None).\n\n"
        "Behaviour\n"
        "---------\n"
        "- Idempotent per image_id (re-attaching the same id is a no-op).\n"
        "- Resolves ids to handles, appends them to this loop’s live registry, and surfaces them in the overview."
    )

    return {
        "live_images_overview": live_images_overview,
        "ask_image": ask_image,
        "attach_image_raw": attach_image_raw,
    }


async def align_images_for(*, args: dict, hints: list[dict]) -> dict:
    """
    Prepare arg‑scoped `images` for an upcoming inner tool call, without manual counting.

    Converts human-friendly substring hints into arg-scoped span keys of the form
    `<arg>[start:end]`, which inner tools that accept `images` can consume directly.
    """
    out: dict[str, int] = {}
    try:
        arg_texts = {str(k): str(v) for k, v in dict(args or {}).items()}
    except Exception:
        arg_texts = {}

    def _extract_id(obj: dict) -> int | None:
        for k in ("image_id", "imageId", "id"):
            if k in obj:
                try:
                    return int(obj[k])
                except Exception:
                    return None
        return None

    def _extract_arg(obj: dict) -> str | None:
        for k in ("arg", "argument", "arg_name", "name"):
            if k in obj:
                return str(obj[k])
        return None

    def _extract_substring(obj: dict) -> str | None:
        for k in ("substring", "text", "span_text"):
            if k in obj:
                return str(obj[k])
        return None

    for item in list(hints or []):
        if not isinstance(item, dict):
            continue
        iid = _extract_id(item)
        arg_name = _extract_arg(item)
        sub = _extract_substring(item)
        if iid is None or not arg_name or sub is None:
            continue
        base = arg_texts.get(arg_name)
        if not isinstance(base, str):
            continue
        try:
            start = base.find(sub)
            if start < 0:
                continue
            end = start + len(sub)
            key = f"{arg_name}[{start}:{end}]"
            out[key] = iid
        except Exception:
            continue

    return {"images": out}


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
                    src, iid_s, span_key = rec.split(":", 2)
                    # Attempt to compute substring if we have source text
                    base_text = (LIVE_IMAGES_SOURCE_TEXTS.get() or {}).get(src, "")
                    _substr = ""
                    try:
                        _substr = substring_from_span(str(base_text), str(span_key))
                    except Exception:
                        _substr = ""
                    prior_lines.append(
                        f"- source={src}, id={int(iid_s)}, span={span_key}, substring={_substr!r}",
                    )
                except Exception:
                    continue
        fn.__doc__ = base_doc + (sep + "\n".join(prior_lines) if prior_lines else "")
    except Exception:
        return
