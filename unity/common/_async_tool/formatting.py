from __future__ import annotations

from contextlib import suppress
from typing import Any, Union
import json

from ..llm_helpers import _dumps, _strip_image_keys, _collect_images


def serialize_tool_content(
    *,
    tool_name: str,
    payload: Any,
    is_final: bool,
) -> Union[str, list]:
    """
    Produce the exact content that will be inserted into the transcript for a tool message.

    - When is_final=True:
      - Serialize payload and promote any embedded base64 images into image_url blocks.
      - If there are images, the content becomes a list of blocks (text first, then image_url items).
      - If there are no images, the content is a pretty-printed JSON string.

    - When is_final=False (progress/notification placeholder):
      - Wrap the payload as {"tool": tool_name, ...} and serialize to a pretty-printed JSON string.
    """

    if not is_final:
        content_payload = (
            payload if isinstance(payload, dict) else {"message": str(payload)}
        )
        # Keep the tool name visible for progress/notification placeholders
        return _dumps(
            {"tool": tool_name, **content_payload},
            indent=4,
            context={"prune_empty": True, "shorthand": True},
        )

    # Final result path – promote embedded images, keep a clean textual view without raw base64
    # Additionally, when payload is a pure string that contains JSON, parse & pretty-print it; otherwise keep as-is.
    parsed_payload = payload
    if isinstance(payload, str):
        with suppress(Exception):
            maybe = json.loads(payload)
            # Only adopt parsed structure when it's a JSON object/array; otherwise treat as plain text
            if isinstance(maybe, (dict, list)):
                parsed_payload = maybe

    images: list[str] = []
    with suppress(Exception):
        _collect_images(parsed_payload, images)

    # Always apply prune + shorthand for final tool results so models that support
    # these modes render compactly and consistently in tool outputs
    if isinstance(parsed_payload, (dict, list)):
        text_repr = _dumps(
            _strip_image_keys(parsed_payload),
            indent=4,
            context={"prune_empty": True, "shorthand": True},
        )
    else:
        # Fallback: always provide a string for non-structured payloads
        # (e.g., numbers, booleans). Preserve plain strings as-is.
        text_repr = payload if isinstance(payload, str) else str(payload)

    if images:
        content_blocks: list = []
        if text_repr and text_repr != "{}":
            content_blocks.append({"type": "text", "text": text_repr})
        for b64 in images:
            # Default to PNG when unknown – mirrors existing behaviour
            content_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            )
        return content_blocks

    return text_repr


def _sanitize_base64_str(value: str) -> str:
    """Redact base64 payloads in data URLs while preserving keys and structure.

    Examples:
        data:image/png;base64,AAAA...  ->  data:image/png;base64,<omitted>
    """
    try:
        prefix = "data:"
        if value.startswith(prefix) and ";base64," in value:
            head, _ = value.split(";base64,", 1)
            return f"{head};base64,<omitted>"
    except Exception:
        pass
    return value


def sanitize_tool_msg_for_logging(msg: dict) -> dict:
    """
    Return a sanitized deep copy of a tool message suitable for human-readable logs.

    - Preserves all keys/structure (including image/image_url keys).
    - Redacts base64 payloads from data URLs and obvious base64 fields in strings.
    - Keeps pretty-printability by leaving content strings intact except for redactions.
    """

    import copy
    import json

    def _sanitize_obj(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == "url" and isinstance(v, str):
                    out[k] = _sanitize_base64_str(v)
                else:
                    out[k] = _sanitize_obj(v)
            return out
        if isinstance(obj, list):
            return [_sanitize_obj(v) for v in obj]
        if isinstance(obj, str):
            # Attempt to catch embedded data URLs in arbitrary strings
            return _sanitize_base64_str(obj)
        return obj

    cloned = copy.deepcopy(msg)
    # If content is a JSON string, parse → sanitize → embed parsed object for readability
    try:
        content = cloned.get("content")
        if isinstance(content, str):
            with suppress(Exception):
                parsed = json.loads(content)
                parsed = _sanitize_obj(parsed)
                # Embed parsed object so outer json.dumps(..., indent=4) pretty-prints it
                cloned["content"] = parsed
        else:
            cloned["content"] = _sanitize_obj(content)
    except Exception:
        pass
    return cloned
