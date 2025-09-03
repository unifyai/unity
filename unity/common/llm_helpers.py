from __future__ import annotations
import os
import json
import asyncio
import functools
import inspect
import secrets
import string
import traceback
from enum import Enum
from pydantic import BaseModel
import time
from typing import (
    Tuple,
    List,
    Dict,
    Set,
    Union,
    Optional,
    Any,
    get_type_hints,
    get_origin,
    get_args,
    Callable,
    Awaitable,
)

import unify
from ..constants import LOGGER
from dataclasses import dataclass
from ..events.event_bus import Event, EVENT_BUS
from contextvars import ContextVar


def short_id(length=4):
    alphabet = string.ascii_lowercase + string.digits  # base36
    return "".join(secrets.choice(alphabet) for _ in range(length))


TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}

# Hierarchical lineage of nested async tool loops (propagates via contextvars)
TOOL_LOOP_LINEAGE: ContextVar[list[str]] = ContextVar("TOOL_LOOP_LINEAGE", default=[])

# ─────────────────────────────────────────────────────────────────────────────
# Image-handling helpers
# ─────────────────────────────────────────────────────────────────────────────


# Recursively collect *every* base-64 image that lives under "image" key
def _collect_images(obj, acc: list[str]) -> None:
    if isinstance(obj, dict):
        if "image" in obj and isinstance(obj["image"], str):
            acc.append(obj["image"])
        for v in obj.values():
            _collect_images(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_images(v, acc)


# Deep-copy *obj* **without** any "image" keys so we can still present the
# textual part of a tool result next to the promoted image blocks.
def _strip_image_keys(obj):
    """
    Recursively deep-copies an object, removing any base64 image data.
    This handles images in `{"image": "b64..."}` format and also the
    more complex `[{"type": "image_url", ...}]` format in tool content.
    """
    if isinstance(obj, dict):
        # Special handling for tool messages with list content
        if obj.get("role") == "tool" and isinstance(obj.get("content"), list):
            new_content = [
                item
                for item in obj["content"]
                if not (isinstance(item, dict) and item.get("type") == "image_url")
            ]
            # If only text remains, flatten content to a simple string for clarity
            if len(new_content) == 1 and new_content[0].get("type") == "text":
                obj_copy = {k: v for k, v in obj.items() if k != "content"}
                obj_copy["content"] = new_content[0].get("text", "")
                return obj_copy
            else:  # Return a new dict with the filtered content
                obj_copy = {k: v for k, v in obj.items() if k != "content"}
                obj_copy["content"] = new_content
                return obj_copy

        # Standard recursive dictionary copy
        return {k: _strip_image_keys(v) for k, v in obj.items() if k != "image"}
    elif isinstance(obj, list):
        return [_strip_image_keys(v) for v in obj]
    else:
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# 0.  metadata wrapper - lets us attach `max_concurrent` to a tool
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ToolSpec:
    """
    Wrap the real *callable* together with optional metadata.

    Only ``max_concurrent`` is required today but we deliberately keep this
    extensible – adding cost caps, rate limits, auth scopes, … later will not
    change any external API.
    """

    fn: Callable
    max_concurrent: Optional[int] = None  # «None» ⇒ unlimited
    # Hidden per-loop quota: when set, the tool will only be callable
    # `max_total_calls` times within a single async tool-use loop. Once
    # exhausted, the tool is silently hidden from the exposed schema and
    # any additional invocations are minimally acknowledged without
    # revealing quota details to the LLM.
    max_total_calls: Optional[int] = None

    # Let a ToolSpec be invoked like the underlying callable (nice for tests)
    def __call__(self, *a, **kw):  # pragma: no cover
        return self.fn(*a, **kw)


def _normalise_tools(
    raw: Dict[str, Union[Callable, "ToolSpec"]],
) -> Dict[str, "ToolSpec"]:
    """
    Accept the *legacy* ``dict[name → callable]`` or the new
    ``dict[name → ToolSpec]`` and always return a *uniform*
    ``dict[name → ToolSpec]``.
    """
    out: Dict[str, ToolSpec] = {}
    for n, v in raw.items():
        out[n] = v if isinstance(v, ToolSpec) else ToolSpec(fn=v)
    return out


# Dynamic-handle helpers ––––––––––––––––––––––––––––––––––––––––––––––––––––––
#  Public methods we *do not* expose again (already wrapped by dedicated helpers
#  or meaningless to the LLM).
_MANAGEMENT_METHOD_NAMES: set[str] = {
    "interject",
    "pause",
    "resume",
    "stop",
    "done",
    "result",
}


def methods_to_tool_dict(
    *methods: Tuple[Union[Callable, "ToolSpec"]],
    include_class_name: bool = True,
) -> Dict[str, Union[Callable, "ToolSpec"]]:
    """
    Build the ``{name → tool}`` mapping from a list of *bound* methods **or**
    :class:`ToolSpec` instances.

    When a ``ToolSpec`` is given we keep its metadata (`max_concurrent`, …)
    but replace ``fn`` with the *bound* method so calls execute on the correct
    object.
    """

    ret: Dict[str, Union[Callable, ToolSpec]] = {}
    for m in methods:
        # ── unwrap, but remember whether we saw a ToolSpec ─────────────────
        if isinstance(m, ToolSpec):
            spec = m
            fn: Callable = spec.fn
        else:  # plain callable
            spec = None
            fn = m

        # ── derive a sensible key (className_method or plain method) ───────
        if (
            include_class_name
            and hasattr(fn, "__self__")
            and hasattr(
                fn.__self__,
                "__class__",
            )
        ):
            key = f"{fn.__self__.__class__.__name__}_{fn.__name__}".replace("__", "_")
        else:
            key = fn.__name__.lstrip("_")

        # ── store ----------------------------------------------------------------
        if spec is None:
            ret[key] = fn
        else:
            # Preserve the metadata but *bind* the function correctly.
            ret[key] = ToolSpec(fn=fn, max_concurrent=spec.max_concurrent)
    return ret


def class_api_overview(cls: type) -> str:
    """Return a Markdown list of all public callables in *cls*."""
    blocks = []
    for name, member in inspect.getmembers(cls, inspect.isroutine):
        if name.startswith("_"):
            continue  # skip dunder/private helpers
        prefix = "async def " if inspect.iscoroutinefunction(member) else "def "
        try:
            sig = inspect.signature(member)
            first_line = (
                (inspect.getdoc(member) or "No description.").strip().split("\n", 1)[0]
            )
            blocks.append(f"- **`{prefix}{name}{sig}`** – {first_line}")
        except ValueError:
            blocks.append(f"- **`{prefix}{name}(...)`** – No description available.")
    return "\n".join(blocks)


def _discover_custom_public_methods(handle) -> dict[str, Callable]:
    """
    Return a mapping ``name → bound_method`` of *public* callables on *handle*:
        • name does **not** start with ``_``  _and_
        • name is not one of the management helpers above.
    """
    import inspect

    methods: dict[str, Callable] = {}
    for name, attr in inspect.getmembers(handle):
        if (
            name.startswith("_")
            or name in _MANAGEMENT_METHOD_NAMES
            or not callable(attr)
        ):
            continue
        # Bind the method to *handle* (important for late-added attributes).
        try:
            bound = handle.__getattribute__(name)
        except Exception:
            # Attribute access raised – treat as non-callable.
            continue

        methods[name] = bound
    return methods


def _dumps(
    obj: Any,
    idx: List[Union[str, int]] = None,
    indent: int = None,
) -> Any:
    # prevents circular import
    from unify.logging.logs import Log

    base = False
    if idx is None:
        base = True
        idx = list()
    if isinstance(obj, BaseModel):
        ret = obj.model_dump(mode="json")
    elif inspect.isclass(obj) and issubclass(obj, BaseModel):
        ret = obj.model_json_schema()
    elif isinstance(obj, Log):
        ret = obj.to_json()
    elif isinstance(obj, dict):
        ret = {k: _dumps(v, idx + ["k"]) for k, v in obj.items()}
    elif isinstance(obj, list):
        ret = [_dumps(v, idx + [i]) for i, v in enumerate(obj)]
    elif isinstance(obj, set):
        # Convert sets to a sorted list for deterministic, JSON-serialisable output
        try:
            ret = sorted(_dumps(v, idx + [i]) for i, v in enumerate(sorted(obj)))
        except Exception:
            # Fallback: best-effort conversion preserving insertion order where possible
            ret = [_dumps(v, idx + [i]) for i, v in enumerate(list(obj))]
    elif isinstance(obj, tuple):
        ret = tuple(_dumps(v, idx + [i]) for i, v in enumerate(obj))
    else:
        ret = obj
    return json.dumps(ret, indent=indent) if base else ret


import re
from textwrap import dedent
from typing import Iterable

# recognised section headings (case-insensitive, colon optional)
_PARAM_SECTIONS = {"args", "arguments", "parameters", "other parameters"}
_OTHER_SECTIONS = {
    "returns",
    "yields",
    "raises",
    "notes",
    "examples",
    "references",
    "see also",
}

# ––– parameter-line pattern that also accepts the "a / b / c : …" variant –––
_PARAM_LINE_RX = re.compile(
    r"""
    ^(?P<indent>\s*)                    # leading spaces
    (?P<names>[^:]+?)                   # everything until " :", *if any*
    (?:\s*:\s*(?P<type>.+))?            # " : type"  ← now OPTIONAL
    $                                   # EOL
    """,
    re.VERBOSE,
)

# dash-only underline used by the NumPy style ("----------")
_DASH_UNDERLINE_RX = re.compile(r"^\s*-{3,}\s*$")


def _strip_hidden_params_from_doc(
    doc: str,
    hidden: set[str] | Iterable[str],
) -> str:
    """Remove the *Parameters* blocks of any parameters listed in *hidden*."""
    hidden = set(hidden)
    if not doc or not hidden:
        return doc

    lines = dedent(doc).splitlines()
    out: list[str] = []

    i = 0
    in_params = False  # are we _currently_ inside a param section?
    skip = False  # are we skipping the current block?
    base_indent = 0  # indent of the "name : type" line we skip

    while i < len(lines):
        ln, stripped = lines[i], lines[i].lstrip()
        lower = stripped.rstrip(":").lower()

        # ───────────────────────────────────────────────────────────────── #
        # 1.  Detect the *start* of a Parameters/Args section
        # ───────────────────────────────────────────────────────────────── #
        if not in_params and lower in _PARAM_SECTIONS:
            in_params = True
            out.append(ln)  # keep the heading
            # keep the NumPy underline if present
            j = i + 1
            if j < len(lines) and _DASH_UNDERLINE_RX.match(lines[j]):
                out.append(lines[j])
                i += 1
            i += 1
            continue

        # ───────────────────────────────────────────────────────────────── #
        # 2.  While inside the section, decide whether to keep or skip
        # ───────────────────────────────────────────────────────────────── #
        if in_params:
            # Heading of some *other* section → we are done with "Parameters"
            if lower in _OTHER_SECTIONS:
                in_params = False
                out.append(ln)
                i += 1
                continue

            # Dash underline that belongs to the *next* section
            if _DASH_UNDERLINE_RX.match(stripped):
                in_params = False
                out.append(ln)
                i += 1
                continue

            # Parameter definition line
            m = _PARAM_LINE_RX.match(ln)
            if m:
                # the spec allows either "a, b" or "a / b" to list synonyms
                names = {
                    n.strip()
                    for part in m.group("names").split("/")
                    for n in part.split(",")
                }
                if names & hidden:
                    skip = True
                    base_indent = len(m.group("indent"))
                    # do *not* append this very line
                    i += 1
                    continue
                else:
                    skip = False  # keep this parameter
            # Parameter description line: keep skipping until indentation drops
            elif skip:
                indent = len(ln) - len(stripped)
                if indent > base_indent:
                    i += 1  # keep swallowing lines of the block
                    continue
                skip = False  # indent dropped → end of block

            if not skip:
                out.append(ln)  # normal, unskipped content
            i += 1
            continue

        # ───────────────────────────────────────────────────────────────── #
        # 3.  Normal line outside any param section
        # ───────────────────────────────────────────────────────────────── #
        out.append(ln)
        i += 1

    # Collapse runs of >2 blank lines that the removals may have created
    doc_clean = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).rstrip()
    return doc_clean


def annotation_to_schema(ann: Any) -> Dict[str, Any]:
    """Convert a Python type annotation into an OpenAI-compatible JSON-Schema
    fragment, including full support for Pydantic BaseModel subclasses.
    """

    # ── 0. Remove typing.Annotated wrapper, if any ────────────────────────────
    origin = get_origin(ann)
    if origin is not None and origin.__name__ == "Annotated":  # Py ≥3.10
        ann = get_args(ann)[0]

    # ── 0a. Explicitly recognise NoneType so Optional[T] collapses correctly ──
    if ann is type(None):
        return {"type": "null"}

    # ── 1. Primitive scalars (str/int/float/bool) ────────────────────────────
    if ann in TYPE_MAP:
        return {"type": TYPE_MAP[ann]}

    # ── 2. Enum subclasses (e.g. ColumnType) ─────────────────────────────────
    if isinstance(ann, type) and issubclass(ann, Enum):
        return {"type": "string", "enum": [member.value for member in ann]}

    # ── 3. Pydantic model ────────────────────────────────────────────────────
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        # Pydantic already produces an OpenAPI/JSON-Schema compliant dictionary.
        # We can embed that verbatim.  (It contains 'title', 'type', 'properties',
        # 'required', etc.  Any 'definitions' block is also allowed by the spec.)
        return ann.model_json_schema()

    # ── 4. typing.Dict[K, V]  → JSON object whose values follow V ────────────
    origin = get_origin(ann)
    if origin is dict or origin is Dict:
        args = get_args(ann)
        # Dict  (i.e., no [K, V] supplied)  →  free-form object
        if len(args) < 2:
            return {"type": "object"}
        _, value_type = args
        return {
            "type": "object",
            "additionalProperties": annotation_to_schema(value_type),
        }

    # ── 5. typing.List[T] or list[T]  → JSON array of T ──────────────────────
    if origin in (list, List):
        (item_type,) = get_args(ann)
        return {
            "type": "array",
            "items": annotation_to_schema(item_type),
        }

    # ── 6. typing.Union / Optional …  → anyOf schemas ────────────────────────
    # Support both typing.Union and PEP 604 unions (types.UnionType)
    _is_union = False
    try:
        import types as _types  # local import to avoid top-level dependency

        _is_union = origin is Union or origin is _types.UnionType
    except Exception:
        _is_union = origin is Union

    if _is_union:
        sub_schemas = [annotation_to_schema(a) for a in get_args(ann)]
        # Collapse trivial Optional[X] (i.e. Union[X, NoneType]) into X
        if len(sub_schemas) == 2 and {"type": "null"} in sub_schemas:
            return next(s for s in sub_schemas if s != {"type": "null"})
        return {"anyOf": sub_schemas}

    # ── 7. Fallback – treat as generic string ────────────────────────────────
    return {"type": "string"}


def method_to_schema(
    bound_method,
    tool_name: Optional[str] = None,
    include_class_name: bool = True,
):
    """
    Convert **bound_method** into an OpenAI-compatible *function*-tool schema.
    """

    sig = inspect.signature(bound_method)
    hints = get_type_hints(bound_method)

    import inspect as _inspect

    props, required, hidden = {}, [], set()

    # Detect whether the callable accepts **kwargs so we can permit extra keys
    has_var_keyword = any(
        p.kind == _inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    for name, param in sig.parameters.items():
        # Skip star-args and star-kwargs – these are not expressible as fixed JSON fields
        if param.kind in (
            _inspect.Parameter.VAR_POSITIONAL,
            _inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        # Determine whether *name* is **hidden** (never exposed to the LLM)
        is_hidden = (
            name.startswith("_") and param.default is not inspect._empty
        ) or name in (
            "parent_chat_context",
            "clarification_up_q",
            "clarification_down_q",
        )

        if is_hidden:
            hidden.add(name)
            continue  # do NOT surface to the model

        ann = hints.get(name, str)
        props[name] = annotation_to_schema(ann)
        if param.default is inspect._empty:
            required.append(name)

    # ── scrub the docstring so hidden args disappear from "Args:"/"Parameters" ──
    raw_doc = bound_method.__doc__ or ""
    cleaned_doc = _strip_hidden_params_from_doc(raw_doc, hidden)

    # … remainder is unchanged …
    if hasattr(bound_method, "__self__") and hasattr(
        bound_method.__self__,
        "__class__",
    ):
        prefix = f"{bound_method.__self__.__class__.__name__}_"
    elif hasattr(bound_method, "__qualname__"):
        parts = bound_method.__qualname__.split(".")
        prefix = f"{parts[-2]}_" if len(parts) > 1 else ""
    else:
        prefix = ""
    if tool_name is None:
        if include_class_name:
            tool_name = f"{prefix}{bound_method.__name__}".replace("__", "_")
        else:
            tool_name = bound_method.__name__.lstrip("_")

    schema: dict = {
        "type": "function",
        "strict": True,
        "function": {
            "name": tool_name,
            "description": cleaned_doc.strip(),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }
    # Allow arbitrary extra keys when the function accepts **kwargs
    if has_var_keyword:
        schema["function"]["parameters"]["additionalProperties"] = True
    return schema


async def _maybe_await(obj):
    """Return *obj* if it is a value, or `await` and return its result if it is
    an awaitable."""
    if inspect.isawaitable(obj):
        return await obj
    return obj


# Shared steering helpers – reduce duplication across dynamic helper tools
def _adopt_signature_and_annotations(from_callable, to_wrapper) -> None:
    """Copy signature and annotations (excluding 'self') from from_callable to to_wrapper."""
    try:
        import inspect as _inspect

        to_wrapper.__signature__ = _inspect.signature(from_callable)
        try:
            ann = dict(getattr(from_callable, "__annotations__", {}))
            ann.pop("self", None)
            to_wrapper.__annotations__ = ann
        except Exception:
            pass
    except Exception:
        pass


def _normalise_kwargs_for_bound_method(bound_method, incoming_kw: dict) -> dict:
    """Normalise kwargs for a bound method: expand nested kwargs, drop noise keys,
    map common aliases when there is a single public param, and filter unknown keys
    unless **kwargs is accepted."""
    try:
        import inspect as _inspect

        sig = _inspect.signature(bound_method)
        params = sig.parameters
        has_varkw = any(
            p.kind == _inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        kw = dict(incoming_kw or {})

        # 1) Expand nested {"kwargs": {...}}
        if "kwargs" in kw and isinstance(kw["kwargs"], dict):
            nested_kw = kw.pop("kwargs")
            for k, v in nested_kw.items():
                kw.setdefault(k, v)

        # 2) Drop common placeholder noise keys when empty
        for _noise in ("a", "kw"):
            if _noise in kw and (kw[_noise] is None or kw[_noise] == ""):
                kw.pop(_noise, None)

        # 3) If exactly one public param, accept common aliases
        public_params = [n for n in params if n != "self"]
        if len(public_params) == 1 and public_params[0] not in kw:
            for alias in (
                "content",
                "message",
                "text",
                "prompt",
                "guidance",
                "instruction",
                "question",
                "query",
            ):
                if alias in kw:
                    kw[public_params[0]] = kw.pop(alias)
                    break

        # 4) Filter unknown keys unless **kwargs is accepted
        if not has_varkw:
            kw = {k: v for k, v in kw.items() if k in params}
        return kw
    except Exception:
        # Best-effort; return original
        return dict(incoming_kw or {})


async def _forward_handle_call(
    handle: Any,
    method_name: str,
    kwargs: dict | None,
    *,
    fallback_positional_keys: list[str] | tuple[str, ...] = (),
):
    """Invoke a steering method on a handle with robust kwargs handling.

    - Filters/normalises kwargs against the bound method's signature.
    - If the method rejects kwargs, tries positional fallback with the first
      available key from fallback_positional_keys (e.g., reason/content).
    - Finally falls back to calling without arguments.
    """
    try:
        bound = getattr(handle, method_name)
    except Exception:
        return None

    try:
        normalised = _normalise_kwargs_for_bound_method(bound, kwargs or {})
        return await _maybe_await(bound(**normalised))
    except TypeError:
        # Fallbacks for legacy signatures
        for k in fallback_positional_keys:
            if kwargs and k in kwargs:
                try:
                    return await _maybe_await(bound(kwargs.get(k)))  # type: ignore[misc]
                except Exception:
                    pass
        try:
            return await _maybe_await(bound())  # type: ignore[misc]
        except Exception:
            return None
    except Exception:
        # Defensive: never let steering failures crash the loop
        return None


def _chat_context_repr(
    parent_ctx: Optional[list[dict]],
    current_msgs: list[dict],
) -> list[dict]:
    """
    Combine **existing** ``parent_ctx`` with the *current* chat history
    (``current_msgs``) into a depth-aware nested structure:

        root_msg0
        root_msg1
        root_msg2
          └── children:
              ├── child_msg0
              └── child_msg1

    Strategy – keep the original list untouched and attach the new
    messages as ``children`` of the *last* element.
    """
    ctx_block = [
        {"role": m.get("role"), "content": m.get("content")} for m in current_msgs
    ]
    if not parent_ctx:
        return ctx_block

    import copy

    combined = copy.deepcopy(parent_ctx)
    combined[-1].setdefault("children", []).extend(ctx_block)
    return combined


# ASYNC TOOL USE INNER HELPERS ────────────────────────────────────────────────


# ── small helper: publish to the EventBus (if configured) ──────────────
async def _to_event_bus(
    messages: Union[Dict, List[Dict]],
    loop_cfg: _AsyncToolLoopConfig,
) -> None:
    """
    Emit *messages* to the shared EventBus (if configured).

    Every `ToolLoop` event now carries **both** the raw chat *message*
    and the *public method* that spawned the loop so downstream
    subscribers can easily group / filter events.
    """
    if not EVENT_BUS:
        return
    if isinstance(messages, dict):
        messages = [messages]
    for message in messages:
        await EVENT_BUS.publish(
            Event(
                type="ToolLoop",
                payload={
                    "message": message,
                    "method": loop_cfg.loop_id,
                    "hierarchy": list(loop_cfg.lineage),
                    "hierarchy_label": loop_cfg.label,
                },
            ),
        )


# Helper: scan transcript for assistant messages that have tool_calls with
# missing tool replies (before the next assistant message).
def _find_unreplied_assistant_entries(client: unify.AsyncUnify) -> list[dict]:
    findings: list[dict] = []
    try:
        for i, m in enumerate(client.messages):
            if m.get("role") != "assistant":
                continue
            tcs = m.get("tool_calls") or []
            if not tcs:
                continue
            ids = [tc.get("id") for tc in tcs if isinstance(tc, dict)]
            if not ids:
                continue
            responded: set[str] = set()
            j = i + 1
            while (
                j < len(client.messages)
                and client.messages[j].get("role") != "assistant"
            ):
                mm = client.messages[j]
                if mm.get("role") == "tool":
                    tcid = mm.get("tool_call_id")
                    if tcid in ids:
                        responded.add(tcid)
                j += 1
            missing = [c for c in ids if c not in responded]
            if missing:
                findings.append(
                    {
                        "assistant_index": i,
                        "assistant_msg": m,
                        "missing": missing,
                    },
                )
    except Exception:
        pass
    return findings


# Helper: call `client.generate` with optional preprocessing
async def _generate_with_preprocess(
    client: unify.AsyncUnify,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]],
    **gen_kwargs,
):
    if preprocess_msgs is None:
        return await _maybe_await(client.generate(**gen_kwargs))

    import copy

    original_msgs = client.messages  # reference to canonical log
    msgs_copy = copy.deepcopy(original_msgs)

    try:
        patched = preprocess_msgs(msgs_copy) or msgs_copy
    except Exception as exc:  # resilience – don't fail the loop
        LOGGER.error(
            f"preprocess_msgs raised {exc!r}; using original messages.",
        )
        patched = msgs_copy

    start_len = len(patched)

    # ------------------------------------------------------------------
    # Some ``AsyncUnify`` implementations (the real one) keep their chat
    # transcript in a **private** attribute ``_messages`` which is what
    # ``.generate`` reads from, while lightweight test doubles (e.g.
    # ``SpyAsyncUnify`` in the test-suite) expose only a public
    # ``messages`` list.  To remain compatible with *both* variants we
    # detect the attribute that is actually consumed by the downstream
    # ``generate`` call and patch **that** for the duration of the call.
    # ------------------------------------------------------------------
    target_attr = "_messages" if hasattr(client, "_messages") else "messages"

    original_container = getattr(client, target_attr)
    setattr(client, target_attr, patched)
    try:
        result = await _maybe_await(client.generate(**gen_kwargs))

        # Append any new messages the LLM produced back to canonical log
        current_msgs = getattr(client, target_attr)
        if len(current_msgs) > start_len:
            original_msgs.extend(copy.deepcopy(current_msgs[start_len:]))

        return result
    finally:
        # Always restore the canonical chat log so the outer loop remains
        # consistent irrespective of whether we patched `_messages` or
        # `messages`.
        setattr(client, target_attr, original_container)


# Helper: detect helper-tool names (continue_/stop_/pause_/resume_/clarify_/interject_)
def _is_helper_tool(name: str) -> bool:
    return (
        name.startswith("continue_")
        or name.startswith("stop_")
        or name.startswith("pause_")
        or name.startswith("resume_")
        or name.startswith("clarify_")
        or name.startswith("interject_")
    )


# Helper: build human-readable acknowledgement content for helper tools
def _build_helper_ack_content(name: str, args_json: Any) -> str:
    ack_content = "Acknowledged."
    try:
        payload = (
            json.loads(args_json or "{}")
            if isinstance(args_json, str)
            else (args_json or {})
        )
    except Exception:
        payload = {}

    if name.startswith("continue_"):
        ack_content = "Continue request acknowledged. Still waiting for the original tool call to finish."
    elif name.startswith("stop_"):
        ack_content = "Stop request acknowledged. If the underlying call is still running, it will be stopped."
    elif name.startswith("pause_"):
        ack_content = "Pause request acknowledged. If the underlying call is still running, it will be paused."
    elif name.startswith("resume_"):
        ack_content = "Resume request acknowledged. If the underlying call was paused, it will be resumed."
    elif name.startswith("clarify_"):
        ans = payload.get("answer")
        ack_content = (
            f"Clarification answer received: {ans!r}. Waiting for the original tool to proceed."
            if ans is not None
            else "Clarification helper acknowledged. Waiting for the original tool to proceed."
        )
    elif name.startswith("interject_"):
        guidance = payload.get("content")
        ack_content = (
            f"Guidance forwarded to the running tool: {guidance!r}."
            if guidance
            else "Interjection acknowledged and forwarded to the running tool."
        )
    else:
        # Default acknowledgement for custom write-only helpers
        ack_content = (
            f"Operation {name!r} acknowledged and forwarded to the running tool."
        )
    return ack_content


class _AsyncToolLoopMessageDispatcher:
    def __init__(
        self,
        client: unify.AsyncUnify,
        cfg: _AsyncToolLoopConfig,
        timer: _TimeoutTimer,
    ):
        self._client = client
        self._cfg = cfg
        self._timer = timer

    async def append_msgs(
        self,
        msgs: list[dict],
    ) -> None:
        self._client.append_messages(msgs)
        await _to_event_bus(msgs, self._cfg)
        self._timer.reset()


# Remove any tool_calls in an assistant message that would exceed the
# hidden per-tool total-call quota. Operates in-place on asst_msg.
def _prune_over_quota_tool_calls(asst_msg: dict, norm_tools, call_counts) -> None:
    try:
        tool_calls = asst_msg.get("tool_calls") or []
        if not isinstance(tool_calls, list) or not tool_calls:
            return

        # Compute remaining budget per base tool (in this loop instance)
        remaining: Dict[str, int] = {}
        for name, spec in norm_tools.items():
            lim = spec.max_total_calls
            if lim is None:
                continue
            remaining[name] = max(0, lim - call_counts.get(name, 0))

        kept: list = []
        for call in tool_calls:
            try:
                fn_name = call.get("function", {}).get("name")
            except Exception:
                fn_name = None

            # Only enforce quota on base tools that define a limit
            if fn_name in remaining:
                if remaining[fn_name] > 0:
                    kept.append(call)
                    remaining[fn_name] -= 1
                else:
                    # drop this over-quota call silently
                    continue
            else:
                kept.append(call)

        # In-place update only if changed
        if len(kept) != len(tool_calls):
            asst_msg["tool_calls"] = kept
    except Exception:
        # Defensive: never let pruning break the loop
        pass


# ── small helper: add completion tool message pair ──────────────
async def _emit_completion_pair(
    result: str,
    call_id: str,
    msg_dispatcher: _AsyncToolLoopMessageDispatcher,
) -> dict:
    """
    Append a synthetic assistant→tool pair that carries the *final*
    outcome for `call_id`.  Returns the tool-message so callers can
    reuse it for logging / event-bus.
    """
    dummy_id = f"{call_id}_status"

    assistant_stub = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": dummy_id,
                "type": "function",
                "function": {
                    "name": f"check_status_{call_id}",
                    "arguments": "{}",
                },
            },
        ],
        "content": "",
    }
    tool_msg = {
        "role": "tool",
        "tool_call_id": dummy_id,
        "name": f"check_status_{call_id}",
        "content": result,
    }

    await msg_dispatcher.append_msgs([assistant_stub, tool_msg])
    return tool_msg


# ── small helper: keep assistant→tool chronology DRY ────────────────────
async def _insert_after_assistant(
    assistant_meta: dict,
    parent_msg: dict,
    tool_msg: dict,
    client,
    msg_dispatcher: _AsyncToolLoopMessageDispatcher,
) -> None:
    """
    Append *tool_msg* and move it directly after *parent_msg*, while
    updating the per-assistant `results_count` bookkeeping.
    """
    meta = assistant_meta.setdefault(
        id(parent_msg),
        {"results_count": 0},
    )
    await msg_dispatcher.append_msgs([tool_msg])
    insert_pos = client.messages.index(parent_msg) + 1 + meta["results_count"]
    client.messages.insert(insert_pos, client.messages.pop())
    meta["results_count"] += 1


# ── *single* authoritative implementation of "task finished" handling ──
async def _process_completed_task(
    # ACTUALLY Task related
    task: asyncio.Task,
    task_info,
    consecutive_failures: _AsyncToolLoopToolFailureTracker,
    # Loop & State related
    pending_tasks,
    outer_handle_container,
    assistant_meta,
    clarification_channels,
    completed_results,
    #
    client,
    msg_dispatcher,
    logger,
) -> bool:
    """
    Deal with a finished tool *task* exactly once:

    1.  Pop bookkeeping (``pending`` / ``task_info``).
    2.  Serialise *success* or *exception* into ``result``.
    3.  Patch or insert the correct **tool** message so the transcript
        stays perfectly chronological.
    4.  Emit the event-bus hook (if configured).
    5.  Record the payload in ``completed_results`` for later
        `_continue_<id>` helpers.
    6.  Enforce the *max_consecutive_failures* safety valve.
    """

    def _at_tail(msg: dict) -> bool:
        """True when *msg* is the very last entry in client.messages."""
        return bool(client.messages) and client.messages[-1] is msg

    pending_tasks.discard(task)
    info = task_info.pop(task)
    name = info["name"]
    call_id = info["call_id"]
    fn = info["call_dict"]["function"]["name"]
    arg = info["call_dict"]["function"]["arguments"]

    # 2️⃣  obtain result -------------------------------------------------
    try:
        raw = task.result()

        # ───────────────────────────────────────────────────────────────
        #  NEW:  the tool *did not really finish* – it returned *another*
        #        AsyncToolLoopHandle.  We:
        #        (1) schedule `handle.result()` as a *new* task,
        #        (2) keep the **same** `call_id` so the continue/-cancel
        #            helpers keep working,
        #        (3) create / patch one placeholder "still running…"
        #            tool-message in the transcript.
        # ───────────────────────────────────────────────────────────────
        # treat ANY AsyncToolLoopHandle (or subclass) as a nested loop
        from unity.common.llm_helpers import SteerableToolHandle

        if isinstance(raw, SteerableToolHandle):
            # If the nested handle explicitly requests pass-through behaviour
            # expose it directly to the outer caller *immediately*.
            if (
                getattr(raw, "__passthrough__", False)
                and outer_handle_container
                and outer_handle_container[0] is not None
            ):
                outer_handle_container[0]._adopt(raw)
            # ── upgrade interject / clarification flags from handle ─────
            if hasattr(raw, "interject"):
                info["is_interjectable"] = True

            h_up_q = getattr(raw, "clarification_up_q", info.get("clar_up_q"))
            h_down_q = getattr(raw, "clarification_down_q", info.get("clar_down_q"))

            if (h_up_q is not None) ^ (h_down_q is not None):
                raise AttributeError(
                    f"Handle returned by tool {info['name']!r} exposes only "
                    "one of 'clarification_up_q' / 'clarification_down_q'. "
                    "Both queues are required (or neither).",
                )

            # 1️⃣ spawn the nested waiter
            #
            # ⤷ `handle.result` can now be **sync OR async**:
            #    • async ⇒ use the coroutine directly,
            #    • sync  ⇒ run it in a worker-thread so the event-loop never blocks.
            if inspect.iscoroutinefunction(raw.result):
                nested_coro = raw.result()  # already a coroutine
            else:
                nested_coro = asyncio.to_thread(raw.result)  # turn sync → coroutine

            nested_task = asyncio.create_task(nested_coro)
            pending_tasks.add(nested_task)

            # 2️⃣ insert / update a single placeholder
            ph = info.get("tool_reply_msg")
            if ph is None:
                ph = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": info["name"],
                    "content": ("Nested async tool loop started… waiting for result."),
                }
                await _insert_after_assistant(
                    assistant_meta,
                    info["assistant_msg"],
                    ph,
                    client,
                    msg_dispatcher,
                )
                info["tool_reply_msg"] = ph  # remember on *parent*
            else:
                ph["content"] = "Nested async tool loop started… waiting for result."

            # 3️⃣ book-keeping for the *new* task (inherit + share placeholder)
            task_info[nested_task] = {
                **info,
                "handle": raw,
                "is_interjectable": hasattr(raw, "interject"),
                "tool_reply_msg": ph,
                "clar_up_q": h_up_q,
                "clar_down_q": h_down_q,
            }
            if h_up_q is not None:
                clarification_channels[call_id] = (h_up_q, h_down_q)
            return False  # ⬅️  no LLM turn required

        # ───────────────────────────────────────────────────────────────
        #  Normal (non-handle) result – unchanged path
        # ───────────────────────────────────────────────────────────────
        # ── finished successfully – promote any embedded images ─────────
        images: list[str] = []
        _collect_images(raw, images)

        text_repr = _dumps(_strip_image_keys(raw), indent=4)

        if images:
            content_blocks: list = []
            if text_repr and text_repr != "{}":
                content_blocks.append({"type": "text", "text": text_repr})
            content_blocks.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
                for b64 in images
            )
            result = content_blocks
        else:
            result = text_repr

        consecutive_failures.reset_failures()
    except Exception:
        consecutive_failures.increment_failures()
        result = traceback.format_exc()
        if logger.log_steps:
            logger.error(
                f"Error: {name} failed "
                f"(attempt {consecutive_failures.current_failures}/{consecutive_failures.max_failures}):\n{result}",
                prefix="❌",
            )
            # Additional debug context: show the exact tool schema and arguments
            # that were presented to the LLM for this failed call. This helps
            # diagnose docstrings/argspec mismatches that cause tool misuse.
            try:
                debug_payload = {
                    "tool_name": name,
                    "call_id": call_id,
                    "llm_function_schema": info.get("tool_schema"),
                    "llm_arguments": info.get("llm_arguments"),
                    "raw_arguments_json": info.get("raw_arguments_json"),
                }
                logger.error(
                    f"FAILED TOOL SCHEMA (as given to LLM):\n{json.dumps(debug_payload, indent=2)}",
                    prefix="🧩",
                )
            except Exception:
                pass

    # 3️⃣  remember so later `_continue_*` helpers can answer instantly
    completed_results[call_id] = result

    # 4️⃣  update / insert tool-result message --------------------------
    asst_msg = info["assistant_msg"]
    continue_msg = info.get("continue_msg")
    clarify_ph = info.get("clarify_placeholder")
    tool_reply_msg = info.get("tool_reply_msg")

    if continue_msg is not None:
        if _at_tail(continue_msg):  # ✅ safe to overwrite
            continue_msg["content"] = result
            continue_msg["name"] = (
                f"{fn}({arg}) completed successfully, "
                "the return values are in the `content` field below."
            )
            tool_msg = continue_msg
        else:  # 🆕 keep history stable
            tool_msg = await _emit_completion_pair(result, call_id, msg_dispatcher)

    elif clarify_ph is not None:
        if _at_tail(clarify_ph):
            clarify_ph["content"] = result
            tool_msg = clarify_ph
        else:
            tool_msg = await _emit_completion_pair(result, call_id, msg_dispatcher)

    elif tool_reply_msg is not None:
        if _at_tail(tool_reply_msg):
            tool_reply_msg["content"] = result
            tool_msg = tool_reply_msg
        else:
            tool_msg = await _emit_completion_pair(result, call_id, msg_dispatcher)

    else:
        tool_msg = {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": result,
        }
        await _insert_after_assistant(
            assistant_meta,
            asst_msg,
            tool_msg,
            client,
            msg_dispatcher,
        )

    # ── optional console logging for every finished tool call ────────────
    #     (mirrors the assistant-message logging above)
    if logger.log_steps:
        # Create a clean version of tool_msg for logging (strip image data)
        tool_msg_for_logging = tool_msg.copy()
        if isinstance(tool_msg_for_logging.get("content"), list):
            # Filter out image_url items and keep only text content
            tool_msg_for_logging["content"] = [
                item
                for item in tool_msg_for_logging["content"]
                if item.get("type") != "image_url"
            ]
        logger.info(
            f"{json.dumps(tool_msg_for_logging, indent=4)}\n",
            prefix="🛠️",
        )

    # 6️⃣  failure guard -------------------------------------------------
    if consecutive_failures.has_exceeded_failures():
        if logger.log_steps:
            logger.error(f"Aborting: too many tool failures.", prefix="🚨")
        raise RuntimeError(
            "Aborted after too many consecutive tool failures.",
        )

    # successful (or failed) *final* result → LLM may need to react
    return True


# helper: register a freshly-minted coroutine as a *temporary* tool
def _register_tool(
    key: str,
    func_name: str,
    doc: str,
    fn: Callable,
    dynamic_tools,
) -> None:
    # prefer the function's own docstring if it exists, else fall back
    existing = inspect.getdoc(fn)
    fn.__doc__ = existing.strip() if existing else doc
    fn.__name__ = func_name[:64]
    fn.__qualname__ = func_name[:64]
    dynamic_tools[key.lstrip("_")] = fn


# Helper: propagate a stop request to any nested SteerableToolHandle returned
# by base tools. This ensures outer stop/cancel signals reach inner loops.
async def _propagate_stop_to_nested_handles(
    task_info,
    reason: Optional[str] = None,
) -> None:
    try:
        for _t, _inf in list(task_info.items()):
            h = _inf.get("handle")
            if h is not None and hasattr(h, "stop"):
                try:
                    await _forward_handle_call(
                        h,
                        "stop",
                        {"reason": reason} if reason is not None else {},
                        fallback_positional_keys=["reason"],
                    )
                except Exception:
                    # Best effort – never let propagation failure crash the loop
                    pass
    except Exception:
        pass


async def _propagate_stop_once(
    task_info,
    stop_forward_once,
    reason: Optional[str],
) -> bool:
    if stop_forward_once:
        return stop_forward_once
    await _propagate_stop_to_nested_handles(task_info, reason)
    return True


# Helper: insert a tool-acknowledgement message for helper tools
async def _acknowledge_helper_call(
    asst_msg: dict,
    call_id: str,
    name: str,
    args_json: Any,
    *,
    assistant_meta,
    client,
    msg_dispatcher,
) -> None:
    tool_msg = {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": _build_helper_ack_content(name, args_json),
    }
    await _insert_after_assistant(
        assistant_meta,
        asst_msg,
        tool_msg,
        client,
        msg_dispatcher,
    )


# Helper: schedule a base tool call (shared by main path and backfill)
async def _schedule_base_tool_call(
    asst_msg: dict,
    *,
    name: str,
    args_json: Any,
    call_id: str,
    call_idx: int,
    call_counts,
    norm_tools,
    task_info,
    pending,
    parent_chat_context,
    propagate_chat_context,
    clarification_channels,
    assistant_meta,
    client,
) -> None:
    # Base tool must exist
    if name not in norm_tools:
        return

    fn = norm_tools[name].fn

    # Enforce hidden per-tool total call quota: should be pre-pruned from
    # the assistant message, but guard here as well and simply skip.
    try:
        lim = norm_tools[name].max_total_calls
        if lim is not None and call_counts.get(name, 0) >= lim:
            return
    except Exception:
        pass

    # Build extra kwargs (chat context, interject/clarification/pause)
    extra_kwargs: dict = {}
    if propagate_chat_context:
        cur_msgs = [m for m in client.messages if not m.get("_ctx_header")]
        ctx_repr = _chat_context_repr(parent_chat_context, cur_msgs)
        extra_kwargs["parent_chat_context"] = ctx_repr

    sig = inspect.signature(fn)
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    sig_accepts_interject_q = "interject_queue" in params or has_varkw
    sig_accepts_pause_event = "pause_event" in params or has_varkw
    sig_accepts_clar_qs = (
        "clarification_up_q" in params and "clarification_down_q" in params
    ) or has_varkw

    pause_ev: Optional[asyncio.Event] = None
    if sig_accepts_pause_event:
        pause_ev = asyncio.Event()
        pause_ev.set()  # start running
        extra_kwargs["pause_event"] = pause_ev

    clar_up_q: Optional[asyncio.Queue[str]] = None
    clar_down_q: Optional[asyncio.Queue[str]] = None
    if sig_accepts_clar_qs:
        clar_up_q = asyncio.Queue()
        clar_down_q = asyncio.Queue()
        extra_kwargs["clarification_up_q"] = clar_up_q
        extra_kwargs["clarification_down_q"] = clar_down_q

    sub_q: Optional[asyncio.Queue[str]] = None
    if sig_accepts_interject_q:
        sub_q = asyncio.Queue()
        extra_kwargs["interject_queue"] = sub_q

    # Parse args
    try:
        call_args = (
            json.loads(args_json) if isinstance(args_json, str) else (args_json or {})
        )
    except Exception:
        call_args = {}

    # Filter extras to match fn signature
    sig = inspect.signature(fn)
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    filtered_extras = {
        k: v for k, v in extra_kwargs.items() if k in params or has_varkw
    }

    # Forward ALL call args verbatim. Let the callee raise if unsupported.
    allowed_call_args = call_args
    merged_kwargs = {**allowed_call_args, **filtered_extras}

    # Build coroutine
    if asyncio.iscoroutinefunction(fn):
        coro = fn(**merged_kwargs)
    else:
        coro = asyncio.to_thread(fn, **merged_kwargs)

    call_dict = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args_json},
    }

    t = asyncio.create_task(coro, name=f"ToolCall_{name}")
    pending.add(t)
    task_info[t] = {
        "name": name,
        "call_id": call_id,
        "assistant_msg": asst_msg,
        "call_dict": call_dict,
        "call_idx": call_idx,
        "is_interjectable": sig_accepts_interject_q,
        "interject_q": sub_q,
        "chat_ctx": extra_kwargs.get("parent_chat_context"),
        "clar_up_q": clar_up_q,
        "clar_down_q": clar_down_q,
        "pause_event": pause_ev,
        # Debug helpers for failure logging
        "tool_schema": method_to_schema(fn, name),
        "llm_arguments": allowed_call_args,
        "raw_arguments_json": args_json,
    }

    # Increment hidden quota counter only once scheduling succeeds
    try:
        call_counts[name] = call_counts.get(name, 0) + 1
    except Exception:
        pass

    if clar_up_q is not None:
        clarification_channels[call_id] = (
            clar_up_q,
            clar_down_q,
        )

    # Ensure assistant meta exists for deterministic insertion ordering
    assistant_meta.setdefault(id(asst_msg), {"results_count": 0})


# Ensure placeholder tool messages exist for pending tasks. If assistant_msg
# is provided, only affects tasks spawned by that assistant turn; otherwise
# applies to all pending tasks. Returns the list of call_ids for which a
# placeholder was created.
async def _ensure_placeholders_for_pending(
    assistant_msg: Optional[dict] = None,
    *,
    content: Optional[str] = None,
    #
    task_info,
    pending,
    assistant_meta,
    client,
    msg_dispatcher,
) -> list[str]:
    created: list[str] = []
    placeholder_content = (
        content
        if content is not None
        else "Pending… tool call accepted. Working on it."
    )
    for _t in list(pending):
        _inf = task_info.get(_t)
        if not _inf:
            continue
        if assistant_msg is not None and _inf.get("assistant_msg") is not assistant_msg:
            continue
        if (
            _inf.get("tool_reply_msg")
            or _inf.get("continue_msg")
            or _inf.get("clarify_placeholder")
        ):
            continue

        placeholder = {
            "role": "tool",
            "tool_call_id": _inf["call_id"],
            "name": _inf["name"],
            "content": placeholder_content,
        }
        await _insert_after_assistant(
            assistant_meta,
            _inf["assistant_msg"],
            placeholder,
            client,
            msg_dispatcher,
        )
        _inf["tool_reply_msg"] = placeholder
        created.append(_inf["call_id"])

    return created


# Helper: schedule a subset of tool_calls on a past assistant message and
# insert placeholders immediately. Skips already-scheduled/finished ids.
async def _schedule_missing_for_message(
    asst_msg: dict,
    only_ids: set[str],
    *,
    clarification_channels,
    call_counts,
    norm_tools,
    pending,
    completed_results,
    parent_chat_context,
    propagate_chat_context,
    task_info,
    assistant_meta,
    client,
    msg_dispatcher,
) -> list[str]:
    scheduled: list[str] = []
    try:
        tool_calls = asst_msg.get("tool_calls") or []
        for idx, call in enumerate(tool_calls):
            cid = call.get("id")
            if cid not in only_ids:
                continue

            # Skip if already pending or completed
            if any(inf.get("call_id") == cid for _t, inf in task_info.items()):
                continue
            if cid in completed_results:
                continue

            name = call["function"]["name"]
            args_json = call["function"].get("arguments", "{}")

            # Handle dynamic helpers similarly to main path
            if _is_helper_tool(name):
                # Do not execute helpers during backfill, only acknowledge
                try:
                    await _acknowledge_helper_call(
                        asst_msg,
                        cid,
                        name,
                        args_json,
                        assistant_meta=assistant_meta,
                        client=client,
                        msg_dispatcher=msg_dispatcher,
                    )
                except Exception:
                    pass
                scheduled.append(cid)
                continue

            # Base tool: locate function
            if name not in norm_tools:
                scheduled.append(cid)
                continue

            await _schedule_base_tool_call(
                asst_msg,
                name=name,
                args_json=args_json,
                call_id=cid,
                call_idx=idx,
                call_counts=call_counts,
                norm_tools=norm_tools,
                task_info=task_info,
                pending=pending,
                parent_chat_context=parent_chat_context,
                propagate_chat_context=propagate_chat_context,
                assistant_meta=assistant_meta,
                clarification_channels=clarification_channels,
                client=client,
            )
            scheduled.append(cid)
    except Exception:
        pass
    # Ensure placeholders are present for backfilled items
    try:
        await _ensure_placeholders_for_pending(
            assistant_msg=asst_msg,
            pending=pending,
            task_info=task_info,
            assistant_meta=assistant_meta,
            client=client,
            msg_dispatcher=msg_dispatcher,
        )
    except Exception:
        pass
    return scheduled


def _check_valid_response_format(response_format: Any):
    # Require a Pydantic model class – anything else is a configuration error.
    if not (
        isinstance(response_format, type) and issubclass(response_format, BaseModel)
    ):
        raise TypeError(
            "response_format must be a Pydantic BaseModel subclass (e.g. MySchema).",
        )

    return response_format.model_json_schema()


# ASYNC TOOL USE LOOP ────────────────────────────────────────────────────────


class _AsyncToolLoopConfig:
    def __init__(self, loop_id, lineage, parent_lineage):
        self._loop_id = loop_id if loop_id is not None else short_id()
        self._lineage = (
            list(lineage) if lineage is not None else [*parent_lineage, self._loop_id]
        )
        self._label = (
            "->".join(self._lineage) if self._lineage else (self._loop_id or "")
        )

    @property
    def loop_id(self):
        return self._loop_id

    @property
    def lineage(self):
        return self._lineage

    @property
    def label(self):
        return self._label


class _AsyncToolLoopLogger:
    def __init__(self, cfg: _AsyncToolLoopConfig, log_steps: bool | str) -> None:
        self._label = cfg.label
        self._log_steps = log_steps

    @property
    def log_steps(self):
        return self._log_steps

    @property
    def log_label(self):
        return self._label

    def info(self, msg, prefix=""):
        txt = f"{prefix} [{self._label}] {msg}"
        LOGGER.info(txt)

    def error(self, msg, prefix=""):
        txt = f"{prefix} [{self._label}] {msg}"
        LOGGER.error(txt)


class _TimeoutTimer:
    def __init__(self, timeout: int, max_steps: int, raise_on_limit: bool, client):
        self._timeout = timeout
        self._client = client
        self._max_steps = max_steps
        self._raise_on_limit = raise_on_limit
        self.reset()

    def remaining_time(self) -> Optional[float]:
        if self._timeout is None:
            return None

        return self._timeout - (time.perf_counter() - self.last_activity_ts)

    def reset(self):
        """Refresh the rolling timeout."""
        self.last_activity_ts = time.perf_counter()
        self.last_msg_count = (
            0 if not self._client.messages else len(self._client.messages)
        )

    def has_exceeded_time(self) -> bool:
        """
        Return whether we exceeded the timeout threshold, raises Exception if raise_on_limit is set
        """
        if self._timeout is None:
            return False

        ret = time.perf_counter() - self.last_activity_ts > self._timeout
        if self._raise_on_limit and ret:
            raise asyncio.TimeoutError(
                f"Loop exceeded {self._timeout}s wall-clock limit",
            )
        return ret

    def has_exceeded_msgs(self) -> bool:
        """
        Return whether we exceeded the messages threshold, raises Exception if raise_on_limit is set
        """
        if self._max_steps is None:
            return False

        ret = len(self._client.messages) >= self._max_steps
        if self._raise_on_limit and ret:
            raise RuntimeError(
                f"Conversation exceeded max_steps={self._max_steps} "
                f"(len(client.messages)={len(self._client.messages)})",
            )
        return ret


class _ToolsData:
    def __init__(self, tools):
        self.norm_tools = _normalise_tools(tools)
        self.pending: Set[asyncio.Task] = set()
        self.info: Dict[asyncio.Task, Dict[str, Any]] = {}
        self.call_counts: Dict[str, int] = {}


# TODO this is not really required, but this just simplifies the extraction of the logic from the loop.
class _AsyncToolLoopToolFailureTracker:
    def __init__(self, max_consecutive_failures: int):
        self._consecutive_failures = 0
        self._max_consecutive_failures = max_consecutive_failures

    @property
    def current_failures(self):
        return self._consecutive_failures

    @property
    def max_failures(self):
        return self._max_consecutive_failures

    def has_exceeded_failures(self) -> bool:
        return self._consecutive_failures >= self._max_consecutive_failures

    def increment_failures(self):
        self._consecutive_failures += 1

    def reset_failures(self):
        self._consecutive_failures = 0


async def _async_tool_use_loop_inner(
    client: unify.AsyncUnify,
    message: str,
    tools: Dict[str, Union[Callable, ToolSpec]],
    *,
    loop_id: Optional[str] = None,
    lineage: Optional[list[str]] = None,
    interject_queue: asyncio.Queue[str],
    cancel_event: asyncio.Event,
    stop_event: asyncio.Event | None = None,
    pause_event: asyncio.Event,
    max_consecutive_failures: int = 3,
    prune_tool_duplicates: bool = True,
    interrupt_llm_with_interjections: bool = True,
    propagate_chat_context: bool = True,
    parent_chat_context: Optional[list[dict]] = None,
    log_steps: Union[bool, str] = True,
    max_steps: Optional[int] = None,
    timeout: Optional[int] = None,
    raise_on_limit: bool = False,
    include_class_in_dynamic_tool_names: bool = False,
    tool_policy: Optional[
        Callable[[int, Dict[str, Callable]], Tuple[str, Dict[str, Callable]]]
    ] = None,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
    outer_handle_container: Optional[list] = None,
    response_format: Optional[Any] = None,
    max_parallel_tool_calls: Optional[int] = None,
    persist: bool = False,
) -> str:
    r"""
    Orchestrate an *interactive* "function-calling" dialogue between an LLM
    and a set of Python callables until the model yields a **final** plain-
    text answer.

    Key design points
    -----------------
    • **Concurrency** – every tool suggested by the model is wrapped in its
      own ``asyncio.Task`` so multiple long-running calls may advance in
      parallel; the loop always waits only for the *first* one to finish.

    • **Interruptibility** – the outer caller may:
        – set ``cancel_event`` → graceful shutdown (all tasks cancelled &
          awaited, then ``asyncio.CancelledError`` is re-raised);
        – queue ``interject_queue.put(text)`` → a new *user* turn injected
          just before the *next* LLM step without disturbing already running
          tools.

    • **Robustness** – exceptions inside tools are caught, serialised, and
      shown to the model; after ``max_consecutive_failures`` consecutive
      crashes the whole loop aborts with ``RuntimeError`` (prevents infinite
      failure ping-pong).

    • **Low coupling** – all transport (e.g. websockets, HTTP) can live
      outside; an optional ``event_bus`` lets a UI or logger subscribe to
      every message without the loop having to know who is listening.

    Parameters
    ----------
    client : ``unify.AsyncUnify``
        Pre-initialised Unify client that provides ``append_messages`` and
        ``generate``.  All tokens sent to / received from the LLM flow
        through this object.

    message : ``str``
        The very first user prompt that kicks-off the whole interactive
        session.

    tools : ``dict[str, Callable]``
        A mapping ``name → function`` describing every callable the LLM may
        invoke.  Each function must be fully type-hinted and have a concise
        docstring – these are automatically converted to an OpenAI *tool
        schema* via :pyfunc:`method_to_schema`.

    interject_queue : ``asyncio.Queue[str]``
        Thread-safe channel through which the *outer* application can push
        additional user turns at any time (e.g. the human changes their
        mind mid-generation).

    cancel_event : ``asyncio.Event``
        Flips to *set* when the outer caller wants graceful shutdown.  The
        loop then cancels every running task and propagates
        ``asyncio.CancelledError`` upstream.

    max_consecutive_failures : ``int``, default ``3``
        Hard safety valve: after this many back-to-back exceptions coming
        from tools the loop bails out with ``RuntimeError`` to avoid an
        infinite crash-and-retry ping-pong.

    ignore_tool_duplicates : ``bool``, default ``True``
        Deduplicates model-requested tool calls that have *identical*
        ``function.name`` **and** argument JSON.  Duplicates are pruned
        **in-place** before ever touching chat history or being scheduled.

    interrupt_llm_with_interjection : ``bool``, default ``True``
        Controls latency to fresh user input.  When *True* any in-flight
        ``client.generate`` is cancelled the moment a new user turn arrives
        so the assistant can pivot instantly.  When *False* the loop waits
        for the model to finish (legacy behaviour).

    propagate_chat_context : ``bool``, default ``True``
        If *True*, the entire conversation state of **this** loop is
        threaded into any child tool that accepts a
        ``parent_chat_context`` keyword argument.
        If *True*, the entire conversation state of **this** loop is threaded
        into any child tool via the *internal-only* ``parent_chat_context``
        argument.  This parameter is added automatically and is **not**
        exposed to the LLM.

     tool_policy : ``Callable | None``, default ``None``
         Optional callable that *dynamically* controls tool exposure **and**
         whether a tool call is **required** on a given turn.  Receives the
         current turn index (starting at ``0``) and the full mapping
         ``{name → callable}``.  It must return a tuple ``(policy, tools)``
         where ``policy`` is either ``"auto"`` or ``"required"`` (fed straight
         into ``tool_choice``) and ``tools`` is the possibly-filtered mapping
         of base tools visible on that turn.

    parent_chat_context : ``list[dict] | None``
        Nested chat structure passed from an **outer** loop.  When
        ``propagate_chat_context`` is on, the helper
        :pyfunc:`_chat_context_repr` merges this with the current
        ``client.messages`` and forwards the result downward.

    log_steps : ``bool | str``, default ``True``
        Controls verbosity of step logging to ``LOGGER``:
          • ``False`` – no logging
          • ``True``  – log everything except system messages
          • ``"full"`` – log everything including system messages

    Returns
    -------
    str
        The assistant's final plain-text reply *after* every tool result has
        been fed back into the conversation.
    """
    # unique id / lineage
    cfg = _AsyncToolLoopConfig(loop_id, lineage, TOOL_LOOP_LINEAGE.get([]))
    logger = _AsyncToolLoopLogger(cfg, log_steps)
    _token = TOOL_LOOP_LINEAGE.set(cfg.lineage)

    # normalise optional graceful stop event
    stop_event = stop_event or asyncio.Event()

    # If structured output is expected, inform the model up-front so it can
    # plan its reasoning with the final JSON shape in mind.  Enforcement via
    # `set_response_format` still happens at the end of the loop.
    if response_format is not None:
        try:
            _schema = _check_valid_response_format(response_format)
            _hint = (
                "\n\nNOTE: After completing all tool calls, your **final** assistant reply must be valid JSON that conforms to the following schema. Do NOT include any extra keys or commentary.\n"
                + json.dumps(_schema, indent=2)
            )

            client.set_system_message((client.system_message or "") + _hint)
        except Exception as _exc:  # noqa: BLE001
            logger.error(f"response_format hint failed: {_exc!r}")

    # ── runtime guards ────────────────────────────────────────────────────
    # rolling timeout ----------------------------------------------------
    timer: _TimeoutTimer = _TimeoutTimer(
        timeout=timeout,
        max_steps=max_steps,
        raise_on_limit=raise_on_limit,
        client=client,
    )
    _msg_dispatcher = _AsyncToolLoopMessageDispatcher(client, cfg, timer)

    if log_steps:
        if log_steps == "full":
            if parent_chat_context:
                logger.info(
                    f"Parent Context: {json.dumps(parent_chat_context, indent=4)}\n",
                    prefix="⬇️",
                )
            logger.info(f"System Message: {client.system_message}\n", prefix="📋")
        logger.info(f"User Message: {message}\n", prefix="🧑‍💻")

    # ── 0-a. Inject **system** header with broader context ───────────────────
    #
    # When a parent context is supplied we prepend a single synthetic system
    # message that *summarises* it.  This offers the LLM immediate awareness
    # of the wider conversation without having to scroll the nested JSON.
    # The special marker ``_ctx_header=True`` lets us later strip it when
    # propagating context further down (avoids duplication).
    # -----------------------------------------------------------------------

    if parent_chat_context:
        sys_msg = {
            "role": "system",
            "_ctx_header": True,
            "content": (
                "Broader context (read-only):\n"
                f"{json.dumps(parent_chat_context, indent=2)}\n\n"
                "Resolve the *next* user request in light of this."
            ),
        }
        await _msg_dispatcher.append_msgs([sys_msg])

    # ── initial prompt ───────────────────────────────────────────────────────
    # ── 0-b. Coerce tools → ToolSpec & helper lambdas ───────────────────────
    #
    # • «norm_tools» holds the *canonical* mapping name → ToolSpec
    # • helper for the active-count of one tool (cheap O(#pending))
    # • helper that answers "may we launch / advertise *this* tool right now?"
    #   by comparing the live count with max_concurrent.
    # -----------------------------------------------------------------------

    # Initialise loop state early so preflight backfill can schedule tasks
    tools_data = _ToolsData(tools)
    consecutive_failures = _AsyncToolLoopToolFailureTracker(max_consecutive_failures)
    clarification_channels: Dict[
        str,
        Tuple[asyncio.Queue[str], asyncio.Queue[str]],
    ] = {}
    completed_results: Dict[str, str] = {}
    assistant_meta: Dict[int, Dict[str, Any]] = {}
    step_index: int = 0  # per assistant turn

    def _active_count(t_name: str) -> int:
        return sum(1 for _t, _inf in tools_data.info.items() if _inf["name"] == t_name)

    def _can_offer_tool(t_name: str) -> bool:
        lim = tools_data.norm_tools[t_name].max_concurrent
        return lim is None or _active_count(t_name) < lim

    # Per-tool hidden total-call quotas (counted per loop instance)
    call_counts: Dict[str, int] = {}

    def _quota_count(t_name: str) -> int:
        return call_counts.get(t_name, 0)

    def _quota_ok(t_name: str) -> bool:
        lim = tools_data.norm_tools[t_name].max_total_calls
        return lim is None or _quota_count(t_name) < lim

    # Expose live task_info mapping on the current Task so outer handles/tests
    # can introspect currently running nested handles (used by ask/stop helpers).
    try:
        _self_task = asyncio.current_task()
        if _self_task is not None:
            setattr(_self_task, "task_info", tools_data.info)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Ensure we forward stop to nested handles at most once, even if multiple
    # branches detect cancellation/stop around the same time.
    _stop_forwarded_once: bool = False

    # Preflight repair: backfill any pre-existing assistant tool_calls without replies
    try:
        unreplied = _find_unreplied_assistant_entries(client)
        if unreplied:
            # backfill for all such assistant messages (oldest → newest)
            for entry in unreplied:
                amsg = entry["assistant_msg"]
                # Before scheduling, drop any over-quota tool calls in this message
                _prune_over_quota_tool_calls(amsg, tools_data.norm_tools, call_counts)
                missing_ids = set(entry["missing"])
                await _schedule_missing_for_message(
                    amsg,
                    missing_ids,
                    clarification_channels=clarification_channels,
                    call_counts=call_counts,
                    norm_tools=tools_data.norm_tools,
                    pending=tools_data.pending,
                    completed_results=completed_results,
                    parent_chat_context=parent_chat_context,
                    propagate_chat_context=propagate_chat_context,
                    task_info=tools_data.info,
                    assistant_meta=assistant_meta,
                    client=client,
                    msg_dispatcher=_msg_dispatcher,
                )
    except Exception:
        pass

    # ── initial **user** message
    if isinstance(message, dict):
        initial_user_msg = message
    else:
        initial_user_msg = {"role": "user", "content": message}

    await _msg_dispatcher.append_msgs([initial_user_msg])

    # ── helper: graceful early-exit when limits are hit ────────────────────
    async def _handle_limit_reached(reason: str) -> str:
        """
        Gracefully terminate the loop when *timeout* or *max_steps* are
        exceeded and `raise_on_limit` is *False*:
          • stop every pending tool (via handle.stop() if available)
          • cancel waiter coroutines
          • append a short assistant notice
        """
        for t in list(tools_data.pending):
            h = tools_data.info.get(t, {}).get("handle")
            try:
                if h is not None and hasattr(h, "stop"):
                    await _maybe_await(h.stop())
            except Exception:
                pass
            if not t.done():
                t.cancel()
        await asyncio.gather(*tools_data.pending, return_exceptions=True)
        tools_data.pending.clear()

        notice = {
            "role": "assistant",
            "content": f"🔚 Terminating early: {reason}",
        }
        await _msg_dispatcher.append_msgs([notice])
        if log_steps:
            logger.info(f"Early exit – {reason}", prefix="⏹️")
        return notice["content"]

    # Set to *True* whenever the loop must grant the LLM an immediate turn
    # before waiting again (user interjection, clarification answer, etc.).
    llm_turn_required = False

    # Last known assistant answer when the model produced a final tool-less reply.
    # Used when `persist=True` to return a stable result upon explicit stop.
    last_final_answer: Optional[str] = None

    try:
        while True:

            # ── 0-α-P. Global *pause* gate  ────────────────────────────
            # Keep handling tool completions & cancellation, but *never*
            # let the LLM speak while we're paused.
            if not pause_event.is_set():
                # Give any pending tool tasks a chance to finish OR wait until the
                # loop is resumed / cancelled.  Every coroutine is wrapped in an
                # asyncio.Task so `asyncio.wait()` is happy.
                if tools_data.pending:
                    pause_waiter = asyncio.create_task(
                        pause_event.wait(),
                        name="PauseEventWait",
                    )
                    cancel_waiter = asyncio.create_task(
                        cancel_event.wait(),
                        name="CancelEventWait",
                    )
                    graceful_stop_waiter = asyncio.create_task(
                        stop_event.wait(),
                        name="StopEventWait",
                    )
                    waiters = tools_data.pending | {
                        pause_waiter,
                        cancel_waiter,
                        graceful_stop_waiter,
                    }

                    done, _ = await asyncio.wait(
                        waiters,
                        timeout=0.1,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # helper-task cleanup so they don't dangle
                    for w in (pause_waiter, cancel_waiter, graceful_stop_waiter):
                        if w not in done and not w.done():
                            w.cancel()
                            await asyncio.gather(w, return_exceptions=True)

                    # tool finished?
                    for t in done & tools_data.pending:
                        await _process_completed_task(
                            task=t,
                            task_info=tools_data.info,
                            consecutive_failures=consecutive_failures,
                            pending_tasks=tools_data.pending,
                            outer_handle_container=outer_handle_container,
                            assistant_meta=assistant_meta,
                            clarification_channels=clarification_channels,
                            completed_results=completed_results,
                            client=client,
                            msg_dispatcher=_msg_dispatcher,
                            logger=logger,
                        )
                    if cancel_event.is_set():
                        # Forward stop to any nested handles before aborting
                        try:
                            _stop_forwarded_once = await _propagate_stop_once(
                                tools_data.info,
                                _stop_forwarded_once,
                                "outer-loop cancelled",
                            )
                        except Exception:
                            pass
                        raise asyncio.CancelledError
                    if stop_event.is_set() and persist:
                        try:
                            _stop_forwarded_once = await _propagate_stop_once(
                                tools_data.info,
                                _stop_forwarded_once,
                                "outer-loop stopped",
                            )
                        except Exception:
                            pass
                        # Graceful stop requested during pause
                        return last_final_answer or ""
                    continue  # remain paused: do not allow the LLM to speak while paused
                else:
                    # nothing running – just idle until resumed or cancelled
                    done, _ = await asyncio.wait(
                        {
                            asyncio.create_task(
                                pause_event.wait(),
                                name="PauseEventWait",
                            ),
                            asyncio.create_task(
                                cancel_event.wait(),
                                name="CancelEventWait",
                            ),
                            asyncio.create_task(
                                stop_event.wait(),
                                name="StopEventWait",
                            ),
                        },
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # resumed?
                    if pause_event.is_set():
                        continue  # back to main loop, un-paused

                    # cancelled?
                    if cancel_event.is_set():
                        try:
                            _stop_forwarded_once = await _propagate_stop_once(
                                tools_data.info,
                                _stop_forwarded_once,
                                "outer-loop cancelled",
                            )
                        except Exception:
                            pass
                        raise asyncio.CancelledError
                    if stop_event.is_set() and persist:
                        try:
                            _stop_forwarded_once = await _propagate_stop_once(
                                tools_data.info,
                                _stop_forwarded_once,
                                "outer-loop stopped",
                            )
                        except Exception:
                            pass
                        return last_final_answer or ""
                        continue  # top-of-loop, still paused

            # 0-α. **Global timeout**
            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            # 0-β. **Chat history length**
            if timer.has_exceeded_msgs():
                return await _handle_limit_reached(
                    f"max_steps ({max_steps}) exceeded",
                )

            # 0-γ. Repair any outstanding assistant tool_calls missing replies
            #      before we allow new user interjections to be appended.
            try:
                unreplied = _find_unreplied_assistant_entries(client)
                # Only consider the very latest assistant with missing replies first
                if unreplied:
                    last_problem = unreplied[-1]
                    amsg = last_problem["assistant_msg"]
                    missing_ids = set(last_problem["missing"])
                    # Skip if we already scheduled for this assistant turn
                    if id(amsg) not in assistant_meta:
                        backfilled = await _schedule_missing_for_message(
                            amsg,
                            missing_ids,
                            clarification_channels=clarification_channels,
                            call_counts=call_counts,
                            norm_tools=tools_data.norm_tools,
                            pending=tools_data.pending,
                            completed_results=completed_results,
                            parent_chat_context=parent_chat_context,
                            propagate_chat_context=propagate_chat_context,
                            task_info=tools_data.info,
                            assistant_meta=assistant_meta,
                            client=client,
                            msg_dispatcher=_msg_dispatcher,
                        )
            except Exception:
                pass

            # ── 0. Drain *all* queued interjections, allowed at any time ──
            # NOTE: We must do this *before* waiting on tool completion so a
            # fast typist can still sneak in a question while long-running
            # tools are in flight.  Doing it here keeps latency <1π loop.
            while True:
                try:
                    extra = interject_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                llm_turn_required = True
                # Build system message based on the user-visible history stored on the outer handle.
                history_lines: list[str] = []
                try:
                    outer_handle = (
                        outer_handle_container[0] if outer_handle_container else None
                    )
                    uvh = (
                        getattr(outer_handle, "_user_visible_history", [])
                        if outer_handle
                        else []
                    )
                    for _m in uvh:
                        role = _m.get("role")
                        content = (_m.get("content") or "").strip()
                        if role in ("user", "assistant") and content:
                            history_lines.append(f"{role}: {content}")
                except Exception:
                    # Fallback to just the original user prompt if available
                    try:
                        first_user = next(
                            (
                                m.get("content", "")
                                for m in client.messages
                                if m.get("role") == "user"
                            ),
                            "",
                        )
                        if first_user:
                            history_lines = [f"user: {first_user}"]
                    except Exception:
                        history_lines = []

                sys_content = (
                    "The user *cannot* see *any* the contents of this ongoing tool use chat context. "
                    "They have just interjected with the following message (in bold at the bottom). "
                    "From their perspective, the conversation thus far is as follows:\n"
                    "--\n" + ("\n".join(history_lines)) + f"\nuser: **{extra}**\n"
                    "--\n"
                    "Please consider and incorporate *all* interjections in your final response to the user. "
                    "Later interjections should always override earlier interjections if there are "
                    "any conflicting comments/requests across the different interjections."
                )
                interjection_msg = {"role": "system", "content": sys_content}
                await _msg_dispatcher.append_msgs([interjection_msg])

                # Append this interjection to the user-visible history for future context
                try:
                    if outer_handle:
                        outer_handle._user_visible_history.append(
                            {"role": "user", "content": extra},
                        )
                except Exception:
                    pass

            # ── A.  Wait for tool completion OR cancellation  ───────────────
            # If a child just asked for clarification we also want to give
            # the LLM a chance to react immediately.
            # Skip this whole block if the model already needs to speak.
            # NOTE: ``asyncio.wait`` lets us race three conditions:
            #       • any tool task finishes
            #       • ``cancel_event`` flips
            #       • a *new* interjection appears
            if tools_data.pending and not llm_turn_required:
                interject_w = asyncio.create_task(
                    interject_queue.get(),
                    name="InterjectQueueGet",
                )
                cancel_waiter = asyncio.create_task(
                    cancel_event.wait(),
                    name="CancelEventWait",
                )
                graceful_stop_waiter = asyncio.create_task(
                    stop_event.wait(),
                    name="StopEventWait",
                )
                clar_waiters: Dict[asyncio.Task, asyncio.Task] = {}
                for _t in tools_data.pending:
                    # Only listen for *new* clarification questions.
                    # If the task is already awaiting an answer,
                    # `waiting_for_clarification` will be True.
                    if tools_data.info[_t].get("waiting_for_clarification"):
                        continue

                    cuq = tools_data.info[_t].get("clar_up_q")
                    if cuq is not None:
                        w = asyncio.create_task(cuq.get(), name="ClarificationQueueGet")
                        clar_waiters[w] = _t
                waiters = (
                    tools_data.pending
                    | set(clar_waiters)
                    | {cancel_waiter, interject_w, graceful_stop_waiter}
                )

                # ── honour global *timeout* while we wait for tools ───────────
                if timer.has_exceeded_time():
                    return await _handle_limit_reached(
                        f"timeout ({timeout}s) exceeded",
                    )

                done, _ = await asyncio.wait(
                    waiters,
                    timeout=timer.remaining_time(),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # ── hit the timeout while waiting? ────────────────────────────
                if not done:
                    # nothing completed → the wait *timed out*
                    if raise_on_limit:
                        raise asyncio.TimeoutError(
                            f"Loop exceeded {timeout}s wall-clock limit",
                        )
                    else:
                        return await _handle_limit_reached(
                            f"timeout ({timeout}s) exceeded",
                        )

                # ── ensure *unused* auxiliary waiters don't linger ──────────
                # If one helper won the race we *must* cancel/await the other
                # so that it cannot consume the next interjection invisibly.
                for aux in (
                    interject_w,
                    cancel_waiter,
                    graceful_stop_waiter,
                    *clar_waiters.keys(),
                ):
                    if aux not in done and not aux.done():
                        aux.cancel()
                        await asyncio.gather(aux, return_exceptions=True)

                if interject_w in done:
                    # re-queue so branch 0 will handle user turn immediately
                    await interject_queue.put(interject_w.result())
                    continue  # → loop, will be processed in 0.

                if cancel_waiter in done:
                    try:
                        _stop_forwarded_once = await _propagate_stop_once(
                            tools_data.info,
                            _stop_forwarded_once,
                            "outer-loop cancelled",
                        )
                    except Exception:
                        pass
                    raise asyncio.CancelledError  # cancellation wins
                if graceful_stop_waiter in done and persist:
                    try:
                        _stop_forwarded_once = await _propagate_stop_once(
                            tools_data.info,
                            _stop_forwarded_once,
                            "outer-loop stopped",
                        )
                    except Exception:
                        pass
                    return last_final_answer or ""

                # ── clarification request bubbled up from a child tool ──────────────
                if done & clar_waiters.keys():
                    for cw in done & clar_waiters.keys():
                        question = cw.result()  # the text from the child
                        src_task = clar_waiters[cw]
                        call_id = tools_data.info[src_task]["call_id"]

                        # 1️⃣ mark the task as waiting
                        tools_data.info[src_task]["waiting_for_clarification"] = True

                        # 2️⃣ REUSE the existing placeholder if we already inserted one
                        ph = tools_data.info[src_task].get("tool_reply_msg")
                        if ph is None:
                            # no placeholder yet → create one exactly once
                            ph = {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": f"clarification_request_{call_id}",
                                "content": "",  # will fill below
                            }
                            await _insert_after_assistant(
                                assistant_meta,
                                tools_data.info[src_task]["assistant_msg"],
                                ph,
                                client,
                                _msg_dispatcher,
                            )
                            tools_data.info[src_task]["tool_reply_msg"] = ph

                        # 3️⃣ turn (or update) the placeholder into the request
                        ph["name"] = f"clarification_request_{call_id}"
                        ph["content"] = (
                            "Tool incomplete, please answer the following to continue "
                            f"tool execution:\n{question}"
                        )
                        tool_msg = ph  # for event_bus

                    # let the assistant answer immediately
                    llm_turn_required = True
                    continue

                needs_turn = False
                for task in done:  # finished tool(s)
                    if await _process_completed_task(
                        task=task,
                        task_info=tools_data.info,
                        consecutive_failures=consecutive_failures,
                        pending_tasks=tools_data.pending,
                        outer_handle_container=outer_handle_container,
                        assistant_meta=assistant_meta,
                        clarification_channels=clarification_channels,
                        completed_results=completed_results,
                        client=client,
                        msg_dispatcher=_msg_dispatcher,
                        logger=logger,
                    ):
                        needs_turn = True

                # Other tools may still be running.
                if tools_data.pending:
                    if needs_turn:  # only when something new
                        llm_turn_required = True
                    continue  # jump to top-of-loop

            # ── B: wait for remaining tools before asking the LLM again,
            # unless the model already deserves a turn
            if tools_data.pending and not llm_turn_required:
                # Ensure placeholders exist for any pending calls before the next assistant turn
                await _ensure_placeholders_for_pending(
                    content=(
                        "Still running… you can use any of the available helper tools "
                        "to interact with this tool call while it is in progress."
                    ),
                    task_info=tools_data.info,
                    pending=tools_data.pending,
                    assistant_meta=assistant_meta,
                    client=client,
                    msg_dispatcher=_msg_dispatcher,
                )
                continue  # still waiting for other tool tasks

            # ── C.  Add temporary tools so the LLM can **continue** or **cancel**
            #       any still‑running tool calls ────────────────────────────────
            #
            # For each pending ``asyncio.Task`` we synthesise two VERY small helper
            # tools and expose them to the model on the *next* LLM step.  Each
            # helper's docstring is a single line that embeds **both** the name of
            # the original function **and** the concrete arguments it was invoked
            # with – this gives the agent just enough context without overwhelming
            # the token budget.
            # ------------------------------------------------------------------

            dynamic_tools: Dict[str, Callable] = {}

            # ------------------------------------------------------------------
            # 1.  Build the *static* part of the toolkit **fresh on every turn**
            #     so that concurrency changes (tasks finishing, stopping, …)
            #     are immediately reflected in what the LLM can see.
            # ------------------------------------------------------------------

            # 0.  Decide policy & tool-subset for this turn  ───────────────
            if tool_policy is not None:
                try:
                    tool_choice_mode, filtered = tool_policy(
                        step_index,
                        {n: s.fn for n, s in tools_data.norm_tools.items()},
                    )
                except Exception as _e:  # never abort the loop on mis-behaving policies
                    logger.error(
                        f"tool_policy raised on turn {step_index}: {_e!r}",
                    )
                    tool_choice_mode, filtered = "auto", {
                        n: s.fn for n, s in tools_data.norm_tools.items()
                    }
                policy_tools_norm = _normalise_tools(filtered)
            else:
                tool_choice_mode = "auto"
                policy_tools_norm = tools_data.norm_tools

            def _concurrency_ok(tn: str) -> bool:
                return tn not in tools_data.norm_tools or _can_offer_tool(tn)

            visible_base_tools_schema = [
                method_to_schema(spec.fn, name)
                for name, spec in policy_tools_norm.items()
                if _concurrency_ok(name) and _quota_ok(name)
            ]

            # Inject `final_answer` tool automatically whenever a `response_format` is
            # supplied. The tool accepts a single `answer` argument whose schema matches
            # the provided Pydantic model.
            if response_format is not None:
                try:
                    _answer_schema = _check_valid_response_format(response_format)

                    visible_base_tools_schema.append(
                        {
                            "type": "function",
                            "strict": True,
                            "function": {
                                "name": "final_answer",
                                "description": (
                                    "Submit your final answer in the required JSON format. "
                                    "Calling this tool marks the conversation as complete."
                                ),
                                "parameters": {
                                    "type": "object",
                                    "properties": {"answer": _answer_schema},
                                    "required": ["answer"],
                                },
                            },
                        },
                    )
                except Exception as _injection_exc:  # noqa: BLE001
                    logger.error(
                        f"Failed to inject final_answer tool: {_injection_exc!r}",
                    )

            for _task in list(tools_data.pending):
                info = tools_data.info[_task]
                handle = info.get("handle")
                ev = info.get("pause_event")

                # ── DYNAMIC capability refresh (handle may change) ─────
                if handle is not None:
                    # 1. interjection
                    info["is_interjectable"] = hasattr(handle, "interject")

                    # 2. clarification queues
                    h_up_q = getattr(
                        handle,
                        "clarification_up_q",
                        info.get("clar_up_q"),
                    )
                    h_dn_q = getattr(
                        handle,
                        "clarification_down_q",
                        info.get("clar_down_q"),
                    )

                    if (h_up_q is not None) ^ (h_dn_q is not None):
                        raise AttributeError(
                            f"Handle of call {info['call_id']} now exposes only one "
                            "of clarification queues; both or neither required.",
                        )

                    # update bookkeeping & channel map
                    prev_up_q = info.get("clar_up_q")
                    if h_up_q is not prev_up_q:
                        # remove old mapping if any
                        clarification_channels.pop(info["call_id"], None)
                        if h_up_q is not None:
                            clarification_channels[info["call_id"]] = (
                                h_up_q,
                                h_dn_q,
                            )
                    info["clar_up_q"] = h_up_q
                    info["clar_down_q"] = h_dn_q

                _call_id: str = info["call_id"]
                # Create a sanitized version of the call_id for use in function names.
                _safe_call_id: str = _call_id.replace("-", "_").split("_")[-1]
                _fn_name: str = info["name"]
                _arg_json: str = info["call_dict"]["function"]["arguments"]
                try:
                    _arg_dict = json.loads(_arg_json)
                    _arg_repr = ", ".join(f"{k}={v!r}" for k, v in _arg_dict.items())
                except Exception:
                    _arg_repr = _arg_json  # fallback: raw JSON string

                # concise, informative, single‑line docs  ----------------------
                _continue_doc = f"Continue waiting for {_fn_name}({_arg_repr})."
                _stop_doc = (
                    f"Stop pending call {_fn_name}({_arg_repr}). "
                    "Accepts any arguments supported by the underlying handle's `stop` method (e.g. `reason`)."
                )

                # ––– 1. continue helper ––––––––––––––––––––––––––––––––––––
                # Skip if the task is blocked waiting for clarification; there's
                # nothing to "continue" until the user answers.
                if not info.get("waiting_for_clarification"):

                    async def _continue() -> Dict[str, str]:
                        return {"status": "continue", "call_id": _call_id}

                    _register_tool(
                        key=f"continue_{_fn_name}_{_safe_call_id}",
                        func_name=f"continue_{_fn_name}_{_safe_call_id}",
                        doc=_continue_doc,
                        fn=_continue,
                        dynamic_tools=dynamic_tools,
                    )

                # ––– 2. stop helper –––––––––––––––––––––––––––––––––––––
                async def _stop(
                    **_kw,
                ) -> Dict[str, str]:
                    # Forward stop intent to the running handle with any extra kwargs
                    if handle is not None and hasattr(handle, "stop"):
                        await _forward_handle_call(
                            handle,
                            "stop",
                            _kw,
                            fallback_positional_keys=["reason"],
                        )
                    if not _task.done():
                        _task.cancel()  # kill the waiter coroutine
                    tools_data.pending.discard(_task)
                    tools_data.info.pop(_task, None)
                    return {"status": "stopped", "call_id": _call_id, **_kw}

                _register_tool(
                    key=f"stop_{_fn_name}_{_safe_call_id}",
                    func_name=f"stop_{_fn_name}_{_safe_call_id}",
                    doc=_stop_doc,
                    fn=_stop,
                    dynamic_tools=dynamic_tools,
                )
                # Expose full argspec of handle.stop in the helper schema
                try:
                    if handle is not None and hasattr(handle, "stop"):
                        _adopt_signature_and_annotations(getattr(handle, "stop"), _stop)
                except Exception:
                    pass

                # ––– 3. interject helper (optional) ––––––––––––––––––––––
                if info.get("is_interjectable"):
                    _interject_doc = (
                        f"Inject additional instructions for {_fn_name}({_arg_repr}). "
                        "Accepts any arguments supported by the underlying handle's `interject` method (e.g. `content`)."
                    )

                    if handle is not None:

                        async def _interject(**_kw) -> Dict[str, str]:
                            # nested async-tool loop: delegate to its public API with full argspec
                            try:
                                await _forward_handle_call(
                                    handle,
                                    "interject",
                                    _kw,
                                    fallback_positional_keys=["content", "message"],
                                )
                            except Exception:
                                pass
                            return {
                                "status": "interjected",
                                "call_id": _call_id,
                                **{k: v for k, v in _kw.items()},
                            }

                        # Expose the downstream handle's signature to the LLM
                        try:
                            _adopt_signature_and_annotations(
                                getattr(handle, "interject"),
                                _interject,
                            )
                        except Exception:
                            pass

                    else:

                        async def _interject(content: str) -> Dict[str, str]:
                            # regular tool: push onto its private queue
                            await info["interject_q"].put(content)
                            return {
                                "status": "interjected",
                                "call_id": _call_id,
                                "content": content,
                            }

                    _register_tool(
                        key=f"interject_{_fn_name}_{_safe_call_id}",
                        func_name=f"interject_{_fn_name}_{_safe_call_id}",
                        doc=_interject_doc,
                        fn=_interject,
                        dynamic_tools=dynamic_tools,
                    )

                # ––– 4. clarification-answer helper (optional) ––––––––––
                if info.get("clar_up_q") is not None:
                    _clarify_doc = (
                        f"Provide an answer to the clarification which was requested by the (currently pending) tool "
                        f"{_fn_name}({_arg_repr}). Takes a single argument `answer`."
                    )

                    async def _clarify(answer: str) -> Dict[str, str]:  # type: ignore[valid-type]
                        return {
                            "status": "clar_answer",
                            "call_id": _call_id,
                            "answer": answer,
                        }

                    _register_tool(
                        key=f"clarify_{_fn_name}_{_safe_call_id}",
                        func_name=f"clarify_{_fn_name}_{_safe_call_id}",
                        doc=_clarify_doc,
                        fn=_clarify,
                        dynamic_tools=dynamic_tools,
                    )

                # ––– 5. pause helper –––––––––––––––––––––––––––––––––––––––––––
                can_pause = (handle is not None and hasattr(handle, "pause")) or ev
                can_resume = (handle is not None and hasattr(handle, "resume")) or ev

                if can_pause:
                    _pause_doc = f"Pause the pending call {_fn_name}({_arg_repr})."

                    if handle is not None and hasattr(handle, "pause"):

                        async def _pause(**_kw) -> Dict[str, str]:
                            try:
                                await _forward_handle_call(handle, "pause", _kw)
                            except Exception:
                                pass
                            return {"status": "paused", "call_id": _call_id, **_kw}

                        # Reflect downstream signature/annotations
                        try:
                            _adopt_signature_and_annotations(
                                getattr(handle, "pause"),
                                _pause,
                            )
                        except Exception:
                            pass

                    else:

                        async def _pause() -> Dict[str, str]:
                            if handle is not None and hasattr(handle, "pause"):
                                await _maybe_await(handle.pause())
                            elif ev is not None:
                                ev.clear()
                            return {"status": "paused", "call_id": _call_id}

                    _register_tool(
                        key=f"pause_{_fn_name}_{_safe_call_id}",
                        func_name=f"pause_{_fn_name}_{_safe_call_id}",
                        doc=_pause_doc,
                        fn=_pause,
                        dynamic_tools=dynamic_tools,
                    )

                # ––– 6. resume helper ––––––––––––––––––––––––––––––––––––––––––
                if can_resume:
                    _resume_doc = (
                        f"Resume the previously paused call {_fn_name}({_arg_repr})."
                    )

                    if handle is not None and hasattr(handle, "resume"):

                        async def _resume(**_kw) -> Dict[str, str]:
                            try:
                                await _forward_handle_call(handle, "resume", _kw)
                            except Exception:
                                pass
                            return {"status": "resumed", "call_id": _call_id, **_kw}

                        try:
                            _adopt_signature_and_annotations(
                                getattr(handle, "resume"),
                                _resume,
                            )
                        except Exception:
                            pass

                    else:

                        async def _resume() -> Dict[str, str]:
                            if handle is not None and hasattr(handle, "resume"):
                                await _maybe_await(handle.resume())
                            elif ev is not None:
                                ev.set()
                            return {"status": "resumed", "call_id": _call_id}

                    _register_tool(
                        key=f"resume_{_fn_name}_{_safe_call_id}",
                        func_name=f"resume_{_fn_name}_{_safe_call_id}",
                        doc=_resume_doc,
                        fn=_resume,
                        dynamic_tools=dynamic_tools,
                    )

                # 7.  expose *all* other public methods of the handle
                if handle is not None:

                    public_methods = _discover_custom_public_methods(handle)

                    # ── honour handle.valid_tools, if present ──────────────
                    if hasattr(handle, "valid_tools"):
                        allowed: set[str] = set(getattr(handle, "valid_tools", []))
                        public_methods = {
                            name: bound
                            for name, bound in public_methods.items()
                            if name in allowed
                        }

                    # Identify write-only helpers declared by the handle
                    write_only_set: set[str] = set()
                    try:
                        wo = getattr(handle, "write_only_methods", None)
                        if wo is not None:
                            write_only_set |= set(wo)
                    except Exception:
                        pass
                    try:
                        wo2 = getattr(handle, "write_only_tools", None)
                        if wo2 is not None:
                            write_only_set |= set(wo2)
                    except Exception:
                        pass

                    for meth_name, bound in public_methods.items():
                        # use the same name we're about to give fn.__name__
                        func_name = f"{meth_name}_{_fn_name}_{_safe_call_id}"
                        helper_key = func_name

                        # Skip if we already generated one this turn (possible when
                        # the loop revisits the same pending task).
                        if helper_key in dynamic_tools:
                            continue

                        # Write-only helpers: fire-and-forget operations
                        if meth_name in write_only_set:

                            async def _invoke_handle_method(
                                _method_name=meth_name,
                                **_kw,
                            ):
                                # Robust forwarding incl. kwargs normalisation and fallbacks
                                try:
                                    await _forward_handle_call(
                                        handle,
                                        _method_name,
                                        _kw,
                                    )
                                except Exception:
                                    pass
                                # Write-only: no result propagation
                                return {"call_id": _call_id, "status": "ack"}

                        else:

                            async def _invoke_handle_method(
                                _method_name=meth_name,
                                **_kw,
                            ):  # default args → capture current method name
                                """
                                Auto-generated wrapper that calls the corresponding
                                method on the live handle and **waits** for the return
                                value (sync or async).
                                """
                                # Use shared forwarding to support flexible args and fallbacks
                                res = await _forward_handle_call(
                                    handle,
                                    _method_name,
                                    _kw,
                                )
                                return {"call_id": _call_id, "result": res}

                        # override the wrapper's signature to match the real method
                        _invoke_handle_method.__signature__ = inspect.signature(bound)

                        _register_tool(
                            key=helper_key,
                            func_name=func_name,
                            doc=(
                                (
                                    f"Perform `{meth_name}` on the running handle (id={_call_id}). "
                                    "Fire-and-forget write-only operation; returns immediately."
                                )
                                if meth_name in write_only_set
                                else (
                                    f"Invoke `{meth_name}` on the running handle (id={_call_id}). "
                                    "Returns when that method finishes."
                                )
                            ),
                            fn=_invoke_handle_method,
                            dynamic_tools=dynamic_tools,
                        )
                        # Mark write-only helpers so scheduling can acknowledge and avoid tracking
                        if meth_name in write_only_set:
                            try:
                                dynamic_tools[helper_key].__write_only__ = True  # type: ignore[attr-defined]
                            except Exception:
                                pass

            # make sure every pending call already has a *tool* reply ──
            #  (a placeholder) before we let the assistant speak again.
            await _ensure_placeholders_for_pending(
                content=(
                    "Still running… you can use any of the available helper tools "
                    "to interact with this tool call while it is in progress."
                ),
                task_info=tools_data.info,
                pending=tools_data.pending,
                assistant_meta=assistant_meta,
                client=client,
                msg_dispatcher=_msg_dispatcher,
            )

            # Merge helpers into the visible toolkit for the upcoming LLM step
            tmp_tools = visible_base_tools_schema + [
                method_to_schema(
                    fn,
                    include_class_name=include_class_in_dynamic_tool_names,
                )
                for fn in dynamic_tools.values()
            ]

            # ── D.  Ask the LLM what to do next  ────────────────────────────
            if log_steps:
                logger.info(f"LLM thinking…", prefix="🔄")

            if interrupt_llm_with_interjections:
                # ––––– new *pre-emptive* mode ––––––––––––––––––––––––––––
                # ➊ start the LLM step …
                _gen_kwargs = {
                    "return_full_completion": True,
                    "tools": tmp_tools,
                    "tool_choice": tool_choice_mode,
                    "stateful": True,
                }
                if max_parallel_tool_calls is not None:
                    _gen_kwargs["max_tool_calls"] = max_parallel_tool_calls

                llm_task = asyncio.create_task(
                    _generate_with_preprocess(client, preprocess_msgs, **_gen_kwargs),
                    name="LLMGenerate",
                )
                interject_w = asyncio.create_task(
                    interject_queue.get(),
                    name="InterjectQueueGet",
                )
                cancel_waiter = asyncio.create_task(
                    cancel_event.wait(),
                    name="CancelEventWait",
                )

                # ➋ …but ALSO watch the tool tasks that were still pending
                pending_snapshot = set(tools_data.pending)

                done, _ = await asyncio.wait(
                    pending_snapshot | {llm_task, interject_w, cancel_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # helper cleanup
                for tsk in (llm_task, interject_w, cancel_waiter):
                    if tsk not in done and not tsk.done():
                        tsk.cancel()
                await asyncio.gather(interject_w, cancel_waiter, return_exceptions=True)

                # 0️⃣ A *different* tool finished before the LLM answered -----
                if done & pending_snapshot:  # ← NEW
                    # — cancel the half-finished reasoning step
                    if not llm_task.done():
                        llm_task.cancel()
                    for aux in (interject_w, cancel_waiter):
                        if aux not in done and not aux.done():
                            aux.cancel()
                    await asyncio.gather(
                        llm_task,
                        interject_w,
                        cancel_waiter,
                        return_exceptions=True,
                    )
                    # — handle each newly-finished task exactly as branch A does
                    needs_turn = False
                    for task in done & pending_snapshot:
                        if await _process_completed_task(
                            task=task,
                            task_info=tools_data.info,
                            consecutive_failures=consecutive_failures,
                            pending_tasks=tools_data.pending,
                            outer_handle_container=outer_handle_container,
                            assistant_meta=assistant_meta,
                            clarification_channels=clarification_channels,
                            completed_results=completed_results,
                            client=client,
                            msg_dispatcher=_msg_dispatcher,
                            logger=logger,
                        ):
                            needs_turn = True

                    # …then restart the main loop so the model sees the new info
                    if needs_turn:  # assistant speaks only if needed
                        llm_turn_required = True
                    continue

                # 1️⃣ user interjected → restart immediately
                if interject_w in done:
                    if not llm_task.done():
                        llm_task.cancel()
                        await asyncio.gather(llm_task, return_exceptions=True)
                    await interject_queue.put(interject_w.result())
                    continue  # top of loop

                # 2️⃣ cancellation requested
                if cancel_waiter in done:
                    if not llm_task.done():
                        llm_task.cancel()
                        await asyncio.gather(llm_task, return_exceptions=True)
                    raise asyncio.CancelledError

                # 3️⃣ LLM finished normally
                if llm_task.exception():
                    raise Exception(
                        f"LLM call failed. Messages at the time:\n{json.dumps(client.messages, indent=4)}, exception: {llm_task.exception()}",
                    )

            else:
                # ––––– legacy *blocking* mode ––––––––––––––––––––––––––––
                try:
                    _gen_kwargs = {
                        "return_full_completion": True,
                        "tools": tmp_tools,
                        "tool_choice": tool_choice_mode,
                        "stateful": True,
                    }
                    if max_parallel_tool_calls is not None:
                        _gen_kwargs["max_tool_calls"] = max_parallel_tool_calls

                    await _generate_with_preprocess(
                        client,
                        preprocess_msgs,
                        **_gen_kwargs,
                    )
                except Exception:
                    raise Exception(
                        f"LLM call failed. Messages at the time:\n{json.dumps(client.messages, indent=4)}",
                    )

            msg = client.messages[-1]
            await _to_event_bus(msg, cfg)

            if log_steps:
                try:
                    logger.info(f"{json.dumps(msg, indent=4)}\n", prefix="🤖")
                except Exception:
                    logger.info(
                        f"Assistant message appended (unserializable)",
                        prefix="🤖",
                    )

            # ── timeout guard (post-LLM) ───────────────────────────────
            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            # LLM has just spoken – reset the flag
            llm_turn_required = False
            # one full assistant turn completed
            step_index += 1

            # ── E.  Launch any new tool calls  ──────────────────────────────
            # NOTE: The model returned `tool_calls`.  For *each* call we:
            #   1. JSON-parse the arguments once (costly in Python – do it
            #      outside the worker thread).
            #   2. Wrap sync functions in `asyncio.to_thread` so the event
            #      loop is never blocked by CPU / I/O.
            #   3. Create an `asyncio.Task` and remember contextual metadata
            #      in `task_info` so we can later insert the result in the
            #      exact chronological position.
            #   4. Keep a pristine copy of the original `tool_calls` list;
            #      step A temporarily hides it to avoid "naked" unresolved
            #      calls flashing in the UI, and restores it once *any*
            #      result for that assistant turn is ready.
            # Finally we `continue` so control jumps back to *branch A*
            # where we wait for the **first** task / cancel / interjection.
            if msg["tool_calls"]:
                # ── De-duplicate tool calls (optional) ────────────────────────
                if prune_tool_duplicates:
                    seen: Set[tuple[str, str]] = set()
                    unique_calls: list = []
                    for call in msg["tool_calls"]:
                        sig = (call["function"]["name"], call["function"]["arguments"])
                        if sig not in seen:
                            seen.add(sig)
                            unique_calls.append(call)
                    if len(unique_calls) != len(msg["tool_calls"]):
                        # mutate in-place so history never contains duplicates
                        msg["tool_calls"] = unique_calls

                # Always ensure over-quota tool calls are removed regardless of
                # deduplication settings, before any scheduling occurs.
                _prune_over_quota_tool_calls(msg, tools_data.norm_tools, call_counts)
                for idx, call in enumerate(msg["tool_calls"]):  # capture index
                    name = call["function"]["name"]
                    args = json.loads(call["function"]["arguments"])

                    # Special-case: handle synthetic `final_answer` tool
                    if name == "final_answer" and response_format is not None:
                        try:
                            payload = (
                                args.get("answer") if isinstance(args, dict) else None
                            )
                            if payload is None:
                                raise ValueError("Missing 'answer' in tool arguments.")

                            # Validate payload with the provided Pydantic model.
                            response_format.model_validate(payload)

                            tool_msg = {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "name": "final_answer",
                                "content": _dumps(payload, indent=4),
                            }
                            await _insert_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )

                            return json.dumps(payload)
                        except Exception as _exc:
                            tool_msg = {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "name": "final_answer",
                                "content": (
                                    "⚠️ Validation failed – proceeding with standard formatting step.\n"
                                    + str(_exc)
                                ),
                            }
                            await _insert_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                            continue

                    # ── Special-case dynamic helpers ──────────────────────
                    # • continue_* → acknowledge, no scheduling
                    # • cancel_*   → cancel underlying task & purge metadata
                    if _is_helper_tool("continue_") and name.startswith("continue_"):
                        # Helper names are of the form: continue_{toolName}_{safeId}
                        call_id_suffix = name.split("_")[-1]

                        tgt_task = next(
                            (
                                t
                                for t, inf in tools_data.info.items()
                                if str(inf.get("call_id", "")).endswith(call_id_suffix)
                            ),
                            None,
                        )

                        orig_fn = (
                            tools_data.info[tgt_task]["name"] if tgt_task else "unknown"
                        )
                        arg_json = (
                            tools_data.info[tgt_task]["call_dict"]["function"][
                                "arguments"
                            ]
                            if tgt_task
                            else "{}"
                        )
                        pretty_name = f"continue {orig_fn}({arg_json})"

                        if tgt_task:  # still running → insert generated placeholder now
                            info = tools_data.info[tgt_task]
                            name = info["name"]
                            arg_json = info["call_dict"]["function"]["arguments"]
                            tool_reply_msg = {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "name": name,
                                "content": (
                                    "The following tool calls are still running. If any of them are no longer "
                                    "relevant to the sequence of user requests, then you can call their "
                                    f"`_cancel_*` helper, otherwise feel free to call the corresponding "
                                    f"`_continue_*` helper to keep waiting:\n"
                                    f" • {name}({arg_json}) → cancel_{call['id']} / continue_{call['id']}"
                                ),
                            }
                            await _insert_after_assistant(
                                assistant_meta,
                                msg,
                                tool_reply_msg,
                                client,
                                _msg_dispatcher,
                            )
                            info["continue_msg"] = tool_reply_msg
                        else:  # the original tool already finished
                            # Lookup finished result by matching call-id suffix
                            _full_id = next(
                                (
                                    k
                                    for k in completed_results.keys()
                                    if k.endswith(call_id_suffix)
                                ),
                                None,
                            )
                            finished = completed_results.get(
                                _full_id,
                                _dumps(
                                    {"status": "not-found", "call_id": call_id_suffix},
                                    indent=4,
                                ),
                            )
                            tool_msg = {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "name": pretty_name,
                                "content": finished,
                            }
                            await _insert_after_assistant(
                                assistant_meta,
                                msg,
                                tool_msg,
                                client,
                                _msg_dispatcher,
                            )
                        continue  # completed handling of this _continue

                    if name.startswith("stop_") and not name.startswith(
                        "_stop_tasks",
                    ):
                        # Helper names are of the form: stop_{toolName}_{safeId}
                        call_id_suffix = name.split("_")[-1]

                        # ── locate & cancel the underlying coroutine ──────
                        task_to_cancel = next(
                            (
                                t
                                for t, info in tools_data.info.items()
                                if str(info.get("call_id", "")).endswith(call_id_suffix)
                            ),
                            None,
                        )

                        orig_fn = (
                            tools_data.info[task_to_cancel]["name"]
                            if task_to_cancel
                            else "unknown"
                        )
                        arg_json = (
                            tools_data.info[task_to_cancel]["call_dict"]["function"][
                                "arguments"
                            ]
                            if task_to_cancel
                            else "{}"
                        )
                        pretty_name = f"stop   {orig_fn}({arg_json})"

                        # Parse payload to forward extras to handle.stop if available
                        try:
                            payload = json.loads(call["function"]["arguments"]) or {}
                        except Exception:
                            payload = {}

                        # ── gracefully shut down any *nested* async-tool loop first ──────
                        if task_to_cancel:
                            nested_handle = tools_data.info[task_to_cancel].get(
                                "handle",
                            )
                            if nested_handle is not None:
                                # public API call – propagates cancellation downwards
                                await _forward_handle_call(
                                    nested_handle,
                                    "stop",
                                    payload,
                                    fallback_positional_keys=["reason"],
                                )

                        # ── then cancel the waiter coroutine itself ───────────────────────────
                        if task_to_cancel and not task_to_cancel.done():
                            task_to_cancel.cancel()
                        if task_to_cancel:
                            tools_data.pending.discard(task_to_cancel)
                            tools_data.info.pop(task_to_cancel, None)

                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": pretty_name,
                            "content": (
                                f"The tool call [{call_id_suffix}] has been stopped successfully."
                            ),
                        }
                        await _insert_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )

                        continue  # nothing else to schedule

                    # ── _pause helper ────────────────────────────────────────────────
                    if name.startswith("pause_") and not name.startswith(
                        "_pause_tasks",
                    ):
                        call_id_suffix = name.split("_")[-1]
                        tgt_task = next(
                            (
                                t
                                for t, info in tools_data.info.items()
                                if call_id_suffix in info["call_id"]
                            ),
                            None,
                        )
                        orig_fn = (
                            tools_data.info[tgt_task]["name"] if tgt_task else "unknown"
                        )
                        arg_json = (
                            tools_data.info[tgt_task]["call_dict"]["function"][
                                "arguments"
                            ]
                            if tgt_task
                            else "{}"
                        )
                        pretty_name = f"pause {orig_fn}({arg_json})"

                        # Forward any extra kwargs to handle.pause if available
                        try:
                            payload = json.loads(call["function"]["arguments"]) or {}
                        except Exception:
                            payload = {}

                        if tgt_task:
                            h = tools_data.info[tgt_task].get("handle")
                            ev = tools_data.info[tgt_task].get("pause_event")
                            if h is not None and hasattr(h, "pause"):
                                await _forward_handle_call(h, "pause", payload)
                            elif ev is not None:
                                ev.clear()

                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": pretty_name,
                            "content": f"The tool call [{call_id_suffix}] has been paused successfully.",
                        }
                        await _insert_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )
                        continue  # helper handled, move on

                    # ── _resume helper ───────────────────────────────────────────────
                    if name.startswith("resume_") and not name.startswith(
                        "_resume_tasks",
                    ):
                        call_id_suffix = name.split("_")[-1]
                        tgt_task = next(
                            (
                                t
                                for t, info in tools_data.info.items()
                                if call_id_suffix in info["call_id"]
                            ),
                            None,
                        )
                        orig_fn = (
                            tools_data.info[tgt_task]["name"] if tgt_task else "unknown"
                        )
                        arg_json = (
                            tools_data.info[tgt_task]["call_dict"]["function"][
                                "arguments"
                            ]
                            if tgt_task
                            else "{}"
                        )
                        pretty_name = f"resume {orig_fn}({arg_json})"

                        # Forward any extra kwargs to handle.resume if available
                        try:
                            payload = json.loads(call["function"]["arguments"]) or {}
                        except Exception:
                            payload = {}

                        if tgt_task:
                            h = tools_data.info[tgt_task].get("handle")
                            ev = tools_data.info[tgt_task].get("pause_event")
                            if h is not None and hasattr(h, "resume"):
                                await _forward_handle_call(h, "resume", payload)
                            elif ev is not None:
                                ev.set()

                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": pretty_name,
                            "content": f"The tool call [{call_id_suffix}] has been resumed successfully.",
                        }
                        await _insert_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )
                        continue  # helper handled

                    if name.startswith("clarify_"):
                        # Helper names are of the form: clarify_{toolName}_{safeId}
                        call_id_suffix = name.split("_")[-1]
                        ans = args["answer"]

                        # ── find the underlying pending task (if still alive) ───────────────
                        tgt_task = next(  # ← NEW
                            (
                                t
                                for t, inf in tools_data.info.items()
                                if str(inf.get("call_id", "")).endswith(call_id_suffix)
                            ),
                            None,
                        )

                        # Find clarification channel by matching call-id suffix
                        _clar_key = next(
                            (
                                k
                                for k in clarification_channels.keys()
                                if k.endswith(call_id_suffix)
                            ),
                            None,
                        )
                        if _clar_key is not None:
                            await clarification_channels[_clar_key][1].put(
                                ans,
                            )  # down-queue
                            # ✔️ the tool is un-blocked – start watching it again
                            for _t, _inf in tools_data.info.items():
                                if str(_inf.get("call_id", "")).endswith(
                                    call_id_suffix,
                                ):
                                    _inf["waiting_for_clarification"] = False
                                    break
                        tool_reply_msg = {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": name,
                            "content": (
                                f"Clarification answer sent upstream: {ans!r}\n"
                                "⏳ Waiting for the original tool to finish…"
                            ),
                        }
                        await _insert_after_assistant(
                            assistant_meta,
                            msg,
                            tool_reply_msg,
                            client,
                            _msg_dispatcher,
                        )
                        if tgt_task is not None:
                            tools_data.info[tgt_task][
                                "clarify_placeholder"
                            ] = tool_reply_msg
                        continue

                    if name.startswith("interject_"):
                        # helper signature mirrors downstream handle.interject (content plus any extras)
                        try:
                            payload = json.loads(call["function"]["arguments"]) or {}
                            new_text = payload.get("content") or payload.get("message")
                            if new_text is None:
                                new_text = ""
                        except Exception:
                            payload = {}
                            new_text = "<unparsable>"

                        # Helper names are of the form: interject_{toolName}_{safeId}
                        call_id_suffix = name.split("_")[-1]

                        # locate the underlying long-running task
                        tgt_task = next(
                            (
                                t
                                for t, inf in tools_data.info.items()
                                if str(inf.get("call_id", "")).endswith(call_id_suffix)
                            ),
                            None,
                        )

                        pretty_name = (
                            f"interject {tools_data.info[tgt_task]['name']}({new_text})"
                            if tgt_task
                            else name
                        )

                        # ― push guidance onto the private queue or forward to handle with full kwargs -------------
                        if tgt_task:
                            iq = tools_data.info[tgt_task]["interject_q"]
                            h = tools_data.info[tgt_task].get("handle")

                            if iq is not None:
                                await iq.put(new_text)
                            elif h is not None and hasattr(h, "interject"):
                                await _forward_handle_call(
                                    h,
                                    "interject",
                                    payload,
                                    fallback_positional_keys=["content", "message"],
                                )

                        # ― emit a tool message so the chat log stays tidy ---
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": pretty_name,
                            "content": f'Guidance "{new_text}" forwarded to the running tool.',
                        }
                        await _insert_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )

                        continue  # nothing else to schedule

                    # Respect hidden per-tool total-call quotas (pre-pruned); guard
                    if (
                        name in tools_data.norm_tools
                        and tools_data.norm_tools[name].max_total_calls is not None
                        and call_counts.get(name, 0)
                        >= tools_data.norm_tools[name].max_total_calls
                    ):
                        continue

                    # Respect *per-tool* concurrency limits  ────────────────
                    if (
                        name in tools_data.norm_tools
                        and tools_data.norm_tools[name].max_concurrent is not None
                        and _active_count(name)
                        >= tools_data.norm_tools[name].max_concurrent
                    ):
                        # Concurrency cap reached → immediately insert a
                        # *tool-error* message and **do not** schedule.
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": name,
                            "content": (
                                f"⚠️ Cannot start '{name}': "
                                f"max_concurrent={tools_data.norm_tools[name].max_concurrent} "
                                "already reached. Wait for an existing call to "
                                "finish or stop one before retrying."
                            ),
                        }
                        await _insert_after_assistant(
                            assistant_meta,
                            msg,
                            tool_msg,
                            client,
                            _msg_dispatcher,
                        )
                        continue

                    # first check any dynamic helpers we generated for long-running handles
                    if name in dynamic_tools:
                        fn = dynamic_tools[name]

                        # ── build **extra** kwargs (chat context + queue) for dynamic helper ──
                        extra_kwargs: dict = {}
                        if propagate_chat_context:
                            cur_msgs = [
                                m for m in client.messages if not m.get("_ctx_header")
                            ]
                            ctx_repr = _chat_context_repr(parent_chat_context, cur_msgs)
                            extra_kwargs["parent_chat_context"] = ctx_repr

                        sig = inspect.signature(fn)
                        params = sig.parameters
                        has_varkw = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in params.values()
                        )
                        filtered_extras = {
                            k: v
                            for k, v in extra_kwargs.items()
                            if k in params or has_varkw
                        }
                        # Forward ALL call args verbatim. Let the callee raise if unsupported.
                        allowed_call_args = args
                        merged_kwargs = {**allowed_call_args, **filtered_extras}

                        if asyncio.iscoroutinefunction(fn):
                            coro = fn(**merged_kwargs)
                        else:
                            coro = asyncio.to_thread(fn, **merged_kwargs)

                        call_dict = {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": call["function"]["arguments"],
                            },
                        }
                        # If this dynamic helper is marked as write-only, acknowledge immediately
                        # and run fire-and-forget without tracking in pending/task_info.
                        if getattr(fn, "__write_only__", False):
                            try:
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": call["id"],
                                    "name": name,
                                    "content": _build_helper_ack_content(
                                        name,
                                        call["function"]["arguments"],
                                    ),
                                }
                                await _insert_after_assistant(
                                    assistant_meta,
                                    msg,
                                    tool_msg,
                                    client,
                                    _msg_dispatcher,
                                )
                            except Exception:
                                pass
                            try:
                                asyncio.create_task(coro, name=f"ToolCall_{name}")
                            except Exception:
                                pass
                            continue

                        # Scheduling dynamic helper call

                        t = asyncio.create_task(coro, name=f"ToolCall_{name}")
                        tools_data.pending.add(t)
                        tools_data.info[t] = {
                            "name": name,
                            "call_id": call["id"],
                            "assistant_msg": msg,
                            "call_dict": call_dict,
                            "call_idx": idx,
                            "is_interjectable": False,
                            "interject_q": None,
                            "chat_ctx": extra_kwargs.get("parent_chat_context"),
                            "clar_up_q": None,
                            "clar_down_q": None,
                            "pause_event": None,
                            # Debug helpers for failure logging
                            "tool_schema": method_to_schema(
                                fn,
                                include_class_name=include_class_in_dynamic_tool_names,
                            ),
                            "llm_arguments": allowed_call_args,
                            "raw_arguments_json": call["function"]["arguments"],
                        }
                    else:
                        # Use shared helper for base tools
                        await _schedule_base_tool_call(
                            msg,
                            name=name,
                            args_json=call["function"]["arguments"],
                            call_id=call["id"],
                            call_idx=idx,
                            call_counts=call_counts,
                            norm_tools=tools_data.norm_tools,
                            task_info=tools_data.info,
                            pending=tools_data.pending,
                            parent_chat_context=parent_chat_context,
                            propagate_chat_context=propagate_chat_context,
                            assistant_meta=assistant_meta,
                            clarification_channels=clarification_channels,
                            client=client,
                        )

                # metadata for orderly insertion
                assistant_meta[id(msg)] = {
                    "results_count": 0,
                }

                # Immediately insert placeholder tool replies for every newly scheduled call
                #  to satisfy API ordering even if a user interjection arrives instantly.
                try:
                    await _ensure_placeholders_for_pending(
                        assistant_msg=msg,
                        content="Pending… tool call accepted. Working on it.",
                        task_info=tools_data.info,
                        pending=tools_data.pending,
                        assistant_meta=assistant_meta,
                        client=client,
                        msg_dispatcher=_msg_dispatcher,
                    )
                except Exception as _ph_exc:
                    logger.error(
                        f"Failed to insert immediate placeholders: {_ph_exc!r}",
                    )

                continue  # finished scheduling tools, back to the very top

            # ── F.  No new tool calls  ──────────────────────────────────────
            # NOTE: Two scenarios reach this block:
            #   • `pending` **non-empty** → older tool tasks are still in
            #     flight; loop back to wait for them.
            #   • `pending` empty        → the model just produced a plain
            #     assistant message; nothing more to do – return it.
            if tools_data.pending:  # still running – stop them proactively, then finish
                try:
                    for t in list(tools_data.pending):
                        info_t = tools_data.info.get(t, {})
                        nested_handle = info_t.get("handle")
                        try:
                            if nested_handle is not None and hasattr(
                                nested_handle,
                                "stop",
                            ):
                                await _maybe_await(nested_handle.stop())
                        except Exception:
                            pass
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(*tools_data.pending, return_exceptions=True)
                except Exception:
                    pass
                finally:
                    tools_data.pending.clear()

            # ── timeout guard (final turn) ──────────────────────────────────
            if timer.has_exceeded_time():
                return await _handle_limit_reached(
                    f"timeout ({timeout}s) exceeded",
                )

            if timer.has_exceeded_msgs():
                return await _handle_limit_reached(
                    f"max_steps ({max_steps}) exceeded",
                )

            final_answer = msg["content"]

            if not persist:
                return final_answer  # DONE!

            # Persist mode: remember latest final answer and enter a lingering state
            last_final_answer = final_answer

            # Wait for either a new interjection (to extend the loop),
            # a graceful stop (to return the last answer), or a hard cancel.
            while True:
                interject_w = asyncio.create_task(
                    interject_queue.get(),
                    name="InterjectQueueGet",
                )
                cancel_waiter = asyncio.create_task(
                    cancel_event.wait(),
                    name="CancelEventWait",
                )
                graceful_stop_waiter = asyncio.create_task(
                    stop_event.wait(),
                    name="StopEventWait",
                )

                done, _ = await asyncio.wait(
                    {interject_w, cancel_waiter, graceful_stop_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # cleanup unused helpers
                for tsk in (interject_w, cancel_waiter, graceful_stop_waiter):
                    if tsk not in done and not tsk.done():
                        tsk.cancel()
                        await asyncio.gather(tsk, return_exceptions=True)

                if interject_w in done:
                    # push back so the standard interjection drain builds system guidance
                    await interject_queue.put(interject_w.result())
                    llm_turn_required = True
                    break  # resume main loop to handle new turn

                if cancel_waiter in done:
                    raise asyncio.CancelledError

                if graceful_stop_waiter in done:
                    return last_final_answer or ""

    except asyncio.CancelledError:  # graceful shutdown
        # NOTE: Caller (or parent task) requested cancellation.  We propagate
        # the signal to *all* running tool tasks first so each can release
        # resources cleanly.  Only after every task has finished/aborted do
        # we re-raise the same `CancelledError`, preserving expected asyncio
        # semantics for upstream callers.
        try:
            _stop_forwarded_once = await _propagate_stop_once(
                tools_data.info,
                _stop_forwarded_once,
                "outer-loop cancelled",
            )
        except Exception:
            pass
        for t in tools_data.pending:
            t.cancel()
        await asyncio.gather(*tools_data.pending, return_exceptions=True)
        raise
    finally:
        try:
            TOOL_LOOP_LINEAGE.reset(_token)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Tiny handle objects exposed to callers
# ─────────────────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod


class SteerableToolHandle(ABC):
    """Abstract base class for steerable tool handles."""

    @abstractmethod
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    async def ask(self, question: str) -> "SteerableToolHandle":
        """
        Ask a question to the running tool.
        """

    @abstractmethod
    def interject(self, message: str) -> Awaitable[Optional[str]] | Optional[str]:
        """Inject an additional *user* turn into the running conversation."""

    @abstractmethod
    def stop(
        self,
        reason: Optional[str] = None,
        *,
        cancel: Optional[bool] = None,
    ) -> Awaitable[Optional[str]] | Optional[str]:
        """Politely request shutdown.

        Semantics
        ---------
        - cancel=True  → hard cancel; pending work is aborted and result() raises.
        - cancel=False → graceful stop; allow a final answer when supported.
        - cancel=None  → default behaviour (treated as cancel=True unless the implementation
          specifies otherwise).
        """

    @abstractmethod
    def pause(self) -> Awaitable[Optional[str]] | Optional[str]:
        """Temporarily freeze the outer loop (tools keep running)."""

    @abstractmethod
    def resume(self) -> Awaitable[Optional[str]] | Optional[str]:
        """Un-freeze a loop that was paused with :pyfunc:`pause`."""

    @abstractmethod
    def done(self) -> Awaitable[bool] | bool:
        """Flag for whether or not this task is done."""

    @abstractmethod
    def result(self) -> Awaitable[str] | str:
        """Wait for the assistant's *final* reply."""


class AsyncToolUseLoopHandle(SteerableToolHandle):
    """
    Returned by `start_async_tool_use_loop`.  Lets you
      • queue extra user messages while the loop runs and
      • stop the loop at any time.
    """

    def __init__(
        self,
        *,
        task: asyncio.Task,
        interject_queue: asyncio.Queue[str],
        cancel_event: asyncio.Event,
        stop_event: asyncio.Event,
        pause_event: Optional[asyncio.Event] = None,
        client: "unify.AsyncUnify | None" = None,
        loop_id: str = "",
        initial_user_message: Optional[str] = None,
        persist: bool = False,
    ):
        self._task = task
        self._queue = interject_queue
        self._cancel_event = cancel_event
        self._stop_event = stop_event
        # "running" ⇢ Event **set**,  "paused" ⇢ Event **cleared**
        self._pause_event = pause_event or asyncio.Event()
        self._client = client
        # Optional live delegate – set via ``_adopt`` when this handle should
        # forward every steering call to another *SteerableToolHandle*.
        self._delegate: Optional["SteerableToolHandle"] = None
        self._pause_event.set()
        self._loop_id: str = loop_id
        self._persist: bool = persist

        # Buffer interjections that may arrive **before** a downstream handle
        # (e.g. an `ActiveTask`) has been adopted.  Once a delegate is ready we
        # forward all queued messages so that no early user guidance is lost.
        self._early_interjects: list[str] = []

        # Maintain a user-visible history (what the end-user would see):
        # Records: original prompt (user), interjections (user), ask Q/A (user/assistant).
        self._user_visible_history: list[dict] = []
        if initial_user_message:
            self._user_visible_history.append(
                {"role": "user", "content": initial_user_message},
            )

    async def ask(
        self,
        question: str,
        *,
        _return_reasoning_steps: bool = False,
    ) -> "SteerableToolHandle":
        """
        Answers *question* about this *pending* tool, associated with this handle.
        The question is read-only (the tool state is not modified whatsoever).
        The calling parent loop is left completely untouched.
        """
        LOGGER.info(f"🕹️ [{self._loop_id}] Ask requested: {question}")
        # Fast-path: delegated handles answer directly.
        if self._delegate is not None:
            return await self._delegate.ask(
                question,
                _return_reasoning_steps=_return_reasoning_steps,
            )

        # Record the user-visible question immediately (even if delegated)
        try:
            self._user_visible_history.append({"role": "user", "content": question})
        except Exception:
            pass

        # 0.  Defensive guard: if the outer loop has already finished we can
        #     just answer from the final transcript without starting another
        #     loop.
        if self.done():
            LOGGER.warning(
                "AsyncToolUseLoopHandle.ask() called on an already-finished "
                "loop – returning a synthetic handle with a static answer.",
            )

            async def _static() -> str:  # type: ignore[return-type]
                return (
                    "Parent loop is already complete; no additional "
                    "information available."
                )

            class _StaticHandle(SteerableToolHandle):
                async def interject(self, message: str): ...

                def stop(self, reason: Optional[str] = None): ...

                def pause(self): ...

                def resume(self): ...

                def done(self):
                    return True

                async def result(self):
                    return await _static()

                async def ask(self, question: str) -> "SteerableToolHandle":
                    return self

            return _StaticHandle()  # pragma: no cover

        # 1.  Gather a *read-only* snapshot of the parent chat.
        parent_ctx = list(self._client.messages) if self._client else []

        # 2.  Prepare an *in-memory* Unify client for the **inspection** loop
        #     (LLM sees only the system header + follow-up user question).
        inspection_client = unify.AsyncUnify(
            "gpt-5@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
            reasoning_effort="high",
            service_tier="priority",
        )
        inspection_client.set_system_message(
            "You are inspecting a running tool-use conversation. The entire "
            "transcript so far is attached below (read-only):\n"
            f"{json.dumps(parent_ctx, indent=2)}\n\n"
            "Answer the user's follow-up question using ONLY this context.\n"
            "Do not attempt to run new tools unless they are exposed to you.\n"
            "Do not ask the user questions or request clarification. If information is missing,\n"
            "state what is known and, if helpful, briefly note assumptions. Respond in a single, concise paragraph.",
        )

        # 3. Recursive visibility ––––––––––––––––––––––––––––––––––––––––
        # Any *currently pending* SteerableToolHandle (deep-nested) should
        # be made available as a tool so the inspection loop can itself ask
        # follow-up questions.  We approximate this by scanning the parent
        # task_info dict that the outer loop stored on our asyncio.Task in
        # its "._task_info" attribute (injected by the inner loop runner).
        #
        # The attribute is deliberately *weakly* referenced to avoid tight
        # coupling; if it is absent we just skip recursion.
        #
        # NOTE: this is best-effort – individual callers can override ask()
        # for richer behaviour if desired.
        try:
            task_info = getattr(self._task, "task_info", {})
        except Exception:
            task_info = {}

        recursive_tools: dict[str, Callable] = {}

        for _t, _inf in task_info.items():
            h = _inf.get("handle")
            if h is None or not isinstance(h, SteerableToolHandle):
                continue

            async def _proxy(
                _q: str,
                _h=h,  # capture now
            ) -> str:
                nested = await _h.ask(_q)
                return await nested.result()

            # tool name encodes the call-id so collisions are impossible
            _proxy.__name__ = f"ask_{_inf['call_id']}"
            recursive_tools[_proxy.__name__] = _proxy
        # ----------------------------------------------------------------

        # 4.  Fire off a *stand-alone* read-only loop.
        # Compose a clear loop identifier so logs show exactly which loop the
        # question refers to, e.g. "Question(TaskScheduler.execute_task)" or
        # "Question(TaskScheduler.execute_task->TaskScheduler.ask)" when a single
        # nested handle is present.
        try:
            parent_label: str = getattr(self, "_loop_id", "unknown") or "unknown"
        except Exception:
            parent_label = "unknown"

        # Best-effort detection of a single nested handle to enrich the label
        child_label: str | None = None
        try:
            _ti = getattr(self._task, "task_info", {})  # type: ignore[attr-defined]
            nested_ids: set[str] = set()
            for _t, _inf in _ti.items() if isinstance(_ti, dict) else []:
                _h = _inf.get("handle")
                _lid = getattr(_h, "_loop_id", None)
                if isinstance(_lid, str) and _lid:
                    nested_ids.add(_lid)
            if len(nested_ids) == 1:
                child_label = next(iter(nested_ids))
        except Exception:
            child_label = None

        if child_label:
            loop_id_label = f"Question({parent_label}->{child_label})"
        else:
            loop_id_label = f"Question({parent_label})"

        helper_handle = start_async_tool_use_loop(
            inspection_client,
            question,
            recursive_tools,  # may be empty
            loop_id=loop_id_label,
            parent_lineage=[],  # keep label concise (do not prepend outer lineage)
            parent_chat_context=parent_ctx,  # ← nested context
            propagate_chat_context=False,
            prune_tool_duplicates=False,
            interrupt_llm_with_interjections=False,
            max_consecutive_failures=1,
            timeout=60,
        )

        # Monkey-patch result() to record the assistant answer when available
        if not _return_reasoning_steps:
            _orig_result = helper_handle.result

            async def _rec_result():  # type: ignore[return-type]
                ans = await _orig_result()
                try:
                    self._user_visible_history.append(
                        {"role": "assistant", "content": ans},
                    )
                except Exception:
                    pass
                return ans

            helper_handle.result = _rec_result  # type: ignore[attr-defined]
            return helper_handle

        async def _wrap():
            answer = await helper_handle.result()
            try:
                self._user_visible_history.append(
                    {"role": "assistant", "content": answer},
                )
            except Exception:
                pass
            return answer, inspection_client.messages

        helper_handle.result = _wrap  # type: ignore[attr-defined]
        return helper_handle

    # -- public API -----------------------------------------------------------
    @functools.wraps(SteerableToolHandle.interject, updated=())
    async def interject(self, message: str) -> None:
        LOGGER.info(f"️ [{self._loop_id}] Interject requested: {message}")
        if self._delegate is not None:
            await self._delegate.interject(message)
            return
        # Buffer then forward to resolver loop.
        self._early_interjects.append(message)
        await self._queue.put(message)

    @functools.wraps(SteerableToolHandle.stop, updated=())
    def stop(
        self,
        reason: Optional[str] = None,
        *,
        cancel: Optional[bool] = None,
    ) -> None:
        LOGGER.info(
            f"🛑 [{self._loop_id}] Stop requested"
            + (f" – reason: {reason}" if reason else ""),
        )
        # Normalise flag: default to explicit cancel unless caller asks for graceful stop
        cancel_flag: bool = True if cancel is None else bool(cancel)

        if self._delegate is not None:
            try:
                # Prefer delegate's modern signature if available
                self._delegate.stop(cancel=cancel_flag, reason=reason)  # type: ignore[misc]
            except TypeError:
                # Delegate may use legacy signature; best-effort forwarding
                self._delegate.stop(reason)  # type: ignore[misc]
            return

        # Determine behaviour for this outer handle
        # cancel=True  → hard cancel (result() raises CancelledError)
        # cancel=False → graceful stop; in persist-mode return last answer, else cancel
        if cancel_flag:
            self._cancel_event.set()
            return

        # Graceful stop: honour persist-mode where the inner loop returns a final answer
        if self._persist:
            self._stop_event.set()
        else:
            # Non-persist legacy mode has no graceful stop – fall back to cancel
            self._cancel_event.set()

    @functools.wraps(SteerableToolHandle.pause, updated=())
    def pause(self) -> None:
        LOGGER.info(f"⏸️ [{self._loop_id}] Pause requested")
        if self._delegate is not None:
            self._delegate.pause()
            return
        self._pause_event.clear()
        # Propagate pause to any nested steerable handles that expose `.pause`
        try:
            task_info = getattr(self._task, "task_info", {})
        except Exception:
            task_info = {}
        try:
            items = task_info.items() if isinstance(task_info, dict) else []
            for _t, _inf in items:
                try:
                    h = _inf.get("handle")
                except Exception:
                    h = None
                if h is not None and hasattr(h, "pause"):
                    try:
                        maybe = h.pause()  # may be sync or async
                        if asyncio.iscoroutine(maybe):
                            asyncio.create_task(maybe)
                    except Exception:
                        # Best-effort propagation – never break outer pause
                        pass
        except Exception:
            # Defensive: do not let propagation errors bubble up
            pass

    @functools.wraps(SteerableToolHandle.resume, updated=())
    def resume(self) -> None:
        LOGGER.info(f"▶️ [{self._loop_id}] Resume requested")
        if self._delegate is not None:
            self._delegate.resume()
            return
        self._pause_event.set()
        # Propagate resume to any nested steerable handles that expose `.resume`
        try:
            task_info = getattr(self._task, "task_info", {})
        except Exception:
            task_info = {}
        try:
            items = task_info.items() if isinstance(task_info, dict) else []
            for _t, _inf in items:
                try:
                    h = _inf.get("handle")
                except Exception:
                    h = None
                if h is not None and hasattr(h, "resume"):
                    try:
                        maybe = h.resume()  # may be sync or async
                        if asyncio.iscoroutine(maybe):
                            asyncio.create_task(maybe)
                    except Exception:
                        # Best-effort propagation – never break outer resume
                        pass
        except Exception:
            # Defensive: do not let propagation errors bubble up
            pass

    @functools.wraps(SteerableToolHandle.done, updated=())
    def done(self) -> bool:
        if self._delegate is not None:
            return self._delegate.done()
        return self._task.done()

    @functools.wraps(SteerableToolHandle.result, updated=())
    async def result(self) -> str:
        """Return the final answer once the conversation loop (or delegate) completes."""
        if self._delegate is not None:
            return await self._delegate.result()
        return await self._task

    # ── internal helper ──────────────────────────────────────────────────────
    def _adopt(self, new_handle: "SteerableToolHandle") -> None:
        """Switch all steering methods to *new_handle* (in-process only).

        Move any *already queued* interjections over to the freshly adopted
        delegate so that early user guidance (issued *before* the delegate was
        ready) is not lost – a common source of hangs during tests that fire
        `interject()` immediately after `execute_task()` returns.
        """
        # Flush queued interjections collected before the delegate became
        # available.  We dispatch them *asynchronously* so that we keep the
        # adopt operation non-blocking and avoid re-entrancy problems if the
        # delegate itself relies on the outer event-loop.
        import asyncio  # local import to dodge unconditional dependency at top-level

        while not self._queue.empty():
            try:
                msg = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            # Forward the message to the delegate.  We purposefully schedule the
            # coroutine in the background – it is semantically equivalent to the
            # original `interject()` call which also runs fire-and-forget.
            try:
                maybe_coro = new_handle.interject(msg)  # type: ignore[attr-defined]
                if asyncio.iscoroutine(maybe_coro):
                    asyncio.create_task(maybe_coro)
            except Exception:
                # Silently swallow to preserve backwards-compat – early
                # interjections are *best-effort* hints rather than critical.
                pass

        # Keep pause / cancel signals in sync – they might have been toggled
        # before we adopted the delegate.
        try:
            if not self._pause_event.is_set() and hasattr(new_handle, "pause"):
                new_handle.pause()  # type: ignore[attr-defined]
            if self._cancel_event.is_set() and hasattr(new_handle, "stop"):
                new_handle.stop()  # type: ignore[attr-defined]
        except Exception:
            # These are advisory only – failing to propagate them should never
            # break the overall execution.
            pass

        self._delegate = new_handle

        # ── Flush any interjections that were consumed by the resolver loop ──
        #     before the delegate became available.
        if self._early_interjects:
            import asyncio as _aio

            for _msg in self._early_interjects:
                try:
                    maybe_coro = new_handle.interject(_msg)  # type: ignore[attr-defined]
                    if _aio.iscoroutine(maybe_coro):
                        _aio.create_task(maybe_coro)
                except Exception:
                    # Advisory only – failure to replay should not break the flow.
                    pass

            self._early_interjects.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  A convenience wrapper that *starts* the loop and returns the handle
# ─────────────────────────────────────────────────────────────────────────────
def start_async_tool_use_loop(
    client: unify.AsyncUnify,
    message: str,
    tools: Dict[str, Callable],
    *,
    loop_id: Optional[str] = None,
    parent_lineage: Optional[list[str]] = None,
    max_consecutive_failures: int = 3,
    prune_tool_duplicates=True,
    interrupt_llm_with_interjections: bool = True,
    propagate_chat_context: bool = True,
    parent_chat_context: Optional[list[dict]] = None,
    log_steps: Union[bool, str] = True,
    max_steps: Optional[int] = 100,
    timeout: Optional[int] = 300,
    raise_on_limit: bool = False,
    include_class_in_dynamic_tool_names: bool = False,
    tool_policy: Optional[
        Callable[[int, Dict[str, Callable]], Tuple[str, Dict[str, Callable]]]
    ] = None,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
    response_format: Optional[Any] = None,
    max_parallel_tool_calls: Optional[int] = None,
    persist: bool = False,
) -> AsyncToolUseLoopHandle:
    """
    Kick off `_async_tool_use_loop_inner` in its own task and give the caller
    a handle for live interaction.

    Parameters
    ----------
    log_steps : bool | str, default True
        Controls verbosity of step logging to `LOGGER`:
          - False: no logging
          - True: log everything except system messages
          - "full": log everything including system messages
    """
    # Ensure a stable loop_id for consistent logging across handle and inner loop
    loop_id = loop_id if loop_id is not None else short_id()
    interject_queue: asyncio.Queue[str] = asyncio.Queue()
    cancel_event = asyncio.Event()
    stop_event = asyncio.Event()
    pause_event = asyncio.Event()
    pause_event.set()  # start un-paused

    # --- enable handle passthrough -----------------------------------------
    # A single-element list is a mutable container that the inner loop can use
    # to call ``_adopt`` on the *real* outer handle once it exists.
    outer_handle_container: list = [None]

    # Determine lineage for this loop start (inherit from context when not provided)
    _parent = (
        parent_lineage if parent_lineage is not None else TOOL_LOOP_LINEAGE.get([])
    )
    _lineage = [*_parent, loop_id]

    task = asyncio.create_task(
        _async_tool_use_loop_inner(
            client,
            message,
            tools,
            loop_id=loop_id,
            lineage=_lineage,
            interject_queue=interject_queue,
            cancel_event=cancel_event,
            stop_event=stop_event,
            pause_event=pause_event,
            max_consecutive_failures=max_consecutive_failures,
            prune_tool_duplicates=prune_tool_duplicates,
            interrupt_llm_with_interjections=interrupt_llm_with_interjections,
            propagate_chat_context=propagate_chat_context,
            parent_chat_context=parent_chat_context,
            log_steps=log_steps,
            max_steps=max_steps,
            timeout=timeout,
            raise_on_limit=raise_on_limit,
            include_class_in_dynamic_tool_names=include_class_in_dynamic_tool_names,
            tool_policy=tool_policy,
            preprocess_msgs=preprocess_msgs,
            outer_handle_container=outer_handle_container,
            response_format=response_format,
            max_parallel_tool_calls=max_parallel_tool_calls,
            persist=persist,
        ),
        name="ToolUseLoop",
    )

    handle = AsyncToolUseLoopHandle(
        task=task,
        interject_queue=interject_queue,
        cancel_event=cancel_event,
        stop_event=stop_event,
        pause_event=pause_event,
        client=client,
        loop_id=loop_id,
        initial_user_message=(
            message["content"] if isinstance(message, dict) else message
        ),
        persist=persist,
    )

    # Attach lineage to handle for optional external inspection
    try:
        handle._lineage = list(_lineage)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Let the inner coroutine discover the outer handle so it can switch
    # steering when a nested handle requests pass-through behaviour.
    outer_handle_container[0] = handle

    return handle


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers used across managers
# ─────────────────────────────────────────────────────────────────────────────


def inject_broader_context(msgs: list[dict]) -> list[dict]:
    """Replace {broader_context} placeholders inside system messages.

    Mirrors the per-manager helpers but centralised so both managers share
    identical behaviour and error handling.
    """
    import copy

    try:
        from unity.memory_manager.memory_manager import (
            MemoryManager,
        )  # local import to avoid cycles
    except Exception:  # pragma: no cover - defensive import guard
        MemoryManager = None  # type: ignore[assignment]

    patched = copy.deepcopy(msgs)

    try:
        broader_ctx = MemoryManager.get_rolling_activity() if MemoryManager else ""
    except Exception:
        broader_ctx = ""

    for m in patched:
        content = m.get("content") or ""
        if m.get("role") == "system" and "{broader_context}" in content:
            m["content"] = content.replace("{broader_context}", broader_ctx)

    return patched


def make_request_clarification_tool(
    up_q: "asyncio.Queue[str]" | None,
    down_q: "asyncio.Queue[str]" | None,
    *,
    on_request: Optional[Callable[[str], Awaitable[None] | None]] = None,
    on_answer: Optional[Callable[[str], Awaitable[None] | None]] = None,
):
    """Return an async tool that bubbles a question up and awaits the answer.

    Behaviour and integration notes
    --------------------------------
    - This tool is only available when clarification queues are provided by the
      outer tool loop. If those queues are not present in a given loop, that
      loop MUST NOT ask the user questions in its final response. Instead, it
      should proceed with sensible defaults or best guesses, and briefly state
      those assumptions. If an inner tool (invoked by this outer loop) asks for
      clarification but the outer loop has no clarification tool, the outer loop
      must explicitly inform the inner tool that no clarification channel is
      available and either (a) instruct the inner tool to use safe defaults, or
      (b) pass down concrete, sensible best‑guess values.

    - Raises RuntimeError if queues are missing at call time.
    - Optionally invokes async/sync callbacks on request/answer events.
    """

    async def _request(question: str) -> str:
        if up_q is None or down_q is None:
            raise RuntimeError(
                "Clarification queues not supplied – cannot request clarification in this context.",
            )
        # Emit request event if provided
        if on_request is not None:
            maybe = on_request(question)
            if asyncio.iscoroutine(maybe):
                await maybe

        await up_q.put(question)
        answer = await down_q.get()

        # Emit answer event if provided
        if on_answer is not None:
            maybe = on_answer(answer)
            if asyncio.iscoroutine(maybe):
                await maybe

        return answer

    return _request
