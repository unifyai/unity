from __future__ import annotations
import json
import asyncio
import inspect
import random
import string
from enum import Enum
from pydantic import BaseModel
from typing import (
    Tuple,
    List,
    Dict,
    Union,
    Optional,
    Any,
    get_type_hints,
    get_origin,
    get_args,
    Callable,
    Awaitable,
)


from .tool_spec import ToolSpec, normalise_tools  # Backward-compatibility


def short_id(length=4):
    alphabet = string.ascii_lowercase + string.digits  # base36
    return "".join(random.choice(alphabet) for _ in range(length))


TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}

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


def _canonical_tool_owner_name(cls: type) -> str:
    """Return a normalised class name for tool exposure.

    Policy:
    - Walk the MRO; if any ancestor's name starts with "Base", strip "Base"
      and use the remainder (e.g., BaseActor → Actor).
    - If any ancestor's name starts with "Simulated", strip "Simulated" and use the remainder
      (e.g., SimulatedConductor → Conductor).
    - Fallback to the class' own __name__ unchanged.
    """
    try:
        for c in getattr(cls, "__mro__", ()):
            if c is object:
                continue
            name = getattr(c, "__name__", "")
            if name.startswith("Base") and len(name) > 4:
                return name[4:]
            elif name.startswith("Simulated") and len(name) > 9:
                return name[9:]
    except Exception:
        pass

    try:
        return getattr(cls, "__name__", "")
    except Exception:
        return ""


def canonicalize_handle_class_name(cls: type) -> str:
    """Return a canonicalized class name for handle display.

    Rules (prefix stripping applied in this order):
    - Strip leading "Simulated" → e.g., SimulatedFoo → Foo
    - Strip leading version prefix "V<digits>" → e.g., V3Foo → Foo
    - Strip leading "Base" → e.g., BaseFoo → Foo
    """
    try:
        name = getattr(cls, "__name__", "") or ""
    except Exception:
        name = ""

    s = str(name)
    try:
        if s.startswith("Simulated") and len(s) > 9:
            s = s[9:]
    except Exception:
        pass
    # Strip version prefix like V3, V12, etc.
    try:
        import re as _re  # local import to avoid polluting module scope

        s = _re.sub(r"^V\d+", "", s)
    except Exception:
        pass
    try:
        if s.startswith("Base") and len(s) > 4:
            s = s[4:]
    except Exception:
        pass
    return s


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
            cls_name = _canonical_tool_owner_name(fn.__self__.__class__)
            key = f"{cls_name}_{fn.__name__}".replace("__", "_")
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


def _dumps(
    obj: Any,
    idx: List[Union[str, int]] = None,
    indent: int = None,
    context: dict | None = None,
) -> Any:
    # prevents circular import
    from unify import Log

    base = False
    if idx is None:
        base = True
        idx = list()
    if isinstance(obj, BaseModel):
        # Thread optional serialization context (ignored by models that don't use it)
        ret = obj.model_dump(mode="json", context=context)
    elif inspect.isclass(obj) and issubclass(obj, BaseModel):
        ret = obj.model_json_schema()
    elif isinstance(obj, Log):
        ret = obj.to_json()
    elif isinstance(obj, dict):
        ret = {k: _dumps(v, idx + ["k"], context=context) for k, v in obj.items()}
    elif isinstance(obj, list):
        ret = [_dumps(v, idx + [i], context=context) for i, v in enumerate(obj)]
    elif isinstance(obj, set):
        # Convert sets to a sorted list for deterministic, JSON-serialisable output
        try:
            ret = sorted(
                _dumps(v, idx + [i], context=context) for i, v in enumerate(sorted(obj))
            )
        except Exception:
            # Fallback: best-effort conversion preserving insertion order where possible
            ret = [
                _dumps(v, idx + [i], context=context) for i, v in enumerate(list(obj))
            ]
    elif isinstance(obj, tuple):
        ret = tuple(_dumps(v, idx + [i], context=context) for i, v in enumerate(obj))
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
    """Remove parameter blocks for any names in `hidden` from a docstring."""
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


def _resolve_doc_with_mro_fallback(bound_method) -> str:
    """
    Return a docstring for `bound_method`, falling back to the first ancestor
    in the MRO that defines a docstring for a method with the same name.
    """
    import inspect as _inspect  # local import to avoid polluting global scope

    # 1) Prefer the method's own docstring (after unwrap)
    try:
        unwrapped = _inspect.unwrap(bound_method)
    except Exception:
        unwrapped = bound_method
    try:
        doc = _inspect.getdoc(unwrapped)
        if isinstance(doc, str) and doc.strip():
            return doc.strip()
    except Exception:
        pass

    # 2) MRO fallback for bound methods: find an ancestor method with a docstring
    try:
        name = getattr(unwrapped, "__name__", None) or getattr(
            bound_method,
            "__name__",
            "",
        )
        owner = getattr(getattr(bound_method, "__self__", None), "__class__", None)
        if not name or owner is None:
            return ""
        for base in getattr(owner, "__mro__", ())[1:]:
            try:
                cand = getattr(base, name, None)
            except Exception:
                cand = None
            if cand is None:
                continue
            fn_obj = getattr(cand, "__func__", cand)
            base_doc = _inspect.getdoc(fn_obj)
            if isinstance(base_doc, str) and base_doc.strip():
                return base_doc.strip()
    except Exception:
        pass

    # 3) No doc found
    return ""


def annotation_to_schema(ann: Any) -> Dict[str, Any]:
    """Convert a Python annotation into a JSON Schema fragment (supports Pydantic)."""

    # Unwrap typing.Annotated
    origin = get_origin(ann)
    if origin is not None and getattr(origin, "__name__", "") == "Annotated":
        ann = get_args(ann)[0]
        origin = get_origin(ann)

    # NoneType
    if ann is type(None):
        return {"type": "null"}

    # Primitive scalars
    if ann in TYPE_MAP:
        return {"type": TYPE_MAP[ann]}

    # Builtin container types without typing metadata
    # Treat builtin dict/list/set/tuple as generic JSON containers rather than strings
    # so tools like `images: dict | None` surface to the LLM with the correct types.
    if ann is dict:
        # Unknown key/value types → allow arbitrary properties
        return {"type": "object", "additionalProperties": True}
    if ann is list:
        # Unknown item type → generic array
        return {"type": "array"}
    if ann is set:
        # Sets map naturally to arrays with unique items
        return {"type": "array", "uniqueItems": True}
    if ann is tuple:
        # Tuples as heterogeneous arrays; without metadata fall back to generic array
        return {"type": "array"}

    # Enum
    if isinstance(ann, type) and issubclass(ann, Enum):
        return {"type": "string", "enum": [member.value for member in ann]}

    # Pydantic model
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return ann.model_json_schema()

    # Dict[K, V]
    if origin is dict or origin is Dict:
        args = get_args(ann)
        if len(args) < 2:
            return {"type": "object", "additionalProperties": True}
        _, value_type = args
        return {
            "type": "object",
            "additionalProperties": annotation_to_schema(value_type),
        }

    # List[T]
    if origin in (list, List):
        (item_type,) = get_args(ann)
        return {"type": "array", "items": annotation_to_schema(item_type)}

    # Union / Optional
    try:
        import types as _types  # local import

        is_union = origin is Union or origin is _types.UnionType
    except Exception:
        is_union = origin is Union

    if is_union:
        sub_schemas = [annotation_to_schema(a) for a in get_args(ann)]
        if len(sub_schemas) == 2 and {"type": "null"} in sub_schemas:
            return next(s for s in sub_schemas if s != {"type": "null"})
        return {"anyOf": sub_schemas}

    # Python's base `object` class: allow any JSON value (empty schema = no constraints)
    if ann is object:
        return {}

    # Fallback
    return {"type": "string"}


def method_to_schema(
    bound_method,
    tool_name: Optional[str] = None,
    include_class_name: bool = True,
):
    """Convert a bound method into an OpenAI-compatible function-tool schema."""

    sig = inspect.signature(bound_method)
    # Be robust to unresolved forward references or missing symbols in
    # annotations. If type-hint evaluation fails, fall back to an empty
    # mapping and infer JSON schema types from defaults.
    try:
        hints = get_type_hints(bound_method)
    except Exception:
        hints = {}

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
            "_parent_chat_context",
            "_clarification_up_q",
            "_clarification_down_q",
            "_notification_up_q",
            "_pause_event",
            "_interject_queue",
        )

        if is_hidden:
            hidden.add(name)
            continue  # do NOT surface to the model

        ann = hints.get(name, str)
        props[name] = annotation_to_schema(ann)
        if param.default is inspect._empty:
            required.append(name)

    # ── resolve docstring with MRO fallback, then scrub hidden args ───────────
    raw_doc = _resolve_doc_with_mro_fallback(bound_method) or ""
    cleaned_doc = _strip_hidden_params_from_doc(raw_doc, hidden)

    if hasattr(bound_method, "__self__") and hasattr(
        bound_method.__self__,
        "__class__",
    ):
        _cls_name = _canonical_tool_owner_name(bound_method.__self__.__class__)
        prefix = f"{_cls_name}_"
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
