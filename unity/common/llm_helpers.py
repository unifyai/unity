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


def short_id(length=4):
    alphabet = string.ascii_lowercase + string.digits  # base36
    return "".join(secrets.choice(alphabet) for _ in range(length))


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
    if isinstance(obj, dict):
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


async def _async_tool_use_loop_inner(
    client: unify.AsyncUnify,
    message: str,
    tools: Dict[str, Union[Callable, ToolSpec]],
    *,
    loop_id: Optional[str] = None,
    interject_queue: asyncio.Queue[str],
    cancel_event: asyncio.Event,
    pause_event: asyncio.Event,
    max_consecutive_failures: int = 3,
    prune_tool_duplicates: bool = True,
    interrupt_llm_with_interjections: bool = True,
    propagate_chat_context: bool = True,
    parent_chat_context: Optional[list[dict]] = None,
    log_steps: bool = True,
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

    log_steps : ``bool``, default ``False``
        When enabled, every significant action (LLM call, tool launch,
        interjection, etc.) is logged to ``LOGGER`` for easier tracing and
        debugging.

    Returns
    -------
    str
        The assistant's final plain-text reply *after* every tool result has
        been fed back into the conversation.
    """
    # unique id
    loop_id = loop_id if loop_id is not None else short_id()

    # If structured output is expected, inform the model up-front so it can
    # plan its reasoning with the final JSON shape in mind.  Enforcement via
    # `set_response_format` still happens at the end of the loop.
    if response_format is not None:
        try:
            from pydantic import BaseModel  # local import

            # Require a Pydantic model class – anything else is a configuration error.
            if not (
                isinstance(response_format, type)
                and issubclass(response_format, BaseModel)
            ):
                raise TypeError(
                    "response_format must be a Pydantic BaseModel subclass (e.g. MySchema).",
                )

            _schema = response_format.model_json_schema()
            _hint = (
                "\n\nNOTE: After completing all tool calls, your **final** assistant reply must be valid JSON that conforms to the following schema. Do NOT include any extra keys or commentary.\n"
                + json.dumps(_schema, indent=2)
            )

            client.set_system_message((client.system_message or "") + _hint)
        except Exception as _exc:  # noqa: BLE001
            LOGGER.error(f"response_format hint failed: {_exc!r}")

    # ── runtime guards ────────────────────────────────────────────────────
    # rolling timeout ----------------------------------------------------
    last_activity_ts: float = time.perf_counter()  # reset every time
    last_msg_count: int = (
        0 if not client.messages else len(client.messages)
    )  # we add a message

    def _reset_timeout_timer() -> None:
        """Refresh the rolling timeout."""
        nonlocal last_activity_ts, last_msg_count
        last_activity_ts = time.perf_counter()
        last_msg_count = 0 if not client.messages else len(client.messages)

    async def _append_msgs(msgs: list[dict]) -> None:
        client.append_messages(msgs)
        await _to_event_bus(msgs)
        _reset_timeout_timer()

    if log_steps:
        if parent_chat_context:
            LOGGER.info(
                f"⬇️ [{loop_id}] Parent Context: {json.dumps(parent_chat_context, indent=4)}\n",
            )
        LOGGER.info(f"📋 [{loop_id}] System Message: {client.system_message}\n")
        LOGGER.info(f"🧑‍💻 [{loop_id}] User Message: {message}\n")

    # ── 0-a. Inject **system** header with broader context ───────────────────
    #
    # When a parent context is supplied we prepend a single synthetic system
    # message that *summarises* it.  This offers the LLM immediate awareness
    # of the wider conversation without having to scroll the nested JSON.
    # The special marker ``_ctx_header=True`` lets us later strip it when
    # propagating context further down (avoids duplication).
    # -----------------------------------------------------------------------

    # ── small helper: keep assistant→tool chronology DRY ────────────────────
    async def _insert_after_assistant(parent_msg: dict, tool_msg: dict) -> None:
        """
        Append *tool_msg* and move it directly after *parent_msg*, while
        updating the per-assistant `results_count` bookkeeping.
        """
        meta = assistant_meta.setdefault(
            id(parent_msg),
            {"results_count": 0},
        )
        await _append_msgs([tool_msg])
        insert_pos = client.messages.index(parent_msg) + 1 + meta["results_count"]
        client.messages.insert(insert_pos, client.messages.pop())
        meta["results_count"] += 1

    # ── small helper: publish to the EventBus (if configured) ──────────────
    async def _to_event_bus(messages: Union[Dict, List[Dict]]) -> None:
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
                        "method": loop_id,
                    },
                ),
            )

    # ── helper: call `client.generate` with optional preprocessing ──
    async def _generate_with_preprocess(**gen_kwargs):
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

    # ── small helper: add completion tool message pair ──────────────
    async def _emit_completion_pair(result: str, call_id: str) -> dict:
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

        await _append_msgs([assistant_stub, tool_msg])
        return tool_msg

    # ── *single* authoritative implementation of "task finished" handling ──
    async def _process_completed_task(task: asyncio.Task) -> bool:
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

        nonlocal consecutive_failures

        pending.discard(task)
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
                pending.add(nested_task)

                # 2️⃣ insert / update a single placeholder
                ph = info.get("tool_reply_msg")
                if ph is None:
                    ph = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": info["name"],
                        "content": (
                            "Nested async tool loop started… waiting for result."
                        ),
                    }
                    await _insert_after_assistant(info["assistant_msg"], ph)
                    info["tool_reply_msg"] = ph  # remember on *parent*
                else:
                    ph["content"] = (
                        "Nested async tool loop started… waiting for result."
                    )

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

            consecutive_failures = 0
        except Exception:
            consecutive_failures += 1
            result = traceback.format_exc()
            if log_steps:
                LOGGER.error(
                    f"❌ [{loop_id}] Error: {name} failed "
                    f"(attempt {consecutive_failures}/{max_consecutive_failures}):\n{result}",
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
                    LOGGER.error(
                        f"🧩 [{loop_id}] FAILED TOOL SCHEMA (as given to LLM):\n{json.dumps(debug_payload, indent=2)}",
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
                tool_msg = await _emit_completion_pair(result, call_id)

        elif clarify_ph is not None:
            if _at_tail(clarify_ph):
                clarify_ph["content"] = result
                tool_msg = clarify_ph
            else:
                tool_msg = await _emit_completion_pair(result, call_id)

        elif tool_reply_msg is not None:
            if _at_tail(tool_reply_msg):
                tool_reply_msg["content"] = result
                tool_msg = tool_reply_msg
            else:
                tool_msg = await _emit_completion_pair(result, call_id)

        else:
            tool_msg = {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": result,
            }
            await _insert_after_assistant(asst_msg, tool_msg)

        # ── optional console logging for every finished tool call ────────────
        #     (mirrors the assistant-message logging above)
        if log_steps:
            LOGGER.info(f"🛠️ [{loop_id}] {json.dumps(tool_msg, indent=4)}\n")

        # 6️⃣  failure guard -------------------------------------------------
        if consecutive_failures >= max_consecutive_failures:
            if log_steps:
                LOGGER.error(f"🚨 [{loop_id}] Aborting: too many tool failures.")
            raise RuntimeError(
                "Aborted after too many consecutive tool failures.",
            )

        # successful (or failed) *final* result → LLM may need to react
        return True

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
        await _append_msgs([sys_msg])

    # ── initial prompt ───────────────────────────────────────────────────────
    # ── 0-b. Coerce tools → ToolSpec & helper lambdas ───────────────────────
    #
    # • «norm_tools» holds the *canonical* mapping name → ToolSpec
    # • helper for the active-count of one tool (cheap O(#pending))
    # • helper that answers "may we launch / advertise *this* tool right now?"
    #   by comparing the live count with max_concurrent.
    # -----------------------------------------------------------------------

    norm_tools: Dict[str, ToolSpec] = _normalise_tools(tools)

    def _active_count(t_name: str) -> int:
        return sum(1 for _t, _inf in task_info.items() if _inf["name"] == t_name)

    def _can_offer_tool(t_name: str) -> bool:
        lim = norm_tools[t_name].max_concurrent
        return lim is None or _active_count(t_name) < lim

    # Helper: scan transcript for assistant messages that have tool_calls with
    # missing tool replies (before the next assistant message).
    def _find_unreplied_assistant_entries() -> list[dict]:
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

    # Ensure placeholder tool messages exist for pending tasks. If assistant_msg
    # is provided, only affects tasks spawned by that assistant turn; otherwise
    # applies to all pending tasks. Returns the list of call_ids for which a
    # placeholder was created.
    async def _ensure_placeholders_for_pending(
        assistant_msg: Optional[dict] = None,
        *,
        reason: str = "",
        content: Optional[str] = None,
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
            if (
                assistant_msg is not None
                and _inf.get("assistant_msg") is not assistant_msg
            ):
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
            await _insert_after_assistant(_inf["assistant_msg"], placeholder)
            _inf["tool_reply_msg"] = placeholder
            created.append(_inf["call_id"])

        return created

    # Helper: schedule a base tool call (shared by main path and backfill)
    async def _schedule_base_tool_call(
        asst_msg: dict,
        *,
        name: str,
        args_json: Any,
        call_id: str,
        call_idx: int,
    ) -> None:
        # Base tool must exist
        if name not in norm_tools:
            return

        fn = norm_tools[name].fn

        # Build extra kwargs (chat context, interject/clarification/pause)
        extra_kwargs: dict = {}
        if propagate_chat_context:
            cur_msgs = [m for m in client.messages if not m.get("_ctx_header")]
            ctx_repr = _chat_context_repr(parent_chat_context, cur_msgs)
            extra_kwargs["parent_chat_context"] = ctx_repr

        sig = inspect.signature(fn)
        params = sig.parameters
        has_varkw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

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
                json.loads(args_json)
                if isinstance(args_json, str)
                else (args_json or {})
            )
        except Exception:
            call_args = {}

        # Filter extras to match fn signature
        sig = inspect.signature(fn)
        params = sig.parameters
        has_varkw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
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

        if clar_up_q is not None:
            clarification_channels[call_id] = (
                clar_up_q,
                clar_down_q,
            )

        # Ensure assistant meta exists for deterministic insertion ordering
        assistant_meta.setdefault(id(asst_msg), {"results_count": 0})

    # Helper: schedule a subset of tool_calls on a past assistant message and
    # insert placeholders immediately. Skips already-scheduled/finished ids.
    async def _schedule_missing_for_message(
        asst_msg: dict,
        only_ids: set[str],
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
                if (
                    name.startswith("continue_")
                    or name.startswith("stop_")
                    or name.startswith("pause_")
                    or name.startswith("resume_")
                    or name.startswith("clarify_")
                    or name.startswith("interject_")
                ):
                    # We don't auto-trigger helpers here; they are orchestration tools.
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
                )
                scheduled.append(cid)
        except Exception:
            pass
        # Ensure placeholders are present for backfilled items
        try:
            await _ensure_placeholders_for_pending(
                assistant_msg=asst_msg,
                reason="backfill",
            )
        except Exception:
            pass
        return scheduled

    # Initialise loop state early so preflight backfill can schedule tasks
    consecutive_failures = 0
    pending: Set[asyncio.Task] = set()
    total_tool_calls_made: int = 0  # base-tool calls actually launched
    task_info: Dict[asyncio.Task, Dict[str, Any]] = {}
    clarification_channels: Dict[
        str,
        Tuple[asyncio.Queue[str], asyncio.Queue[str]],
    ] = {}
    completed_results: Dict[str, str] = {}
    assistant_meta: Dict[int, Dict[str, Any]] = {}
    step_index: int = 0  # per assistant turn

    # Preflight repair: backfill any pre-existing assistant tool_calls without replies
    try:
        unreplied = _find_unreplied_assistant_entries()
        if unreplied:
            # backfill for all such assistant messages (oldest → newest)
            for entry in unreplied:
                amsg = entry["assistant_msg"]
                missing_ids = set(entry["missing"])
                await _schedule_missing_for_message(amsg, missing_ids)
    except Exception:
        pass

    # ── initial **user** message
    if isinstance(message, dict):
        initial_user_msg = message
    else:
        initial_user_msg = {"role": "user", "content": message}

    await _append_msgs([initial_user_msg])

    # ── helper: graceful early-exit when limits are hit ────────────────────
    async def _handle_limit_reached(reason: str) -> str:
        """
        Gracefully terminate the loop when *timeout* or *max_steps* are
        exceeded and `raise_on_limit` is *False*:
          • stop every pending tool (via handle.stop() if available)
          • cancel waiter coroutines
          • append a short assistant notice
        """
        for t in list(pending):
            h = task_info.get(t, {}).get("handle")
            try:
                if h is not None and hasattr(h, "stop"):
                    await _maybe_await(h.stop())
            except Exception:
                pass
            if not t.done():
                t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        pending.clear()

        notice = {
            "role": "assistant",
            "content": f"🔚 Terminating early: {reason}",
        }
        await _append_msgs([notice])
        if log_steps:
            LOGGER.info(f"⏹️ [{loop_id}] Early exit – {reason}")
        return notice["content"]

    # Set to *True* whenever the loop must grant the LLM an immediate turn
    # before waiting again (user interjection, clarification answer, etc.).
    llm_turn_required = False

    try:
        while True:

            # ── 0-α-P. Global *pause* gate  ────────────────────────────
            # Keep handling tool completions & cancellation, but *never*
            # let the LLM speak while we're paused.
            if not pause_event.is_set():
                # Give any pending tool tasks a chance to finish OR wait until the
                # loop is resumed / cancelled.  Every coroutine is wrapped in an
                # asyncio.Task so `asyncio.wait()` is happy.
                if pending:
                    pause_waiter = asyncio.create_task(
                        pause_event.wait(),
                        name="PauseEventWait",
                    )
                    cancel_waiter = asyncio.create_task(
                        cancel_event.wait(),
                        name="CancelEventWait",
                    )
                    waiters = pending | {pause_waiter, cancel_waiter}

                    done, _ = await asyncio.wait(
                        waiters,
                        timeout=0.1,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # helper-task cleanup so they don't dangle
                    for w in (pause_waiter, cancel_waiter):
                        if w not in done and not w.done():
                            w.cancel()
                            await asyncio.gather(w, return_exceptions=True)

                    # tool finished?
                    for t in done & pending:
                        await _process_completed_task(t)
                    if cancel_event.is_set():
                        raise asyncio.CancelledError
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
                        },
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # resumed?
                    if pause_event.is_set():
                        continue  # back to main loop, un-paused

                    # cancelled?
                    if cancel_event.is_set():
                        raise asyncio.CancelledError
                        continue  # top-of-loop, still paused

            # 0-α. **Global timeout**
            if timeout is not None and time.perf_counter() - last_activity_ts > timeout:
                if raise_on_limit:
                    raise asyncio.TimeoutError(
                        f"Loop exceeded {timeout}s wall-clock limit",
                    )
                else:
                    return await _handle_limit_reached(
                        f"timeout ({timeout}s) exceeded",
                    )

            # 0-β. **Chat history length**
            if max_steps is not None and len(client.messages) >= max_steps:
                if raise_on_limit:
                    raise RuntimeError(
                        f"Conversation exceeded max_steps={max_steps} "
                        f"(len(client.messages)={len(client.messages)})",
                    )
                else:
                    return await _handle_limit_reached(
                        f"max_steps ({max_steps}) exceeded",
                    )

            # 0-γ. Repair any outstanding assistant tool_calls missing replies
            #      before we allow new user interjections to be appended.
            try:
                unreplied = _find_unreplied_assistant_entries()
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
                await _append_msgs([interjection_msg])

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
            if pending and not llm_turn_required:
                interject_w = asyncio.create_task(
                    interject_queue.get(),
                    name="InterjectQueueGet",
                )
                cancel_waiter = asyncio.create_task(
                    cancel_event.wait(),
                    name="CancelEventWait",
                )
                clar_waiters: Dict[asyncio.Task, asyncio.Task] = {}
                for _t in pending:
                    # Only listen for *new* clarification questions.
                    # If the task is already awaiting an answer,
                    # `waiting_for_clarification` will be True.
                    if task_info[_t].get("waiting_for_clarification"):
                        continue

                    cuq = task_info[_t].get("clar_up_q")
                    if cuq is not None:
                        w = asyncio.create_task(cuq.get(), name="ClarificationQueueGet")
                        clar_waiters[w] = _t
                waiters = pending | set(clar_waiters) | {cancel_waiter, interject_w}

                # ── honour global *timeout* while we wait for tools ───────────
                wait_timeout: Optional[float] = None
                if timeout is not None:
                    wait_timeout = timeout - (time.perf_counter() - last_activity_ts)
                    # already exceeded?
                    if wait_timeout <= 0:
                        if raise_on_limit:
                            raise asyncio.TimeoutError(
                                f"Loop exceeded {timeout}s wall-clock limit",
                            )
                        else:
                            return await _handle_limit_reached(
                                f"timeout ({timeout}s) exceeded",
                            )

                done, _ = await asyncio.wait(
                    waiters,
                    timeout=wait_timeout,
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
                for aux in (interject_w, cancel_waiter, *clar_waiters.keys()):
                    if aux not in done and not aux.done():
                        aux.cancel()
                        await asyncio.gather(aux, return_exceptions=True)

                if interject_w in done:
                    # re-queue so branch 0 will handle user turn immediately
                    await interject_queue.put(interject_w.result())
                    continue  # → loop, will be processed in 0.

                if cancel_waiter in done:
                    raise asyncio.CancelledError  # cancellation wins

                # ── clarification request bubbled up from a child tool ──────────────
                if done & clar_waiters.keys():
                    for cw in done & clar_waiters.keys():
                        question = cw.result()  # the text from the child
                        src_task = clar_waiters[cw]
                        call_id = task_info[src_task]["call_id"]

                        # 1️⃣ mark the task as waiting
                        task_info[src_task]["waiting_for_clarification"] = True

                        # 2️⃣ REUSE the existing placeholder if we already inserted one
                        ph = task_info[src_task].get("tool_reply_msg")
                        if ph is None:
                            # no placeholder yet → create one exactly once
                            ph = {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": f"clarification_request_{call_id}",
                                "content": "",  # will fill below
                            }
                            await _insert_after_assistant(
                                task_info[src_task]["assistant_msg"],
                                ph,
                            )
                            task_info[src_task]["tool_reply_msg"] = ph

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
                    if await _process_completed_task(task):
                        needs_turn = True

                # Other tools may still be running.
                if pending:
                    if needs_turn:  # only when something new
                        llm_turn_required = True
                    continue  # jump to top-of-loop

            # ── B: wait for remaining tools before asking the LLM again,
            # unless the model already deserves a turn
            if pending and not llm_turn_required:
                # Ensure placeholders exist for any pending calls before the next assistant turn
                await _ensure_placeholders_for_pending(
                    reason="pre_llm_wait",
                    content=(
                        "Still running… you can use any of the available helper tools "
                        "to interact with this tool call while it is in progress."
                    ),
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
                        {n: s.fn for n, s in norm_tools.items()},
                    )
                except Exception as _e:  # never abort the loop on mis-behaving policies
                    LOGGER.error(
                        f"tool_policy raised on turn {step_index}: {_e!r}",
                    )
                    tool_choice_mode, filtered = "auto", {
                        n: s.fn for n, s in norm_tools.items()
                    }
                policy_tools_norm = _normalise_tools(filtered)
            else:
                tool_choice_mode = "auto"
                policy_tools_norm = norm_tools

            def _concurrency_ok(tn: str) -> bool:
                return tn not in norm_tools or _can_offer_tool(tn)

            visible_base_tools_schema = [
                method_to_schema(spec.fn, name)
                for name, spec in policy_tools_norm.items()
                if _concurrency_ok(name)
            ]

            # Inject `final_answer` tool automatically whenever a `response_format` is
            # supplied. The tool accepts a single `answer` argument whose schema matches
            # the provided Pydantic model.
            if response_format is not None:
                try:
                    from pydantic import BaseModel  # local import

                    if not (
                        isinstance(response_format, type)
                        and issubclass(response_format, BaseModel)
                    ):
                        raise TypeError(
                            "response_format must be a Pydantic BaseModel subclass.",
                        )

                    _answer_schema = response_format.model_json_schema()

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
                    LOGGER.error(
                        f"Failed to inject final_answer tool: {_injection_exc!r}",
                    )

            # helper: register a freshly-minted coroutine as a *temporary* tool
            def _reg_tool(key: str, func_name: str, doc: str, fn: Callable) -> None:
                # prefer the function's own docstring if it exists, else fall back
                existing = inspect.getdoc(fn)
                fn.__doc__ = existing.strip() if existing else doc
                fn.__name__ = func_name[:64]
                fn.__qualname__ = func_name[:64]
                dynamic_tools[key.lstrip("_")] = fn

            for _task in list(pending):
                info = task_info[_task]
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
                _fn_name: str = info["name"]
                _arg_json: str = info["call_dict"]["function"]["arguments"]
                try:
                    _arg_dict = json.loads(_arg_json)
                    _arg_repr = ", ".join(f"{k}={v!r}" for k, v in _arg_dict.items())
                except Exception:
                    _arg_repr = _arg_json  # fallback: raw JSON string

                # concise, informative, single‑line docs  ----------------------
                _continue_doc = f"Continue waiting for {_fn_name}({_arg_repr})."
                _stop_doc = f"Stop pending call {_fn_name}({_arg_repr})."

                # ––– 1. continue helper ––––––––––––––––––––––––––––––––––––
                # Skip if the task is blocked waiting for clarification; there's
                # nothing to "continue" until the user answers.
                if not info.get("waiting_for_clarification"):

                    async def _continue() -> Dict[str, str]:
                        return {"status": "continue", "call_id": _call_id}

                    _reg_tool(
                        key=f"continue_{_fn_name}_{_call_id}",
                        func_name=f"continue_{_fn_name}_{_call_id}",
                        doc=_continue_doc,
                        fn=_continue,
                    )

                # ––– 2. stop helper –––––––––––––––––––––––––––––––––––––
                async def _stop() -> Dict[str, str]:
                    if handle is not None and hasattr(handle, "stop"):
                        await _maybe_await(handle.stop())  # graceful nested shutdown
                    if not _task.done():
                        _task.cancel()  # kill the waiter coroutine
                    pending.discard(_task)
                    task_info.pop(_task, None)
                    return {"status": "stopped", "call_id": _call_id}

                _reg_tool(
                    key=f"stop_{_fn_name}_{_call_id}",
                    func_name=f"stop_{_fn_name}_{_call_id}",
                    doc=_stop_doc,
                    fn=_stop,
                )

                # ––– 3. interject helper (optional) ––––––––––––––––––––––
                if info.get("is_interjectable"):
                    _interject_doc = (
                        f"Inject additional instructions for {_fn_name}({_arg_repr}). "
                        "Takes a single argument `content` containing plain-English guidance."
                    )

                    if handle is not None:

                        async def _interject(content: str) -> Dict[str, str]:
                            # nested async-tool loop: delegate to its public API
                            await _maybe_await(handle.interject(content))
                            return {
                                "status": "interjected",
                                "call_id": _call_id,
                                "content": content,
                            }

                    else:

                        async def _interject(content: str) -> Dict[str, str]:
                            # regular tool: push onto its private queue
                            await info["interject_q"].put(content)
                            return {
                                "status": "interjected",
                                "call_id": _call_id,
                                "content": content,
                            }

                    _reg_tool(
                        key=f"interject_{_fn_name}_{_call_id}",
                        func_name=f"interject_{_fn_name}_{_call_id}",
                        doc=_interject_doc,
                        fn=_interject,
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

                    _reg_tool(
                        key=f"clarify_{_fn_name}_{_call_id}",
                        func_name=f"clarify_{_fn_name}_{_call_id}",
                        doc=_clarify_doc,
                        fn=_clarify,
                    )

                # ––– 5. pause helper –––––––––––––––––––––––––––––––––––––––––––
                can_pause = (handle is not None and hasattr(handle, "pause")) or ev
                can_resume = (handle is not None and hasattr(handle, "resume")) or ev

                if can_pause:
                    _pause_doc = f"Pause the pending call {_fn_name}({_arg_repr})."

                    async def _pause() -> Dict[str, str]:
                        if handle is not None and hasattr(handle, "pause"):
                            await _maybe_await(handle.pause())
                        elif ev is not None:
                            ev.clear()
                        return {"status": "paused", "call_id": _call_id}

                    _reg_tool(
                        key=f"pause_{_fn_name}_{_call_id}",
                        func_name=f"pause_{_fn_name}_{_call_id}",
                        doc=_pause_doc,
                        fn=_pause,
                    )

                # ––– 6. resume helper ––––––––––––––––––––––––––––––––––––––––––
                if can_resume:
                    _resume_doc = (
                        f"Resume the previously paused call {_fn_name}({_arg_repr})."
                    )

                    async def _resume() -> Dict[str, str]:
                        if handle is not None and hasattr(handle, "resume"):
                            await _maybe_await(handle.resume())
                        elif ev is not None:
                            ev.set()
                        return {"status": "resumed", "call_id": _call_id}

                    _reg_tool(
                        key=f"resume_{_fn_name}_{_call_id}",
                        func_name=f"resume_{_fn_name}_{_call_id}",
                        doc=_resume_doc,
                        fn=_resume,
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

                    for meth_name, bound in public_methods.items():
                        # use the same name we're about to give fn.__name__
                        func_name = f"{meth_name}_{_fn_name}_{_call_id}"
                        helper_key = func_name

                        # Skip if we already generated one this turn (possible when
                        # the loop revisits the same pending task).
                        if helper_key in dynamic_tools:
                            continue

                        async def _invoke_handle_method(
                            _bound=bound,
                            **_kw,
                        ):  # default args → capture current bound method
                            """
                            Auto-generated wrapper that calls the corresponding
                            method on the live handle and **waits** for the return
                            value (sync or async).
                            """
                            # ── normalise/validate incoming arguments against the bound method ──
                            try:
                                import inspect as _inspect  # local to avoid polluting module ns

                                sig = _inspect.signature(_bound)
                                params = sig.parameters
                                has_varkw = any(
                                    p.kind == _inspect.Parameter.VAR_KEYWORD
                                    for p in params.values()
                                )

                                # 1) Expand nested {"kwargs": {...}} if present
                                if "kwargs" in _kw and isinstance(_kw["kwargs"], dict):
                                    nested_kw = _kw.pop("kwargs")
                                    for k, v in nested_kw.items():
                                        _kw.setdefault(k, v)

                                # 2) Drop common placeholder noise keys when empty (e.g. "a", "kw")
                                for _noise in ("a", "kw"):
                                    if _noise in _kw and (
                                        _kw[_noise] is None or _kw[_noise] == ""
                                    ):
                                        _kw.pop(_noise, None)

                                # 3) Map positional array → named params if provided under "args"
                                if "args" in _kw and isinstance(_kw["args"], list):
                                    pos_params = [
                                        name
                                        for name, p in params.items()
                                        if name != "self"
                                        and p.kind
                                        in (
                                            _inspect.Parameter.POSITIONAL_ONLY,
                                            _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                        )
                                    ]
                                    for idx, val in enumerate(_kw["args"]):
                                        if (
                                            idx < len(pos_params)
                                            and pos_params[idx] not in _kw
                                        ):
                                            _kw[pos_params[idx]] = val
                                    _kw.pop("args", None)

                                # 4) If the method has exactly one public parameter, accept common aliases
                                public_params = [
                                    name
                                    for name, p in params.items()
                                    if name not in ("self",)
                                ]
                                # exclude underscored keyword-only helpers from public consideration
                                public_params = [
                                    n for n in public_params if not n.startswith("_")
                                ]
                                if (
                                    len(public_params) == 1
                                    and public_params[0] not in _kw
                                ):
                                    for alias in (
                                        "question",
                                        "query",
                                        "text",
                                        "message",
                                        "prompt",
                                        "content",
                                    ):
                                        if alias in _kw:
                                            _kw[public_params[0]] = _kw.pop(alias)
                                            break

                                # 5) Unless the method accepts **kwargs, drop unknown keys
                                if not has_varkw:
                                    _kw = {k: v for k, v in _kw.items() if k in params}
                            except Exception:
                                # Best-effort normalisation – never fail the call because of sanitisation
                                pass

                            res = await _maybe_await(_bound(**_kw))
                            return {"call_id": _call_id, "result": res}

                        # override the wrapper's signature to match the real method
                        _invoke_handle_method.__signature__ = inspect.signature(bound)

                        _reg_tool(
                            key=helper_key,
                            func_name=func_name,
                            doc=(
                                f"Invoke `{meth_name}` on the running handle (id={_call_id}). "
                                "Returns when that method finishes."
                            ),
                            fn=_invoke_handle_method,
                        )

            # make sure every pending call already has a *tool* reply ──
            #  (a placeholder) before we let the assistant speak again.
            await _ensure_placeholders_for_pending(
                reason="pre_llm_merge_helpers",
                content=(
                    "Still running… you can use any of the available helper tools "
                    "to interact with this tool call while it is in progress."
                ),
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
                LOGGER.info(f"🔄 [{loop_id}] LLM thinking…")

            if interrupt_llm_with_interjections:
                # ––––– new *pre-emptive* mode ––––––––––––––––––––––––––––
                # ➊ start the LLM step …
                llm_task = asyncio.create_task(
                    _generate_with_preprocess(
                        return_full_completion=True,
                        tools=tmp_tools,
                        tool_choice=tool_choice_mode,
                        stateful=True,
                    ),
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
                pending_snapshot = set(pending)

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
                        if await _process_completed_task(task):
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
                    await _generate_with_preprocess(
                        return_full_completion=True,
                        tools=tmp_tools,
                        tool_choice=tool_choice_mode,
                        stateful=True,
                    )
                except Exception:
                    raise Exception(
                        f"LLM call failed. Messages at the time:\n{json.dumps(client.messages, indent=4)}",
                    )

            msg = client.messages[-1]
            await _to_event_bus(msg)

            if log_steps:
                try:
                    LOGGER.info(f"🤖 [{loop_id}] {json.dumps(msg, indent=4)}\n")
                except Exception:
                    LOGGER.info(
                        f"🤖 [{loop_id}] Assistant message appended (unserializable)",
                    )

            # ── timeout guard (post-LLM) ───────────────────────────────
            if timeout is not None and time.perf_counter() - last_activity_ts > timeout:
                if raise_on_limit:
                    raise asyncio.TimeoutError(
                        f"Loop exceeded {timeout}s wall-clock limit",
                    )
                else:
                    return await _handle_limit_reached(
                        f"timeout ({timeout}s) exceeded",
                    )

            # LLM has just spoken – reset the flag
            llm_turn_required = False
            # one full assistant turn completed
            step_index += 1

            # ── De-duplicate tool calls (optional) ────────────────────────
            if prune_tool_duplicates and msg.get("tool_calls"):
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

                pass  # removed: original_tool_calls collector no longer used
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
                            await _insert_after_assistant(msg, tool_msg)

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
                            await _insert_after_assistant(msg, tool_msg)
                            continue

                    # ── Special-case dynamic helpers ──────────────────────
                    # • continue_* → acknowledge, no scheduling
                    # • cancel_*   → cancel underlying task & purge metadata
                    if name.startswith("continue_"):
                        call_id = "_".join(name.split("_")[-2:])

                        tgt_task = next(
                            (
                                t
                                for t, inf in task_info.items()
                                if call_id in inf["call_id"]
                            ),
                            None,
                        )

                        orig_fn = task_info[tgt_task]["name"] if tgt_task else "unknown"
                        arg_json = (
                            task_info[tgt_task]["call_dict"]["function"]["arguments"]
                            if tgt_task
                            else "{}"
                        )
                        pretty_name = f"continue {orig_fn}({arg_json})"

                        if tgt_task:  # still running → insert generated placeholder now
                            info = task_info[tgt_task]
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
                            await _insert_after_assistant(msg, tool_reply_msg)
                            info["continue_msg"] = tool_reply_msg
                        else:  # the original tool already finished
                            finished = completed_results.get(
                                call_id,
                                _dumps(
                                    {"status": "not-found", "call_id": call_id},
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
                                info["assistant_msg"],
                                tool_msg,
                            )
                        continue  # completed handling of this _continue

                    if name.startswith("stop_") and not name.startswith(
                        "_stop_tasks",
                    ):
                        call_id = "_".join(name.split("_")[-2:])

                        # ── locate & cancel the underlying coroutine ──────
                        task_to_cancel = next(
                            (
                                t
                                for t, info in task_info.items()
                                if info["call_id"] == call_id
                            ),
                            None,
                        )

                        orig_fn = (
                            task_info[task_to_cancel]["name"]
                            if task_to_cancel
                            else "unknown"
                        )
                        arg_json = (
                            task_info[task_to_cancel]["call_dict"]["function"][
                                "arguments"
                            ]
                            if task_to_cancel
                            else "{}"
                        )
                        pretty_name = f"stop   {orig_fn}({arg_json})"

                        # ── gracefully shut down any *nested* async-tool loop first ──────
                        if task_to_cancel:
                            nested_handle = task_info[task_to_cancel].get("handle")
                            if nested_handle is not None:
                                # public API call – propagates cancellation downwards
                                await _maybe_await(nested_handle.stop())

                        # ── then cancel the waiter coroutine itself ───────────────────────────
                        if task_to_cancel and not task_to_cancel.done():
                            task_to_cancel.cancel()
                        if task_to_cancel:
                            pending.discard(task_to_cancel)
                            task_info.pop(task_to_cancel, None)

                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": pretty_name,
                            "content": (
                                f"The tool call [{call_id}] has been stopped successfully."
                            ),
                        }
                        await _insert_after_assistant(msg, tool_msg)

                        continue  # nothing else to schedule

                    # ── _pause helper ────────────────────────────────────────────────
                    if name.startswith("pause_") and not name.startswith(
                        "_pause_tasks",
                    ):
                        call_id = "_".join(name.split("_")[-2:])
                        tgt_task = next(
                            (
                                t
                                for t, info in task_info.items()
                                if info["call_id"] == call_id
                            ),
                            None,
                        )
                        orig_fn = task_info[tgt_task]["name"] if tgt_task else "unknown"
                        arg_json = (
                            task_info[tgt_task]["call_dict"]["function"]["arguments"]
                            if tgt_task
                            else "{}"
                        )
                        pretty_name = f"pause {orig_fn}({arg_json})"

                        if tgt_task:
                            h = task_info[tgt_task].get("handle")
                            ev = task_info[tgt_task].get("pause_event")
                            if h is not None and hasattr(h, "pause"):
                                await _maybe_await(h.pause())
                            elif ev is not None:
                                ev.clear()

                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": pretty_name,
                            "content": f"The tool call [{call_id}] has been paused successfully.",
                        }
                        await _insert_after_assistant(msg, tool_msg)
                        continue  # helper handled, move on

                    # ── _resume helper ───────────────────────────────────────────────
                    if name.startswith("resume_") and not name.startswith(
                        "_resume_tasks",
                    ):
                        call_id = "_".join(name.split("_")[-2:])
                        tgt_task = next(
                            (
                                t
                                for t, info in task_info.items()
                                if info["call_id"] == call_id
                            ),
                            None,
                        )
                        orig_fn = task_info[tgt_task]["name"] if tgt_task else "unknown"
                        arg_json = (
                            task_info[tgt_task]["call_dict"]["function"]["arguments"]
                            if tgt_task
                            else "{}"
                        )
                        pretty_name = f"resume {orig_fn}({arg_json})"

                        if tgt_task:
                            h = task_info[tgt_task].get("handle")
                            ev = task_info[tgt_task].get("pause_event")
                            if h is not None and hasattr(h, "resume"):
                                await _maybe_await(h.resume())
                            elif ev is not None:
                                ev.set()

                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": pretty_name,
                            "content": f"The tool call [{call_id}] has been resumed successfully.",
                        }
                        await _insert_after_assistant(msg, tool_msg)
                        continue  # helper handled

                    if name.startswith("clarify_"):
                        call_id = "_".join(name.split("_")[-2:])
                        ans = args["answer"]

                        # ── find the underlying pending task (if still alive) ───────────────
                        tgt_task = next(  # ← NEW
                            (
                                t
                                for t, inf in task_info.items()
                                if call_id in inf["call_id"]
                            ),
                            None,
                        )

                        if call_id in clarification_channels:
                            await clarification_channels[call_id][1].put(
                                ans,
                            )  # down-queue
                            # ✔️ the tool is un-blocked – start watching it again
                            for _t, _inf in task_info.items():
                                if call_id in _inf["call_id"]:
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
                        await _insert_after_assistant(msg, tool_reply_msg)
                        if tgt_task is not None:
                            task_info[tgt_task]["clarify_placeholder"] = tool_reply_msg
                        continue

                    if name.startswith("interject_"):
                        # helper signature: {"content": "..."}
                        try:
                            payload = json.loads(call["function"]["arguments"])
                            new_text = payload["content"]
                        except Exception:
                            new_text = "<unparsable>"

                        call_id = "_".join(name.split("_")[-2:])

                        # locate the underlying long-running task
                        tgt_task = next(
                            (
                                t
                                for t, inf in task_info.items()
                                if call_id in inf["call_id"]
                            ),
                            None,
                        )

                        pretty_name = (
                            f"interject {task_info[tgt_task]['name']}({new_text})"
                            if tgt_task
                            else name
                        )

                        # ― push guidance onto the private queue -------------
                        if tgt_task:
                            iq = task_info[tgt_task]["interject_q"]
                            h = task_info[tgt_task].get("handle")

                            if iq is not None:
                                await iq.put(new_text)
                            elif h is not None and hasattr(h, "interject"):
                                await _maybe_await(h.interject(new_text))

                        # ― emit a tool message so the chat log stays tidy ---
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": pretty_name,
                            "content": f'Guidance "{new_text}" forwarded to the running tool.',
                        }
                        await _insert_after_assistant(msg, tool_msg)

                        continue  # nothing else to schedule

                    # Respect *per-tool* concurrency limits  ────────────────
                    if (
                        name in norm_tools
                        and norm_tools[name].max_concurrent is not None
                        and _active_count(name) >= norm_tools[name].max_concurrent
                    ):
                        # Concurrency cap reached → immediately insert a
                        # *tool-error* message and **do not** schedule.
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": name,
                            "content": (
                                f"⚠️ Cannot start '{name}': "
                                f"max_concurrent={norm_tools[name].max_concurrent} "
                                "already reached. Wait for an existing call to "
                                "finish or stop one before retrying."
                            ),
                        }
                        await _insert_after_assistant(msg, tool_msg)
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
                        # original_tool_calls removed

                        t = asyncio.create_task(coro, name=f"ToolCall_{name}")
                        pending.add(t)
                        task_info[t] = {
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
                        # ⇢ counts only "real" (base) tool invocations
                        total_tool_calls_made += 1

                        # Use shared helper for base tools
                        await _schedule_base_tool_call(
                            msg,
                            name=name,
                            args_json=call["function"]["arguments"],
                            call_id=call["id"],
                            call_idx=idx,
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
                        reason="post_schedule_immediate",
                        content="Pending… tool call accepted. Working on it.",
                    )
                except Exception as _ph_exc:
                    LOGGER.error(
                        f"Failed to insert immediate placeholders: {_ph_exc!r}",
                    )

                continue  # finished scheduling tools, back to the very top

            # ── F.  No new tool calls  ──────────────────────────────────────
            # NOTE: Two scenarios reach this block:
            #   • `pending` **non-empty** → older tool tasks are still in
            #     flight; loop back to wait for them.
            #   • `pending` empty        → the model just produced a plain
            #     assistant message; nothing more to do – return it.
            if pending:  # still running – stop them proactively, then finish
                try:
                    for t in list(pending):
                        info_t = task_info.get(t, {})
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
                    await asyncio.gather(*pending, return_exceptions=True)
                except Exception:
                    pass
                finally:
                    pending.clear()

            # ── timeout guard (final turn) ──────────────────────────────────
            if timeout is not None and time.perf_counter() - last_activity_ts > timeout:
                if raise_on_limit:
                    raise asyncio.TimeoutError(
                        f"Loop exceeded {timeout}s wall-clock limit",
                    )
                else:
                    return await _handle_limit_reached(
                        f"timeout ({timeout}s) exceeded",
                    )

            if max_steps is not None and len(client.messages) >= max_steps:
                if raise_on_limit:
                    raise RuntimeError(
                        f"Conversation exceeded max_steps={max_steps} "
                        f"(len(client.messages)={len(client.messages)})",
                    )
                else:
                    return await _handle_limit_reached(
                        f"max_steps ({max_steps}) exceeded",
                    )

            final_answer = msg["content"]

            return final_answer  # DONE!

    except asyncio.CancelledError:  # graceful shutdown
        # NOTE: Caller (or parent task) requested cancellation.  We propagate
        # the signal to *all* running tool tasks first so each can release
        # resources cleanly.  Only after every task has finished/aborted do
        # we re-raise the same `CancelledError`, preserving expected asyncio
        # semantics for upstream callers.
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise


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
    def stop(self) -> Awaitable[Optional[str]] | Optional[str]:
        """Politely ask the loop to shut down (gracefully)."""

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
        pause_event: Optional[asyncio.Event] = None,
        client: "unify.AsyncUnify | None" = None,
        loop_id: str = "",
        initial_user_message: Optional[str] = None,
    ):
        self._task = task
        self._queue = interject_queue
        self._cancel_event = cancel_event
        # "running" ⇢ Event **set**,  "paused" ⇢ Event **cleared**
        self._pause_event = pause_event or asyncio.Event()
        self._client = client
        # Optional live delegate – set via ``_adopt`` when this handle should
        # forward every steering call to another *SteerableToolHandle*.
        self._delegate: Optional["SteerableToolHandle"] = None
        self._pause_event.set()
        self._loop_id: str = loop_id

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

                def stop(self): ...

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
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )
        inspection_client.set_system_message(
            "You are inspecting a running tool-use conversation. The entire "
            "transcript so far is attached below (read-only):\n"
            f"{json.dumps(parent_ctx, indent=2)}\n\n"
            "Answer the user's follow-up question using ONLY this context. "
            "Do not attempt to run new tools unless they are exposed to you.",
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
        helper_handle = start_async_tool_use_loop(
            inspection_client,
            question,
            recursive_tools,  # may be empty
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
    def stop(self) -> None:
        LOGGER.info(f"🛑 [{self._loop_id}] Stop requested")
        if self._delegate is not None:
            self._delegate.stop()
            return
        self._cancel_event.set()

    @functools.wraps(SteerableToolHandle.pause, updated=())
    def pause(self) -> None:
        LOGGER.info(f"⏸️ [{self._loop_id}] Pause requested")
        if self._delegate is not None:
            self._delegate.pause()
            return
        self._pause_event.clear()

    @functools.wraps(SteerableToolHandle.resume, updated=())
    def resume(self) -> None:
        LOGGER.info(f"▶️ [{self._loop_id}] Resume requested")
        if self._delegate is not None:
            self._delegate.resume()
            return
        self._pause_event.set()

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
    max_consecutive_failures: int = 3,
    prune_tool_duplicates=True,
    interrupt_llm_with_interjections: bool = True,
    propagate_chat_context: bool = True,
    parent_chat_context: Optional[list[dict]] = None,
    log_steps: bool = True,
    max_steps: Optional[int] = 100,
    timeout: Optional[int] = 300,
    raise_on_limit: bool = False,
    include_class_in_dynamic_tool_names: bool = False,
    tool_policy: Optional[
        Callable[[int, Dict[str, Callable]], Tuple[str, Dict[str, Callable]]]
    ] = None,
    preprocess_msgs: Optional[Callable[[list[dict]], list[dict]]] = None,
    response_format: Optional[Any] = None,
) -> AsyncToolUseLoopHandle:
    """
    Kick off `_async_tool_use_loop_inner` in its own task and give the caller
    a handle for live interaction.
    """
    # Ensure a stable loop_id for consistent logging across handle and inner loop
    loop_id = loop_id if loop_id is not None else short_id()
    interject_queue: asyncio.Queue[str] = asyncio.Queue()
    cancel_event = asyncio.Event()
    pause_event = asyncio.Event()
    pause_event.set()  # start un-paused

    # --- enable handle passthrough -----------------------------------------
    # A single-element list is a mutable container that the inner loop can use
    # to call ``_adopt`` on the *real* outer handle once it exists.
    outer_handle_container: list = [None]

    task = asyncio.create_task(
        _async_tool_use_loop_inner(
            client,
            message,
            tools,
            loop_id=loop_id,
            interject_queue=interject_queue,
            cancel_event=cancel_event,
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
        ),
        name="ToolUseLoop",
    )

    handle = AsyncToolUseLoopHandle(
        task=task,
        interject_queue=interject_queue,
        cancel_event=cancel_event,
        pause_event=pause_event,
        client=client,
        loop_id=loop_id,
        initial_user_message=(
            message["content"] if isinstance(message, dict) else message
        ),
    )

    # Let the inner coroutine discover the outer handle so it can switch
    # steering when a nested handle requests pass-through behaviour.
    outer_handle_container[0] = handle

    return handle
