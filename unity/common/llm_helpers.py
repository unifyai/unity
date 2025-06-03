import json
import asyncio
import functools
import inspect
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
)

import unify
from ..constants import LOGGER
from ..events.event_bus import EventBus, Event


TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}


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
    *methods: Tuple[Callable],
    include_class_name: bool = True,
) -> Dict[str, Callable]:
    ret = dict()
    for m in methods:
        if include_class_name:
            key = f"{m.__self__.__class__.__name__}_{m.__name__}".replace("__", "_")
        else:
            key = m.__name__.lstrip("_")
        ret[key] = m
    return ret


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
        ret = obj.model_dump()
    elif inspect.isclass(obj) and issubclass(obj, BaseModel):
        ret = obj.model_json_schema()
    elif isinstance(obj, Log):
        ret = obj.to_json()
    elif isinstance(obj, dict):
        ret = {k: _dumps(v, idx + ["k"]) for k, v in obj.items()}
    elif isinstance(obj, list):
        ret = [_dumps(v, idx + [i]) for i, v in enumerate(obj)]
    elif isinstance(obj, tuple):
        ret = tuple(_dumps(v, idx + [i]) for i, v in enumerate(obj))
    else:
        ret = obj
    return json.dumps(ret, indent=indent) if base else ret


# ─────────────────────────────────────────────────────────────────────────────
#  strip doc-lines that describe “hidden” parameters
# ─────────────────────────────────────────────────────────────────────────────
import re
from textwrap import dedent

_PARAM_LINE_RX = re.compile(
    r"""
    ^
    (?P<indent>\s*)                  #     any amount of leading WS
    (?P<names>[^\s:]+(?:\s*,\s*[^\s:]+)*)   # 1 or more names (a, b, …)
    (\s*\([^)]*\))?                 #     optional "(type)"
    \s*:
    """,
    re.VERBOSE,
)


def _strip_hidden_params_from_doc(doc: str, hidden: set[str]) -> str:
    """
    Remove *parameter documentation blocks* whose parameter name is in
    **hidden** from a Google- or NumPy-style docstring.

    We look only inside the “Args:” or “Parameters” section; everything
    else is kept verbatim.
    """
    if not doc or not hidden:
        return doc

    lines = dedent(doc).splitlines()
    out: list[str] = []

    inside_param_section = False
    skip_block = False
    current_indent = 0

    for ln in lines:
        stripped = ln.lstrip()

        # ── detect section headers ────────────────────────────────────────
        if stripped.lower() in ("args:", "parameters"):
            inside_param_section = True
            skip_block = False
            out.append(ln)
            continue
        if inside_param_section and not stripped:
            # blank line inside section
            out.append(ln)
            continue
        if inside_param_section and not stripped.startswith(" "):
            # left the section
            inside_param_section = False
            skip_block = False

        if inside_param_section:
            m = _PARAM_LINE_RX.match(ln)
            if m:
                # Are *any* of the listed names hidden?
                names = {n.strip() for n in m.group("names").split(",")}
                if names & hidden:
                    # start skipping this parameter-block
                    skip_block = True
                    current_indent = len(m.group("indent"))
                    continue  # swallow the parameter line itself
            if skip_block:
                # keep skipping as long as the line is *more indented*
                indent = len(ln) - len(stripped)
                if indent > current_indent:
                    continue
                # description block ended
                skip_block = False

        if not skip_block:
            out.append(ln)

    # normalise consecutive blank lines
    cleaned: list[str] = []
    prev_blank = False
    for ln in out:
        blank = not ln.strip()
        if blank and prev_blank:
            continue
        cleaned.append(ln)
        prev_blank = blank

    if hidden:
        pat = re.compile(
            r"^\s*(?:"
            + "|".join(re.escape(name) for name in hidden)
            + r")\s*:.*(?:\n\s+.*)*",
            flags=re.MULTILINE,
        )
        cleaned = pat.sub("", "\n".join(cleaned))
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # collapse big gaps

    return "\n".join(cleaned).rstrip()


def annotation_to_schema(ann: Any) -> Dict[str, Any]:
    """Convert a Python type annotation into an OpenAI-compatible JSON-Schema
    fragment, including full support for Pydantic BaseModel subclasses.
    """

    # ── 0. Remove typing.Annotated wrapper, if any ────────────────────────────
    origin = get_origin(ann)
    if origin is not None and origin.__name__ == "Annotated":  # Py ≥3.10
        ann = get_args(ann)[0]

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
        # ignore key type; JSON object keys are always strings
        _, value_type = get_args(ann)
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
    if origin is Union:
        sub_schemas = [annotation_to_schema(a) for a in get_args(ann)]
        # Collapse trivial Optional[X] (i.e. Union[X, NoneType]) into X
        if len(sub_schemas) == 2 and {"type": "null"} in sub_schemas:
            return next(s for s in sub_schemas if s != {"type": "null"})
        return {"anyOf": sub_schemas}

    # ── 7. Fallback – treat as generic string ────────────────────────────────
    return {"type": "string"}


def method_to_schema(bound_method):
    """
    Convert **bound_method** into an OpenAI-compatible *function*-tool schema.
    """

    sig = inspect.signature(bound_method)
    hints = get_type_hints(bound_method)

    props, required, hidden = {}, [], set()

    for name, param in sig.parameters.items():
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

    # ── scrub the docstring so hidden args disappear from “Args:”/“Parameters” ──
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
    tool_name = f"{prefix}{bound_method.__name__}".replace("__", "_")

    return {
        "type": "function",
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
    tools: Dict[str, Callable],
    event_type: Optional[str] = None,
    event_bus: Optional[EventBus] = None,
    *,
    interject_queue: asyncio.Queue[str],
    cancel_event: asyncio.Event,
    pause_event: asyncio.Event,
    max_consecutive_failures: int = 3,
    prune_tool_duplicates: bool = True,
    interrupt_llm_with_interjections: bool = True,
    propagate_chat_context: bool = True,
    parent_chat_context: Optional[list[dict]] = None,
    log_steps: bool = False,
    max_steps: Optional[int] = None,
    timeout: Optional[int] = None,
    raise_on_limit: bool = False,
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

    event_type, event_bus : ``str | None``, ``EventBus | None``
        Optional pub-sub hooks.  When both are provided every message
        exchanged inside the loop is emitted as
        ``Event(type=event_type, payload={"message": msg})`` which lets a
        UI or logger stay in sync without tight coupling.

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
    # ── runtime guards ────────────────────────────────────────────────────
    start_ts = time.perf_counter()

    assert (event_bus and event_type) or (
        not event_bus and not event_type
    ), "event_bus and event_type must either both be specified or both be unspecified"

    if log_steps:
        LOGGER.info(f"\n🧑‍💻 {message}\n")

    # ── 0-a. Inject **system** header with broader context ───────────────────
    #
    # When a parent context is supplied we prepend a single synthetic system
    # message that *summarises* it.  This offers the LLM immediate awareness
    # of the wider conversation without having to scroll the nested JSON.
    # The special marker ``_ctx_header=True`` lets us later strip it when
    # propagating context further down (avoids duplication).
    # -----------------------------------------------------------------------

    # ── small helper: keep assistant→tool chronology DRY ────────────────────
    def _insert_after_assistant(parent_msg: dict, tool_msg: dict) -> None:
        """
        Append *tool_msg* and move it directly after *parent_msg*, while
        updating the per-assistant `results_count` bookkeeping.
        """
        meta = assistant_meta.setdefault(
            id(parent_msg),
            {"original_tool_calls": [], "results_count": 0},
        )
        client.append_messages([tool_msg])
        insert_pos = client.messages.index(parent_msg) + 1 + meta["results_count"]
        client.messages.insert(insert_pos, client.messages.pop())
        meta["results_count"] += 1

    # ── small helper: publish to the EventBus (if configured) ──────────────
    async def _to_event_bus(message: dict) -> None:
        """Emit *message* to the EventBus iff one was provided."""
        if event_bus:
            await event_bus.publish(
                Event(type=event_type, payload={"message": message}),
            )

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

        nonlocal consecutive_failures

        pending.discard(task)
        info = task_info.pop(task)
        name = info["name"]
        call_id = info["call_id"]

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
                # ── upgrade interject / clarification flags from handle ─────
                if hasattr(raw, "interject"):
                    info["is_interjectable"] = True

                h_up_q = getattr(raw, "clarification_up_q", None)
                h_down_q = getattr(raw, "clarification_down_q", None)

                if (h_up_q is not None) ^ (h_down_q is not None):
                    raise AttributeError(
                        f"Handle returned by tool {info['name']!r} exposes only "
                        "one of 'clarification_up_q' / 'clarification_down_q'. "
                        "Both queues are required (or neither).",
                    )

                # 1️⃣ spawn the nested waiter
                nested_task = asyncio.create_task(raw.result())
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
                    _insert_after_assistant(info["assistant_msg"], ph)
                    info["tool_reply_msg"] = ph  # remember on *parent*
                else:
                    ph["content"] = (
                        "Nested async tool loop started… waiting for result."
                    )

                await _to_event_bus(ph)

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
            result = _dumps(raw, indent=4)
            consecutive_failures = 0
            if log_steps:
                LOGGER.info(f"\n🛠️ {name} = {result}\n")
        except Exception:
            consecutive_failures += 1
            result = traceback.format_exc()
            if log_steps:
                LOGGER.error(
                    f"\n❌ {name} failed "
                    f"(attempt {consecutive_failures}/{max_consecutive_failures}):\n{result}",
                )

        # 3️⃣  remember so later `_continue_*` helpers can answer instantly
        completed_results[call_id] = result

        # 4️⃣  update / insert tool-result message --------------------------
        asst_msg = info["assistant_msg"]
        meta = assistant_meta[id(asst_msg)]

        continue_msg = info.get("continue_msg")
        if continue_msg is not None:
            continue_msg["content"] = result
            fn = info["call_dict"]["function"]["name"]
            arg = info["call_dict"]["function"]["arguments"]
            continue_msg["name"] = (
                f"{fn}({arg}) completed successfully, "
                "the return values are in the `content` field below."
            )
            tool_msg = continue_msg
        else:
            clarify_ph = info.get("clarify_placeholder")
            if clarify_ph is not None:
                clarify_ph["content"] = result
                tool_msg = clarify_ph
            else:
                tool_reply_msg = info.get("tool_reply_msg")
                if tool_reply_msg is not None:
                    tool_reply_msg["content"] = result
                    tool_msg = tool_reply_msg
                else:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": result,
                    }
                    _insert_after_assistant(asst_msg, tool_msg)

        await _to_event_bus(tool_msg)

        # 6️⃣  failure guard -------------------------------------------------
        if consecutive_failures >= max_consecutive_failures:
            if log_steps:
                LOGGER.error("🚨 Aborting: too many tool failures.")
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
        client.append_messages([sys_msg])

    # ── initial prompt ───────────────────────────────────────────────────────
    base_tools_schema = [method_to_schema(v) for v in tools.values()]
    msg = {"role": "user", "content": message}
    await _to_event_bus(msg)
    client.append_messages([msg])

    consecutive_failures = 0
    pending: Set[asyncio.Task] = set()
    task_info: Dict[asyncio.Task, Dict[str, Any]] = {}
    clarification_channels: Dict[
        str,
        Tuple[asyncio.Queue[str], asyncio.Queue[str]],
    ] = {}
    completed_results: Dict[str, str] = {}
    assistant_meta: Dict[int, Dict[str, Any]] = {}

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
        client.append_messages([notice])
        await _to_event_bus(notice)
        if log_steps:
            LOGGER.info(f"⏹️  Early exit – {reason}")
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
                    pause_waiter = asyncio.create_task(pause_event.wait())
                    cancel_waiter = asyncio.create_task(cancel_event.wait())
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
                else:
                    # nothing running – just idle until resumed or cancelled
                    done, _ = await asyncio.wait(
                        {
                            asyncio.create_task(pause_event.wait()),
                            asyncio.create_task(cancel_event.wait()),
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
            if timeout is not None and time.perf_counter() - start_ts > timeout:
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
                msg = {"role": "user", "content": extra}
                await _to_event_bus(msg)
                client.append_messages([msg])

            # ── A.  Wait for tool completion OR cancellation  ───────────────
            # If a child just asked for clarification we also want to give
            # the LLM a chance to react immediately.
            # Skip this whole block if the model already needs to speak.
            # NOTE: ``asyncio.wait`` lets us race three conditions:
            #       • any tool task finishes
            #       • ``cancel_event`` flips
            #       • a *new* interjection appears
            if pending and not llm_turn_required:
                interject_w = asyncio.create_task(interject_queue.get())
                cancel_waiter = asyncio.create_task(cancel_event.wait())
                clar_waiters: Dict[asyncio.Task, asyncio.Task] = {}
                for _t in pending:
                    # Only listen for *new* clarification questions.
                    # If the task is already awaiting an answer,
                    # `waiting_for_clarification` will be True.
                    if task_info[_t].get("waiting_for_clarification"):
                        continue

                    cuq = task_info[_t].get("clar_up_q")
                    if cuq is not None:
                        w = asyncio.create_task(cuq.get())
                        clar_waiters[w] = _t
                waiters = pending | set(clar_waiters) | {cancel_waiter, interject_w}

                # ── honour global *timeout* while we wait for tools ───────────
                wait_timeout: Optional[float] = None
                if timeout is not None:
                    wait_timeout = timeout - (time.perf_counter() - start_ts)
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
                for aux in (interject_w, cancel_waiter):
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
                            _insert_after_assistant(
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
                        await _to_event_bus(tool_msg)

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

            # helper: register a freshly-minted coroutine as a *temporary* tool
            def _reg_tool(key: str, func_name: str, doc: str, fn: Callable) -> None:
                # prefer the function’s own docstring if it exists, else fall back
                existing = inspect.getdoc(fn)
                fn.__doc__ = existing.strip() if existing else doc
                fn.__name__ = func_name[:64]
                fn.__qualname__ = func_name[:64]
                dynamic_tools[key] = fn

            for _task in list(pending):
                info = task_info[_task]
                handle = info.get("handle")
                ev = info.get("pause_event")

                # ── DYNAMIC capability refresh (handle may change) ─────
                if handle is not None:
                    # 1. interjection
                    info["is_interjectable"] = hasattr(handle, "interject")

                    # 2. clarification queues
                    h_up_q = getattr(handle, "clarification_up_q", None)
                    h_dn_q = getattr(handle, "clarification_down_q", None)

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
                        key=f"continue_{_call_id}",
                        func_name=f"_continue_{_fn_name}_{_call_id}",
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
                    key=f"stop_{_call_id}",
                    func_name=f"_stop_{_fn_name}_{_call_id}",
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
                            await handle.interject(content)  # type: ignore[arg-type]
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
                        key=f"interject_{_call_id}",
                        func_name=f"_interject_{_fn_name}_{_call_id}",
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
                        key=f"clarify_{_call_id}",
                        func_name=f"_clarify_{_fn_name}_{_call_id}",
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
                        key=f"pause_{_call_id}",
                        func_name=f"_pause_{_fn_name}_{_call_id}",
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
                        key=f"resume_{_call_id}",
                        func_name=f"_resume_{_fn_name}_{_call_id}",
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
                        # use the same name we’re about to give fn.__name__
                        func_name = f"_{meth_name}_{_fn_name}_{_call_id}"
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
                            res = await _maybe_await(_bound(**_kw))
                            return {"call_id": _call_id, "result": res}

                        # override the wrapper’s signature to match the real method
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
            for _task in list(pending):
                inf = task_info[_task]
                if (
                    inf.get("tool_reply_msg")
                    or inf.get("continue_msg")
                    or inf.get("clarify_placeholder")
                ):
                    continue  # already covered

                name = inf["name"]
                call_id = inf["call_id"]
                arg_json = inf["call_dict"]["function"]["arguments"]
                asst_msg = inf["assistant_msg"]
                meta = assistant_meta[id(asst_msg)]

                placeholder = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": (
                        "Still running… you can use any of the available helper tools "
                        "to interact with this tool call while it is in progress."
                    ),
                }
                _insert_after_assistant(asst_msg, placeholder)

                # remember so we can patch later
                inf["tool_reply_msg"] = placeholder

            # Merge helpers into the visible toolkit for the upcoming LLM step
            tmp_tools = base_tools_schema + [
                method_to_schema(fn) for fn in dynamic_tools.values()
            ]

            # ── D.  Ask the LLM what to do next  ────────────────────────────
            if log_steps:
                LOGGER.info("🔄 LLM thinking…")

            if interrupt_llm_with_interjections:
                # ––––– new *pre-emptive* mode ––––––––––––––––––––––––––––
                # ➊ start the LLM step …
                llm_task = asyncio.create_task(
                    _maybe_await(
                        client.generate(
                            return_full_completion=True,
                            tools=tmp_tools,
                            tool_choice="auto",
                            stateful=True,
                        ),
                    ),
                )
                interject_w = asyncio.create_task(interject_queue.get())
                cancel_waiter = asyncio.create_task(cancel_event.wait())

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
                    await _maybe_await(
                        client.generate(
                            return_full_completion=True,
                            tools=tmp_tools,
                            tool_choice="auto",
                            stateful=True,
                        ),
                    )
                except Exception:
                    raise Exception(
                        f"LLM call failed. Messages at the time:\n{json.dumps(client.messages, indent=4)}",
                    )

            msg = client.messages[-1]

            # ── timeout guard (post-LLM) ───────────────────────────────
            if timeout is not None and time.perf_counter() - start_ts > timeout:
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
            await _to_event_bus(msg)

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
            #      step A temporarily hides it to avoid “naked” unresolved
            #      calls flashing in the UI, and restores it once *any*
            #      result for that assistant turn is ready.
            # Finally we `continue` so control jumps back to *branch A*
            # where we wait for the **first** task / cancel / interjection.
            if msg["tool_calls"]:

                original_tool_calls: list = []
                for idx, call in enumerate(msg["tool_calls"]):  # capture index
                    name = call["function"]["name"]
                    args = json.loads(call["function"]["arguments"])

                    # ── Special-case dynamic helpers ──────────────────────
                    # • continue_* → acknowledge, no scheduling
                    # • cancel_*   → cancel underlying task & purge metadata
                    if name.startswith("_continue"):
                        call_id = "_".join(name.split("_")[-2:])

                        tgt_task = next(
                            (
                                t
                                for t, inf in task_info.items()
                                if inf["call_id"] == call_id
                            ),
                            None,
                        )

                        orig_fn = task_info[tgt_task]["name"] if tgt_task else "unknown"
                        arg_json = (
                            task_info[tgt_task]["call_dict"]["function"]["arguments"]
                            if tgt_task
                            else "{}"
                        )
                        pretty_name = f"_continue {orig_fn}({arg_json})"

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
                            _insert_after_assistant(msg, tool_reply_msg)
                            info["continue_msg"] = tool_reply_msg
                            if log_steps:
                                LOGGER.info(
                                    f"↩️  {pretty_name} placeholder inserted – still waiting",
                                )

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
                            _insert_after_assistant(info["assistant_msg"], tool_msg)
                            if log_steps:
                                LOGGER.info(
                                    f"↩️  {pretty_name} answered immediately (already finished)",
                                )
                        continue  # completed handling of this _continue

                    if name.startswith("_stop") and not name.startswith(
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
                        pretty_name = f"_stop   {orig_fn}({arg_json})"

                        # ── gracefully shut down any *nested* async-tool loop first ──────
                        if task_to_cancel:
                            nested_handle = task_info[task_to_cancel].get("handle")
                            if nested_handle is not None:
                                # public API call – propagates cancellation downwards
                                nested_handle.stop()

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
                        _insert_after_assistant(msg, tool_msg)

                        if log_steps:
                            LOGGER.info(f"🚫  {name} executed – task stopped")
                        continue  # nothing else to schedule

                    # ── _pause helper ────────────────────────────────────────────────
                    if name.startswith("_pause") and not name.startswith(
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
                        pretty_name = f"_pause {orig_fn}({arg_json})"

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
                        _insert_after_assistant(msg, tool_msg)
                        if log_steps:
                            LOGGER.info(f"⏸️  {pretty_name} executed – task paused")
                        continue  # helper handled, move on

                    # ── _resume helper ───────────────────────────────────────────────
                    if name.startswith("_resume") and not name.startswith(
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
                        pretty_name = f"_resume {orig_fn}({arg_json})"

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
                        _insert_after_assistant(msg, tool_msg)
                        if log_steps:
                            LOGGER.info(f"▶️  {pretty_name} executed – task resumed")
                        continue  # helper handled

                    if name.startswith("_clarify_"):
                        call_id = "_".join(name.split("_")[-2:])
                        ans = args["answer"]

                        # ── find the underlying pending task (if still alive) ───────────────
                        tgt_task = next(  # ← NEW
                            (
                                t
                                for t, inf in task_info.items()
                                if inf["call_id"] == call_id
                            ),
                            None,
                        )

                        if call_id in clarification_channels:
                            await clarification_channels[call_id][1].put(
                                ans,
                            )  # down-queue
                            # ✔️ the tool is un-blocked – start watching it again
                            for _t, _inf in task_info.items():
                                if _inf["call_id"] == call_id:
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
                        _insert_after_assistant(msg, tool_reply_msg)
                        if tgt_task is not None:
                            task_info[tgt_task]["clarify_placeholder"] = tool_reply_msg
                        continue

                    if name.startswith("_interject"):
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
                                if inf["call_id"] == call_id
                            ),
                            None,
                        )

                        pretty_name = (
                            f"_interject {task_info[tgt_task]['name']}({new_text})"
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
                        _insert_after_assistant(msg, tool_msg)

                        if log_steps:
                            LOGGER.info(f"💬  Interjection delivered → {new_text!r}")
                        continue  # nothing else to schedule

                    # first check any dynamic helpers we generated for long-running handles
                    if name in dynamic_tools:
                        fn = dynamic_tools[name]
                    else:
                        fn = tools[name]

                    # ── build **extra** kwargs (chat context + queue) ───
                    extra_kwargs: dict = {}

                    # ------- build nested chat context --------------------
                    # Combine any *inherited* context with the messages of
                    # this loop, but strip the synthetic system header
                    # (marked `_ctx_header`) so it isn't duplicated further
                    # down the call-stack.
                    if propagate_chat_context:
                        cur_msgs = [
                            m for m in client.messages if not m.get("_ctx_header")
                        ]
                        ctx_repr = _chat_context_repr(
                            parent_chat_context,
                            cur_msgs,
                        )
                        extra_kwargs["parent_chat_context"] = ctx_repr

                    # ── interjection capability (implicit) ─────────────
                    sig = inspect.signature(fn)
                    params = sig.parameters
                    has_varkw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                    )

                    sig_accepts_interject_q = "interject_queue" in params or has_varkw
                    sig_accepts_pause_event = "pause_event" in params or has_varkw

                    # ── dedicated pause/resume event (optional) ────────────────────
                    pause_ev: Optional[asyncio.Event] = None
                    if sig_accepts_pause_event:
                        pause_ev = asyncio.Event()
                        pause_ev.set()  # start *running*
                        extra_kwargs["pause_event"] = pause_ev

                    # ── clarification capability (implicit) ─────────────
                    sig_accepts_clar_qs = (
                        "clarification_up_q" in params
                        and "clarification_down_q" in params
                    ) or has_varkw

                    # sanity: must provide *both* queues or neither
                    if (
                        ("clarification_up_q" in params)
                        ^ ("clarification_down_q" in params)
                    ) and not has_varkw:
                        raise TypeError(
                            f"Tool {name!r} provides only one of "
                            "'clarification_up_q' / 'clarification_down_q'. "
                            "Both are required for clarification support.",
                        )

                    sub_q: Optional[asyncio.Queue[str]] = None
                    is_interj = sig_accepts_interject_q  # may be upgraded later
                    # ── per-call clarification queues (optional) ─────────
                    clar_up_q: Optional[asyncio.Queue[str]] = None
                    clar_down_q: Optional[asyncio.Queue[str]] = None
                    # Always provide queues when the tool supports them – the LLM
                    # never passes them explicitly anymore.
                    if sig_accepts_clar_qs:
                        clar_up_q = asyncio.Queue()
                        clar_down_q = asyncio.Queue()
                        extra_kwargs["clarification_up_q"] = clar_up_q
                        extra_kwargs["clarification_down_q"] = clar_down_q
                    if sig_accepts_interject_q:
                        sub_q = asyncio.Queue()
                        extra_kwargs["interject_queue"] = sub_q

                    # ---- filter extras to match fn signature ----------
                    sig = inspect.signature(fn)
                    params = sig.parameters
                    has_varkw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                    )

                    filtered_extras = {
                        k: v
                        for k, v in extra_kwargs.items()
                        if k in params or has_varkw
                    }

                    # 1️⃣ re-fetch the real signature of the chosen fn
                    sig = inspect.signature(fn)
                    params = sig.parameters

                    # 2️⃣ only keep those caller-supplied args the fn actually declares
                    allowed_call_args = {k: v for k, v in args.items() if k in params}

                    # 3️⃣ filter any extra_kwargs the same way
                    has_varkw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                    )
                    filtered_extras = {
                        k: v
                        for k, v in extra_kwargs.items()
                        if k in params or has_varkw
                    }

                    # 4️⃣ finally merge
                    merged_kwargs = {**allowed_call_args, **filtered_extras}

                    # avoid double-passing the queue if the model already
                    # supplied an `interject_queue` argument
                    if "interject_queue" in args and sig_accepts_interject_q:
                        merged_kwargs["interject_queue"] = sub_q

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
                    original_tool_calls.append(call_dict)

                    t = asyncio.create_task(coro)
                    pending.add(t)
                    task_info[t] = {
                        "name": name,
                        "call_id": call["id"],
                        "assistant_msg": msg,
                        "call_dict": call_dict,
                        "call_idx": idx,
                        "is_interjectable": is_interj,  # may later become True automatically
                        "interject_q": sub_q,
                        "chat_ctx": extra_kwargs.get("parent_chat_context"),
                        "clar_up_q": clar_up_q,
                        "clar_down_q": clar_down_q,
                        "pause_event": pause_ev,
                    }

                    if clar_up_q is not None:
                        clarification_channels[call["id"]] = (
                            clar_up_q,
                            clar_down_q,
                        )

                # metadata for orderly insertion of results
                assistant_meta[id(msg)] = {
                    "original_tool_calls": original_tool_calls,
                    "results_count": 0,
                }

                if log_steps:
                    LOGGER.info("✅ Step finished (tool calls scheduled)")
                continue  # finished scheduling tools, back to the very top

            # ── F.  No new tool calls  ──────────────────────────────────────
            # NOTE: Two scenarios reach this block:
            #   • `pending` **non-empty** → older tool tasks are still in
            #     flight; loop back to wait for them.
            #   • `pending` empty        → the model just produced a plain
            #     assistant message; nothing more to do – return it.
            if pending:  # still waiting for others
                continue

            # ── timeout guard (final turn) ──────────────────────────────────
            if timeout is not None and time.perf_counter() - start_ts > timeout:
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

            if log_steps:
                LOGGER.info(f"\n🤖 {msg['content']}\n")
                LOGGER.info("✅ Step finished (final answer)")
            return msg["content"]  # DONE!

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
    async def interject(self, message: str) -> None:
        """Inject an additional *user* turn into the running conversation."""

    @abstractmethod
    def stop(self) -> None:
        """Politely ask the loop to shut down (gracefully)."""

    @abstractmethod
    def pause(self) -> None:
        """Temporarily freeze the outer loop (tools keep running)."""

    @abstractmethod
    def resume(self) -> None:
        """Un-freeze a loop that was paused with :pyfunc:`pause`."""

    @abstractmethod
    def done(self) -> bool:
        """Flag for whether or not this task is done."""

    @abstractmethod
    async def result(self) -> str:
        """Wait for the assistant’s *final* reply."""


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
    ):
        self._task = task
        self._queue = interject_queue
        self._cancel_event = cancel_event
        # “running” ⇢ Event **set**,  “paused” ⇢ Event **cleared**
        self._pause_event = pause_event or asyncio.Event()
        self._pause_event.set()

    # -- public API -----------------------------------------------------------
    @functools.wraps(SteerableToolHandle.interject, updated=())
    async def interject(self, message: str) -> None:
        await self._queue.put(message)

    @functools.wraps(SteerableToolHandle.stop, updated=())
    def stop(self) -> None:
        self._cancel_event.set()

    @functools.wraps(SteerableToolHandle.pause, updated=())
    def pause(self) -> None:
        self._pause_event.clear()

    @functools.wraps(SteerableToolHandle.resume, updated=())
    def resume(self) -> None:
        self._pause_event.set()

    @functools.wraps(SteerableToolHandle.done, updated=())
    def done(self) -> bool:
        return self._task.done()

    @functools.wraps(SteerableToolHandle.result, updated=())
    async def result(self) -> str:
        return await self._task


# ─────────────────────────────────────────────────────────────────────────────
# 3.  A convenience wrapper that *starts* the loop and returns the handle
# ─────────────────────────────────────────────────────────────────────────────
def start_async_tool_use_loop(
    client: unify.AsyncUnify,
    message: str,
    tools: Dict[str, Callable],
    *,
    event_type: Optional[str] = None,
    event_bus: Optional[EventBus] = None,
    max_consecutive_failures: int = 3,
    prune_tool_duplicates=True,
    interrupt_llm_with_interjections: bool = True,
    propagate_chat_context: bool = True,
    parent_chat_context: Optional[list[dict]] = None,
    log_steps: bool = False,
    max_steps: Optional[int] = None,
    timeout: Optional[int] = None,
    raise_on_limit: bool = False,
) -> AsyncToolUseLoopHandle:
    """
    Kick off `_async_tool_use_loop_inner` in its own task and give the caller
    a handle for live interaction.
    """
    interject_queue: asyncio.Queue[str] = asyncio.Queue()
    cancel_event = asyncio.Event()
    pause_event = asyncio.Event()
    pause_event.set()  # start un-paused

    task = asyncio.create_task(
        _async_tool_use_loop_inner(
            client,
            message,
            tools,
            event_type=event_type,
            event_bus=event_bus,
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
        ),
    )

    return AsyncToolUseLoopHandle(
        task=task,
        interject_queue=interject_queue,
        cancel_event=cancel_event,
        pause_event=pause_event,
    )
