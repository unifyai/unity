import json
import inspect
import asyncio
import inspect
import traceback
from enum import Enum
from collections import defaultdict
from pydantic import BaseModel
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
    sig = inspect.signature(bound_method)
    hints = get_type_hints(bound_method)
    props, required = {}, []
    for name, param in sig.parameters.items():
        ann = hints.get(name, str)
        props[name] = annotation_to_schema(ann)
        if param.default is inspect._empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": bound_method.__name__,
            "description": (bound_method.__doc__ or "").strip(),
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

async def _async_tool_use_loop_inner(
    client: unify.AsyncUnify,
    message: str,
    tools: Dict[str, Callable],
    event_type: Optional[str] = None,
    event_bus: Optional[EventBus] = None,
    *,
    interject_queue: asyncio.Queue[str],
    cancel_event: asyncio.Event,
    max_consecutive_failures: int = 3,
    log_steps: bool = False,
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

    Returns
    -------
    str
        The assistant's final plain-text reply *after* every tool result has
        been fed back into the conversation.
    """

    assert (event_bus and event_type) or (
        not event_bus and not event_type
    ), "event_bus and event_type must either both be specified or both be unspecified"

    if log_steps:
        LOGGER.info(f"\n🧑‍💻 {message}\n")

    # ── initial prompt ───────────────────────────────────────────────────────
    base_tools_schema = [method_to_schema(v) for v in tools.values()]
    msg = {"role": "user", "content": message}
    if event_bus:
        await event_bus.publish(Event(type=event_type, payload={"message": msg}))
    client.append_messages([msg])

    consecutive_failures = 0
    pending: Set[asyncio.Task] = set()
    task_info: Dict[asyncio.Task, Dict[str, Any]] = {}
    assistant_meta: Dict[int, Dict[str, Any]] = {}

    try:
        while True:
            # ── 0. Drain *all* queued interjections, allowed at any time ──
            # NOTE: We must do this *before* waiting on tool completion so a
            # fast typist can still sneak in a question while long-running
            # tools are in flight.  Doing it here keeps latency <1π loop.
            had_interjection = False
            while True:
                try:
                    extra = interject_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                # hide unresolved calls in the *previous* assistant turn
                # ------------------------------------------------------------------
                # Each assistant message may own **several** tool calls.  We build a
                # map  msg → {indices still pending}  and prune by position, removing
                # the whole key only when the very last call has finished.
                pending_idx_by_msg: Dict[int, Set[int]] = defaultdict(set)
                msg_by_id: Dict[int, dict] = {}
                for info in task_info.values():
                    asst_msg = info["assistant_msg"]
                    mid = id(asst_msg)
                    pending_idx_by_msg[mid].add(info["call_idx"])
                    msg_by_id[mid] = asst_msg

                for mid, keep_idx in pending_idx_by_msg.items():
                    asst_msg = msg_by_id[mid]
                    tool_calls = asst_msg.get("tool_calls")
                    if tool_calls is None:
                        continue

                    pruned = [tc for i, tc in enumerate(tool_calls) if i in keep_idx]
                    if pruned:
                        asst_msg["tool_calls"] = pruned # keep unresolved ones
                    else:
                        asst_msg.pop("tool_calls", None) # drop field when empty

                had_interjection = True
                msg = {"role": "user", "content": extra}
                if event_bus:
                    await event_bus.publish(Event(type=event_type,
                                                  payload={"message": msg}))
                client.append_messages([msg])

            # ── A.  Wait for tool completion OR cancellation  ───────────────
            # NOTE: ``asyncio.wait`` lets us race three conditions:
            #       • any tool task finishes
            #       • ``cancel_event`` flips
            #       • a *new* interjection appears
            if pending and not had_interjection:
                interject_w = asyncio.create_task(interject_queue.get())
                waiters = (
                    pending
                    | {asyncio.create_task(cancel_event.wait()), interject_w}
                )
                done, _ = await asyncio.wait(
                    waiters,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if interject_w in done:
                    await interject_queue.put(interject_w.result())
                    continue            # → loop, will be processed in 0.

                if any(t for t in done if t not in pending):
                    raise asyncio.CancelledError  # cancellation wins

                for task in done:  # finished tool(s)
                    # NOTE: Ordered insertion dance starts here:
                    #   – we first restore the original assistant turn
                    #     (``tool_calls`` list) so its *own* content is
                    #     visible again,
                    #   – then we inject the matching “tool → result” message
                    #     *immediately after* that assistant turn so the chat
                    #     history remains perfectly chronological.
                    pending.remove(task)
                    info = task_info.pop(task)
                    name = info["name"]
                    call_id = info["call_id"]

                    try:
                        raw = task.result()
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

                    # --- retro-patch + ordered insertion --------------------
                    asst_msg = info["assistant_msg"]
                    meta = assistant_meta[id(asst_msg)]

                    # 1. restore full tool_calls list exactly once
                    asst_msg["tool_calls"] = meta["original_tool_calls"]

                    # 2. build the tool-result message
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": result,
                    }
                    # 3a. let AsyncUnify know about the new message
                    client.append_messages([tool_msg])

                    # 3b. move it to sit right after the other results that
                    #     belong to this assistant turn
                    insert_pos = (
                        client.messages.index(asst_msg)
                        + 1
                        + meta["results_count"]
                    )
                    moved = client.messages.pop()                # last element
                    client.messages.insert(insert_pos, moved)
                    meta["results_count"] += 1

                    if event_bus:
                        await event_bus.publish(
                            Event(type=event_type, payload={"message": tool_msg}),
                        )

                    if consecutive_failures >= max_consecutive_failures:
                        if log_steps:
                            LOGGER.error("🚨 Aborting: too many tool failures.")
                        raise RuntimeError(
                            "Aborted after too many consecutive tool failures.",
                        )

            # ── B: wait for remaining tools before asking the LLM again
            if pending and not had_interjection:
                continue  # still waiting for other tool tasks

            #  An interjection to handle, or no pending tool calls
            while True:
                try:
                    extra = interject_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if log_steps:
                    LOGGER.info(f"\n⚡ Interjection → {extra!r}\n")
                msg = {"role": "user", "content": extra}
                if event_bus:
                    await event_bus.publish(
                        Event(type=event_type, payload={"message": msg}),
                    )
                client.append_messages([msg])

            # ── C.  Cancel check before calling the LLM  ────────────────────
            # NOTE: Light-weight early-exit guard – we do **not** want to pay
            # for a needless ``client.generate`` if the caller has already
            # decided to abort.
            if cancel_event.is_set():
                raise asyncio.CancelledError

            # ── D.  Ask the LLM what to do next  ────────────────────────────
            if log_steps:
                LOGGER.info("🔄 LLM thinking…")
            # NOTE: No tool tasks are running **right now** (else we would
            # have hit the early-continue above) so it is the cheapest moment
            # to query the model.  We let it decide freely between:
            #   * another function call                → branch E
            #   * plain-text assistant response        → branch F
            # `_maybe_await` shields us from the fact that some back-ends
            # expose `generate` as a coroutine and others as a normal def.

            await _maybe_await(
                client.generate(
                    return_full_completion=True,
                    tools=base_tools_schema,
                    tool_choice="auto",
                    stateful=True,
                )
            )
            msg = client.messages[-1]
            if event_bus:
                await event_bus.publish(
                    Event(type=event_type, payload={"message": msg}),
                )

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
                for idx, call in enumerate(msg["tool_calls"]): # capture index
                    name = call["function"]["name"]
                    args = json.loads(call["function"]["arguments"])
                    fn = tools[name]
                    coro = (
                        fn(**args)
                        if asyncio.iscoroutinefunction(fn)
                        else asyncio.to_thread(fn, **args)
                    )

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
                    }

                # metadata for orderly insertion of results
                assistant_meta[id(msg)] = {
                    "original_tool_calls": original_tool_calls,
                    "results_count": 0,
                }

                if log_steps:
                    LOGGER.info("✅ Step finished (tool calls scheduled)")
                continue  # back to the top

            # ── F.  No new tool calls  ──────────────────────────────────────
            # NOTE: Two scenarios reach this block:
            #   • `pending` **non-empty** → older tool tasks are still in
            #     flight; loop back to wait for them.
            #   • `pending` empty        → the model just produced a plain
            #     assistant message; nothing more to do – return it.
            if pending:  # still waiting for others
                continue

            if log_steps:
                LOGGER.info(f"\n🤖 {msg["content"]}\n")
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
# 2.  A tiny handle object exposed to callers
# ─────────────────────────────────────────────────────────────────────────────
class AsyncToolLoopHandle:
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
    ):
        self._task = task
        self._queue = interject_queue
        self._cancel_event = cancel_event

    # -- public API -----------------------------------------------------------
    @unify.traced
    async def interject(self, message: str) -> None:
        """Inject an additional *user* turn into the running conversation."""
        await self._queue.put(message)

    @unify.traced
    def stop(self) -> None:
        """Politely ask the loop to shut down (gracefully)."""
        self._cancel_event.set()

    # Optional helpers --------------------------------------------------------
    @unify.traced
    def done(self) -> bool:
        return self._task.done()

    @unify.traced
    async def result(self) -> str:
        """Wait for the assistant’s *final* reply."""
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
    log_steps: bool = False,
) -> AsyncToolLoopHandle:
    """
    Kick off `_async_tool_use_loop_inner` in its own task and give the caller
    a handle for live interaction.
    """
    interject_queue: asyncio.Queue[str] = asyncio.Queue()
    cancel_event = asyncio.Event()

    task = asyncio.create_task(
        _async_tool_use_loop_inner(
            client,
            message,
            tools,
            event_type=event_type,
            event_bus=event_bus,
            interject_queue=interject_queue,
            cancel_event=cancel_event,
            max_consecutive_failures=max_consecutive_failures,
            log_steps=log_steps,
        ),
    )

    return AsyncToolLoopHandle(
        task=task,
        interject_queue=interject_queue,
        cancel_event=cancel_event,
    )
