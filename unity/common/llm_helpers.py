import json
import inspect
import asyncio
import inspect
import traceback
import functools
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
    prune_tool_duplicates: bool = True,
    interrupt_llm_with_interjections: bool = True,
    interjectable_tools: Optional[Set[str]] = None,
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

    interjectable_tools : ``set[str] | None``
        Names of long-running tools that are *steerable* post-launch.  For
        each pending call the loop exposes a helper
        ``_interject_<call-id>(content: str)``.  The original tool must
        accept an ``interject_queue`` keyword argument (an
        ``asyncio.Queue[str]``) to receive these live instructions.

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
    completed_results: Dict[str, str] = {}
    assistant_meta: Dict[int, Dict[str, Any]] = {}

    if interjectable_tools is None:
        interjectable_tools = set()

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

                had_interjection = True
                msg = {"role": "user", "content": extra}
                if event_bus:
                    await event_bus.publish(
                        Event(type=event_type, payload={"message": msg}),
                    )
                client.append_messages([msg])

            # ── A.  Wait for tool completion OR cancellation  ───────────────
            # NOTE: ``asyncio.wait`` lets us race three conditions:
            #       • any tool task finishes
            #       • ``cancel_event`` flips
            #       • a *new* interjection appears
            if pending and not had_interjection:
                interject_w = asyncio.create_task(interject_queue.get())
                cancel_waiter = asyncio.create_task(cancel_event.wait())
                waiters = pending | {cancel_waiter, interject_w}
                done, _ = await asyncio.wait(
                    waiters,
                    return_when=asyncio.FIRST_COMPLETED,
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

                    # 1. Prefer patching the *continue* placeholder (if any);
                    #    otherwise fall back to the original tool placeholder.
                    continue_msg = info.get("continue_msg")
                    if continue_msg is not None:
                        continue_msg["content"] = result
                        name = info["call_dict"]["function"]["name"]
                        args = info["call_dict"]["function"]["arguments"]
                        continue_msg["name"] = f"{name}({args}) completed successfully, the return values are in the `content` field below."
                        tool_msg = continue_msg
                    else:
                        placeholder_msg = info.get("placeholder_msg")
                        if placeholder_msg is not None:
                            placeholder_msg["content"] = result
                            tool_msg = placeholder_msg
                        else:
                            tool_msg = {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": name,
                                "content": result,
                            }
                            client.append_messages([tool_msg])
                            insert_pos = (
                                client.messages.index(asst_msg)
                                + 1
                                + meta["results_count"]
                            )
                            moved = client.messages.pop()
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

            # ── B: wait for remaining tools before asking the LLM again,
            # unless there was an interjection
            if pending and not had_interjection:
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

            for _task in list(pending):
                info = task_info[_task]
                _call_id: str = info["call_id"]
                _fn_name: str = info["name"]
                _arg_json: str = info["call_dict"]["function"]["arguments"]
                try:
                    _arg_dict = json.loads(_arg_json)
                    _arg_repr = ", ".join(f"{k}={v!r}" for k, v in _arg_dict.items())
                except Exception:
                    _arg_repr = _arg_json  # fallback: raw JSON string

                # concise, informative, single‑line docs  ----------------------
                _continue_doc = (
                    f"Continue waiting for {_fn_name}({_arg_repr})."
                )
                _cancel_doc = (
                    f"Cancel pending call {_fn_name}({_arg_repr})."
                )

                # ––– 1. continue helper ––––––––––––––––––––––––––––––––––––
                async def _continue() -> Dict[str, str]:
                    return {"status": "continue", "call_id": _call_id}
                _continue.__doc__ = _continue_doc  # type: ignore[attr-defined]
                _continue.__name__ = f"_continue_{_fn_name}_{_call_id}"
                dynamic_tools[f"continue_{_call_id}"] = _continue

                # ––– 2. cancel helper –––––––––––––––––––––––––––––––––––––
                async def _cancel() -> Dict[str, str]:
                    if not _task.done():
                        _task.cancel()
                    pending.discard(_task)
                    task_info.pop(_task, None)
                    return {"status": "cancelled", "call_id": _call_id}
                _cancel.__doc__ = _cancel_doc  # type: ignore[attr-defined]
                _cancel.__name__ = f"_cancel_{_fn_name}_{_call_id}"
                dynamic_tools[f"cancel_{_call_id}"] = _cancel

                # ––– 3. interject helper (optional) ––––––––––––––––––––––
                if info.get("is_interjectable"):
                    _interject_doc = (
                        f"Inject additional instructions for {_fn_name}({_arg_repr}). "
                        "Takes a single argument `content` containing plain-English guidance."
                    )

                    async def _interject(content: str) -> Dict[str, str]:  # type: ignore[valid-type]
                        await info["interject_q"].put(content)
                        return {
                            "status": "interjected",
                            "call_id": _call_id,
                            "content": content,
                        }

                    _interject.__doc__ = _interject_doc  # type: ignore[attr-defined]
                    _interject.__name__ = f"_interject_{_fn_name}_{_call_id}"
                    dynamic_tools[f"interject_{_call_id}"] = _interject

            # Merge helpers into the visible toolkit for the upcoming LLM step
            tmp_tools = base_tools_schema + [method_to_schema(fn) for fn in dynamic_tools.values()]

            # ── D.  Ask the LLM what to do next  ────────────────────────────
            if log_steps:
                LOGGER.info("🔄 LLM thinking…")

            if interrupt_llm_with_interjections:
                # ––––– new *pre-emptive* mode ––––––––––––––––––––––––––––
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

                done, _ = await asyncio.wait(
                    {llm_task, interject_w, cancel_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # helper cleanup
                for tsk in (llm_task, interject_w, cancel_waiter):
                    if tsk not in done and not tsk.done():
                        tsk.cancel()
                await asyncio.gather(interject_w, cancel_waiter, return_exceptions=True)

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
                        f"LLM call failed. Messages at the time:\n{json.dumps(client.messages, indent=4)}",
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

            if event_bus:
                await event_bus.publish(
                    Event(type=event_type, payload={"message": msg}),
                )

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
                            (t for t, inf in task_info.items() if inf["call_id"] == call_id),
                            None,
                        )

                        orig_fn = (
                            task_info[tgt_task]["name"] if tgt_task else "unknown"
                        )
                        arg_json = (
                            task_info[tgt_task]["call_dict"]["function"]["arguments"]
                            if tgt_task
                            else "{}"
                        )
                        pretty_name = f"_continue {orig_fn}({arg_json})"

                        if tgt_task:  # still running → insert placeholder
                            placeholder_msg = {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "name": pretty_name,
                                "content": (
                                    "Still waiting for the final result. "
                                    "This message will be updated automatically."
                                ),
                            }
                            client.append_messages([placeholder_msg])

                            meta = assistant_meta.setdefault(
                                id(msg), {"original_tool_calls": [], "results_count": 0},
                            )
                            insert_pos = client.messages.index(msg) + 1 + meta["results_count"]
                            client.messages.insert(insert_pos, client.messages.pop())
                            meta["results_count"] += 1

                            # Section A will patch this content later
                            task_info[tgt_task]["continue_msg"] = placeholder_msg

                            if log_steps:
                                LOGGER.info(f"↩️  {pretty_name} placeholder inserted – still waiting")

                        else:  # the original tool already finished
                            finished = completed_results.get(
                                call_id,
                                _dumps({"status": "not-found", "call_id": call_id}, indent=4),
                            )
                            tool_msg = {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "name": pretty_name,
                                "content": finished,
                            }
                            client.append_messages([tool_msg])

                            meta = assistant_meta.setdefault(
                                id(msg), {"original_tool_calls": [], "results_count": 0},
                            )
                            insert_pos = client.messages.index(msg) + 1 + meta["results_count"]
                            client.messages.insert(insert_pos, client.messages.pop())
                            meta["results_count"] += 1

                            if log_steps:
                                LOGGER.info(f"↩️  {pretty_name} answered immediately (already finished)")

                        continue  # completed handling of this _continue

                    if name.startswith("_cancel"):
                        call_id = "_".join(name.split("_")[-2:])

                        # ── locate & cancel the underlying coroutine ──────
                        task_to_cancel = next(
                            (t for t, info in task_info.items() if info["call_id"] == call_id),
                            None,
                        )

                        orig_fn = (
                            task_info[task_to_cancel]["name"] if task_to_cancel else "unknown"
                        )
                        arg_json = (
                            task_info[task_to_cancel]["call_dict"]["function"]["arguments"]
                            if task_to_cancel
                            else "{}"
                        )
                        pretty_name = f"_cancel {orig_fn}({arg_json})"

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
                                f"The tool call [{call_id}] has been cancelled successfully."
                            ),
                        }
                        client.append_messages([tool_msg])

                        meta = assistant_meta.setdefault(
                            id(msg), {"original_tool_calls": [], "results_count": 0},
                        )
                        insert_pos = client.messages.index(msg) + 1 + meta["results_count"]
                        client.messages.insert(insert_pos, client.messages.pop())
                        meta["results_count"] += 1

                        if log_steps:
                            LOGGER.info(f"🚫  {name} executed – task cancelled")
                        continue  # nothing else to schedule

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
                            (t for t, inf in task_info.items() if inf["call_id"] == call_id),
                            None,
                        )

                        pretty_name = f"_interject {task_info[tgt_task]['name']}({new_text})" if tgt_task else name

                        # ― push guidance onto the private queue -------------
                        if tgt_task and task_info[tgt_task]["interject_q"] is not None:
                            await task_info[tgt_task]["interject_q"].put(new_text)

                        # ― emit a tool message so the chat log stays tidy ---
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": pretty_name,
                            "content": f'Guidance "{new_text}" forwarded to the running tool.',
                        }
                        client.append_messages([tool_msg])

                        meta = assistant_meta.setdefault(
                            id(msg), {"original_tool_calls": [], "results_count": 0},
                        )
                        insert_pos = client.messages.index(msg) + 1 + meta["results_count"]
                        client.messages.insert(insert_pos, client.messages.pop())
                        meta["results_count"] += 1

                        if log_steps:
                            LOGGER.info(f"💬  Interjection delivered → {new_text!r}")
                        continue  # nothing else to schedule

                    fn = tools[name]

                    # ── wrap interjectable tools with a private queue ────
                    if name in interjectable_tools:
                        sub_q: asyncio.Queue[str] = asyncio.Queue()
                        if "interject_queue" in args:
                            # the model already provided the kw-arg → overwrite
                            args["interject_queue"] = sub_q
                            if asyncio.iscoroutinefunction(fn):
                                coro = fn(**args)
                            else:
                                coro = asyncio.to_thread(fn, **args)
                        else:
                            # inject our queue explicitly
                            if asyncio.iscoroutinefunction(fn):
                                coro = fn(interject_queue=sub_q, **args)
                            else:
                                coro = asyncio.to_thread(
                                    fn,
                                    interject_queue=sub_q,
                                    **args,
                                )
                        is_interj = True
                    else:
                        coro = (
                            fn(**args)
                            if asyncio.iscoroutinefunction(fn)
                            else asyncio.to_thread(fn, **args)
                        )
                        sub_q = None
                        is_interj = False

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
                        "is_interjectable": is_interj,
                        "interject_q": sub_q,
                    }

                    # ── Insert a placeholder tool message so the chat history
                    #    keeps the correct assistant → tool ordering without
                    #    touching the assistant turn itself.  Once the tool
                    #    finishes, this message will be updated in place.
                    placeholder_content = (
                        "The following tool calls are still running. If any of them are no longer "
                        "relevant to the sequence of user requests, then you can call their "
                        "`_cancel_*` helper, otherwise feel free to call the corresponding "
                        "`_continue_*` helper to keep waiting:\n"
                        f" • {name}({call['function']['arguments']}) → "
                        f"cancel_{call['id']} / continue_{call['id']}"
                    )
                    placeholder_msg = {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": placeholder_content,
                    }
                    client.append_messages([placeholder_msg])

                    meta = assistant_meta.setdefault(
                        id(msg), {"original_tool_calls": [], "results_count": 0},
                    )
                    insert_pos = client.messages.index(msg) + 1 + meta["results_count"]
                    moved = client.messages.pop()          # last element is the placeholder
                    client.messages.insert(insert_pos, moved)
                    meta["results_count"] += 1

                    task_info[t]["placeholder_msg"] = placeholder_msg

                    if event_bus:
                        await event_bus.publish(
                            Event(type=event_type, payload={"message": placeholder_msg}),
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
    prune_tool_duplicates=True,
    interrupt_llm_with_interjections: bool = True,
    interjectable_tools: Optional[Set[str]] = None,
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
            prune_tool_duplicates=prune_tool_duplicates,
            interrupt_llm_with_interjections=interrupt_llm_with_interjections,
            interjectable_tools=interjectable_tools,
            log_steps=log_steps,
        ),
    )

    return AsyncToolLoopHandle(
        task=task,
        interject_queue=interject_queue,
        cancel_event=cancel_event,
    )
