import json
import asyncio
import inspect
import traceback
from enum import Enum
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


async def _async_tool_use_loop_inner(
    client: "unify.AsyncUnify",
    message: str,
    tools: Dict[str, Callable],
    *,
    interject_queue: asyncio.Queue[str],
    cancel_event: asyncio.Event,
    max_consecutive_failures: int = 3,
    log_steps: bool = False,
) -> str:
    r"""
     Drive a structured-tool conversation with an LLM until it produces a
     *final* textual answer, executing tool calls concurrently along the way
     and remaining interruptible at any moment via ``cancel_event``.

     ----------
     High-level behaviour
     --------------------
     1. **Seed the conversation** with the user's ``message`` and the JSON
        schemas for every tool in ``tools``.
     2. Enter an **event loop** that interleaves two concerns:

        *Listening* – wait for **either**
          • the first of the in-flight tool tasks to finish **or**
          • an external cancellation signal.

        *Thinking / Acting* – whenever no pending tool task completes the loop
        checks with the LLM (`client.generate`) to learn whether it should:

          • launch new tool calls (**branch D**) – they are scheduled as
            asyncio tasks and execution jumps back to *Listening*, **or**

          • emit ordinary assistant text (**branch E**) – if *no* tool calls
            are outstanding the loop returns that text as the function result;
            otherwise it keeps listening for the remaining tasks.

     3. **Failure handling** – any exception raised by a tool counts as a
        *failure*; after ``max_consecutive_failures`` in a row the loop aborts
        with :class:`RuntimeError`.

     4. **Graceful cancellation** – if ``cancel_event`` is set (or the caller
        cancels the outer task) all running tools are cancelled and awaited
        before propagating :class:`asyncio.CancelledError`.

    5. **On-the-fly user interjections** – a small wrapper
       :func:`start_async_tool_use_loop` now launches the loop in its own task
       and returns a *handle* object that lets a caller **modify** the
       conversation while it is still in flight:

         • ``await handle.interject(text)`` queues an additional *user* turn
           which is merged into the dialogue just before the next LLM step,
           so already-running tool calls are not disturbed.

         • ``handle.stop()`` triggers the same graceful-shutdown path as
           manually setting ``cancel_event`` – useful when the wrapper is
           nested inside another loop that wants to cancel it from “above”.

       Because the handle is an ordinary Python object these two methods can
       themselves be exposed as *tools* to a parent loop, giving you **nested
       conversations** whose inner loops can be steered or halted by the
       outer assistant.

     ----------
     Execution branches in detail
     ----------------------------
     **A – Listen for task completion or cancellation**

         * If at least one tool task is pending the loop waits for the first
           task **or** ``cancel_event``.
         * A set ``pending`` keeps track of every scheduled task and
           ``task_info`` maps each task to the corresponding *(tool-name,
           call-id)* pair supplied by the LLM.
         * If the *cancel* waiter exits first the loop raises
           :class:`asyncio.CancelledError` immediately.
         * Otherwise the finished tool's result (or traceback on error) is
           appended to the conversation as a ``"role": "tool"`` message and
           the consecutive-failure counter is updated.

     **B – Early cancellation check**
         Skip the LLM step altogether if ``cancel_event`` *has already* been
         set while no tasks were pending.

     **C – Drain queued interjections**

         * At the very start of each iteration the loop empties an internal
           ``asyncio.Queue`` fed by ``handle.interject`` and appends every
           payload as a ``"role": "user"`` message.  This guarantees any new
           clarifications reach the model before it decides on the next tool
           calls.

     **D – Ask the LLM what to do next**
         ``client.generate`` is called with:

         * the accumulated conversation,
         * ``tools_schema`` describing every available function,
         * ``tool_choice="auto"`` – the model decides whether to call a tool
           or to speak.

     **E – Launch new tool calls**
         For every tool call proposed in ``msg.tool_calls`` a coroutine is
         prepared (executed in a thread if the function is synchronous),
         wrapped in ``asyncio.create_task`` and added to ``pending``.
         Control returns to *Listening* immediately so tools can run
         concurrently.

     **F – No new tool calls**

         * If some tool calls are *still* running: loop back to *Listening*.
         * Otherwise – no tasks pending **and** the LLM produced ordinary text –
           the function returns that text to the caller and terminates.

     ----------
     Parameters
     ----------
     client : :class:`unify.AsyncUnify`
         Stateful chat-completion client that must expose
         ``append_messages`` and an async ``generate`` method compatible with
         the OpenAI ChatCompletion API.
     message : str
         The very first user message that kicks off the assistant dialogue.
     tools : Dict[str, Callable]
         Mapping **name → callable** for every function the assistant may
         invoke.  Each callable must be JSON-serialisable via
         :pyfunc:`method_to_schema`; asynchronous functions are awaited,
         synchronous ones are wrapped in :func:`asyncio.to_thread`.
     cancel_event : Optional[:class:`asyncio.Event`], default ``None``
         If provided, the caller can set this event to *politely* abort the
         loop.  A fresh, unset :class:`~asyncio.Event` is created when ``None``
         is given so that callers may also cancel by simply cancelling the
         outer task.
     max_consecutive_failures : int, default ``3``
         After this many **back-to-back** exceptions from tool calls the loop
         aborts with :class:`RuntimeError` to avoid an infinite crash cycle.
     log_steps : bool, default ``False``
         Placeholder for future instrumentation; currently unused.

     ----------
     Returns
     -------
     str
         The assistant's final plain-text reply **after** all required tool
         interactions have completed.
         When ``return_history=True`` the function instead yields
         ``Tuple[str, List[Dict[str, Any]]]`` – the assistant reply **and** the
         complete chat transcript up to that point.

     ----------
     Raises
     ------
     asyncio.CancelledError
         Raised as soon as cancellation is requested, *after* any running tool
         tasks have been cancelled and awaited.
     RuntimeError
         When ``consecutive_failures`` reaches ``max_consecutive_failures``.

     Using the live handle for interjections
     --------------------------------------
     >>> handle = start_async_tool_use_loop(
     ...     client,
     ...     "Which task is most important?",
     ...     task_tools,
     ... )
     >>> # 500 ms later we realise we forgot a constraint
     >>> await handle.interject("Which is also scheduled for this week?")
     >>> answer = await handle.result()
     >>> print(answer)
     'Task XYZ is both high-priority and due this week.'
    """

    if log_steps:
        LOGGER.info(f"\n🧑‍💻 {message}\n")

    # ── initial prompt ───────────────────────────────────────────────────────
    tools_schema = [method_to_schema(v) for v in tools.values()]
    client.append_messages([{"role": "user", "content": message}])

    consecutive_failures = 0
    pending: Set[asyncio.Task] = set()
    task_info: Dict[asyncio.Task, Tuple[str, str]] = {}

    try:
        while True:
            # ── 0.  Drain queued *user* interjections (but **only** if all
            #        previous tool calls have been satisfied).  Injecting a
            #        user turn while the API still expects tool-role messages
            #        would violate the OpenAI protocol and trigger a 400.
            if not pending:
                while True:
                    try:
                        extra = interject_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if log_steps:
                        LOGGER.info(f"\n⚡ Interjection → {extra!r}\n")
                    client.append_messages([{"role": "user", "content": extra}])

            # ── A.  Wait for tool completion OR cancellation  ───────────────
            if pending:
                waiters = pending | {asyncio.create_task(cancel_event.wait())}
                done, _ = await asyncio.wait(
                    waiters, return_when=asyncio.FIRST_COMPLETED
                )

                if any(t for t in done if t not in pending):
                    raise asyncio.CancelledError  # cancellation wins

                for task in done:  # finished tool(s)
                    pending.remove(task)
                    name, call_id = task_info.pop(task)

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

                    client.append_messages(
                        [
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": name,
                                "content": result,
                            }
                        ],
                    )

                    if consecutive_failures >= max_consecutive_failures:
                        if log_steps:
                            LOGGER.error("🚨 Aborting: too many tool failures.")
                        raise RuntimeError(
                            "Aborted after too many consecutive tool failures."
                        )

            # ── B: wait for remaining tools before asking the LLM again
            if pending:
                continue  # still waiting for other tool tasks

            #  (no pending tool calls → safe to inject new user input)
            while True:
                try:
                    extra = interject_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if log_steps:
                    LOGGER.info(f"\n⚡ Interjection → {extra!r}\n")
                client.append_messages([{"role": "user", "content": extra}])

            # ── C.  Cancel check before calling the LLM  ────────────────────
            if cancel_event.is_set():
                raise asyncio.CancelledError

            # ── D.  Ask the LLM what to do next  ────────────────────────────
            if log_steps:
                LOGGER.info("🔄 LLM thinking…")

            response = await client.generate(
                return_full_completion=True,
                tools=tools_schema,
                tool_choice="auto",
                stateful=True,
            )
            msg = response.choices[0].message

            # ── E.  Launch any new tool calls  ──────────────────────────────
            if msg.tool_calls:
                for call in msg.tool_calls:
                    name = call.function.name
                    args = json.loads(call.function.arguments)
                    fn = tools[name]
                    coro = (
                        fn(**args)
                        if asyncio.iscoroutinefunction(fn)
                        else asyncio.to_thread(fn, **args)
                    )

                    t = asyncio.create_task(coro)
                    pending.add(t)
                    task_info[t] = (name, call.id)

                if log_steps:
                    LOGGER.info("✅ Step finished (tool calls scheduled)")
                continue  # back to the top

            # ── F.  No new tool calls  ──────────────────────────────────────
            if pending:  # still waiting for others
                continue

            if log_steps:
                LOGGER.info(f"\n🤖 {msg.content}\n")
                LOGGER.info("✅ Step finished (final answer)")
            return msg.content  # DONE!

    except asyncio.CancelledError:  # graceful shutdown
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
    async def interject(self, message: str) -> None:
        """Inject an additional *user* turn into the running conversation."""
        await self._queue.put(message)

    def stop(self) -> None:
        """Politely ask the loop to shut down (gracefully)."""
        self._cancel_event.set()

    # Optional helpers --------------------------------------------------------
    def done(self) -> bool:
        return self._task.done()

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
            interject_queue=interject_queue,
            cancel_event=cancel_event,
            max_consecutive_failures=max_consecutive_failures,
            log_steps=log_steps,
        )
    )

    return AsyncToolLoopHandle(
        task=task,
        interject_queue=interject_queue,
        cancel_event=cancel_event,
    )
