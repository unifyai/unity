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
from constants import LOGGER


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


def tool_use_loop(
    client: unify.Unify,
    message: str,
    tools: Dict[str, Callable],
    *,
    name: Optional[str] = None,
    max_consecutive_failures: int = 3,
    log_steps: bool = False,
):
    """
    Keep invoking the model until it stops calling tools **or**
    `max_consecutive_failures` is exceeded.

    A *failed* attempt = the Python callable raised an Exception.
    Every failure's full stack-trace is returned to the model as the
    content of the relevant ``tool`` message.

    The failure counter resets to 0 after any *successful* tool call.
    """

    if log_steps:
        LOGGER.info(f"\n🧑‍💻 {message}\n")

    tools_schema = [method_to_schema(v) for v in tools.values()]
    msg = {"role": "user", "content": message}
    if name is not None:
        # apparently this works: https://chatgpt.com/share/681af318-91b0-8012-aeb5-1159f5c250a4
        msg["name"] = name
    client.append_messages([msg])

    consecutive_failures = 0

    while True:
        if log_steps:
            LOGGER.info("🔄 LLM thinking…")

        response = client.generate(
            return_full_completion=True,
            tools=tools_schema,
            tool_choice="auto",
            stateful=True,
        )

        msg = response.choices[0].message
        new_msgs = []

        if msg.tool_calls:
            # ── Iterate over all tool calls produced in this turn ────────────
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)

                try:
                    raw_result = tools[name](**args)
                    result = _dumps(raw_result, indent=4)
                    consecutive_failures = 0  # reset on success
                    if log_steps:
                        LOGGER.info(f"\n🛠️ {name}({args}) = {result}\n")

                except Exception:
                    consecutive_failures += 1
                    result = traceback.format_exc()
                    if log_steps:
                        LOGGER.error(
                            f"\n❌ {name}({args}) raised an exception "
                            f"(attempt {consecutive_failures}/{max_consecutive_failures}):\n{result}",
                        )

                # Feed the (successful result **or** stack trace) back to the model
                new_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": result,
                    },
                )

            client.append_messages(new_msgs)

            # Abort if we hit the error limit
            if consecutive_failures >= max_consecutive_failures:
                LOGGER.error(
                    f"🚨 Aborting: reached "
                    f"{max_consecutive_failures} consecutive tool failures.",
                )
                raise Exception(
                    "Aborted after too many consecutive tool failures.\n"
                    f"Last stack trace:\n{result}\n"
                    f"Conversation history:\n{json.dumps(client.messages, indent=4)}\n",
                )

            if log_steps:
                LOGGER.info("✅ Step finished (tool calls executed)")

        else:
            # ── No tool call – final answer ─────────────────────────────────
            if log_steps:
                LOGGER.info(f"\n🤖 {msg.content}\n")
                LOGGER.info("✅ Step finished (final answer)")
            return msg.content



async def async_tool_use_loop(
    client: unify.AsyncUnify,
    message: str,
    tools: Dict[str, Callable],
    *,
    name: Optional[str] = None,
    max_consecutive_failures: int = 3,
    log_steps: bool = False,
):
    """
    Converse with the LLM, running its requested tools *concurrently*.

    Behaviour
    ---------
    • Launch every tool-call in its own asyncio.Task.
    • After *any* task finishes, feed its result back to the LLM immediately.
    • Keep looping until the first model turn that contains **no** tool calls.
      At that point wait for any still-running tool tasks to finish, then
      return the model’s content.
    • Abort after `max_consecutive_failures` tool exceptions *in a row*.
    """

    # ── prompt setup ────────────────────────────────────────────────────────
    if log_steps:
        LOGGER.info(f"\n🧑‍💻 {message}\n")

    tools_schema = [method_to_schema(v) for v in tools.values()]
    user_msg = {"role": "user", "content": message}
    if name is not None:
        user_msg["name"] = name
    client.append_messages([user_msg])

    consecutive_failures = 0
    pending_tasks: Set[asyncio.Task] = set()
    task_info: Dict[asyncio.Task, Tuple[str, str]] = {}   # task → (tool_name, call_id)

    # ── conversation loop ───────────────────────────────────────────────────
    while True:

        # ── 1️⃣  If tasks exist, wait for *one* to finish ──────────────────
        if pending_tasks:
            done, _ = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                pending_tasks.remove(task)
                tool_name, call_id = task_info.pop(task)

                try:
                    raw = task.result()
                    result = _dumps(raw, indent=4)
                    consecutive_failures = 0
                    if log_steps:
                        LOGGER.info(f"\n🛠️ {tool_name} (…) = {result}\n")
                except Exception:
                    consecutive_failures += 1
                    result = traceback.format_exc()
                    if log_steps:
                        LOGGER.error(
                            f"\n❌ {tool_name} (…) failed "
                            f"(attempt {consecutive_failures}/{max_consecutive_failures}):\n{result}"
                        )

                client.append_messages(
                    [
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name,
                            "content": result,
                        }
                    ]
                )

                if consecutive_failures >= max_consecutive_failures:
                    raise RuntimeError(
                        "Aborted after too many consecutive tool failures.\n"
                        f"Last stack trace:\n{result}"
                    )
            # Fall through – we’ll ask the LLM for the next move right away.

        # ── 2️⃣  Ask the LLM what to do next ───────────────────────────────
        if log_steps:
            LOGGER.info("🔄 LLM thinking…")

        response = await client.generate(
            return_full_completion=True,
            tools=tools_schema,
            tool_choice="auto",
            stateful=True,
        )
        msg = response.choices[0].message

        # ── 3️⃣  Launch any new tool calls ─────────────────────────────────
        if msg.tool_calls:
            for call in msg.tool_calls:
                tool_name = call.function.name
                args = json.loads(call.function.arguments)
                fn = tools[tool_name]

                if asyncio.iscoroutinefunction(fn):
                    coro = fn(**args)
                else:
                    coro = asyncio.to_thread(fn, **args)

                task = asyncio.create_task(coro)
                pending_tasks.add(task)
                task_info[task] = (tool_name, call.id)

            if log_steps:
                LOGGER.info(
                    "🚀 Launched "
                    f"{len(msg.tool_calls)} task(s) "
                    f"({len(pending_tasks)} pending total)"
                )
            # Loop again (some tasks are now running)
            continue

        # ── 4️⃣  Model returned *no* tool calls → final answer ─────────────
        if pending_tasks:
            # Wait for all still-running tasks (ignore results / exceptions)
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        if log_steps:
            LOGGER.info(f"\n🤖 {msg.content}\n✅ Final answer")
        return msg.content
