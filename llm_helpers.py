import json
import inspect
import traceback
from enum import Enum
from pydantic import BaseModel
from typing import (
    List,
    Dict,
    Union,
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
    max_consecutive_failures: int = 3,
):
    """
    Keep invoking the model until it stops calling tools **or**
    `max_consecutive_failures` is exceeded.

    A *failed* attempt = the Python callable raised an Exception.
    Every failure’s full stack-trace is returned to the model as the
    content of the relevant ``tool`` message.

    The failure counter resets to 0 after any *successful* tool call.
    """

    LOGGER.info(f"\n🧑‍💻 {message}\n")

    tools_schema = [method_to_schema(v) for v in tools.values()]
    client.append_messages([{"role": "user", "content": message}])

    consecutive_failures = 0

    while True:
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
                    LOGGER.info(f"\n🛠️ {name}({args}) = {result}\n")

                except Exception:
                    consecutive_failures += 1
                    result = traceback.format_exc()
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
                    f"Last stack trace:\n{result}",
                )

        else:
            # ── No tool call – final answer ─────────────────────────────────
            LOGGER.info(f"\n🤖 {msg.content}\n")
            return msg.content
