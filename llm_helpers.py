import json
import inspect
from pydantic import BaseModel
from typing import List, Dict, Union, Any, get_type_hints

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


def method_to_schema(bound_method):
    sig = inspect.signature(bound_method)
    hints = get_type_hints(bound_method)
    props, required = {}, []
    for name, param in sig.parameters.items():
        props[name] = {"type": TYPE_MAP.get(hints.get(name, str), "string")}
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


def tool_use_loop(client: unify.Unify, message: str, tools: Dict[str, callable]):
    """
    Loops the agent until no more tools are called, and the agent is satisfied.
    """
    LOGGER.info(f"\n🧑‍💻 {message}\n")
    tools_schema = [method_to_schema(v) for v in tools.values()]
    client.append_messages([{"role": "user", "content": message}])
    while True:
        response = client.generate(
            return_full_completion=True,
            tools=tools_schema,
            tool_choice="auto",
            stateful=True,
        )

        new_msgs = list()
        msg = response.choices[0].message
        if msg.tool_calls:
            # iterate over *all* tool calls returned in this turn
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)
                result = _dumps(tools[name](**args), indent=4)
                LOGGER.info(f"\n🛠️ {name}({args}) = {result}\n")

                # feed result back so the model can think again
                new_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": result,
                    },
                )
            client.append_messages(new_msgs)
        else:
            LOGGER.info(f"\n🤖 {response.choices[0].message.content}\n")
            return response.choices[0].message.content
