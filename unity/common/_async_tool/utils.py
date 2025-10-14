import inspect
import json


async def maybe_await(obj):
    """Return *obj* if it is a value, or `await` and return its result if it is
    an awaitable."""
    if inspect.isawaitable(obj):
        return await obj
    return obj


def try_parse_json(value):
    """Return JSON-parsed value when `value` is a JSON string; otherwise return value unchanged."""
    try:
        if isinstance(value, str):
            return json.loads(value)
    except Exception:
        pass
    return value
