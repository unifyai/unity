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


def get_handle_paused_state(handle) -> bool | None:
    """Check if a SteerableToolHandle is paused by inspecting its _pause_event.

    This is the canonical way to determine whether a handle is currently paused.
    The pattern follows the async tool loop convention where:
    - Event **set** = running (not paused)
    - Event **cleared** = paused

    Args:
        handle: A SteerableToolHandle or any object with a `_pause_event` attribute.

    Returns:
        True if paused (event cleared), False if running (event set),
        None if unknown (no _pause_event, not an Event, or error).
    """
    try:
        pev = getattr(handle, "_pause_event", None)
        if pev is not None and hasattr(pev, "is_set"):
            return not pev.is_set()  # running ⇢ set, paused ⇢ cleared
    except Exception:
        pass
    return None
