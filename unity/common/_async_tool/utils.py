import copy
import inspect
import json
import textwrap

from unillm.logger import _expand_string_newlines


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


def _add_code_delimiters(args: dict) -> None:
    """Add visual delimiters around the code field in execute_code arguments."""
    code = args.get("code", "")
    lang = args.get("language", "")
    if not code:
        return
    banner = f"┄┄┄┄┄┄┄┄ {lang} ┄┄┄┄┄┄┄┄" if lang else "┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄"
    args["code"] = f"\n{banner}\n{textwrap.dedent(code).strip()}\n{banner}"


def format_json_for_log(body: dict) -> str:
    """Format a dict as human-readable JSON for terminal logging.

    Expands escaped newlines in string values so that multi-line content
    (prompts, markdown, code) renders naturally. For execute_code tool calls,
    adds visual delimiters around the code block.
    """
    return _expand_string_newlines(
        json.dumps(body, indent=4, default=str, ensure_ascii=False),
    )


def format_llm_response_for_log(msg: dict) -> str:
    """Format an LLM assistant message for terminal logging.

    Parses stringified tool-call arguments into dicts for pretty-printing
    and adds visual delimiters around execute_code code blocks.
    """
    msg = copy.deepcopy(msg)
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        fn["arguments"] = try_parse_json(fn.get("arguments"))
        if fn.get("name") == "execute_code" and isinstance(fn.get("arguments"), dict):
            _add_code_delimiters(fn["arguments"])
    return format_json_for_log(msg)


def get_handle_paused_state(handle) -> bool | None:
    """Check if a SteerableToolHandle is paused by inspecting its _pause_event.

    This is the canonical way to determine whether a handle is currently paused.
    The pattern follows the async tool loop convention where:
    - Event **set** = running (not paused)
    - Event **cleared** = paused

    All steerable handles should expose a `_pause_event` attribute (or property)
    that follows this convention. For handles that track state differently
    (e.g., via an enum), they should expose a `_pause_event` property that
    returns a proxy object with an `is_set()` method.

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
