import asyncio
from typing import List
import unify

# --------------------------------------------------------------------------- #
#  ASYNC TOOL LOOP – TEST HELPERS                                             #
# --------------------------------------------------------------------------- #


_ASSISTANT_PREFIX_COUNTS: dict[tuple[int, str], int] = {}
_TOOL_PREFIX_COUNTS: dict[tuple[int, str], int] = {}


@unify.traced
def count_assistant_tool_calls(msgs: List[dict], tool_name: str) -> int:
    """Return the number of *assistant* turns whose visible ``tool_calls``
    reference *tool_name* (exact match).

    Used by the async-tool-loop tests to synchronise on a specific tool
    request appearing in the transcript.
    """
    return sum(
        1
        for m in msgs
        if m.get("role") == "assistant"
        and any(
            tc.get("function", {}).get("name") == tool_name
            for tc in (m.get("tool_calls") or [])
        )
    )


async def _wait_for_condition(predicate, poll: float, timeout: float):
    """Generic helper – poll *predicate()* until it returns ``True`` or the
    *timeout* (seconds) elapses.
    """
    import time as _time

    start = _time.perf_counter()
    while _time.perf_counter() - start < timeout:
        if await predicate():
            return
        await asyncio.sleep(poll)
    raise TimeoutError("Timed out waiting for condition to become true.")


@unify.traced
async def _wait_for_tool_request(
    client: "unify.AsyncUnify",
    tool_name: str,
    *,
    timeout: float = 300.0,
    poll: float = 0.05,
) -> None:
    """Block until at least one visible *assistant* tool-call to *tool_name*
    exists in *client.messages* or *timeout* seconds have passed.
    """

    async def _predicate():
        msgs = client.messages or []  # unify may return None initially
        return count_assistant_tool_calls(msgs, tool_name) >= 1

    await _wait_for_condition(_predicate, poll=poll, timeout=timeout)


@unify.traced
async def _wait_for_tool_result(
    client: "unify.AsyncUnify",
    tool_name: str | None = None,
    *,
    min_results: int = 1,
    timeout: float = 300.0,
    poll: float = 0.05,
) -> None:
    """Wait until *min_results* tool result messages are present.

    If *tool_name* is given, only results **whose ``name`` matches exactly**
    are counted.  This mirrors the behaviour required by several tests that
    must synchronise with a tool finishing before proceeding.
    """

    async def _predicate():
        msgs = client.messages or []
        n_seen = sum(
            1
            for m in msgs
            if m.get("role") == "tool"
            and (tool_name is None or m.get("name") == tool_name)
        )
        return n_seen >= min_results

    await _wait_for_condition(_predicate, poll=poll, timeout=timeout)


@unify.traced
async def _wait_for_assistant_call_prefix(
    client: "unify.AsyncUnify",
    prefix: str,
    *,
    timeout: float = 300.0,
    poll: float = 0.05,
) -> None:
    """Poll for a NEW assistant tool-call whose function name starts with ``prefix``.

    Uses a per-(client,prefix) baseline so repeated waits only return on fresh events.
    """
    import time as _time

    def _assistant_calls_prefix(msgs, pref):
        return sum(
            1
            for m in (msgs or [])
            if m.get("role") == "assistant"
            and any(
                (tc.get("function", {}) or {}).get("name", "").startswith(pref)
                for tc in (m.get("tool_calls") or [])
            )
        )

    start_ts = _time.perf_counter()
    key = (id(client), prefix)
    try:
        current = _assistant_calls_prefix(client.messages or [], prefix)
    except Exception:
        current = 0
    baseline = _ASSISTANT_PREFIX_COUNTS.get(key)
    if baseline is None:
        _ASSISTANT_PREFIX_COUNTS[key] = current
        if current > 0:
            return
    while _time.perf_counter() - start_ts < timeout:
        msgs = client.messages or []
        cnt = _assistant_calls_prefix(msgs, prefix)
        if baseline is None:
            if cnt > 0:
                _ASSISTANT_PREFIX_COUNTS[key] = cnt
                return
        else:
            if cnt > baseline:
                _ASSISTANT_PREFIX_COUNTS[key] = cnt
                return
        await asyncio.sleep(poll)
    raise TimeoutError(
        f"Timed out after {timeout}s waiting for assistant to request a helper starting with {prefix!r}.",
    )


@unify.traced
async def _wait_for_tool_message_prefix(
    client: "unify.AsyncUnify",
    prefix: str,
    *,
    timeout: float = 300.0,
    poll: float = 0.05,
) -> None:
    """Poll until a NEW tool message whose ``name`` starts with ``prefix`` appears."""
    import time as _time

    def _count_tool_msgs(_msgs, _pref):
        return sum(
            1
            for m in (_msgs or [])
            if (m.get("role") == "tool")
            and isinstance(m.get("name"), str)
            and m["name"].startswith(_pref)
        )

    start_ts = _time.perf_counter()
    key = (id(client), prefix)
    try:
        current = _count_tool_msgs(client.messages or [], prefix)
    except Exception:
        current = 0
    baseline = _TOOL_PREFIX_COUNTS.get(key)
    if baseline is None:
        _TOOL_PREFIX_COUNTS[key] = current
        if current > 0:
            return
    while _time.perf_counter() - start_ts < timeout:
        msgs = client.messages or []
        cnt = _count_tool_msgs(msgs, prefix)
        if baseline is None:
            if cnt > 0:
                _TOOL_PREFIX_COUNTS[key] = cnt
                return
        else:
            if cnt > baseline:
                _TOOL_PREFIX_COUNTS[key] = cnt
                return
        await asyncio.sleep(poll)
    raise TimeoutError(
        f"Timed out after {timeout}s waiting for a tool message with name starting with {prefix!r}.",
    )


# --------------------------------------------------------------------------- #
#  TEST UTILITIES – GATED TOOLS                                               #
# --------------------------------------------------------------------------- #


def make_gated_sync_tool(return_value: str = "ok", timeout: float = 300):
    """
    Return (gate, tool_fn) where tool_fn blocks until gate.set() is called,
    then returns `return_value`. Useful to keep a sync tool running until a
    deterministic trigger is observed in the outer loop.
    """
    from threading import Event

    gate = Event()

    def _tool():
        gate.wait(timeout=timeout)
        return return_value

    return gate, _tool
