from typing import List, Sequence, Any
import unillm
import asyncio

# --------------------------------------------------------------------------- #
#  ASYNC TOOL LOOP – TEST HELPERS                                             #
# --------------------------------------------------------------------------- #


_ASSISTANT_PREFIX_COUNTS: dict[tuple[int, str], int] = {}
_TOOL_PREFIX_COUNTS: dict[tuple[int, str], int] = {}


def count_assistant_tool_calls(msgs: Sequence[Any], tool_name: str) -> int:
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


async def _wait_for_tool_request(
    client: "unillm.AsyncUnify",
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


async def _wait_for_tool_result(
    client: "unillm.AsyncUnify",
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


async def _wait_for_assistant_call_prefix(
    client: "unillm.AsyncUnify",
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


async def _wait_for_tool_message_prefix(
    client: "unillm.AsyncUnify",
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


def make_gated_async_tool(return_value: str = "ok", timeout: float = 300):
    """
    Return (gate, tool_fn) where tool_fn is an async function that blocks
    until gate.set() is called, then returns `return_value`.

    Useful for async tool loops to keep a tool running until a deterministic
    trigger is observed in the outer test.
    """
    gate = asyncio.Event()

    async def _tool():
        await asyncio.wait_for(gate.wait(), timeout=timeout)
        return return_value

    return gate, _tool


# --------------------------------------------------------------------------- #
#  TRANSCRIPT SCANNING HELPERS (INDEX-AGNOSTIC)                                #
# --------------------------------------------------------------------------- #


def first_user_message(msgs: List[dict]) -> dict:
    """Return the first user message in a chat transcript."""
    for m in msgs:
        if m.get("role") == "user":
            return m
    raise AssertionError("No user message found in transcript")


def first_assistant_tool_call(msgs: List[dict], tool_name: str) -> tuple[dict, dict]:
    """Return (assistant_message, tool_call) for the first assistant turn that calls tool_name."""
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", {}) or {}
            if f.get("name") == tool_name:
                return m, tc
    raise AssertionError(f"Assistant tool call not found: {tool_name}")


def first_assistant_tool_call_by_prefix(
    msgs: List[dict],
    name_prefix: str,
) -> tuple[dict, dict]:
    """Return (assistant_message, tool_call) for the first assistant turn that calls a tool whose name startswith prefix."""
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", {}) or {}
            n = f.get("name") or ""
            if isinstance(n, str) and n.startswith(name_prefix):
                return m, tc
    raise AssertionError(f"Assistant tool call not found with prefix: {name_prefix}")


def last_plain_assistant_message(msgs: List[dict]) -> dict:
    """Return the last assistant message that has no tool_calls."""
    for m in reversed(msgs):
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            return m
    raise AssertionError("No plain assistant message (without tool_calls) found")


def first_tool_message_by_name_prefix(msgs: List[dict], prefix: str) -> dict:
    """Return the first tool message whose name startswith prefix."""
    for m in msgs:
        if m.get("role") == "tool" and isinstance(m.get("name"), str):
            if m["name"].startswith(prefix):
                return m
    raise AssertionError(f"No tool message found with name prefix: {prefix}")


def first_tool_message_by_name(msgs: List[dict], name: str) -> dict:
    """Return the first tool message whose name equals the given name."""
    for m in msgs:
        if m.get("role") == "tool" and m.get("name") == name:
            return m
    raise AssertionError(f"No tool message found with name: {name}")


# --------------------------------------------------------------------------- #
#  POLLING-BASED WAIT HELPERS (detect NEW occurrences via baseline counting)   #
# --------------------------------------------------------------------------- #

# Per-(client, key) baseline tracking for detecting NEW occurrences
_ASSISTANT_TOOL_CALL_COUNTS: dict[tuple[int, str], int] = {}
_ASSISTANT_RESPONSE_COUNTS: dict[int, int] = {}


def _count_assistant_tool_calls_for_name(msgs: Sequence[Any], tool_name: str) -> int:
    """Count assistant turns that call the specified tool_name."""
    return sum(
        1
        for m in msgs or []
        if m.get("role") == "assistant"
        and any(
            (tc.get("function") or {}).get("name") == tool_name
            for tc in (m.get("tool_calls") or [])
        )
    )


def _is_synthetic_check_status_stub(msg: dict) -> bool:
    """Check if this assistant message is a synthetic check_status_ stub.

    Synthetic stubs are emitted by the loop to preserve chronological ordering
    when a tool's placeholder is not at the transcript tail. They're internal
    bookkeeping, not actual LLM responses.
    """
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        return False
    return all(
        (tc.get("function", {}).get("name", "") or "").startswith("check_status_")
        for tc in tool_calls
    )


def _count_non_synthetic_assistant_messages(msgs: Sequence[Any]) -> int:
    """Count assistant messages that are not synthetic check_status_ stubs."""
    count = 0
    for m in msgs or []:
        if m.get("role") != "assistant":
            continue
        if _is_synthetic_check_status_stub(m):
            continue
        count += 1
    return count


async def _wait_for_any_assistant_tool_call(
    client: "unillm.AsyncUnify",
    tool_name: str,
    *,
    timeout: float = 300.0,
    poll: float = 0.05,
):
    """Poll for a NEW assistant tool_call to `tool_name`.

    Uses baseline counting to detect new occurrences since the wait started.
    This differs from `_wait_for_tool_request` which just checks count >= 1.

    Args:
        client: The LLM client whose messages to monitor.
        tool_name: The exact tool name to wait for.
        timeout: Maximum time to wait in seconds.
        poll: Polling interval in seconds.
    """
    import time as _time

    start_ts = _time.perf_counter()
    key = (id(client), tool_name)

    # Capture baseline count at start
    try:
        current = _count_assistant_tool_calls_for_name(client.messages or [], tool_name)
    except Exception:
        current = 0

    baseline = _ASSISTANT_TOOL_CALL_COUNTS.get(key)
    if baseline is None:
        _ASSISTANT_TOOL_CALL_COUNTS[key] = current
        baseline = current

    while _time.perf_counter() - start_ts < timeout:
        msgs = client.messages or []
        cnt = _count_assistant_tool_calls_for_name(msgs, tool_name)
        if cnt > baseline:
            _ASSISTANT_TOOL_CALL_COUNTS[key] = cnt
            return
        await asyncio.sleep(poll)

    raise TimeoutError(
        f"Timed out after {timeout}s waiting for assistant to call {tool_name!r}",
    )


async def _wait_for_next_assistant_response_event(
    client: "unillm.AsyncUnify",
    *,
    timeout: float = 300.0,
    poll: float = 0.05,
):
    """Poll for a NEW assistant message (non-synthetic) in the transcript.

    Uses baseline counting to detect new occurrences since the wait started.
    Designed to be called after a tool result is available, to detect when
    the LLM has responded to that result.

    Note: Synthetic check_status_* assistant stubs (used for chronological
    ordering in the transcript) are skipped - this only triggers on real
    LLM responses.

    Args:
        client: The LLM client whose messages to monitor.
        timeout: Maximum time to wait in seconds.
        poll: Polling interval in seconds.
    """
    import time as _time

    start_ts = _time.perf_counter()
    key = id(client)

    # Capture baseline count at start
    try:
        current = _count_non_synthetic_assistant_messages(client.messages or [])
    except Exception:
        current = 0

    baseline = _ASSISTANT_RESPONSE_COUNTS.get(key)
    if baseline is None:
        _ASSISTANT_RESPONSE_COUNTS[key] = current
        baseline = current

    while _time.perf_counter() - start_ts < timeout:
        msgs = client.messages or []
        cnt = _count_non_synthetic_assistant_messages(msgs)
        if cnt > baseline:
            _ASSISTANT_RESPONSE_COUNTS[key] = cnt
            return
        await asyncio.sleep(poll)

    raise TimeoutError(
        f"Timed out after {timeout}s waiting for next assistant response",
    )
