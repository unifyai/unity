from typing import List
import unillm
import asyncio

# --------------------------------------------------------------------------- #
#  ASYNC TOOL LOOP – TEST HELPERS                                             #
# --------------------------------------------------------------------------- #


_ASSISTANT_PREFIX_COUNTS: dict[tuple[int, str], int] = {}
_TOOL_PREFIX_COUNTS: dict[tuple[int, str], int] = {}


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


async def _wait_for_tool_scheduled(
    outer_handle,
    tool_name: str,
    *,
    timeout: float = 300.0,
    poll: float = 0.05,
) -> None:
    """Wait until the async tool loop has actually scheduled `tool_name`
    into its live `task_info` mapping (not just visible in assistant tool_calls).
    """
    import time as _time

    start = _time.perf_counter()
    while _time.perf_counter() - start < timeout:
        try:
            ti = getattr(outer_handle._task, "task_info", {})  # type: ignore[attr-defined]
            if isinstance(ti, dict):
                if any(
                    getattr(_inf, "name", None) == tool_name for _inf in ti.values()
                ):
                    return
        except Exception:
            pass
        await asyncio.sleep(poll)
    raise TimeoutError(
        f"Timed out after {timeout}s waiting for {tool_name!r} to be scheduled",
    )


async def _wait_for_tools_scheduled(
    outer_handle,
    tool_names: list[str],
    *,
    timeout: float = 300.0,
    poll: float = 0.05,
) -> None:
    """Wait until all `tool_names` are present in the loop's live `task_info`."""
    import time as _time

    pending = set(tool_names or [])
    start = _time.perf_counter()
    while _time.perf_counter() - start < timeout:
        try:
            ti = getattr(outer_handle._task, "task_info", {})  # type: ignore[attr-defined]
            if isinstance(ti, dict):
                have = {getattr(_inf, "name", None) for _inf in ti.values()}
                if pending.issubset(have):
                    return
        except Exception:
            pass
        await asyncio.sleep(poll)
    raise TimeoutError(
        f"Timed out after {timeout}s waiting for tools to be scheduled: {sorted(pending)}",
    )


async def _wait_for_tool_requested_and_scheduled(
    client: "unillm.AsyncUnify",
    outer_handle,
    tool_name: str,
    *,
    timeout: float = 300.0,
    poll: float = 0.05,
) -> None:
    """Wait until an assistant tool_call for `tool_name` is visible AND the loop
    has scheduled it (present in task_info)."""
    await _wait_for_tool_request(client, tool_name, timeout=timeout, poll=poll)
    await _wait_for_tool_scheduled(
        outer_handle,
        tool_name,
        timeout=timeout,
        poll=poll,
    )


async def _wait_for_tools_requested_and_scheduled(
    client: "unillm.AsyncUnify",
    outer_handle,
    tool_names: list[str],
    *,
    timeout: float = 300.0,
    poll: float = 0.05,
) -> None:
    """Wait until assistant has requested all `tool_names` AND the loop
    has scheduled each one into task_info."""
    for name in tool_names or []:
        await _wait_for_tool_request(client, name, timeout=timeout, poll=poll)
    await _wait_for_tools_scheduled(
        outer_handle,
        tool_names or [],
        timeout=timeout,
        poll=poll,
    )


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
#  EVENT-BASED WAIT HELPERS (no polling)                                       #
# --------------------------------------------------------------------------- #


async def _wait_for_interjection_event(
    *,
    contains: str | None = None,
    timeout: float = 300.0,
):
    """Await the next ToolLoop event whose message is a user interjection.

    Interjections are now sent as simple user messages (not system messages)
    for Claude/Gemini compatibility. We subscribe to the EventBus and trigger
    on the first matching user message event after registration.
    """
    from unity.events.event_bus import EVENT_BUS

    done: asyncio.Event = asyncio.Event()

    # Build a safe filter expression evaluated against evt.model_dump() namespace
    # Payload shape published by LoopMessageDispatcher.to_event_bus: {"message": <dict>, ...}
    # Interjections are now user messages (not system messages)
    base = "(payload['message'].get('role') == 'user')"
    if contains is not None:
        # substring match in content without relying on builtins
        sub = contains.replace("'", "\\'")
        base += f" and ('{sub}' in (payload['message'].get('content') or ''))"

    async def _cb(_events):  # noqa: D401 – small event marker
        try:
            done.set()
        except Exception:
            pass

    # Register a count-based trigger so only the next matching event fires
    await EVENT_BUS.register_callback(
        event_type="ToolLoop",
        callback=_cb,
        filter=base,
        every_n=1,
    )

    await asyncio.wait_for(done.wait(), timeout=timeout)


# Backwards compatibility alias
_wait_for_system_interjection_event = _wait_for_interjection_event


async def _wait_for_any_assistant_tool_call(
    tool_name: str,
    *,
    timeout: float = 300.0,
):
    """Await the next assistant ToolLoop event that calls `tool_name`."""
    from unity.events.event_bus import EVENT_BUS

    done: asyncio.Event = asyncio.Event()

    async def _cb(events):
        try:
            for evt in events or []:
                payload = getattr(evt, "payload", {})
                msg = payload.get("message") if isinstance(payload, dict) else None
                if not isinstance(msg, dict) or msg.get("role") != "assistant":
                    continue
                for tc in msg.get("tool_calls") or []:
                    nm = (tc.get("function") or {}).get("name")
                    if nm == tool_name:
                        done.set()
                        return
        except Exception:
            pass

    await EVENT_BUS.register_callback(
        event_type="ToolLoop",
        callback=_cb,
        every_n=1,
    )
    await asyncio.wait_for(done.wait(), timeout=timeout)


async def _wait_for_any_tool_message_by_name(
    tool_name: str,
    *,
    timeout: float = 300.0,
):
    """Await the next tool message with name == tool_name across all loops."""
    from unity.events.event_bus import EVENT_BUS

    done: asyncio.Event = asyncio.Event()

    async def _cb(events):
        try:
            for evt in events or []:
                payload = getattr(evt, "payload", {})
                msg = payload.get("message") if isinstance(payload, dict) else None
                if not isinstance(msg, dict) or msg.get("role") != "tool":
                    continue
                if msg.get("name") == tool_name:
                    done.set()
                    return
        except Exception:
            pass

    await EVENT_BUS.register_callback(
        event_type="ToolLoop",
        callback=_cb,
        every_n=1,
    )
    await asyncio.wait_for(done.wait(), timeout=timeout)


async def _wait_for_any_tool_message_prefix(
    prefix: str,
    *,
    timeout: float = 300.0,
):
    """Await the next tool message whose name startswith ``prefix`` across all loops (event-based).

    Uses the EventBus to subscribe to ToolLoop events and triggers on the next
    tool message that satisfies the prefix condition.
    """
    from unity.events.event_bus import EVENT_BUS

    done: asyncio.Event = asyncio.Event()

    async def _cb(events):
        try:
            for evt in events or []:
                payload = getattr(evt, "payload", {})
                msg = payload.get("message") if isinstance(payload, dict) else None
                if not isinstance(msg, dict) or msg.get("role") != "tool":
                    continue
                name = msg.get("name")
                if isinstance(name, str) and name.startswith(prefix):
                    done.set()
                    return
        except Exception:
            pass

    await EVENT_BUS.register_callback(
        event_type="ToolLoop",
        callback=_cb,
        every_n=1,
    )
    await asyncio.wait_for(done.wait(), timeout=timeout)


async def _wait_for_assistant_tool_calls(
    tool_names: list[str],
    *,
    timeout: float = 300.0,
):
    """Wait until assistant has called all tools in `tool_names` at least once.

    Uses a single EventBus subscription to avoid dedupe collisions and races.
    """
    from unity.events.event_bus import EVENT_BUS

    required = set(tool_names or [])
    seen: set[str] = set()
    done: asyncio.Event = asyncio.Event()

    async def _cb(events):
        try:
            for evt in events or []:
                payload = getattr(evt, "payload", {})
                msg = payload.get("message") if isinstance(payload, dict) else None
                if not isinstance(msg, dict) or msg.get("role") != "assistant":
                    continue
                calls = msg.get("tool_calls") or []
                for tc in calls:
                    try:
                        nm = (tc.get("function") or {}).get("name")
                        if isinstance(nm, str):
                            if nm in required and nm not in seen:
                                seen.add(nm)
                                if seen >= required:
                                    done.set()
                                    return
                    except Exception:
                        continue
        except Exception:
            pass

    await EVENT_BUS.register_callback(
        event_type="ToolLoop",
        callback=_cb,
        every_n=1,
    )
    await asyncio.wait_for(done.wait(), timeout=timeout)


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


async def _wait_for_next_assistant_response_event(
    *,
    timeout: float = 300.0,
):
    """Wait until the async tool loop publishes an assistant message to the EventBus.

    This helper registers a callback and waits for the NEXT assistant message
    event to be published. It's designed to be called after a tool result is
    available, to detect when the LLM has responded to that result.

    This uses the EventBus to detect when the LLM has actually responded,
    avoiding race conditions with fixed delays.

    Note: Synthetic check_status_* assistant stubs (used for chronological
    ordering in the transcript) are skipped - this only triggers on real
    LLM responses.
    """
    from unity.events.event_bus import EVENT_BUS

    done: asyncio.Event = asyncio.Event()

    async def _cb(events):
        try:
            for evt in events or []:
                payload = getattr(evt, "payload", {})
                msg = payload.get("message") if isinstance(payload, dict) else None
                if not isinstance(msg, dict) or msg.get("role") != "assistant":
                    continue
                # Skip synthetic check_status_ stubs (internal bookkeeping)
                if _is_synthetic_check_status_stub(msg):
                    continue
                done.set()
                return
        except Exception:
            pass

    await EVENT_BUS.register_callback(
        event_type="ToolLoop",
        callback=_cb,
        every_n=1,
    )
    await asyncio.wait_for(done.wait(), timeout=timeout)
