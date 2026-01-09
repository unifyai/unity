from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Literal, Optional
from unittest.mock import AsyncMock

from tests.test_async_tool_loop.async_helpers import _wait_for_condition
from unity.actor.code_act_actor import CodeActActor
from unity.actor.hierarchical_actor import HierarchicalActor
from unity.actor.environments import StateManagerEnvironment
from unity.function_manager.primitives import Primitives

try:
    from unity.function_manager.function_manager import FunctionManager
except Exception:  # pragma: no cover
    FunctionManager = Any  # type: ignore[misc,assignment]


def get_state_manager_tools(handle: Any) -> list[str]:
    """Extract all state manager tool calls from idempotency_cache."""
    cache = getattr(handle, "idempotency_cache", {}) or {}
    return [
        entry.get("meta", {}).get("tool", "")
        for entry in cache.values()
        if entry.get("meta", {}).get("tool", "").startswith("primitives.")
    ]


# ---------------------------------------------------------------------------
# CodeActActor helpers (simulated routing parity)
# ---------------------------------------------------------------------------


def _wrap_primitives_method_for_trace(
    *,
    manager: Any,
    method_name: str,
    fq_tool_name: str,
    sink: list[str],
) -> None:
    if not hasattr(manager, method_name):
        return
    orig = getattr(manager, method_name)
    if not callable(orig):
        return

    if asyncio.iscoroutinefunction(orig):

        async def _wrapped(*args: Any, **kwargs: Any) -> Any:
            sink.append(fq_tool_name)
            return await orig(*args, **kwargs)

    else:

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            sink.append(fq_tool_name)
            return orig(*args, **kwargs)

    setattr(manager, method_name, _wrapped)


def instrument_basic_primitives_calls(primitives: Primitives) -> list[str]:
    """Wrap a minimal state-manager surface to record which primitives were invoked."""
    calls: list[str] = []
    targets: list[tuple[str, list[str]]] = [
        ("contacts", ["ask", "update"]),
        ("tasks", ["ask", "update", "execute"]),
        ("knowledge", ["ask", "update", "refactor"]),
        ("transcripts", ["ask"]),
        ("guidance", ["ask", "update"]),
        ("web", ["ask"]),
    ]
    for manager_attr, methods in targets:
        try:
            mgr = getattr(primitives, manager_attr)
        except Exception:
            continue
        for m in methods:
            _wrap_primitives_method_for_trace(
                manager=mgr,
                method_name=m,
                fq_tool_name=f"primitives.{manager_attr}.{m}",
                sink=calls,
            )
    return calls


async def wait_for_recorded_primitives_call(
    calls: list[str],
    tool_name: str,
    *,
    timeout: float = 60.0,
    poll: float = 0.05,
) -> None:
    """Wait until `tool_name` appears in `calls` (best-effort for CodeAct routing tests)."""

    async def _predicate() -> bool:
        return tool_name in set(calls)

    try:
        await asyncio.wait_for(
            _wait_for_condition(_predicate, poll=poll, timeout=timeout),
            timeout=timeout + 10.0,
        )
    except TimeoutError as e:
        raise AssertionError(
            f"Tool '{tool_name}' not recorded within {timeout}s. Calls seen: {calls}",
        ) from e


def _iter_tool_calls_from_chat_history(chat_history: list[dict[str, Any]]):
    for msg in chat_history:
        tool_calls = msg.get("tool_calls") or []
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                yield tc


def get_code_act_tool_calls(handle: Any) -> list[str]:
    """Extract tool call names from a CodeActActor handle's chat history."""
    try:
        chat_history = list(getattr(handle, "chat_history", []) or [])
    except Exception:
        chat_history = []

    names: list[str] = []
    for tc in _iter_tool_calls_from_chat_history(chat_history):
        fn = tc.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.append(fn["name"])
            continue
        if isinstance(tc.get("name"), str):
            names.append(tc["name"])
            continue
    return names


def extract_code_act_execute_python_code_snippets(handle: Any) -> list[str]:
    """Extract the `code` field from execute_python_code tool calls (best-effort)."""
    try:
        chat_history = list(getattr(handle, "chat_history", []) or [])
    except Exception:
        chat_history = []

    snippets: list[str] = []
    for tc in _iter_tool_calls_from_chat_history(chat_history):
        fn = tc.get("function") or {}
        name = None
        args = None
        if isinstance(fn, dict):
            name = fn.get("name")
            args = fn.get("arguments")
        else:
            name = tc.get("name")
            args = tc.get("arguments")

        if name != "execute_python_code":
            continue

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = None
        if isinstance(args, dict):
            code = args.get("code")
            if isinstance(code, str) and code.strip():
                snippets.append(code)
    return snippets


def assert_code_act_tool_called(handle: Any, tool_name: str) -> None:
    names = get_code_act_tool_calls(handle)
    assert tool_name in set(names), f"Expected tool call '{tool_name}', saw: {names}"


def assert_code_act_function_manager_used(handle: Any) -> None:
    """Assert that CodeAct used at least one FunctionManager tool call."""
    names = get_code_act_tool_calls(handle)
    assert any(n.startswith("FunctionManager_") for n in names), (
        "Expected CodeAct to call at least one FunctionManager tool, "
        f"but saw tool calls: {names}"
    )


@asynccontextmanager
async def make_code_act_actor(
    *,
    impl: Literal["real", "simulated"],
    include_function_manager_tools: bool = False,
    function_manager: Optional["FunctionManager"] = None,
    primitives: Optional[Primitives] = None,
    exposed_managers: Optional[set[str]] = None,
) -> AsyncIterator[tuple[CodeActActor, Primitives, list[str]]]:
    """
    Create a CodeActActor wired to a provided Primitives in primitives-only mode.

    NOTE: IMPL selection ("real" vs "simulated") is controlled by the autouse fixtures
    in `tests/test_actor/test_state_managers/conftest.py`, keyed off test path.
    This argument is kept as an assertion/documentation aid.
    """
    primitives = primitives or Primitives()
    calls = instrument_basic_primitives_calls(primitives)

    env = StateManagerEnvironment(primitives, exposed_managers=exposed_managers)
    actor = CodeActActor(environments=[env], function_manager=function_manager)

    # Optionally strip FunctionManager tools to focus on on-the-fly routing via primitives.
    if not include_function_manager_tools:
        actor._tools = {"execute_python_code": actor._tools["execute_python_code"]}

    try:
        yield actor, primitives, calls
    finally:
        try:
            await actor.close()
        except Exception:
            pass


def assert_tool_called(handle: Any, tool_name: str) -> None:
    """Verify a specific tool was called by checking idempotency_cache."""
    cache_entries = [
        entry
        for entry in (getattr(handle, "idempotency_cache", {}) or {}).values()
        if (entry or {}).get("meta", {}).get("tool") == tool_name
    ]
    assert cache_entries, (
        f"Tool '{tool_name}' was not called. "
        f"Cache keys: {list((getattr(handle, 'idempotency_cache', {}) or {}).keys())}"
    )


def assert_memoized_function_used(
    handle: Any,
    function_name: str | None = None,
) -> None:
    """Verify actor used a memoized FunctionManager function via `can_compose=True`."""
    log_text = "\n".join(getattr(handle, "action_log", []) or [])

    used_llm_plan = "Generating plan from goal..." in log_text or (
        "Initial plan generated successfully." in log_text
    )
    assert used_llm_plan, (
        "Expected LLM plan generation (can_compose=True path). "
        f"Log tail:\n{log_text[-800:]}"
    )

    if function_name:
        plan_source = getattr(handle, "plan_source_code", None) or ""
        assert plan_source.strip(), (
            "Expected plan_source_code to be present for can_compose=True path. "
            f"Log tail:\n{log_text[-800:]}"
        )

        # Use AST to robustly check for calls and definitions
        import ast

        try:
            tree = ast.parse(plan_source)
        except SyntaxError as e:
            raise AssertionError(
                f"Plan source code has syntax errors: {e}\n"
                f"Plan tail:\n{plan_source[-800:]}",
            )

        # Collect all function calls
        called_functions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_functions.add(node.func.id)

        # Assertions using AST-based detection
        assert function_name in called_functions, (
            f"Expected plan to call '{function_name}(...)' but no call was found. "
            f"Called functions: {sorted(called_functions)}\n"
            f"Plan tail:\n{plan_source[-800:]}"
        )


def extract_action_log_entries(handle: Any, pattern: str) -> list[str]:
    """Extract action log entries matching a regex pattern."""
    return [
        line
        for line in (getattr(handle, "action_log", []) or [])
        if re.search(pattern, line)
    ]


async def wait_for_tool_call(handle: Any, tool_name: str, timeout: int = 60) -> None:
    """Poll until a specific tool appears in idempotency_cache."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if any(
            (entry or {}).get("meta", {}).get("tool") == tool_name
            for entry in (getattr(handle, "idempotency_cache", {}) or {}).values()
        ):
            return
        await asyncio.sleep(0.5)
    raise AssertionError(f"Tool '{tool_name}' not called within {timeout}s")


# ---------------------------------------------------------------------------
# Actor construction helpers (mirror legacy orchestrator-style tests: construct in test body)
# ---------------------------------------------------------------------------


class NoKeychainBrowser:
    """Minimal browser stub to prevent Keychain prompts during tests."""

    def __init__(self, *, url: str = "", screenshot: str = "") -> None:
        self._url = url
        self._screenshot = screenshot
        # Some codepaths introspect backend attributes; keep a simple object.
        self.backend = object()

    async def get_current_url(self) -> str:
        return self._url

    async def get_screenshot(self) -> str:
        return self._screenshot

    def stop(self) -> None:
        return None


async def _mock_observe(*args: Any, **kwargs: Any) -> Any:
    """Return a reasonable observation for browser plans without invoking real backends."""
    import inspect
    from typing import get_origin, get_args

    response_format = kwargs.get("response_format", str)
    try:
        from pydantic import BaseModel

        if isinstance(response_format, type) and issubclass(response_format, BaseModel):

            def create_default(annotation):
                """Recursively create default values for complex types."""
                origin = get_origin(annotation)
                args_tuple = get_args(annotation)

                # Handle Optional[X] (Union[X, None]) → extract X
                if origin is type(None):
                    return None
                if origin and type(None) in args_tuple:
                    # Optional[X] is Union[X, None], so get the non-None type
                    non_none_args = [a for a in args_tuple if a is not type(None)]
                    if non_none_args:
                        return create_default(non_none_args[0])
                    return None

                # Handle List[X] → return empty list
                if origin is list:
                    return []

                # Handle nested BaseModel
                if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                    # Recursively create nested model
                    nested_data = {}
                    for name, field in annotation.model_fields.items():
                        nested_data[name] = create_default(field.annotation)
                    return annotation(**nested_data)

                # Handle primitives
                if annotation is bool:
                    return True
                elif annotation is int:
                    return 0
                elif annotation is float:
                    return 0.0
                elif annotation is str:
                    return ""

                return None

            data: dict[str, Any] = {}
            for name, field in response_format.model_fields.items():
                data[name] = create_default(field.annotation)

            return response_format(**data)
    except Exception as e:
        # Log the exception for debugging
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"_mock_observe failed to create {response_format}: {e}",
            exc_info=True,
        )
    return "observed"


async def _mock_reason(*args: Any, **kwargs: Any) -> Any:
    """Avoid provider-specific calls inside ComputerPrimitives.reason (Vertex)."""
    import inspect
    from typing import get_origin, get_args

    response_format = kwargs.get("response_format", str)
    try:
        from pydantic import BaseModel

        if isinstance(response_format, type) and issubclass(response_format, BaseModel):

            def create_default(annotation):
                """Recursively create default values for complex types."""
                origin = get_origin(annotation)
                args_tuple = get_args(annotation)

                # Handle Optional[X] (Union[X, None]) → extract X
                if origin is type(None):
                    return None
                if origin is type(None) or (origin and type(None) in args_tuple):
                    # Optional[X] is Union[X, None], so get the non-None type
                    non_none_args = [a for a in args_tuple if a is not type(None)]
                    if non_none_args:
                        return create_default(non_none_args[0])
                    return None

                # Handle List[X] → return empty list
                if origin is list:
                    return []

                # Handle nested BaseModel
                if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                    # Recursively create nested model
                    nested_data = {}
                    for name, field in annotation.model_fields.items():
                        nested_data[name] = create_default(field.annotation)
                    return annotation(**nested_data)

                # Handle primitives
                if annotation is bool:
                    return False
                elif annotation is int:
                    return 0
                elif annotation is float:
                    return 0.0
                elif annotation is str:
                    return ""

                return None

            data: dict[str, Any] = {}
            for name, field in response_format.model_fields.items():
                data[name] = create_default(field.annotation)

            return response_format(**data)
    except Exception as e:
        # Log the exception for debugging
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"_mock_reason failed to create {response_format}: {e}",
            exc_info=True,
        )
    return "reasoned"


@asynccontextmanager
async def make_actor(
    *,
    impl: Literal["real", "simulated"],
    can_compose: bool = True,
    can_store: bool = False,
) -> AsyncIterator[HierarchicalActor]:
    """
    Create a HierarchicalActor with immediate browser mocks, for use inside tests.

    NOTE: IMPL selection ("real" vs "simulated") is controlled by the autouse fixtures
    in `tests/test_actor/test_state_managers/conftest.py`, keyed off test path.
    This argument is kept as an assertion/documentation aid.
    """
    actor = HierarchicalActor(
        headless=True,
        browser_mode="legacy",
        connect_now=False,
        can_compose=can_compose,
        can_store=can_store,
    )

    # Mock browser immediately (before any handle creation).
    actor.computer_primitives._browser = NoKeychainBrowser()
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value="acted")
    actor.computer_primitives.observe = AsyncMock(side_effect=_mock_observe)
    actor.computer_primitives.reason = AsyncMock(side_effect=_mock_reason)

    try:
        yield actor
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Steerability helpers (SteerableToolPane)
# ---------------------------------------------------------------------------


def pick_handle_id_by_origin_tool(
    handles: list[dict[str, Any]],
    *,
    origin_tool_prefix: str,
) -> str:
    """Pick a `handle_id` from `pane.list_handles()` output by matching origin_tool prefix."""
    match = next(
        (
            h
            for h in handles
            if str(h.get("origin_tool") or "").startswith(origin_tool_prefix)
        ),
        None,
    )
    if match is None:
        raise AssertionError(
            f"No handle found with origin_tool_prefix={origin_tool_prefix!r}. Handles={handles}",
        )
    return str(match["handle_id"])


def get_pane_events(obj: Any, *, n: int = 500) -> list[dict[str, Any]]:
    """Get recent pane events from either a pane-like object or a handle with `.pane`."""
    pane = getattr(obj, "pane", obj)
    return list(pane.get_recent_events(n=n) or [])


def get_pane_steering_events(
    obj: Any,
    *,
    n: int = 500,
    method: str | None = None,
) -> list[dict[str, Any]]:
    """Return pane events of type `steering_applied`, optionally filtered by method."""
    evs = [e for e in get_pane_events(obj, n=n) if e.get("type") == "steering_applied"]
    if method is None:
        return evs
    return [e for e in evs if (e.get("payload") or {}).get("method") == method]


async def wait_for_pane_handle_count(
    obj: Any,
    *,
    expected: int,
    timeout: float = 30.0,
    poll: float = 0.05,
) -> None:
    """Wait until pane registers at least `expected` handles."""
    pane = getattr(obj, "pane", obj)

    async def _has_n() -> bool:
        return len(await pane.list_handles()) >= expected

    await asyncio.wait_for(
        _wait_for_condition(_has_n, poll=poll, timeout=timeout),
        timeout=timeout + 10.0,
    )


async def wait_for_pane_event(
    obj: Any,
    *,
    predicate: Callable[[dict[str, Any]], bool],
    timeout: float = 30.0,
    poll: float = 0.05,
) -> dict[str, Any]:
    """Wait for a pane event matching `predicate` and return it."""

    found: dict[str, Any] | None = None

    async def _seen() -> bool:
        nonlocal found
        for e in get_pane_events(obj, n=500):
            if predicate(e):
                found = e
                return True
        return False

    await asyncio.wait_for(
        _wait_for_condition(_seen, poll=poll, timeout=timeout),
        timeout=timeout + 10.0,
    )
    assert found is not None
    return found


async def wait_for_pane_steering_event(
    obj: Any,
    *,
    handle_id: str | None = None,
    method: str | None = None,
    status: str | None = None,
    timeout: float = 30.0,
    poll: float = 0.05,
) -> dict[str, Any]:
    """Wait for a `steering_applied` event matching the given filters and return it."""

    def _pred(e: dict[str, Any]) -> bool:
        if e.get("type") != "steering_applied":
            return False
        if handle_id is not None and str(e.get("handle_id")) != str(handle_id):
            return False
        payload = e.get("payload") or {}
        if method is not None and payload.get("method") != method:
            return False
        if status is not None and payload.get("status") != status:
            return False
        return True

    return await wait_for_pane_event(
        obj,
        predicate=_pred,
        timeout=timeout,
        poll=poll,
    )


async def wait_for_clarification_event(
    obj: Any,
    *,
    timeout: float = 30.0,
    poll: float = 0.05,
) -> dict[str, Any]:
    """Wait for a `clarification` pane event and return it."""
    return await wait_for_pane_event(
        obj,
        predicate=lambda e: e.get("type") == "clarification",
        timeout=timeout,
        poll=poll,
    )


async def get_pending_clarification_count(obj: Any) -> int:
    """Return the number of pending clarifications currently indexed by the pane."""
    pane = getattr(obj, "pane", obj)
    pending = await pane.get_pending_clarifications()
    return len(pending)


def extract_clarification_details(event: dict[str, Any]) -> tuple[str, str, str]:
    """Extract (handle_id, call_id, question) from a clarification pane event."""
    handle_id = str(event.get("handle_id") or "")
    payload = event.get("payload") or {}
    call_id = str(payload.get("call_id") or "")
    question = str(payload.get("question") or "")
    return handle_id, call_id, question
