from __future__ import annotations

import asyncio
import functools
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal, Optional

from tests.async_helpers import _wait_for_condition
from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments import StateManagerEnvironment
from unity.function_manager.primitives import Primitives

try:
    from unity.function_manager.function_manager import FunctionManager
except Exception:  # pragma: no cover
    FunctionManager = Any  # type: ignore[misc,assignment]


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

    # ``orig`` may be a bound method (on a real manager) or a plain function
    # (returned by _AsyncPrimitiveWrapper.__getattr__).  In both cases it is
    # already fully callable without an extra ``self`` argument, so the
    # wrapper simply records the call and delegates.
    is_async = asyncio.iscoroutinefunction(orig)

    if is_async:

        @functools.wraps(orig)
        async def _traced(*args: Any, **kwargs: Any) -> Any:
            sink.append(fq_tool_name)
            return await orig(*args, **kwargs)

    else:

        @functools.wraps(orig)
        def _traced(*args: Any, **kwargs: Any) -> Any:
            sink.append(fq_tool_name)
            return orig(*args, **kwargs)

    setattr(manager, method_name, _traced)


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
        (
            "data",
            [
                "filter",
                "search",
                "reduce",
                "join",
                "insert_rows",
                "update_rows",
                "delete_rows",
                "vectorize",
                "plot",
                "create_table",
                "describe_table",
            ],
        ),
        (
            "files",
            [
                "ask",
                "ask_about_file",
                "describe",
                "list_columns",
                "reduce",
                "filter_files",
                "search_files",
                "visualize",
            ],
        ),
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
    chat_history = list(handle.get_history() or [])

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


def extract_code_act_execute_code_snippets(handle: Any) -> list[str]:
    """Extract the `code` field from execute_code tool calls (best-effort)."""
    chat_history = list(handle.get_history() or [])

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

        if name != "execute_code":
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
    exposed_managers: Optional[set[str]] = None,
) -> AsyncIterator[tuple[CodeActActor, Primitives, list[str]]]:
    """
    Create a CodeActActor wired to a provided Primitives in primitives-only mode.

    NOTE: IMPL selection ("real" vs "simulated") is controlled by the autouse fixtures
    in `tests/actor/state_managers/conftest.py`, keyed off test path.
    This argument is kept as an assertion/documentation aid.
    """
    if exposed_managers:
        from unity.function_manager.primitives import PrimitiveScope

        scope = PrimitiveScope(scoped_managers=frozenset(exposed_managers))
        primitives = Primitives(primitive_scope=scope)
    else:
        primitives = Primitives()
    calls = instrument_basic_primitives_calls(primitives)
    env = StateManagerEnvironment(primitives)
    actor = CodeActActor(environments=[env], function_manager=function_manager)

    # Optionally strip FunctionManager tools to focus on on-the-fly routing via primitives.
    if not include_function_manager_tools:
        act_tools = actor.get_tools("act")
        actor.add_tools(
            "act",
            {"execute_code": act_tools["execute_code"]},
        )

    try:
        yield actor, primitives, calls
    finally:
        try:
            await actor.close()
        except Exception:
            pass
