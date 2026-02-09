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


def instrument_basic_primitives_calls(primitives: Primitives) -> list[str]:
    """Wrap every exposed primitive method to record which primitives were invoked.

    Uses the ``ToolSurfaceRegistry`` to discover all primitive methods for
    each manager in scope, so the tracing list is always in sync with the
    actual primitive surface -- no manual curation needed.

    For managers that are returned directly (not wrapped by
    ``_AsyncPrimitiveWrapper``), we must avoid mutating the singleton
    because other code (e.g. ``TaskScheduler.__init__``) introspects
    the manager's bound methods.  Instead, we insert a thin tracing
    proxy into the ``Primitives._managers`` cache so tracing is
    transparent to both the CodeActActor sandbox and the manager itself.
    """
    from unity.function_manager.primitives.registry import get_registry

    calls: list[str] = []
    registry = get_registry()
    scope = primitives.primitive_scope

    targets: list[tuple[str, list[str]]] = []
    for alias in sorted(scope.scoped_managers):
        methods = registry.primitive_methods(manager_alias=alias)
        if methods:
            targets.append((alias, methods))

    for manager_attr, methods in targets:
        try:
            # Eagerly resolve so the manager is cached in primitives._managers.
            mgr = getattr(primitives, manager_attr)
        except Exception:
            continue

        # Build a tracing proxy that intercepts only the listed methods
        # and forwards everything else to the real manager.
        proxy = _TracingProxy(mgr, methods, manager_attr, calls)
        primitives._managers[manager_attr] = proxy
    return calls


class _TracingProxy:
    """Lightweight proxy that records calls to specified methods.

    All attribute access is forwarded to the wrapped manager, so the
    proxy is transparent to code that inspects the manager (e.g.
    ``methods_to_tool_dict`` checking ``__self__``).
    """

    __slots__ = ("_real", "_traced_methods")

    def __init__(
        self,
        real_manager: Any,
        method_names: list[str],
        manager_alias: str,
        sink: list[str],
    ) -> None:
        object.__setattr__(self, "_real", real_manager)
        traced: dict[str, Any] = {}
        for m in method_names:
            orig = getattr(real_manager, m, None)
            if orig is None or not callable(orig):
                continue
            fq = f"primitives.{manager_alias}.{m}"
            if asyncio.iscoroutinefunction(orig):

                @functools.wraps(orig)
                async def _t(*a, _fq=fq, _o=orig, **kw):
                    sink.append(_fq)
                    return await _o(*a, **kw)

                traced[m] = _t
            else:

                @functools.wraps(orig)
                def _t(*a, _fq=fq, _o=orig, **kw):
                    sink.append(_fq)
                    return _o(*a, **kw)

                traced[m] = _t
        object.__setattr__(self, "_traced_methods", traced)

    def __getattr__(self, name: str) -> Any:
        traced = object.__getattribute__(self, "_traced_methods")
        if name in traced:
            return traced[name]
        return getattr(object.__getattribute__(self, "_real"), name)


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
