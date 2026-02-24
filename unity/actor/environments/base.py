from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel


def matches_segment(pattern: str, canonical_name: str) -> bool:
    """Check if *pattern* matches *canonical_name* using dotted-segment rules.

    A pattern matches a canonical name if it is either an exact match or a
    dotted ancestor (i.e., a complete prefix up to a ``.`` boundary).

    Examples::

        matches_segment("primitives", "primitives.contacts.ask")      # True
        matches_segment("primitives.contacts", "primitives.contacts.ask")  # True
        matches_segment("primitives.contacts.ask", "primitives.contacts.ask")  # True
        matches_segment("primitives.con", "primitives.contacts.ask")   # False
        matches_segment("functions", "functions.alpha")                 # True
        matches_segment("functions.alpha", "functions.alpha")           # True
    """
    return canonical_name == pattern or canonical_name.startswith(pattern + ".")


def resolve_directly_callable(
    patterns: List[str],
    all_tool_names: Set[str],
) -> Set[str]:
    """Expand a list of dotted-segment patterns into matching canonical tool names.

    Args:
        patterns: List of patterns (e.g., ``["primitives.contacts", "alpha"]``).
        all_tool_names: Complete set of canonical tool names across all environments.

    Returns:
        Set of canonical tool names matched by the patterns.

    Raises:
        ValueError: If any pattern matches zero tool names (likely a typo or
            a function the agent hasn't encountered).
    """
    matched: Set[str] = set()
    for pat in patterns:
        hits = {name for name in all_tool_names if matches_segment(pat, name)}
        if not hits:
            raise ValueError(
                f"environment pattern {pat!r} did not match any known "
                f"tool. Available tools: {sorted(all_tool_names)}",
            )
        matched |= hits
    return matched


def build_filtered_method_docs(
    allowed_methods: frozenset[str],
    namespace: str = "primitives",
) -> str:
    """Build method-level documentation for only the specified fully-qualified methods.

    Reusable across all environment types (state managers, computer, actor).
    Uses the ``ToolSurfaceRegistry`` to introspect method signatures and
    docstrings for each allowed method.
    """
    from unity.function_manager.primitives.registry import get_registry

    registry = get_registry()

    allowed_aliases: dict[str, list[str]] = {}
    for fq in sorted(allowed_methods):
        parts = fq.split(".")
        if len(parts) != 3 or parts[0] != namespace:
            continue
        alias, method = parts[1], parts[2]
        allowed_aliases.setdefault(alias, []).append(method)

    if not allowed_aliases:
        return ""

    lines = ["### Method Reference\n"]
    for alias in sorted(allowed_aliases):
        spec = registry.get_manager_spec(alias)
        mgr_cls = (
            registry._load_manager_class(spec.primitive_class_path) if spec else None
        )

        lines.append(f"\n#### `{namespace}.{alias}`")
        if spec:
            lines.append(f"*{spec.domain}* — {spec.description}")

        for method_name in sorted(allowed_aliases[alias]):
            sig_str = registry._format_method_signature(mgr_cls, method_name)
            full_doc = registry._extract_method_docstring(mgr_cls, method_name)
            compact_doc = registry._extract_summary_and_params(full_doc)
            lines.append(f"\n**`.{method_name}{sig_str}`**")
            if compact_doc:
                for doc_line in compact_doc.splitlines():
                    lines.append(f"  {doc_line}")

    return "\n".join(lines)


class ToolMetadata(BaseModel):
    """Metadata describing a tool's behavior and safety characteristics.

    Attributes:
        name: Fully-qualified tool name as used in the Actor execution sandbox.
        is_impure: True if the tool can cause side effects.
        is_steerable: True if calling the tool may return a steerable handle.
        docstring: Tool documentation string (if available).
        signature: Human-readable signature string (if available).
        function_id: Optional cross-reference to a stored FunctionManager function.
            When set, indicates this environment tool corresponds to a function in
            the FunctionManager backend, enabling automatic exclusion from
            FunctionManager search/list/filter results to prevent overlap.
            Must be paired with ``function_context`` to identify which DB context
            the ID belongs to (IDs are only unique within a context).
        function_context: Which FunctionManager DB context ``function_id``
            belongs to. Required when ``function_id`` is set.
            ``"primitive"`` for state manager methods (``Functions/Primitives``),
            ``"compositional"`` for user-defined functions
            (``Functions/Compositional``).
    """

    name: str
    is_impure: bool
    is_steerable: bool = False
    docstring: Optional[str] = None
    signature: Optional[str] = None
    function_id: Optional[int] = None
    function_context: Optional[Literal["primitive", "compositional"]] = None


def _callable_accepts_clarification_kwargs(fn: Any) -> bool:
    """
    Return True if `fn` appears to accept clarification queue kwargs.

    We only inject queues into callables that declare `_clarification_up_q` /
    `_clarification_down_q` explicitly or accept `**kwargs`. This avoids breaking
    other async utilities (e.g., FileManager wrappers) that do not accept these
    keyword arguments.
    """
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False

    params = sig.parameters
    if "_clarification_up_q" in params or "_clarification_down_q" in params:
        return True

    for p in params.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True

    return False


class _ClarificationQueueInjector:
    """
    Lightweight wrapper that injects clarification queues into manager calls.

    This is intentionally minimal:
    - No caching
    - No logging
    - No pane registration
    - Just queue injection (when supported by the target callable)
    """

    _DO_NOT_WRAP_TYPES: tuple[type, ...] = (
        str,
        bytes,
        bytearray,
        int,
        float,
        bool,
        dict,
        list,
        tuple,
        set,
        frozenset,
        type(None),
    )

    def __init__(
        self,
        *,
        target: Any,
        clarification_up_q: asyncio.Queue[str],
        clarification_down_q: Optional[asyncio.Queue[str]],
    ):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_clar_up_q", clarification_up_q)
        object.__setattr__(self, "_clar_down_q", clarification_down_q)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)

        # Pass through private/dunder attributes directly.
        if name.startswith("_"):
            return attr

        # Wrap callables so we can inject queues at call time.
        if callable(attr):
            return self._wrap_callable(attr)

        # For nested objects (e.g. `primitives.contacts` returning a manager), return
        # another injector so `primitives.contacts.ask(...)` also gets queue injection.
        return self._maybe_wrap_object(attr)

    def __setattr__(self, name: str, value: Any) -> None:
        # Keep wrapper transparent to normal attribute assignment.
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._target, name, value)

    def __repr__(self) -> str:
        return f"<ClarificationQueueInjector target={type(self._target).__name__}>"

    def _maybe_wrap_object(self, obj: Any) -> Any:
        if isinstance(obj, self._DO_NOT_WRAP_TYPES):
            return obj
        if inspect.ismodule(obj):
            return obj
        if isinstance(obj, type):
            return obj
        if isinstance(obj, _ClarificationQueueInjector):
            return obj
        return _ClarificationQueueInjector(
            target=obj,
            clarification_up_q=self._clar_up_q,
            clarification_down_q=self._clar_down_q,
        )

    def _inject_queues(self, *, fn: Any, kwargs: Dict[str, Any]) -> None:
        if "_clarification_up_q" in kwargs or "_clarification_down_q" in kwargs:
            return
        if not _callable_accepts_clarification_kwargs(fn):
            return
        kwargs["_clarification_up_q"] = self._clar_up_q
        kwargs["_clarification_down_q"] = self._clar_down_q

    def _wrap_callable(self, fn: Any) -> Any:
        if asyncio.iscoroutinefunction(fn):

            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                self._inject_queues(fn=fn, kwargs=kwargs)
                return await fn(*args, **kwargs)

            return _async_wrapper

        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            self._inject_queues(fn=fn, kwargs=kwargs)
            return fn(*args, **kwargs)

        return _sync_wrapper


class BaseEnvironment(ABC):
    """Abstract interface for execution environments.

    An environment encapsulates a domain of tools (computer/web control, state managers,
    custom adapters) and provides:
    - a namespace to inject into the plan execution sandbox
    - metadata for tools (purity/steerability)
    - a prompt context section describing usage patterns for those tools

    NOTE: proxying/caching/logging is owned by the Actor, not the environment.
    """

    def __init__(
        self,
        *,
        instance: Any,
        namespace: str,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> None:
        self._instance = instance
        self._namespace = namespace
        self._clarification_up_q = clarification_up_q
        self._clarification_down_q = clarification_down_q

    @property
    def namespace(self) -> str:
        """Global variable name injected into the sandbox (e.g. "primitives")."""
        return self._namespace

    def get_instance(self) -> Any:
        """Return the object injected into the sandbox under `namespace`."""
        return self._instance

    def get_sandbox_instance(self) -> Any:
        """
        Return instance for sandbox injection.

        If clarification queues are configured, returns a lightweight wrapper
        that injects `_clarification_up_q` / `_clarification_down_q` into manager
        method calls (when supported).
        """
        instance = self.get_instance()
        clar_up_q = getattr(self, "_clarification_up_q", None)
        if clar_up_q is None:
            return instance

        clar_down_q = getattr(self, "_clarification_down_q", None)
        return _ClarificationQueueInjector(
            target=instance,
            clarification_up_q=clar_up_q,
            clarification_down_q=clar_down_q,
        )

    @abstractmethod
    def get_tools(self) -> Dict[str, ToolMetadata]:
        """Return metadata for tools exposed by this environment.

        The returned keys MUST be fully-qualified tool names as used in execution,
        so callers can look up metadata by the same string that appears in logs.
        """

    @abstractmethod
    def get_prompt_context(self) -> str:
        """Return Markdown-formatted rules/examples for using this environment."""

    @abstractmethod
    async def capture_state(self) -> Dict[str, Any]:
        """Capture environment-specific evidence for verification.

        This is used by the Actor's verification system to gather a structured
        snapshot of the environment's observable state before/after executing a
        plan function.

        Implementations should be best-effort and never raise; if state capture
        fails, return a structured error payload (e.g. `{"type": "...", "error": "..."}`).
        """


class _CompositeEnvironment(BaseEnvironment):
    """Merges multiple environments sharing the same namespace.

    When several environments share the ``"primitives"`` namespace (e.g.
    ``StateManagerEnvironment``, ``ComputerEnvironment``, ``ActorEnvironment``),
    this wrapper aggregates their tool metadata and prompt context while
    injecting a single ``Primitives`` instance into the sandbox.
    """

    def __init__(
        self,
        envs: List["BaseEnvironment"],
        *,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> None:
        self._envs = envs
        primary = self._build_primary_instance(envs)
        super().__init__(
            instance=primary,
            namespace=envs[0].namespace,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    @staticmethod
    def _build_primary_instance(envs: List["BaseEnvironment"]) -> Any:
        """Return the broadest-scoped Primitives instance from sub-envs."""
        from unity.function_manager.primitives import Primitives, PrimitiveScope

        merged_managers: Set[str] = set()
        for env in envs:
            instance = env.get_instance()
            if isinstance(instance, Primitives):
                merged_managers |= instance.primitive_scope.scoped_managers

        if merged_managers:
            return Primitives(
                primitive_scope=PrimitiveScope(
                    scoped_managers=frozenset(merged_managers),
                ),
            )
        # Fallback: first environment's instance.
        return envs[0].get_instance()

    @property
    def sub_environments(self) -> List["BaseEnvironment"]:
        """Expose wrapped environments for introspection."""
        return list(self._envs)

    def get_tools(self) -> Dict[str, ToolMetadata]:
        merged: Dict[str, ToolMetadata] = {}
        for env in self._envs:
            merged.update(env.get_tools())
        return merged

    def get_prompt_context(self) -> str:
        parts = [env.get_prompt_context() for env in self._envs]
        return "\n\n".join(p for p in parts if p and p.strip())

    async def capture_state(self) -> Dict[str, Any]:
        states: List[Dict[str, Any]] = []
        for env in self._envs:
            try:
                states.append(await env.capture_state())
            except Exception:
                pass
        if len(states) == 1:
            return states[0]
        return {"type": "composite", "states": states}
