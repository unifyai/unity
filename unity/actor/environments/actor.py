"""Actor execution environment for CodeActActor.

Provides an ``actor`` namespace in the sandbox with a single ``act()``
method that spawns isolated inner CodeActActors for focused sub-tasks.
Because actor invocations are regular sandbox code, they can be saved
as compositional functions for reuse.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from unity.actor.environments.base import BaseEnvironment, ToolMetadata
from unity.function_manager.primitives.registry import get_registry

if TYPE_CHECKING:
    from unity.common.async_tool_loop import SteerableToolHandle
    from unity.function_manager.function_manager import FunctionManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB-based environment resolution
# ---------------------------------------------------------------------------


def _build_scoped_fm(
    discovery_scope: str | None,
) -> "FunctionManager":
    """Build a fresh FunctionManager with an optional discovery filter.

    Constructs a default FunctionManager with all managers in scope.
    If *discovery_scope* is provided it is applied as the ``filter_scope``
    so the inner actor's search/list/filter results are restricted
    accordingly.
    """
    from unity.function_manager.function_manager import FunctionManager as _FM

    return _FM(
        filter_scope=discovery_scope,
        include_primitives=True,
    )


def _build_environments_from_db(
    prompt_functions: list[str] | None,
    function_manager: Optional["FunctionManager"],
) -> List[BaseEnvironment]:
    """Resolve *prompt_functions* patterns against the FunctionManager DB.

    Returns a list of environments for the inner actor. Only state-manager
    primitives and compositional functions are supported; computer primitives
    and custom environments are out of scope.
    """
    from unity.actor.environments.base import resolve_directly_callable
    from unity.actor.environments.function_store import FunctionStoreEnvironment
    from unity.actor.environments.state_managers import StateManagerEnvironment
    from unity.function_manager.primitives import Primitives, PrimitiveScope

    if not prompt_functions:
        return []

    # Collect all known names from the DB via FunctionManager.
    all_known_names: set[str] = set()
    if function_manager is not None:
        try:
            fm_listing = function_manager.list_functions()
            all_known_names = set(fm_listing.keys())
        except Exception:
            pass

    if not all_known_names:
        return []

    # Resolve dotted-segment patterns to canonical names.
    matched_names = resolve_directly_callable(prompt_functions, all_known_names)

    # Bucket by type: primitives (dotted, starting with "primitives.") vs compositional (bare).
    primitive_methods: set[str] = set()
    fm_function_names: list[str] = []

    for name in matched_names:
        if name.startswith("primitives."):
            primitive_methods.add(name)
        elif "." not in name:
            fm_function_names.append(name)
        else:
            logger.debug(
                "Skipping dotted name %r in prompt_functions — only "
                "primitives.* and bare compositional names can be resolved from DB",
                name,
            )

    # Build environments.
    envs: list[BaseEnvironment] = []

    if primitive_methods:
        needed_managers: set[str] = set()
        for fq in primitive_methods:
            parts = fq.split(".")
            if len(parts) >= 2:
                needed_managers.add(parts[1])
        scope = PrimitiveScope(scoped_managers=frozenset(needed_managers))
        envs.append(
            StateManagerEnvironment(
                Primitives(primitive_scope=scope),
                allowed_methods=primitive_methods,
            ),
        )

    if fm_function_names and function_manager is not None:
        envs.append(
            FunctionStoreEnvironment(
                function_manager,
                function_names=fm_function_names,
            ),
        )

    return envs


# ---------------------------------------------------------------------------
# Actor runner (injected into the sandbox as ``actor``)
# ---------------------------------------------------------------------------


class _ActorRunner:
    """Runtime object injected into the sandbox as ``actor``.

    Fully stateless: constructs its own FunctionManager from defaults.
    No ambient ContextVars required -- can be called in total isolation.
    """

    _PRIMITIVE_METHODS = ("act",)

    async def act(
        self,
        request: str,
        *,
        guidelines: str | None = None,
        prompt_functions: list[str] | None = None,
        discovery_scope: str | None = None,
        timeout: float | None = None,
        can_compose: bool = True,
        can_store: bool = False,
        can_spawn_sub_agents: bool = False,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> "SteerableToolHandle":
        """Spawn an actor to work on a focused sub-task.

        The actor is an independent CodeActActor with its own sandbox,
        prompt, and (optionally) a curated set of directly callable
        functions.  It returns a steerable handle, allowing the caller
        to monitor progress and steer (stop, pause, resume, interject) the
        actor mid-flight.

        Actors are **steerable** — once spawned, dynamic steering helpers
        appear (stop, pause, resume, interject) so you can monitor progress and
        redirect the actor mid-flight, just like any other steerable handle.

        When to use
        -----------
        - The overall task decomposes naturally into independent sub-problems
          that benefit from focused, isolated reasoning.
        - A sub-task requires multi-step work that would clutter or distract
          the main agent's context window.
        - You want to isolate a sub-task's execution state (sessions,
          variables) from the main agent's sandbox.
        - You want to keep the main agent's context clean for high-level
          orchestration.

        When NOT to use
        ---------------
        - The task is simple enough to handle directly with ``execute_code``.
        - You need intermediate results from the sub-task to inform logic in
          the same code block (use ``execute_code`` with stateful sessions).
        - The sub-task is trivial (single tool call) — the overhead of an
          actor is not worth it.

        Best practices
        --------------
        - Write a **clear, self-contained request description**. The actor
          does not see the parent's conversation or sandbox state. Include
          all relevant context in the request string.
        - Use **guidelines** to inject persistent behavioral directives
          (e.g., "Be extremely thorough", "Respond in JSON") that the
          actor must follow throughout its session.
        - Set an appropriate **timeout** for the expected complexity.

        Parameters
        ----------
        request : str
            A clear, self-contained description of what the actor should
            accomplish. Be specific and include all necessary context, because
            the actor does **not** share the parent agent's conversation
            history or session state.
        guidelines : str, optional
            Persistent behavioral directives injected into the actor's
            system prompt. Unlike the request (which describes *what* to
            do), guidelines describe *how* to behave throughout the
            session — e.g., output format, thoroughness level, tone, or
            constraints. When omitted, no additional guidelines are set.
        prompt_functions : list[str], optional
            Functions to place directly in the actor's system prompt,
            making them immediately callable without any discovery step.

            Use this for the functions most critical to the sub-task —
            the ones you actively want the actor to reach for.
            Because they appear directly in the prompt, they receive the
            actor's full attention and are the first tools it will
            consider.

            Prompt-injected functions are automatically excluded from the
            actor's FunctionManager search index, so they will not
            appear as duplicate results during discovery.

            Names use dotted-segment matching against the function names
            stored in the database:

            - ``"primitives"`` — all state manager primitives
            - ``"primitives.contacts"`` — all contacts methods
            - ``"primitives.contacts.ask"`` — just contacts.ask
            - ``"alpha"`` — a specific stored function

            Any function stored in the database (primitives or
            compositional) can be listed here.
            Functions not listed are still discoverable via
            FunctionManager search (subject to ``discovery_scope``).

            When omitted, the actor receives no prompt-injected
            functions and relies entirely on FunctionManager discovery.
        discovery_scope : str, optional
            A boolean filter expression that restricts which functions
            the actor can discover via FunctionManager
            search/list/filter (e.g., ``"language == 'python'"`` or
            ``"'data' in docstring"``).

            When provided, only functions matching this expression are
            visible to the actor's discovery tools.  When omitted, all
            stored functions are discoverable.
        timeout : float, optional
            Maximum seconds for the actor to complete.  When omitted
            the actor runs without a time limit.
        can_compose : bool, default True
            Whether the actor can write and execute arbitrary code via
            ``execute_code``. Set to False to restrict the actor to
            only discovering and executing stored functions.
        can_store : bool, default False
            Whether a post-completion review loop should run to identify
            and store reusable functions and guidance from the actor's
            trajectory. Storage is always deferred to a dedicated second
            loop after the main task completes.
        can_spawn_sub_agents : bool, default False
            Whether the actor can itself spawn deeper actors.
            Use with caution to avoid excessive nesting.

        Returns
        -------
        SteerableToolHandle
            A live handle to the running actor.  The handle supports
            mid-flight steering (stop, pause, resume, interject).  The
            final string result is surfaced when the actor completes.
        """
        from unity.actor.code_act_actor import CodeActActor
        from unity.actor.execution import _PARENT_CHAT_CONTEXT

        effective_timeout = timeout if timeout is not None else 1000

        # Pick up parent chat context if running inside a CodeActActor
        # sandbox.  Gracefully degrades to None in standalone usage.
        _parent_chat_context = _PARENT_CHAT_CONTEXT.get(None)

        # Build a fresh FM scoped by discovery_scope (no parent inheritance).
        inner_fm = _build_scoped_fm(discovery_scope)

        # Resolve prompt_functions against DB.
        inner_envs = _build_environments_from_db(prompt_functions, inner_fm)

        # Optionally allow nested actor spawning.
        if can_spawn_sub_agents:
            inner_envs.append(ActorEnvironment())

        # Create inner CodeActActor.
        inner_actor = CodeActActor(
            environments=inner_envs,
            function_manager=inner_fm,
            can_compose=bool(can_compose),
            can_store=bool(can_store),
            timeout=effective_timeout,
        )

        handle = await inner_actor.act(
            request,
            guidelines=guidelines,
            clarification_enabled=True,
            _parent_chat_context=_parent_chat_context,
            _clarification_up_q=_clarification_up_q,
            _clarification_down_q=_clarification_down_q,
        )

        # Attach inner actor cleanup to the handle's lifecycle so
        # inner_actor.close() runs after the actor completes.
        # The handle may be wrapped by a logging proxy (_LoggedHandle)
        # with __slots__, so patch the unwrapped inner handle directly.
        _unwrapped = getattr(handle, "__wrapped__", handle)
        _original_result = _unwrapped.result

        async def _result_with_cleanup():
            try:
                return await _original_result()
            finally:
                await inner_actor.close()

        _unwrapped.result = _result_with_cleanup  # type: ignore[assignment]

        return handle


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------


class ActorEnvironment(BaseEnvironment):
    """Environment that provides actor spawning via the ``actor`` namespace.

    Injects an ``actor`` object into the sandbox with a single ``act()``
    method for spawning isolated inner CodeActActors.
    """

    NAMESPACE = "actor"

    def __init__(
        self,
        *,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> None:
        runner = _ActorRunner()
        super().__init__(
            instance=runner,
            namespace=self.NAMESPACE,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    def get_tools(self) -> Dict[str, ToolMetadata]:
        registry = get_registry()
        return {
            f"{self.NAMESPACE}.act": ToolMetadata(
                name=f"{self.NAMESPACE}.act",
                is_impure=True,
                is_steerable=True,
                function_id=registry.get_function_id("actor", "act"),
                function_context="primitive",
            ),
        }

    def get_prompt_context(self) -> str:
        """Generate prompt context from the ``act()`` method's docstring."""
        registry = get_registry()
        sig_str = registry._format_method_signature(
            _ActorRunner,
            "act",
        )
        full_doc = inspect.getdoc(_ActorRunner.act) or ""
        filtered_doc = registry._filter_internal_params_from_docstring(full_doc)

        lines = [f"### `{self.NAMESPACE}` — Actor Delegation\n"]
        lines.append(f"**`{self.NAMESPACE}.act{sig_str}`**")
        if filtered_doc:
            for doc_line in filtered_doc.splitlines():
                lines.append(f"  {doc_line}")
        return "\n".join(lines)

    async def capture_state(self) -> Dict[str, Any]:
        return {"type": "actor"}
