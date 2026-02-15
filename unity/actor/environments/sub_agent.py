"""Sub-agent execution environment for CodeActActor.

Provides a ``sub_agent`` namespace in the sandbox with a single ``run()``
method that spawns isolated inner CodeActActors for focused sub-tasks.
Because sub-agent invocations are regular sandbox code, they can be saved
as compositional functions for reuse.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, Optional, TYPE_CHECKING

from unity.actor.environments.base import BaseEnvironment, ToolMetadata
from unity.function_manager.primitives.registry import get_registry

if TYPE_CHECKING:
    from unity.common.async_tool_loop import SteerableToolHandle
    from unity.function_manager.function_manager import FunctionManager


class _SubAgentRunner:
    """Runtime object injected into the sandbox as ``sub_agent``.

    Constructed without parent context; the owning ``CodeActActor`` back-fills
    it via :meth:`_bind` after the environment dict is built.
    """

    def __init__(self) -> None:
        self._parent_environments: Optional[Dict[str, BaseEnvironment]] = None
        self._function_manager: Optional["FunctionManager"] = None
        self._parent_can_compose: bool = True
        self._parent_can_store: bool = True
        self._model: Optional[str] = None
        self._preprocess_msgs: Any = None
        self._prompt_caching: Any = None
        self._parent_timeout: float = 1000
        self._bound: bool = False

    def _bind(
        self,
        *,
        parent_environments: Dict[str, BaseEnvironment],
        function_manager: Optional["FunctionManager"],
        parent_can_compose: bool,
        parent_can_store: bool,
        model: Optional[str],
        preprocess_msgs: Any,
        prompt_caching: Any,
        parent_timeout: float,
    ) -> None:
        """Back-fill parent actor context after the environment dict is built."""
        self._parent_environments = parent_environments
        self._function_manager = function_manager
        self._parent_can_compose = parent_can_compose
        self._parent_can_store = parent_can_store
        self._model = model
        self._preprocess_msgs = preprocess_msgs
        self._prompt_caching = prompt_caching
        self._parent_timeout = parent_timeout
        self._bound = True

    async def run(
        self,
        task: str,
        *,
        prompt_functions: list[str] | None = None,
        discovery_scope: str | None = None,
        timeout: float | None = None,
        can_compose: bool = True,
        can_store: bool = False,
        can_spawn_sub_agents: bool = False,
        storage_check_on_return: bool = False,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> "SteerableToolHandle":
        """Spawn a sub-agent to work on a focused sub-task.

        The sub-agent is an independent CodeActActor with its own sandbox,
        prompt, and (optionally) a curated set of directly callable
        functions.  It returns a steerable handle, allowing the caller
        to monitor progress and steer (stop, pause, resume, interject) the
        sub-agent mid-flight.

        Sub-agents are **steerable** — once spawned, dynamic steering helpers
        appear (stop, pause, resume, interject) so you can monitor progress and
        redirect the sub-agent mid-flight, just like any other steerable handle.

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
        - The sub-task is trivial (single tool call) — the overhead of a
          sub-agent is not worth it.

        Best practices
        --------------
        - Write a **clear, self-contained task description**. The sub-agent
          does not see the parent's conversation or sandbox state. Include
          all relevant context in the task string.
        - Set an appropriate **timeout** for the expected complexity.

        Parameters
        ----------
        task : str
            A clear, self-contained description of what the sub-agent should
            accomplish. Be specific and include all necessary context, because
            the sub-agent does **not** share the parent agent's conversation
            history or session state.
        prompt_functions : list[str], optional
            Functions to place directly in the sub-agent's system prompt,
            making them immediately callable without any discovery step.

            Use this for the functions most critical to the sub-task —
            the ones you actively want the sub-agent to reach for.
            Because they appear directly in the prompt, they receive the
            sub-agent's full attention and are the first tools it will
            consider.

            Prompt-injected functions are automatically excluded from the
            sub-agent's FunctionManager search index, so they will not
            appear as duplicate results during discovery.

            Names use dotted-segment matching:

            - ``"primitives"`` — all state manager primitives
            - ``"primitives.contacts"`` — all contacts methods
            - ``"primitives.contacts.ask"`` — just contacts.ask
            - ``"alpha"`` — a specific stored function
            - ``"my_service"`` — all methods from a custom environment
            - ``"my_service.do_something"`` — a specific custom method

            Any function you have seen (in your prompt, in search
            results, or in your environment) can be listed here.
            Functions not listed are still discoverable via
            FunctionManager search (subject to ``discovery_scope``).

            When omitted, the sub-agent receives no prompt-injected
            functions and relies entirely on FunctionManager discovery.
        discovery_scope : str, optional
            A boolean filter expression that restricts which functions
            the sub-agent can discover via FunctionManager
            search/list/filter (e.g., ``"language == 'python'"`` or
            ``"'data' in docstring"``).

            The sub-agent automatically inherits the parent agent's
            existing scope.  This parameter strictly narrows that
            inherited scope further (ANDed with the parent's filter),
            so the sub-agent's discoverable function library is always
            a subset of the parent's.  Use this to keep the sub-agent
            focused on only the functions relevant to its specific task.
        timeout : float, optional
            Maximum seconds for the sub-agent to complete.  When omitted
            the sub-agent runs without a time limit.
        can_compose : bool, default True
            Whether the sub-agent can write and execute arbitrary code via
            ``execute_code``. Set to False to restrict the sub-agent to
            only discovering and executing stored functions.
            Capped by the parent agent's own ``can_compose`` setting.
        can_store : bool, default False
            Whether the sub-agent can persist new functions to the
            FunctionManager via ``FunctionManager_add_functions``.
            Capped by the parent agent's own ``can_store`` setting.
        can_spawn_sub_agents : bool, default False
            Whether the sub-agent can itself spawn deeper sub-agents.
            Use with caution to avoid excessive nesting.
        storage_check_on_return : bool, default False
            Whether a post-completion review loop should run to identify
            and store reusable functions from the sub-agent's trajectory.

        Returns
        -------
        SteerableToolHandle
            A live handle to the running sub-agent.  The handle supports
            mid-flight steering (stop, pause, resume, interject).  The
            final string result is surfaced when the sub-agent completes.
        """
        if not self._bound:
            raise RuntimeError(
                "SubAgentEnvironment has not been bound to a parent actor. "
                "Pass it in the environments list to a CodeActActor so that "
                "bind_parent_context() is called during initialization.",
            )

        from unity.actor.code_act_actor import (
            CodeActActor,
            _build_sub_agent_environments,
        )
        from unity.actor.execution import _PARENT_CHAT_CONTEXT

        # Read parent chat context from the ContextVar (set by execute_code).
        _parent_chat_context = _PARENT_CHAT_CONTEXT.get(None)

        # Privilege escalation prevention: cap inner permissions by parent's.
        effective_can_compose = can_compose and self._parent_can_compose
        effective_can_store = can_store and self._parent_can_store

        effective_timeout = timeout if timeout is not None else self._parent_timeout

        # ── Create a fresh FunctionManager for the sub-agent ──
        inner_fm = None
        if self._function_manager is not None:
            from unity.function_manager.function_manager import (
                FunctionManager as _FM,
            )

            parent_scope = self._function_manager.filter_scope
            if discovery_scope and parent_scope:
                combined_scope = f"({parent_scope}) and ({discovery_scope})"
            elif discovery_scope:
                combined_scope = discovery_scope
            else:
                combined_scope = parent_scope

            inner_fm = _FM(
                primitive_scope=self._function_manager.primitive_scope,
                filter_scope=combined_scope,
                exclude_primitive_ids=self._function_manager.exclude_primitive_ids,
                exclude_compositional_ids=self._function_manager.exclude_compositional_ids,
                include_primitives=self._function_manager._include_primitives,
            )

        # ── Build inner actor environments from prompt_functions patterns ──
        # Filter out the sub_agent namespace to prevent circular matching.
        assert self._parent_environments is not None
        filtered_parent_envs = {
            ns: env
            for ns, env in self._parent_environments.items()
            if ns != SubAgentEnvironment.NAMESPACE
        }

        if prompt_functions:
            inner_envs = _build_sub_agent_environments(
                environment=prompt_functions,
                parent_environments=filtered_parent_envs,
                function_manager=inner_fm,
            )
        else:
            inner_envs = []

        # If the caller wants nested sub-agent spawning, include a fresh
        # SubAgentEnvironment in the inner actor's environments.
        if can_spawn_sub_agents:
            inner_envs.append(SubAgentEnvironment())

        # ── Create inner CodeActActor ──
        inner_actor = CodeActActor(
            environments=inner_envs or [],
            function_manager=inner_fm,
            can_compose=bool(effective_can_compose),
            can_store=bool(effective_can_store),
            storage_check_on_return=bool(storage_check_on_return),
            timeout=effective_timeout,
            model=self._model,
            preprocess_msgs=self._preprocess_msgs,
            prompt_caching=self._prompt_caching,
        )

        handle = await inner_actor.act(
            task,
            clarification_enabled=True,
            _parent_chat_context=_parent_chat_context,
            _clarification_up_q=_clarification_up_q,
            _clarification_down_q=_clarification_down_q,
        )

        # Attach inner actor cleanup to the handle's lifecycle so
        # inner_actor.close() runs after the sub-agent completes.
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


class SubAgentEnvironment(BaseEnvironment):
    """Environment that provides sub-agent spawning via the ``sub_agent`` namespace.

    Injects a ``sub_agent`` object into the sandbox with a single ``run()``
    method for spawning isolated inner CodeActActors.

    Constructed without parent context; ``CodeActActor.__init__`` calls
    :meth:`bind_parent_context` to back-fill the runner after the
    environment dict is built.
    """

    NAMESPACE = "sub_agent"

    def __init__(
        self,
        *,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ) -> None:
        runner = _SubAgentRunner()
        super().__init__(
            instance=runner,
            namespace=self.NAMESPACE,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    def bind_parent_context(
        self,
        *,
        parent_environments: Dict[str, BaseEnvironment],
        function_manager: Optional["FunctionManager"],
        parent_can_compose: bool,
        parent_can_store: bool,
        model: Optional[str],
        preprocess_msgs: Any,
        prompt_caching: Any,
        parent_timeout: float,
    ) -> None:
        """Back-fill parent actor context after the environment dict is built."""
        self.get_instance()._bind(
            parent_environments=parent_environments,
            function_manager=function_manager,
            parent_can_compose=parent_can_compose,
            parent_can_store=parent_can_store,
            model=model,
            preprocess_msgs=preprocess_msgs,
            prompt_caching=prompt_caching,
            parent_timeout=parent_timeout,
        )

    def get_tools(self) -> Dict[str, ToolMetadata]:
        return {
            f"{self.NAMESPACE}.run": ToolMetadata(
                name=f"{self.NAMESPACE}.run",
                is_impure=True,
                is_steerable=True,
            ),
        }

    def get_prompt_context(self) -> str:
        """Generate prompt context from the ``run()`` method's docstring."""
        registry = get_registry()
        sig_str = registry._format_method_signature(
            _SubAgentRunner,
            "run",
        )
        full_doc = inspect.getdoc(_SubAgentRunner.run) or ""
        filtered_doc = registry._filter_internal_params_from_docstring(full_doc)

        lines = [f"### `{self.NAMESPACE}` — Sub-Agent Delegation\n"]
        lines.append(f"**`{self.NAMESPACE}.run{sig_str}`**")
        if filtered_doc:
            for doc_line in filtered_doc.splitlines():
                lines.append(f"  {doc_line}")
        return "\n".join(lines)

    async def capture_state(self) -> Dict[str, Any]:
        return {"type": "sub_agent"}
