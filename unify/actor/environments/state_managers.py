"""State manager environment for CodeActActor.

Exposes state manager primitives (contacts, files, tasks, etc.) for use in
generated plan code via the `primitives` namespace.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Set

from unify.actor.environments.base import (
    BaseEnvironment,
    ToolMetadata,
    _ClarificationQueueInjector,
    build_filtered_method_docs,
)
from unify.function_manager.primitives import Primitives, PrimitiveScope, get_registry


class StateManagerEnvironment(BaseEnvironment):
    """State manager environment backed by scoped Primitives.

    Exposes state manager methods like `primitives.contacts.ask(...)` for use inside
    generated plan code.

    Parameters
    ----------
    primitives : Primitives | None
        The Primitives instance to wrap. If None, a default instance exposing
        all managers is created. The instance is already scoped at construction
        time via ``Primitives(primitive_scope=...)``.
    allowed_methods : set[str] | None
        Optional set of fully-qualified method names to expose (e.g.,
        ``{"primitives.contacts.ask", "primitives.tasks.update"}``). When
        set, only these methods appear in ``get_tools()`` and
        ``get_prompt_context()``. When ``None`` (default), all methods
        from scoped managers are exposed.
    clarification_up_q : asyncio.Queue | None
        Queue for sending clarification requests to the user.
    clarification_down_q : asyncio.Queue | None
        Queue for receiving clarification responses from the user.
    """

    def __init__(
        self,
        primitives: Optional[Primitives] = None,
        *,
        allowed_methods: Optional[Set[str]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ):
        primitives = primitives or Primitives()

        super().__init__(
            instance=primitives,
            namespace="primitives",
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
        self._primitives = primitives
        self._primitive_scope = primitives.primitive_scope
        self._allowed_methods = frozenset(allowed_methods) if allowed_methods else None
        self._registry = get_registry()
        # Tools are not filtered — everything stays callable.
        # `prompt_documented_names` tells the actor's search-overlap
        # exclusion what the prompt documents inline. In the default
        # (unfiltered) mode no method docs are inlined at all, so nothing
        # is excluded and every primitive — including the core `ask` /
        # `update` methods — stays discoverable via FunctionManager search.
        # In the filtered (`allowed_methods`) mode the docs are fully
        # inlined, so the attribute stays None and the actor excludes all
        # of this environment's tools from search.
        self.prompt_documented_names: Optional[frozenset[str]] = None
        if self._allowed_methods is None:
            self.prompt_documented_names = frozenset()

    @property
    def namespace(self) -> str:
        return "primitives"

    @property
    def primitive_scope(self) -> PrimitiveScope:
        """The scope controlling which managers are exposed."""
        return self._primitive_scope

    def get_instance(self) -> Primitives:
        """Return the primitives instance."""
        return self._primitives

    def get_sandbox_instance(self) -> Any:
        """Return the instance for sandbox injection.

        The Primitives instance is already scoped, so no additional filtering needed.
        Optionally wraps for clarification queue injection.
        """
        instance: Any = self._primitives

        # Optionally wrap for clarification queue injection.
        if getattr(self, "_clarification_up_q", None) is None:
            return instance
        return _ClarificationQueueInjector(
            target=instance,
            clarification_up_q=self._clarification_up_q,
            clarification_down_q=self._clarification_down_q,
        )

    def get_tools(self) -> Dict[str, ToolMetadata]:
        """Get tool metadata for exposed managers."""
        # IMPORTANT: We are intentionally conservative with purity:
        # - Only clearly read-only methods are treated as pure (cacheable).
        # - Unknown methods default to impure to avoid incorrectly caching side effects.
        pure_methods = {
            "ask",
            "ask_about_file",
            "get",
            "list",
            "search",
            "exists",
            "parse",
            "preview",
            "reduce",
            "filter_files",
            "search_files",
            "describe",
            "list_columns",
            "get_run_event_children",
            "get_run_event",
        }

        tools: Dict[str, ToolMetadata] = {}

        for alias in sorted(self._primitive_scope.scoped_managers):
            method_names = self._registry.primitive_methods(manager_alias=alias)
            for method_name in method_names:
                fq_name = f"{self.namespace}.{alias}.{method_name}"
                if (
                    self._allowed_methods is not None
                    and fq_name not in self._allowed_methods
                ):
                    continue
                tools[fq_name] = ToolMetadata(
                    name=fq_name,
                    is_impure=(method_name not in pure_methods),
                    is_steerable=True,
                    docstring=None,
                    signature=None,
                    function_id=self._registry.get_function_id(alias, method_name),
                    function_context="primitive",
                )

        return tools

    def get_prompt_context(self) -> str:
        """Generate self-contained prompt context: rules, method docs, and examples."""
        parts: list[str] = []

        parts.append("""\
### State Manager Rules

- **Do not answer from scratch when `primitives` is available**: prefer \
calling the relevant state manager via `await primitives.<manager>.ask(...)` \
over answering purely from memory — even when you think you "already know" \
the answer. Treat manager return values as the primary ground truth.

- **Read vs write**: `.ask(...)` is typically **pure** (read-only); \
`.update(...)`, `.execute(...)`, `.refactor(...)` are **impure** (they \
mutate state or start work).

- **Mutation methods are self-contained**: go straight to `.update(...)` / \
`.execute(...)` / `.refactor(...)` — do NOT first call `.ask(...)` on the \
**same manager** to check existing state (mutation methods already inspect \
existing records). Bundle the full intent, including any "check if exists" \
logic, into the mutation call's natural-language `text` argument.

- **Steerable handles**: Manager calls are nested tool loops returning \
`SteerableToolHandle` objects for in-flight control. Default to \
**returning the handle as the last expression** of `execute_code` so \
outer-loop steering/progress stays available; await `.result()` only for \
immediate in-code composition. If intent is neutral or uncertain, default \
to returning the handle. If a manager asks for clarification, answer via \
the handle's API.

- **Progress notifications**: surface progress with the `send_notification` \
tool between `execute_code` blocks, never from inside generated code; keep \
messages user-facing and high-level (diagnostics belong in stdlib `logging` \
with `PHASE`/`SKIP`/`SOFT_FAIL` markers), and do not announce the final \
result this way — your response text handles that.

  **SteerableToolHandle API:**

  | Method | Returns | Purpose |
  |--------|---------|---------|
  | `await handle.result()` | `str` | Wait for the final result |
  | `await handle.ask(question)` | `SteerableToolHandle` | Query status |
  | `await handle.interject(message)` | `None` | Inject corrections mid-flight |
  | `await handle.pause()` | `str | None` | Pause at the next safe point |
  | `await handle.resume()` | `str | None` | Resume a paused operation |
  | `await handle.stop(reason=None)` | `None` | Terminate immediately |
  | `handle.done()` | `bool` | Completed? |

  ```python
  handle = await primitives.tasks.execute(task_id=123)
  await handle.interject("Also include the Q2 numbers")  # mid-flight
  result = await handle.result()  # wait for completion
  ```""")

        if self._allowed_methods is not None:
            # Per-method filtering: build method docs only for allowed methods.
            parts.append(self._build_filtered_method_docs())
        else:
            # Routing overview + discovery/introspection pointer; the
            # method surface is discovered via search and read via
            # help()/inspect.signature at run time.
            registry_ctx = self._registry.prompt_context(self._primitive_scope)
            if registry_ctx:
                parts.append(registry_ctx)

        return "\n\n".join(p for p in parts if p and p.strip())

    def _build_filtered_method_docs(self) -> str:
        """Build method-level documentation for only the allowed methods."""
        assert self._allowed_methods is not None
        return build_filtered_method_docs(
            self._allowed_methods,
            self.namespace,
            exposed_aliases=self._primitive_scope.scoped_managers,
        )

    async def capture_state(self) -> Dict[str, Any]:
        """State manager state is primarily evidenced via return values."""
        return {
            "type": "return_value",
            "note": "State manager evidence is captured via function return values.",
        }
