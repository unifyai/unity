"""State manager environment for CodeActActor.

Exposes state manager primitives (contacts, files, tasks, etc.) for use in
generated plan code via the `primitives` namespace.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Set

from unity.actor.environments.base import (
    BaseEnvironment,
    ToolMetadata,
    _ClarificationQueueInjector,
    build_filtered_method_docs,
)
from unity.function_manager.primitives import Primitives, PrimitiveScope, get_registry


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
            "visualize",
            "describe",
            "list_columns",
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

- **Do not answer from scratch when `primitives` is available**:
  - If the user asks an information question, prefer calling the relevant \
state manager via `await primitives.<manager>.ask(...)` instead of answering \
purely from memory.
  - This applies even when you think you "already know" the answer \
— use the manager as evidence/ground truth.

- **Read vs write**:
  - `await primitives.<manager>.ask(...)` is typically **pure** (read-only).
  - `await primitives.<manager>.update(...)`, `.execute(...)`, `.refactor(...)` \
are **impure** (they mutate state or start work).

- **Prefer return values as evidence**: treat return values from state managers \
as the primary ground truth.

- **Steerable handles**: Manager calls return `SteerableToolHandle` objects for \
in-flight control.
  You can either **await the result** for immediate use, or **return the handle \
as the last expression** of `execute_code` to hand steering control back to the \
outer loop (see `execute_code` docstring).
  Prefer returning the handle when the operation may be long-running or likely \
to need user steering (progress updates, corrections, cancellation). Prefer \
awaiting when you need the result immediately for additional logic in the same \
code block. If intent is neutral or uncertain, default to returning the handle \
and only await when same-block composition truly requires it.

- **Progress notifications around primitives calls**:
  - Treat `primitives.*` calls as potentially long-running by default.
  - Emit `notify({...})` before each primitives call so the outer loop can surface progress.
  - If you await a primitives call and continue with additional steps, emit another \
`notify({...})` with concrete completion details.
  - If you return a handle directly, send one kickoff notification before returning \
the handle.
  - Keep notifications user-facing and high-level; avoid internal diagnostics.

  **SteerableToolHandle API:**

  | Method | Returns | Purpose |
  |--------|---------|---------|
  | `await handle.result()` | `str` | Wait for the final result |
  | `await handle.ask(question)` | `SteerableToolHandle` | Query status without modifying execution |
  | `await handle.interject(message)` | `None` | Inject corrections or context mid-flight |
  | `await handle.pause()` | `str | None` | Pause at the next safe point |
  | `await handle.resume()` | `str | None` | Resume a paused operation |
  | `await handle.stop(reason=None)` | `None` | Terminate immediately |
  | `handle.done()` | `bool` | Check if execution has completed |

  ```python
  handle = await primitives.tasks.execute(task_id=123)
  result = await handle.result()  # wait for completion

  # Mid-flight steering (while handle is running):
  await handle.interject("Also include the Q2 numbers")
  await handle.pause()   # pause if needed
  await handle.resume()  # continue later
  await handle.stop()    # cancel if no longer needed
  ```""")

        if self._allowed_methods is not None:
            # Per-method filtering: build method docs only for allowed methods.
            parts.append(self._build_filtered_method_docs())
        else:
            # Full registry-generated context (all methods for scoped managers).
            registry_ctx = self._registry.prompt_context(self._primitive_scope)
            if registry_ctx:
                parts.append(registry_ctx)

            examples = self._registry.prompt_examples(self._primitive_scope)
            if examples:
                parts.append(f"### Implementation Examples\n\n{examples}")

        return "\n\n".join(p for p in parts if p and p.strip())

    def _build_filtered_method_docs(self) -> str:
        """Build method-level documentation for only the allowed methods."""
        assert self._allowed_methods is not None
        return build_filtered_method_docs(self._allowed_methods, self.namespace)

    async def capture_state(self) -> Dict[str, Any]:
        """State manager state is primarily evidenced via return values."""
        return {
            "type": "return_value",
            "note": "State manager evidence is captured via function return values.",
        }
