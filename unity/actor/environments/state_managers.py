"""State manager environment for CodeActActor.

Exposes state manager primitives (contacts, files, tasks, etc.) for use in
generated plan code via the `primitives` namespace.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from unity.actor.environments.base import (
    BaseEnvironment,
    ToolMetadata,
    _ClarificationQueueInjector,
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
    clarification_up_q : asyncio.Queue | None
        Queue for sending clarification requests to the user.
    clarification_down_q : asyncio.Queue | None
        Queue for receiving clarification responses from the user.
    """

    def __init__(
        self,
        primitives: Optional[Primitives] = None,
        *,
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
            # Skip ComputerPrimitives; those belong to the `computer_primitives` environment.
            # Note: "computer" is not in VALID_MANAGER_ALIASES so this is a defensive check.
            if alias == "computer":
                continue

            method_names = self._registry.primitive_methods(manager_alias=alias)
            for method_name in method_names:
                fq_name = f"{self.namespace}.{alias}.{method_name}"
                tools[fq_name] = ToolMetadata(
                    name=fq_name,
                    is_impure=(method_name not in pure_methods),
                    is_steerable=True,
                    docstring=None,
                    signature=None,
                )

        return tools

    def get_prompt_context(self) -> str:
        """Generate prompt context from the registry for exposed managers."""
        return self._registry.prompt_context(self._primitive_scope)

    async def capture_state(self) -> Dict[str, Any]:
        """State manager state is primarily evidenced via return values."""
        return {
            "type": "return_value",
            "note": "State manager evidence is captured via function return values.",
        }
