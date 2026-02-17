from __future__ import annotations

import inspect
from typing import Any, Dict, Optional
import asyncio

from unity.actor.environments.base import (
    BaseEnvironment,
    ToolMetadata,
    build_filtered_method_docs,
)
from unity.function_manager.primitives import ComputerPrimitives, get_registry


class ComputerEnvironment(BaseEnvironment):
    """Computer (web/desktop) control environment backed by `ComputerPrimitives`.

    Exposes web control methods like `primitives.computer.act(instruction)` for use inside
    generated plan code.  Lives under the unified ``primitives`` namespace alongside
    state managers and actor delegation.

    Parameters
    ----------
    computer_primitives : ComputerPrimitives
        The backend instance to wrap.
    allowed_methods : set[str] | None
        Optional set of fully-qualified method names to expose (e.g.,
        ``{"primitives.computer.act", "primitives.computer.observe"}``).
        When set, only these methods appear in ``get_tools()`` and
        ``get_prompt_context()``.  When ``None`` (default), all methods
        are exposed.
    """

    NAMESPACE = "primitives"
    MANAGER_ALIAS = "computer"

    def __init__(
        self,
        computer_primitives: ComputerPrimitives,
        *,
        allowed_methods: Optional[set[str]] = None,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ):
        from unity.function_manager.primitives import Primitives, PrimitiveScope

        self._computer_primitives = computer_primitives
        self._allowed_methods = frozenset(allowed_methods) if allowed_methods else None
        primitives = Primitives(
            primitive_scope=PrimitiveScope(
                scoped_managers=frozenset({self.MANAGER_ALIAS}),
            ),
        )
        # Pre-seed so primitives.computer returns the caller-provided instance
        # (important when the instance is a mock or pre-configured singleton).
        primitives._managers[self.MANAGER_ALIAS] = computer_primitives
        super().__init__(
            instance=primitives,
            namespace=self.NAMESPACE,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )

    @property
    def namespace(self) -> str:
        return self.NAMESPACE

    def get_instance(self) -> Any:
        return self._instance

    def get_tools(self) -> Dict[str, ToolMetadata]:
        # Explicit categorization avoids brittle substring heuristics.
        # (This list should track the actual public methods exposed on ComputerPrimitives.)
        impure = {"navigate", "act"}
        steerable = (
            set()
        )  # computer primitives sometimes return handles, but actor proxies detect dynamically

        tool_names = [
            "navigate",
            "act",
            "observe",
            "query",
            "get_links",
            "get_content",
            "get_screenshot",
        ]

        registry = get_registry()
        tools: Dict[str, ToolMetadata] = {}
        for name in tool_names:
            fq_name = f"{self.NAMESPACE}.{self.MANAGER_ALIAS}.{name}"
            if (
                self._allowed_methods is not None
                and fq_name not in self._allowed_methods
            ):
                continue
            if not hasattr(self._computer_primitives, name):
                continue
            fn = getattr(self._computer_primitives, name)
            if not callable(fn):
                continue

            try:
                signature = str(inspect.signature(fn))
            except Exception:
                signature = None

            tools[fq_name] = ToolMetadata(
                name=fq_name,
                is_impure=name in impure,
                is_steerable=name in steerable,
                docstring=getattr(fn, "__doc__", None),
                signature=signature,
                function_id=registry.get_function_id(self.MANAGER_ALIAS, name),
                function_context="primitive",
            )

        return tools

    def get_prompt_context(self) -> str:
        """Generate self-contained prompt context: rules, method docs, and examples."""
        parts: list[str] = []

        parts.append(
            "### Viewing Computer State\n\n"
            "To see the current screen state after a computer action, call "
            "`get_screenshot()` and `display()` the result:\n\n"
            "```python\n"
            "screenshot = await primitives.computer.get_screenshot()\n"
            "display(screenshot)\n"
            "```\n\n"
            "`get_screenshot()` returns a PIL Image. `display()` renders it as "
            "visual output you can inspect on the next turn.\n\n"
            "Use **stateful sessions** for multi-step computer workflows "
            "(e.g., navigate then observe).",
        )

        parts.append(
            "### Progress Notifications for Computer Actions\n\n"
            "- Treat `primitives.computer.*` calls as potentially long-running by default.\n"
            "- Emit `notify({...})` before each major computer step (for example: navigate, act, observe).\n"
            "- If you await a computer step and continue with more work, emit a completion update with concrete progress.\n"
            "- Keep notification messages user-facing and high-level (what was accomplished and what happens next).\n"
            "- Avoid low-level diagnostics in notifications (internal IDs, schema/debug details, stack traces).",
        )

        if self._allowed_methods is not None:
            filtered_docs = build_filtered_method_docs(
                self._allowed_methods,
                self.NAMESPACE,
            )
            if filtered_docs:
                parts.append(filtered_docs)
        else:
            registry_ctx = get_registry().computer_prompt_context()
            if registry_ctx:
                parts.append(registry_ctx)

            from unity.actor.prompt_examples import get_computer_examples

            examples = get_computer_examples()
            if examples:
                parts.append(f"### Computer Examples\n\n{examples}")

        return "\n\n".join(p for p in parts if p and p.strip())

    async def capture_state(self) -> Dict[str, Any]:
        """Captures visual computer state (screenshot + URL)."""
        try:
            screenshot = await self._computer_primitives.backend.get_screenshot()
            url = await self._computer_primitives.backend.get_current_url()
            return {
                "type": "visual",
                "screenshot": screenshot,
                "url": url,
            }
        except Exception as e:
            return {
                "type": "visual",
                "error": str(e),
            }
