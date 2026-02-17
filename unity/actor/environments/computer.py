from __future__ import annotations

import inspect
from typing import Any, Dict, Optional
import asyncio

from unity.actor.environments.base import BaseEnvironment, ToolMetadata
from unity.function_manager.primitives import ComputerPrimitives, get_registry


class ComputerEnvironment(BaseEnvironment):
    """Computer (web/desktop) control environment backed by `ComputerPrimitives`.

    Exposes web control methods like `computer_primitives.act(instruction)` for use inside
    generated plan code.
    """

    def __init__(
        self,
        computer_primitives: ComputerPrimitives,
        *,
        clarification_up_q: Optional[asyncio.Queue[str]] = None,
        clarification_down_q: Optional[asyncio.Queue[str]] = None,
    ):
        super().__init__(
            instance=computer_primitives,
            namespace="computer_primitives",
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
        )
        self._computer_primitives = computer_primitives

    @property
    def namespace(self) -> str:
        return "computer_primitives"

    def get_instance(self) -> ComputerPrimitives:
        return self._computer_primitives

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
            if not hasattr(self._computer_primitives, name):
                continue
            fn = getattr(self._computer_primitives, name)
            if not callable(fn):
                continue

            try:
                signature = str(inspect.signature(fn))
            except Exception:
                signature = None

            tools[f"{self.namespace}.{name}"] = ToolMetadata(
                name=f"{self.namespace}.{name}",
                is_impure=name in impure,
                is_steerable=name in steerable,
                docstring=getattr(fn, "__doc__", None),
                signature=signature,
                function_id=registry.get_function_id("computer", name),
                function_context="primitive",
            )

        return tools

    def get_prompt_context(self) -> str:
        """Generate self-contained prompt context: rules, method docs, and examples."""
        from unity.function_manager.primitives import get_registry
        from unity.actor.prompt_examples import get_computer_examples

        parts: list[str] = []

        parts.append(
            "### Viewing Computer State\n\n"
            "To see the current screen state after a computer action, call "
            "`get_screenshot()` and `display()` the result:\n\n"
            "```python\n"
            "screenshot = await computer_primitives.get_screenshot()\n"
            "display(screenshot)\n"
            "```\n\n"
            "`get_screenshot()` returns a PIL Image. `display()` renders it as "
            "visual output you can inspect on the next turn.\n\n"
            "Use **stateful sessions** for multi-step computer workflows "
            "(e.g., navigate then observe).",
        )

        registry_ctx = get_registry().computer_prompt_context()
        if registry_ctx:
            parts.append(registry_ctx)

        examples = get_computer_examples()
        if examples:
            parts.append(f"### Computer Examples\n\n{examples}")

        return "\n\n".join(p for p in parts if p and p.strip())

    async def capture_state(self) -> Dict[str, Any]:
        """Captures visual computer state (screenshot + URL)."""
        try:
            screenshot = await self._computer_primitives.computer.get_screenshot()
            url = await self._computer_primitives.computer.get_current_url()
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
