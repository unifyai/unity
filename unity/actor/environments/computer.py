from __future__ import annotations

import inspect
from typing import Any, Dict, Optional
import asyncio

from unity.actor.environments.base import BaseEnvironment, ToolMetadata
from unity.function_manager.primitives import ComputerPrimitives


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
            "reason",
        ]

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
            )

        return tools

    def get_prompt_context(self) -> str:
        """Generate prompt context from registry for computer methods."""
        from unity.function_manager.primitives import get_registry

        return get_registry().computer_prompt_context()

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
