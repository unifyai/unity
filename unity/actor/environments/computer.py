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
    """Computer control environment backed by ``ComputerPrimitives``.

    Exposes two interfaces for generated plan code:

    - ``primitives.computer.desktop.*``  -- singleton desktop control (mouse/keyboard)
    - ``primitives.computer.web.new_session(visible=...)``  -- factory for browser sessions

    Lives under the unified ``primitives`` namespace alongside state managers
    and actor delegation.
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
        impure = {"navigate", "act", "new_session"}
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

        # Desktop namespace -- singleton, full method set
        desktop_ns = self._computer_primitives.desktop
        for name in tool_names:
            fq_name = f"{self.NAMESPACE}.{self.MANAGER_ALIAS}.desktop.{name}"
            if (
                self._allowed_methods is not None
                and fq_name not in self._allowed_methods
            ):
                continue
            fn = getattr(desktop_ns, name, None)
            if fn is None or not callable(fn):
                continue
            try:
                signature = str(inspect.signature(fn))
            except Exception:
                signature = None
            tools[fq_name] = ToolMetadata(
                name=fq_name,
                is_impure=name in impure,
                is_steerable=False,
                docstring=getattr(fn, "__doc__", None),
                signature=signature,
                function_id=registry.get_function_id(self.MANAGER_ALIAS, name),
                function_context="primitive",
            )

        # Web factory -- new_session() only
        web_factory = self._computer_primitives.web
        fq_name = f"{self.NAMESPACE}.{self.MANAGER_ALIAS}.web.new_session"
        if self._allowed_methods is None or fq_name in self._allowed_methods:
            fn = web_factory.new_session
            try:
                signature = str(inspect.signature(fn))
            except Exception:
                signature = None
            tools[fq_name] = ToolMetadata(
                name=fq_name,
                is_impure=True,
                is_steerable=False,
                docstring=getattr(fn, "__doc__", None),
                signature=signature,
                function_id=registry.get_function_id(self.MANAGER_ALIAS, "new_session"),
                function_context="primitive",
            )

        return tools

    def get_prompt_context(self) -> str:
        """Generate prompt context with desktop + web factory guidance."""
        parts: list[str] = []

        parts.append(
            "### Computer Control\n\n"
            "Two interfaces for controlling browsers and the desktop:\n\n"
            "#### `primitives.computer.desktop` -- Desktop Control (singleton)\n\n"
            "Controls the full VM desktop via mouse and keyboard.  There is "
            "exactly one desktop session -- it persists for the lifetime of the "
            "assistant.  Suitable for native desktop apps, terminal operations, "
            "and also straightforward single-site web browsing.\n\n"
            "```python\n"
            "await primitives.computer.desktop.act('Open the Terminal app')\n"
            "await primitives.computer.desktop.navigate('https://example.com')\n"
            "display(await primitives.computer.desktop.get_screenshot())\n"
            "```\n\n"
            "#### `primitives.computer.web.new_session()` -- Web Sessions (factory)\n\n"
            "Creates independent browser sessions.  Each session is an isolated "
            "Chromium process with its own cookies, storage, and browsing context.  "
            "Multiple sessions can run in parallel.  Always call `stop()` when done.\n\n"
            "```python\n"
            "session = await primitives.computer.web.new_session()  # visible=True by default\n"
            "await session.navigate('https://example.com')\n"
            "data = await session.observe('Extract the main heading')\n"
            "display(await session.get_screenshot())\n"
            "await session.stop()\n"
            "```\n\n"
            "The `visible` parameter controls where the browser runs:\n"
            "- `visible=True` (default): browser window appears on the VM desktop "
            "(user can see it via screen sharing / noVNC).  Controlled via CDP -- "
            "no mouse or keyboard involved.\n"
            "- `visible=False`: headless browser on the host.  Faster, but "
            "invisible to the user.\n\n"
            "Session handles have the same methods as the desktop namespace: "
            "`act`, `observe`, `query`, `navigate`, `get_links`, `get_content`, "
            "`get_screenshot`, plus `stop()`.\n\n"
            "#### When to Consider Each\n\n"
            "- **Simple single-site browsing** -- `primitives.computer.desktop` "
            "works fine and is the simplest option.\n"
            "- **Multiple sites in parallel, or isolated browser state** -- use "
            "`web.new_session()`.  Each session has fresh cookies/storage and "
            "runs independently.\n"
            "- **Quick background lookup where the user doesn't need to see** -- "
            "`web.new_session(visible=False)` for speed.\n"
            "- **Interactive session where the user is watching and you need "
            "multiple concurrent browser tasks** -- "
            "`web.new_session(visible=True)` so the user can observe each "
            "browser window.\n"
            "- **Native desktop apps, terminal, file operations** -- "
            "`primitives.computer.desktop`.",
        )

        parts.append(
            "### Viewing Computer State\n\n"
            "`get_screenshot()` returns a PIL Image.  `display()` renders it as "
            "visual output you can inspect on the next turn.\n\n"
            "```python\n"
            "# Desktop\n"
            "display(await primitives.computer.desktop.get_screenshot())\n\n"
            "# Web session\n"
            "display(await session.get_screenshot())\n"
            "```",
        )

        parts.append(
            "### Progress Notifications for Computer Actions\n\n"
            "- Treat computer calls as potentially long-running by default.\n"
            "- Emit `notify({...})` before each major computer step.\n"
            "- Keep notification messages user-facing and high-level.",
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
        """Captures visual computer state from the desktop session."""
        try:
            session = await self._computer_primitives.backend.get_session("desktop")
            screenshot = await session.get_screenshot()
            url = await session.get_current_url()
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
