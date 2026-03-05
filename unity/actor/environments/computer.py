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

        # Web factory -- new_session() and list_sessions()
        web_factory = self._computer_primitives.web
        for factory_method_name in ("new_session", "list_sessions"):
            fq_name = f"{self.NAMESPACE}.{self.MANAGER_ALIAS}.web.{factory_method_name}"
            if (
                self._allowed_methods is not None
                and fq_name not in self._allowed_methods
            ):
                continue
            fn = getattr(web_factory, factory_method_name, None)
            if fn is None or not callable(fn):
                continue
            try:
                signature = str(inspect.signature(fn))
            except Exception:
                signature = None
            tools[fq_name] = ToolMetadata(
                name=fq_name,
                is_impure=factory_method_name == "new_session",
                is_steerable=False,
                docstring=getattr(fn, "__doc__", None),
                signature=signature,
                function_id=registry.get_function_id(
                    self.MANAGER_ALIAS,
                    factory_method_name,
                ),
                function_context="primitive",
            )

        return tools

    def get_prompt_context(self) -> str:
        """Generate prompt context with desktop + web factory guidance."""
        parts: list[str] = []

        parts.append(
            "### Computer Control\n\n"
            "The VM desktop is accessed through a VNC connection.  The "
            "`primitives.computer.desktop` namespace drives the desktop via a "
            "headless Playwright browser connected to the noVNC VNC viewer.  "
            "This means methods like `navigate()`, `get_links()`, and "
            "`get_content()` operate on this headless browser (the VNC viewer "
            "page), not on the X11 desktop itself.  Screenshots are captured "
            "natively from the X11 display, so only changes visible on the "
            "physical desktop surface appear in screenshots.\n\n"
            "Two interfaces for controlling the desktop:\n\n"
            "#### `primitives.computer.desktop` -- Desktop Control (singleton)\n\n"
            "Sends mouse and keyboard actions to the VM desktop through the VNC "
            "connection.  There is exactly one desktop session -- it persists "
            "for the lifetime of the assistant.  Use this for native desktop "
            "apps, terminal operations, file managers, and any interaction with "
            "windows already visible on the desktop.\n\n"
            "```python\n"
            "await primitives.computer.desktop.act('Open the Terminal app')\n"
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
            "Session handles expose: "
            "`act`, `observe`, `query`, `navigate`, `get_links`, `get_content`, "
            "`get_screenshot`, plus `stop()`.\n\n"
            "#### When to Use Each\n\n"
            "- **Any task involving a web browser** -- "
            "`web.new_session(visible=True)`.  This is the default for all "
            "browser work and the only way to get a browser window visible on "
            "the desktop and in screenshots.  If the task involves a browser "
            "in any way, always use `web.new_session()`, not `desktop`.\n"
            "- **Background web lookup the user doesn't need to see** -- "
            "`web.new_session(visible=False)` for speed.\n"
            "- **Native desktop apps, terminal, file operations** -- "
            "`primitives.computer.desktop`.  Only for non-browser interactions.",
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
            "Notify once per **logical task**, not per API call.  A task like "
            '"search Google for X" is one notification at the start and one at '
            "completion -- the sub-steps (open browser, navigate, type query, "
            "press Enter) are implementation details the user does not need to "
            "hear individually.\n\n"
            "Reserve intermediate notifications for genuinely long workflows "
            "that span multiple unrelated sites or take more than ~30 seconds "
            "(e.g., comparing prices across five stores).  For a single-site "
            "interaction that completes in a few seconds, one kickoff + one "
            "completion is sufficient.",
        )

        parts.append(
            "### Latency: Act and Observe Concurrently\n\n"
            "Computer actions are the single biggest latency bottleneck — "
            "especially during interactive sessions where the user is waiting "
            "in real time.  **Never** follow a sequential observe → act → "
            "observe pattern (three separate round trips).  Instead:\n\n"
            "1. **Act immediately** based on known or assumed state.  If the "
            "user says \"open Chrome\", just call `act('Open Chrome')` — do "
            "not take a screenshot first to confirm the desktop is visible.\n"
            "2. **Observe after acting** to verify the outcome.  If the state "
            "is not what you expected, course-correct with a follow-up action.\n"
            "3. **Combine observe + act in one turn** when possible.  If you "
            "need both a screenshot and an action, issue them concurrently "
            "rather than waiting for the screenshot before deciding to act.\n\n"
            "The principle: **assume the likely state and act on it; verify "
            "and correct afterwards.**  One optimistic action + one "
            "verification is almost always faster than observe → plan → act → "
            "verify, and the cost of an occasional correction is far less "
            "than the cost of an extra round trip on every single interaction.\n\n"
            "**Multi-step automated workflows are different.**  The above "
            "applies to individual interactive actions where latency matters "
            "(user is watching).  For multi-step automated work — loops, "
            "sequential data extraction, form-filling pipelines — work "
            "incrementally: execute one iteration, verify the result, then "
            "proceed.  The cost of one wrong action is small; the cost of "
            "repeating it in an unverified loop is not.",
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
