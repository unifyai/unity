from __future__ import annotations

import inspect
from typing import Any, Dict, Optional
import asyncio

from unify.actor.environments.base import (
    BaseEnvironment,
    ToolMetadata,
    build_filtered_method_docs,
)
from unify.function_manager.primitives import ComputerPrimitives, get_registry

_NO_DESKTOP_ROUTES = (
    "Routes that need no desktop and cover most of what a browser "
    "would have been used for:\n"
    "- `primitives.web.fetch(url)` -- download a public URL, "
    "including a folder or file shared as 'anyone with the link', "
    "and return a local path ready to parse or ingest.\n"
    "- The connected workspace (Microsoft 365 / Google) for "
    "anything the account can already see, including content "
    "shared into it by another organisation.\n"
    "- Connected integration apps for content held in them."
)

_USER_DESKTOP_RULES = (
    "`primitives.computer.user_desktop` drives a **user's own physical "
    "machine**: only on explicit request, clarify when unsure which "
    "machine is meant, confirm before consequential actions, stop "
    "immediately on `PermissionError` (control revoked), and never "
    "modify their machine to work around an error.  Discover machines "
    "with `user_desktop.list_linked()`; select with "
    "`session(user_id=...)`.  The full rules of engagement are in "
    'guidance -- search for "driving desktops and reading screens".'
)

_USER_DESKTOP_FILES_CONTRACT = (
    "**Their files: use `primitives.computer.user_desktop.files` "
    "(`list`/`pull`/`push`, bulk `sync`) -- it mirrors what you pull "
    "into `~/Unity/Remote/<user_id>/` and returns local paths you can "
    "parse.  Never harvest their files by running shell "
    "`find`/`cat`/`tar`/`base64`/`cp`/`scp`/`rclone` on "
    '`surface="user_desktop"`** -- that surface is only for commands '
    "the user explicitly wants executed on their machine.  The full "
    "lazy-read procedure is in guidance -- search for "
    '"user desktop files".'
)

_PROGRESS_NOTIFICATIONS = (
    "### Progress Notifications for Computer Actions\n\n"
    "Notify once per **logical task**, not per API call.  A task like "
    '"search Google for X" is one notification at the start and one at '
    "completion -- the sub-steps (open browser, navigate, type query, "
    "press Enter) are implementation details the user does not need to "
    "hear individually.\n\n"
    "Reserve intermediate notifications for genuinely long procedures "
    "that span multiple unrelated sites or take more than ~30 seconds "
    "(e.g., comparing prices across five stores).  For a single-site "
    "interaction that completes in a few seconds, one kickoff + one "
    "completion is sufficient."
)

_LATENCY_GUIDANCE = (
    "### Latency: Act and Observe Concurrently\n\n"
    "Computer actions are the biggest latency bottleneck in "
    "interactive sessions.  **Never** follow a sequential observe → "
    "act → observe pattern: act immediately on the likely state "
    "(if the user says \"open Chrome\", call `act('Open Chrome')` "
    "without a confirming screenshot first), observe after acting to "
    "verify, and issue observe + act concurrently when you need "
    "both.  One optimistic action + one verification beats observe → "
    "plan → act → verify.  **Multi-step automated procedures are "
    "different**: for loops, sequential extraction, and form-filling "
    "pipelines, work incrementally — execute one iteration, verify, "
    "then proceed."
)


def _viewing_state_section(example: str, coordinate_spaces: str) -> str:
    """Render the Viewing Computer State section around a shape-appropriate
    screenshot example and coordinate-space note."""
    return (
        "### Viewing Computer State\n\n"
        "`get_screenshot()` returns a PIL Image; `display()` renders it "
        "for the **user** to see:\n\n"
        "```python\n"
        f"{example}\n"
        "```\n\n"
        "To **interpret** a screen yourself, send it to the vision model "
        "with `query_llm(prompt, images=[png_bytes_or_path])`.  "
        "Coordinate spaces are NOT interchangeable: read `click(x, y)` "
        "coordinates from the SAME session's `get_screenshot()` "
        f"({coordinate_spaces}).  Observation-space "
        "scaling and the full coordinate rules are in guidance -- search "
        'for "driving desktops and reading screens".'
    )


class ComputerEnvironment(BaseEnvironment):
    """Computer control environment backed by ``ComputerPrimitives``.

    Exposes three interfaces for generated plan code:

    - ``primitives.computer.desktop.*``  -- singleton desktop control (mouse/keyboard)
    - ``primitives.computer.web.new_session(visible=...)``  -- factory for browser sessions
    - ``primitives.computer.user_desktop.session(user_id=...)``  -- a user's own linked machine

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
        from unify.function_manager.primitives import Primitives, PrimitiveScope

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
        desktop_tool_names = [
            "navigate",
            "act",
            "observe",
            "query",
            "get_links",
            "get_screenshot",
        ]

        registry = get_registry()
        tools: Dict[str, ToolMetadata] = {}

        # Desktop namespace -- singleton, full method set
        desktop_ns = self._computer_primitives.desktop
        for name in desktop_tool_names:
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

        # Web factory -- new_session(), list_sessions(), and get_session()
        web_factory = self._computer_primitives.web
        for factory_method_name in ("new_session", "list_sessions", "get_session"):
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

        # User-desktop factory -- session() and list_linked()
        user_desktop_factory = self._computer_primitives.user_desktop
        for factory_method_name in ("session", "list_linked"):
            fq_name = (
                f"{self.NAMESPACE}.{self.MANAGER_ALIAS}.user_desktop."
                f"{factory_method_name}"
            )
            if (
                self._allowed_methods is not None
                and fq_name not in self._allowed_methods
            ):
                continue
            fn = getattr(user_desktop_factory, factory_method_name, None)
            if fn is None or not callable(fn):
                continue
            try:
                signature = str(inspect.signature(fn))
            except Exception:
                signature = None
            tools[fq_name] = ToolMetadata(
                name=fq_name,
                is_impure=False,
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
        from unify.session_details import SESSION_DETAILS

        entitled = SESSION_DETAILS.assistant.managed_desktop_entitled
        linked = bool(SESSION_DETAILS.assistant.user_desktops)

        parts: list[str] = []

        if not entitled and not linked:
            # Say so before describing any of it. The methods below exist on
            # the namespace whether or not a machine backs them, so an actor
            # told only what they do will reach for one and be refused -- and
            # it will read as a fault rather than as an add-on this assistant
            # does not have. Naming the routes that do work is what stops it
            # retrying.
            return (
                "### Computer Control -- NOT AVAILABLE\n\n"
                "This assistant has no managed desktop, so nothing under "
                "`primitives.computer.desktop` or `primitives.computer.web` "
                "can run: calls fail immediately rather than waiting.  Do not "
                "attempt them, and do not treat this as a transient error.\n\n"
                f"{_NO_DESKTOP_ROUTES}\n\n"
                "If a task genuinely requires driving a graphical application, "
                "say that the Desktop Computer add-on is needed rather than "
                "attempting a workaround."
            )

        if not entitled:
            # No managed desktop, but a user has linked their own machine.
            # `primitives.computer.user_desktop` is gated on links and
            # per-user consent, never on the add-on -- so it must still be
            # taught here, together with its rules of engagement; otherwise
            # the actor holds a live, sensitive capability it was never shown
            # the safety contract for.
            parts.append(
                "### Computer Control -- Managed Desktop NOT AVAILABLE\n\n"
                "This assistant has no managed desktop of its own, so "
                "nothing under `primitives.computer.desktop` or "
                "`primitives.computer.web` can run: calls fail immediately "
                "rather than waiting.  Do not attempt them, and do not treat "
                "this as a transient error.  The one live computer-control "
                "route is `primitives.computer.user_desktop` -- a user's own "
                "linked machine, covered below.\n\n"
                f"{_NO_DESKTOP_ROUTES}\n\n"
                "If a task genuinely requires driving a graphical "
                "application anywhere other than a machine a user has linked "
                "and explicitly asked you onto, say that the Desktop "
                "Computer add-on is needed rather than attempting a "
                "workaround.",
            )
            parts.append(
                "### A User's Linked Desktop\n\n"
                f"{_USER_DESKTOP_RULES}  "
                "Session handles expose the desktop method set (`act`, "
                "`observe`, `query`, `type_text`, `click`, "
                "`get_screenshot`, ...) -- read the full docs from inside "
                "`execute_code` with `help(session.<method>)` / "
                "`inspect.signature(...)` before any non-obvious call.\n\n"
                f"{_USER_DESKTOP_FILES_CONTRACT}",
            )
            parts.append(
                _viewing_state_section(
                    "session = primitives.computer.user_desktop.session()\n"
                    "display(await session.get_screenshot())",
                    "each linked machine is its own display space",
                ),
            )
            parts.append(_PROGRESS_NOTIFICATIONS)
            parts.append(_LATENCY_GUIDANCE)
            if self._allowed_methods is not None:
                filtered_docs = build_filtered_method_docs(
                    self._allowed_methods,
                    self.NAMESPACE,
                )
                if filtered_docs:
                    parts.append(filtered_docs)
            return "\n\n".join(p for p in parts if p and p.strip())

        parts.append(
            "### Computer Control\n\n"
            "The VM desktop runs on X11 display `:99` (1920×1080 physical).  "
            "Desktop and visible-web interactions use native input via "
            "DisplayHarness: xdotool for mouse/keyboard, scrot for screenshots.  "
            "`observe()` on the desktop performs vision extraction over a "
            "full-display screenshot -- the correct way to verify what is "
            "currently visible.  `get_content()` is intentionally not exposed "
            "on the desktop namespace because there is no DOM to query.\n\n"
            "Two interfaces for controlling the desktop:\n\n"
            "#### `primitives.computer.desktop` -- Desktop Control (singleton)\n\n"
            "Sends mouse and keyboard actions to the VM desktop via native X11 "
            "input (xdotool).  There is exactly one desktop session -- it "
            "persists for the lifetime of the assistant.  Use this for native "
            "desktop apps, terminal operations, file managers, and any "
            "interaction with windows already visible on the desktop.\n\n"
            "```python\n"
            "await primitives.computer.desktop.act('Open the Terminal app')\n"
            "display(await primitives.computer.desktop.get_screenshot())\n"
            "```\n\n"
            "#### `primitives.computer.web` -- Web Sessions (factory + registry)\n\n"
            "Creates independent browser sessions and lets you reattach to "
            "existing ones.  Each session is an isolated Chromium process with "
            "its own cookies, storage, and browsing context.  Multiple sessions "
            "can run in parallel.  Always call `stop()` when done.\n\n"
            "```python\n"
            "session = await primitives.computer.web.new_session()  # visible=True by default\n"
            "await session.navigate('https://example.com')\n"
            "data = await session.observe('Extract the main heading')\n"
            "display(await session.get_screenshot())\n"
            "await session.stop()\n"
            "```\n\n"
            "To continue working in a browser that already exists, reattach by "
            "session ID instead of opening a duplicate browser:\n\n"
            "```python\n"
            "session = primitives.computer.web.get_session(0)\n"
            "display(await session.get_screenshot())\n"
            "await session.act('Click the Continue button')\n"
            "```\n\n"
            "Use `primitives.computer.web.list_sessions(visible_only=True, active_only=True)` "
            "when you need to discover reusable sessions programmatically.  In "
            "voice/screen-share flows, visible session IDs are often surfaced in "
            "`<active_web_sessions>` for direct reuse.\n\n"
            "The `visible` parameter: `visible=True` (default) puts the "
            "browser window on the VM desktop (native xdotool input; "
            "Playwright/CDP retained for `navigate`, `get_content`, "
            "`get_links`, captchas); `visible=False` is a headless "
            "Playwright-only browser — faster, invisible to the user, "
            "viewport screenshots only.\n\n"
            "Session handles expose: "
            "`act`, `observe`, `query`, `navigate`, `get_links`, `get_content`, "
            "`get_screenshot`, plus `stop()`.\n\n"
            "#### When to Use Each\n\n"
            "- **Any task involving a web browser** -- "
            "`web.new_session(visible=True)`, never `desktop` — the only way "
            "to get a browser window visible on the desktop and in "
            "screenshots.\n"
            "- **Background web lookup the user doesn't need to see** -- "
            "`web.new_session(visible=False)` for speed.\n"
            "- **Native desktop apps, terminal, file operations** -- "
            "`primitives.computer.desktop`.  Only for non-browser interactions.\n\n"
            "Note: mouse clicks on native-input sessions hit the topmost "
            "window at the given coordinates — bring the target window to "
            "the front first, or use `act()` with an instruction that "
            "includes focusing it.\n\n"
            "#### Managed desktop filesystem\n\n"
            "The managed VM desktop is a separate filesystem from the local "
            "`execute_code` workspace.  Layout:\n"
            "- Desktop login name: `unityuser`.  Home is **`HOME=/Unity`** -- "
            "not `/home/unityuser` (that is not the desktop home).\n"
            "- Browser / XDG Downloads: **`/Unity/Downloads`**, which is a "
            "symlink into the synced tree at `/Unity/Local/Downloads`.  A file "
            "downloaded here therefore reaches the pod workspace on the next "
            "sync and can be parsed, read and ingested from there -- you do "
            "**not** need to move it out of Downloads first.\n"
            "- Synced workspace on the VM: **`/Unity/Local`** (bisync with the "
            "local pod workspace under Filesystem Context).\n"
            "- Other XDG dirs sit beside Downloads under `/Unity/` "
            "(Desktop, Documents, Pictures, ...).\n\n"
            "Path reading rules (on this VM desktop only):\n"
            "- A file-dialog or Thunar breadcrumb like `Unity > Downloads` "
            "means absolute **`/Unity/Downloads`** -- the home folder itself is "
            "named `Unity`.  Do not expand that breadcrumb as "
            "`~/Unity/Downloads` or `/home/unityuser/Unity/Downloads`.\n"
            "- Do not invent `/home/unityuser/...` from the panel username.\n"
            "- When reporting a saved or opened desktop path, use the absolute "
            "`/Unity/...` path you actually used.",
        )

        parts.append(
            "### Your Desktop vs. a User's Desktop\n\n"
            "`primitives.computer.desktop` and `primitives.computer.web` always "
            "drive **your own** managed desktop (the Console live view) -- the "
            "default workspace for every task.  "
            f"{_USER_DESKTOP_RULES}\n\n"
            f"{_USER_DESKTOP_FILES_CONTRACT}",
        )

        parts.append(
            _viewing_state_section(
                "display(await primitives.computer.desktop.get_screenshot())",
                "desktop "
                "and visible web share the full-display space; headless "
                "`visible=False` screenshots are viewport-only",
            ),
        )

        parts.append(_PROGRESS_NOTIFICATIONS)
        parts.append(_LATENCY_GUIDANCE)

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
