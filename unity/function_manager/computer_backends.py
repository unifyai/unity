import inspect
import json
import os
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Optional, List, Dict
import logging
import aiohttp
from unify.utils import http
from pydantic import BaseModel, PydanticUserError
import asyncio
import websockets
from unity.session_details import SESSION_DETAILS
from unity.image_manager.utils import make_solid_png_base64
from unity.logger import LOGGER as _UNITY_LOGGER
from unity.common._async_tool.loop_config import TOOL_LOOP_LINEAGE

logger = logging.getLogger("websockets")

_MAG_DEBUG_PREFIX = "__MAG_DEBUG__ "
_MAGNITUDE_LOG_DIR = os.environ.get("MAGNITUDE_LOG_DIR", "")


def _handle_magnitude_debug_payload(raw: str) -> None:
    """Parse and persist a TEXT debug payload from agent-service.

    Screenshots and traces are saved directly to the filesystem by
    agent-service (same or VM container) — they never flow over the
    WebSocket.  Only lightweight TEXT payloads are handled here.

    Payload format: ``"TEXT JSON_BODY"``.
    """
    if not _MAGNITUDE_LOG_DIR:
        return

    log_dir = Path(_MAGNITUDE_LOG_DIR)

    space_idx = raw.find(" ")
    if space_idx == -1:
        return
    ptype = raw[:space_idx]
    body_str = raw[space_idx + 1 :]

    if ptype != "TEXT":
        return

    try:
        body = json.loads(body_str)
    except json.JSONDecodeError:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "magnitude.log", "a") as f:
        f.write(body.get("line", "") + "\n")


def _get_current_lineage() -> list[str]:
    """Read the current TOOL_LOOP_LINEAGE for propagation to agent-service."""
    try:
        val = TOOL_LOOP_LINEAGE.get([])
        return list(val) if isinstance(val, list) else []
    except LookupError:
        return []


@dataclass
class ActResult:
    """Result of a ``desktop.act()`` call with post-completion context."""

    summary: str
    screenshot: str  # base64 PNG

    def __str__(self) -> str:
        return self.summary

    def __repr__(self) -> str:
        screenshot_preview = (
            f"{self.screenshot[:20]}..."
            if len(self.screenshot) > 20
            else self.screenshot
        )
        return f"ActResult(summary={self.summary!r}, screenshot={screenshot_preview!r})"


class ComputerAgentError(Exception):
    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(f"[{error_type}] {message}")


class _LowLevelActionsMixin:
    """Mixin providing low-level browser action convenience methods.

    Each method builds a single-action payload and delegates to
    ``execute_actions``, which must be provided by the concrete class.
    """

    async def click(self, x: int, y: int) -> dict:
        """
        Click at exact pixel coordinates on the current page.

        Performs a single left-click at the specified (x, y) viewport
        coordinates.  This is a **direct, low-level action** -- it bypasses
        the LLM planning layer entirely and executes immediately via
        Playwright.

        Guidance
        --------
        Use this when you already know the precise coordinates of the
        element you want to click.  Determine coordinates beforehand by
        calling ``get_screenshot()`` and visually inspecting the image, or
        by using ``observe()`` to extract element positions.

        Prefer ``act()`` when the target is best described in natural
        language (e.g. "click the Submit button") and you don't know the
        coordinates.

        Parameters
        ----------
        x : int
            Horizontal pixel coordinate (from the left edge of the viewport).
        y : int
            Vertical pixel coordinate (from the top edge of the viewport).

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot`` (base64 PNG
            of the page after the click).
        """
        return await self.execute_actions([{"variant": "mouse:click", "x": x, "y": y}])

    async def double_click(self, x: int, y: int) -> dict:
        """
        Double-click at exact pixel coordinates on the current page.

        Performs a double left-click at the specified (x, y) viewport
        coordinates.  Useful for selecting words in text fields, opening
        files in file-manager UIs, or any interaction that requires a
        double-click.

        Parameters
        ----------
        x : int
            Horizontal pixel coordinate (from the left edge of the viewport).
        y : int
            Vertical pixel coordinate (from the top edge of the viewport).

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions(
            [{"variant": "mouse:double_click", "x": x, "y": y}],
        )

    async def right_click(self, x: int, y: int) -> dict:
        """
        Right-click at exact pixel coordinates to open a context menu.

        Performs a single right-click at the specified (x, y) viewport
        coordinates, which typically opens a context menu.

        Parameters
        ----------
        x : int
            Horizontal pixel coordinate (from the left edge of the viewport).
        y : int
            Vertical pixel coordinate (from the top edge of the viewport).

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions(
            [{"variant": "mouse:right_click", "x": x, "y": y}],
        )

    async def drag(self, from_x: int, from_y: int, to_x: int, to_y: int) -> dict:
        """
        Click and drag from one point to another.

        Presses the mouse button at (from_x, from_y), moves to
        (to_x, to_y), then releases.  Useful for drag-and-drop
        interactions, slider adjustments, drawing, and resizing elements.

        Parameters
        ----------
        from_x : int
            Starting horizontal pixel coordinate.
        from_y : int
            Starting vertical pixel coordinate.
        to_x : int
            Ending horizontal pixel coordinate.
        to_y : int
            Ending vertical pixel coordinate.

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions(
            [
                {
                    "variant": "mouse:drag",
                    "from": {"x": from_x, "y": from_y},
                    "to": {"x": to_x, "y": to_y},
                },
            ],
        )

    async def scroll(
        self,
        x: int,
        y: int,
        delta_x: int = 0,
        delta_y: int = -500,
    ) -> dict:
        """
        Scroll the page at a specific position.

        Moves the mouse to (x, y) and then scrolls by the specified pixel
        deltas.  Positive ``delta_y`` scrolls **down**, negative scrolls
        **up**.  Positive ``delta_x`` scrolls right, negative scrolls left.

        Guidance
        --------
        Position the mouse over the scrollable element before scrolling.
        For example, to scroll a sidebar, place (x, y) inside the sidebar
        area.  A typical scroll increment is 300-500 pixels.

        Parameters
        ----------
        x : int
            Horizontal pixel coordinate to hover over before scrolling.
        y : int
            Vertical pixel coordinate to hover over before scrolling.
        delta_x : int, default 0
            Pixels to scroll horizontally (positive = right).
        delta_y : int, default -500
            Pixels to scroll vertically (positive = down, negative = up).

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions(
            [
                {
                    "variant": "mouse:scroll",
                    "x": x,
                    "y": y,
                    "deltaX": delta_x,
                    "deltaY": delta_y,
                },
            ],
        )

    async def type_text(self, content: str) -> dict:
        """
        Type text into the currently focused element.

        Sends keyboard input character-by-character into whichever element
        currently has focus.  This does **not** click before typing --
        make sure you ``click()`` the target input field first.

        Guidance
        --------
        Always click the target text field before calling ``type_text()``.
        For example::

            await session.click(300, 200)   # focus the input field
            await session.type_text("hello world")

        To clear existing text before typing, call ``select_all()`` then
        ``type_text()`` with the new value.

        Parameters
        ----------
        content : str
            The text to type.  May include literal characters only --
            use ``press_enter()``, ``press_tab()``, or ``press_backspace()``
            for special keys.

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions(
            [{"variant": "keyboard:type", "content": content}],
        )

    async def press_enter(self) -> dict:
        """
        Press the Enter key.

        Sends a single Enter/Return keypress to the currently focused
        element.  Commonly used to submit forms, confirm dialogs, or
        trigger search after typing a query.

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions([{"variant": "keyboard:enter"}])

    async def press_tab(self) -> dict:
        """
        Press the Tab key.

        Sends a single Tab keypress, which typically moves focus to the
        next form field or interactive element on the page.

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions([{"variant": "keyboard:tab"}])

    async def press_backspace(self) -> dict:
        """
        Press the Backspace key.

        Deletes the character before the cursor in the currently focused
        text field.  Call multiple times or combine with ``select_all()``
        to delete larger amounts of text.

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions([{"variant": "keyboard:backspace"}])

    async def select_all(self) -> dict:
        """
        Select all content in the active text area (Ctrl+A).

        Sends a Ctrl+A (or Cmd+A on macOS) keypress to select all text
        in the currently focused element.  Useful before ``type_text()``
        to replace existing content, or before ``press_backspace()`` to
        clear a field.

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions([{"variant": "keyboard:select_all"}])

    async def switch_tab(self, index: int) -> dict:
        """
        Switch to a browser tab by its index.

        Activates the tab at the given zero-based index.  Use
        ``get_current_url()`` or ``observe()`` to discover which tabs
        are open and their indices.

        Parameters
        ----------
        index : int
            Zero-based index of the tab to switch to.

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions(
            [{"variant": "browser:tab:switch", "index": index}],
        )

    async def close_tab(self, index: int) -> dict:
        """
        Close a browser tab by its index.

        Closes the tab at the given zero-based index.  If the active tab
        is closed, the browser will automatically switch to an adjacent
        tab.

        Parameters
        ----------
        index : int
            Zero-based index of the tab to close.

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions(
            [{"variant": "browser:tab:close", "index": index}],
        )

    async def new_tab(self) -> dict:
        """
        Open a new empty browser tab and switch to it.

        Creates a new blank tab and makes it the active tab.  Follow up
        with ``navigate()`` to load a URL in the new tab.

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions([{"variant": "browser:tab:new"}])

    async def go_back(self) -> dict:
        """
        Navigate back in the browser history.

        Equivalent to pressing the browser's back button.  Navigates to
        the previous page in the current tab's history stack.

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions([{"variant": "browser:nav:back"}])

    async def wait_for(self, seconds: float) -> dict:
        """
        Wait for a specified number of seconds.

        Pauses execution for the given duration.  Most actions include
        smart waiting automatically, so only use this when a significant
        additional wait is clearly required (e.g. waiting for an animation
        to complete or a delayed network response).

        Parameters
        ----------
        seconds : float
            Number of seconds to wait.

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions([{"variant": "wait", "seconds": seconds}])

    async def save_browser_state(self, name: str) -> dict:
        """
        Save the current browser state to a named file.

        Persists cookies, localStorage, and sessionStorage to disk so
        they can be restored in future sessions.  Useful for preserving
        authentication state across session restarts.

        This is a single, instant operation with no visual changes to the
        page.  Do not call it multiple times or wait for page changes.

        Parameters
        ----------
        name : str
            Name for the state file (e.g. ``'my_app_auth'``).  Used as
            the filename when saving to disk.

        Returns
        -------
        dict
            Execution result with ``status`` and ``screenshot``.
        """
        return await self.execute_actions(
            [{"variant": "browser:state:save", "name": name}],
        )

    async def execute_actions(self, actions: list[dict]) -> dict:
        """
        Execute one or more low-level browser actions directly.

        Bypasses the LLM planning layer entirely -- actions are executed
        immediately via the browser automation engine (Playwright) with
        zero LLM calls.  This is the fastest path for performing known
        UI interactions.

        Each action is a dict with a ``variant`` key specifying the action
        type, plus variant-specific parameters.  Multiple actions in a
        single call are executed sequentially.

        Supported variants and their parameters:

        - ``{"variant": "mouse:click", "x": int, "y": int}``
        - ``{"variant": "mouse:double_click", "x": int, "y": int}``
        - ``{"variant": "mouse:right_click", "x": int, "y": int}``
        - ``{"variant": "mouse:drag", "from": {"x": int, "y": int}, "to": {"x": int, "y": int}}``
        - ``{"variant": "mouse:scroll", "x": int, "y": int, "deltaX": int, "deltaY": int}``
        - ``{"variant": "keyboard:type", "content": str}``
        - ``{"variant": "keyboard:enter"}``
        - ``{"variant": "keyboard:tab"}``
        - ``{"variant": "keyboard:backspace"}``
        - ``{"variant": "keyboard:select_all"}``
        - ``{"variant": "browser:tab:switch", "index": int}``
        - ``{"variant": "browser:tab:close", "index": int}``
        - ``{"variant": "browser:tab:new"}``
        - ``{"variant": "browser:nav:back"}``
        - ``{"variant": "wait", "seconds": float}``
        - ``{"variant": "browser:state:save", "name": str}``

        Parameters
        ----------
        actions : list[dict]
            List of action dicts to execute in sequence.

        Returns
        -------
        dict
            Result dict with ``status`` and ``screenshot`` (base64 PNG
            of the page after all actions have been executed).
        """
        raise NotImplementedError


class ComputerBackend(_LowLevelActionsMixin, ABC):
    """
    Abstract Base Class defining the interface for any computer use backend.

    Supports two interfaces:
    - desktop: singleton full desktop control (mouse/keyboard via noVNC)
    - web sessions: independent browser sessions created via factory,
      either visible on the VM (mode=web-vm) or headless (mode=web)
    """

    @abstractmethod
    async def act(self, instruction: str, verify: bool = False) -> "ActResult":
        """
        Perform an autonomous action on the current page or screen.

        Executes a natural language instruction by interpreting the current
        visual state and performing the necessary UI interactions (clicks,
        typing, scrolling, navigation, form filling, etc.). The agent is
        **autonomous and can perform multiple steps** to achieve the goal
        described in the instruction. It operates based on a visual
        understanding of the current page/screen.

        Guidance
        --------
        Write instructions at the **goal level**, not the action level. The
        agent figures out the individual steps (find fields, type, click, etc.)
        on its own.

        **Good instructions (goal-level):**
        - "Log into the account using username 'testuser' and password 'password123'."
          (Agent finds the fields, types, and clicks the login button.)
        - "Find the cheapest blue t-shirt on the page and add it to the cart."
          (Agent visually scans, finds the item, clicks 'Add to Cart'.)
        - "Click the 'Promotions' link in the navigation bar."

        **Bad instructions (too low-level):**
        - "Move the mouse to coordinate 250, 400, then click."
          (Avoid pixel-level commands — let the agent handle element targeting.)

        When to use verify
        ------------------
        **Default to verify=False.** Single-pass execution is ~2x faster and
        is correct for the vast majority of tasks: clicking a button, typing
        into a field, opening an application, navigating to a page, filling
        a short form, etc.

        **Use verify=True only** for complex, multi-step tasks where a single
        planning pass is unlikely to achieve the full goal in one shot —
        e.g. completing a long multi-page wizard, filling an extensive form
        across multiple sections, or a task with conditional branches the
        agent cannot predict ahead of time.

        **During live demos / interactive sessions** where the user is
        watching in real time, strongly prefer verify=False. The latency
        cost of verification (extra screenshot + LLM round-trip per
        iteration) is directly felt by the user. Only use verify=True
        interactively when the task is genuinely complex enough that
        retrying from scratch would be worse than the verification overhead.

        Parameters
        ----------
        instruction : str
            High-level, natural language description of the desired outcome.
            The agent autonomously determines the steps needed.
        verify : bool, optional
            When True, the agent re-observes the screen after executing its
            planned actions and re-plans in a loop until it confirms the task
            is complete (up to an internal iteration cap). Defaults to False
            (single-pass execution). See the guidance above for when to enable.

        Returns
        -------
        ActResult
            Contains ``summary`` (the agent's description of what was done)
            and ``screenshot`` (base64 PNG of the screen after completion).
            ``str(result)`` returns the summary for backward compatibility.
        """

    @abstractmethod
    async def observe(self, query: str, response_format: Any = str) -> Any:
        """
        Observe and extract information from the current page or screen state.

        Uses vision-based analysis to answer questions about what is currently
        visible on the page/screen. This is a read-only operation that does not
        modify the page state. The agent examines the visual content and extracts
        the requested information. This is the primary tool for **perception**.

        Guidance
        --------
        The agent uses a vision-language model, so its success depends on the
        quality and clarity of the query.

        **Key Principles for an Effective Query:**

        1. **Be Specific and Descriptive**: Don't ask "what's on the page."
           Instead of "get the product details," prefer "Extract the product
           name from the top, the price listed in bold, and the author's name
           below the title."

        2. **Provide a Strategy for Non-Textual Elements**: For visual elements
           like star ratings, progress bars, or icons, provide a method for
           interpretation.
           - Good: "For the star_rating, visually count the number of filled
             yellow stars and provide it as a number (e.g., 4.0)."
           - Bad: "Get the star rating." (Fails if the rating is not plain text.)

        3. **Request Specific Data Types**: Guide the model to return the
           correct data type (e.g., "Extract the number of reviews as an
           integer", "Get the price as a float, without the currency symbol").

        4. **Leverage Pydantic for Structure**: For any non-trivial extraction,
           use a Pydantic model via ``response_format``. This forces the agent
           to return clean, structured, and validated data.

        5. **Embrace Optional Fields for Robustness**: Web pages are
           unpredictable. Define fields that might not always be present as
           ``Optional`` in Pydantic models to prevent failures.

        6. **Resolve Visual Ambiguity**: If the page presents conflicting
           information, instruct the model on how to resolve the conflict.
           Prioritize the element that reflects the true state of the page.
           - Bad: "Get the number of servings."
           - Good: "Determine the active serving size multiplier. Identify
             which button ('1/2X', '1X', '2X') is visually selected. IGNORE
             any nearby static text like 'Original recipe yields...'."

        **Bad Queries (HTML/DOM Specific):**
        - "Get the href attribute of the 'About Us' link."
          Instead, ask: "What is the destination URL of the 'About Us' link?"

        Parameters
        ----------
        query : str
            Natural language question about what to extract, and if necessary,
            a strategy for visual interpretation.
        response_format : type, default str
            Expected return type. Can be ``str`` for plain text responses, or a
            Pydantic model class for structured data extraction. When a Pydantic
            model is provided, the response will be parsed and validated against
            that schema. **Highly recommended for reliable extraction.**

        Returns
        -------
        str | BaseModel
            The extracted information. Returns a string when ``response_format=str``,
            or an instance of the specified Pydantic model when a model class is
            provided.
        """

    @abstractmethod
    async def query(self, query: str, response_format: Any = str) -> Any:
        """
        Query the agent's memory and action history.

        Retrieves information from the agent's internal memory about past actions,
        observations, and page states encountered during the current session. This
        enables the agent to recall what it has done and seen, supporting multi-step
        workflows that require context from earlier interactions.

        **Key characteristics:**
        - **Memory-focused**: Uses the agent's accumulated memory and context from
          past actions.
        - **Historical analysis**: Analyzes what happened during previous ``act()``
          calls.
        - **Context-aware**: Includes full agent memory context in the query.
        - **No fresh content**: Does not capture new page content; works with
          existing observations.

        **Good queries (what the agent has done):**
        - "Did the login attempt succeed?"
        - "What were the steps you took to add the item to the cart?"
        - "Summarize the actions you have performed so far."

        **Bad queries (require live page content):**
        - "What is the current price of the item on the page?" (Use ``observe``.)
        - "Click the 'Submit' button." (Use ``act``.)

        Parameters
        ----------
        query : str
            Natural language question about the agent's history or memory.
        response_format : type, default str
            Expected return type. Can be ``str`` for plain text responses, or a
            Pydantic model class for structured data extraction.

        Returns
        -------
        str | BaseModel
            Information from the agent's memory. Returns a string when
            ``response_format=str``, or an instance of the specified Pydantic model
            when a model class is provided.
        """

    @abstractmethod
    async def get_screenshot(self) -> str:
        """
        Capture a screenshot of the current page or screen.

        Takes a visual snapshot of the current browser viewport (web mode) or
        entire screen (desktop mode) and returns it as a base64-encoded PNG image.
        This provides a visual record of the current state for debugging, logging,
        or further analysis.

        Returns
        -------
        str
            Base64-encoded PNG image data representing the current screen state.
            Can be decoded and saved as a .png file or embedded in HTML/markdown.
        """

    @abstractmethod
    async def get_current_url(self) -> str:
        """
        Get the current URL or active window information.

        In web mode, returns the current browser URL. In desktop mode, returns
        information about the currently active window (title, application name,
        or other identifying details). This helps track navigation state and
        context across multi-step workflows.

        Returns
        -------
        str
            In web mode: the current page URL (e.g., "https://example.com/page").
            In desktop mode: active window information (e.g., window title or
            application identifier).
        """

    @abstractmethod
    async def navigate(self, url: str) -> str:
        """
        Navigate to a specific URL in the browser.

        Directs the browser to load the specified URL. This is the primary method
        for moving between pages in web mode. The method waits for the page to
        load before returning. In desktop mode, this method may not be applicable
        depending on the backend implementation.

        Parameters
        ----------
        url : str
            The target URL to navigate to. Should be a fully-qualified URL
            including protocol (e.g., "https://example.com/page"). Relative URLs
            may be supported depending on the backend implementation.

        Returns
        -------
        str
            Confirmation message indicating successful navigation, or an error
            message if navigation failed (e.g., invalid URL, network error,
            timeout).
        """

    @abstractmethod
    async def get_links(
        self,
        same_domain: bool = True,
        selector: str = None,
        **kwargs,
    ) -> dict:
        """
        Extract all links from the current page.

        Scans the current page for hyperlinks and returns them in a structured
        format. Supports filtering by domain and CSS selector to narrow results.
        This is useful for discovering navigation options, scraping link
        collections, or building site maps.

        Parameters
        ----------
        same_domain : bool, default True
            When True, only return links that point to the same domain as the
            current page. When False, include all links regardless of domain
            (including external links).
        selector : str, optional
            CSS selector to filter which elements are scanned for links. When
            provided, only links within elements matching this selector are
            returned. When None, all links on the page are considered.
        **kwargs
            Additional backend-specific filtering or formatting options.

        Returns
        -------
        dict
            Dictionary containing extracted links. Structure depends on backend
            implementation but typically includes link URLs, anchor text, and
            metadata about each link.
        """

    @abstractmethod
    async def get_content(self, format: str = "markdown", **kwargs) -> dict:
        """
        Get raw page content without LLM processing.

        Extracts the current page's content in a specified format without using
        vision or LLM interpretation. This provides direct access to the page's
        text, structure, or markup for parsing, analysis, or storage. Faster and
        more deterministic than vision-based observation.

        Parameters
        ----------
        format : str, default "markdown"
            Desired output format for the page content. Common values include:
            - "markdown": Convert page to markdown representation
            - "html": Return raw HTML source
            - "text": Extract plain text only
            Backend implementations may support additional formats.
        **kwargs
            Additional backend-specific options for content extraction or
            formatting.

        Returns
        -------
        dict
            Dictionary containing the extracted content. Structure depends on
            backend implementation but typically includes the formatted content
            string and metadata about the extraction.
        """

    @abstractmethod
    async def execute_actions(self, actions: list[dict]) -> dict:
        """Execute low-level actions directly (see _LowLevelActionsMixin)."""

    @abstractmethod
    def stop(self):
        """Cleanly shut down the backend."""

    @abstractmethod
    async def pause(self) -> None:
        """Pause the agent's action loop at the next safe checkpoint."""

    @abstractmethod
    async def resume(self) -> None:
        """Resume a paused agent's action loop."""


# A valid 32x32 white PNG image encoded as base64 - used as default mock screenshot
# This ensures screenshot values don't cause "invalid image format" errors when sent to LLMs
VALID_MOCK_SCREENSHOT_PNG = make_solid_png_base64(32, 32, (255, 255, 255))


class MockComputerBackend(ComputerBackend):
    """
    A lightweight mock backend for testing Actor logic without external services.

    This backend requires no Playwright, no Magnitude service. It returns
    configurable canned responses and is designed for testing Actor logic without
    requiring external services.

    The mock also implements the additional methods that MagnitudeBackend
    provides (barrier, interrupt_current_action, clear_pending_commands) so tests
    can use it without modification.

    Usage:
        # Basic usage - returns default canned responses
        backend = MockComputerBackend()

        # Configured responses
        backend = MockComputerBackend(
            url="https://example.com",
            screenshot="base64_encoded_screenshot",
            act_response="done",
            observe_response="Page shows login form",
        )
    """

    def __init__(
        self,
        *,
        url: str = "https://google.com",
        screenshot: str = VALID_MOCK_SCREENSHOT_PNG,
        act_response: ActResult | None = None,
        observe_response: str = "Mock observation",
        query_response: str = "Mock query response",
        **kwargs,
    ):
        """
        Initialize the mock backend with configurable responses.

        Args:
            url: URL to return from get_current_url()
            screenshot: Base64 string to return from get_screenshot()
            act_response: Response to return from act()
            observe_response: Response to return from observe()
            query_response: Response to return from query()
            **kwargs: Ignored (for compatibility with other backends)
        """
        self._url = url
        self._screenshot = screenshot
        self._act_response = act_response or ActResult(
            summary="done",
            screenshot=VALID_MOCK_SCREENSHOT_PNG,
        )
        self._observe_response = observe_response
        self._query_response = query_response

        # Sequence tracking (for barrier compatibility)
        self._seq = 0

        self._on_session_closed = None

    @property
    def backend(self) -> "MockComputerBackend":
        """
        Self-reference for compatibility with code that expects `computer.backend`.

        When MockComputerBackend is used as a drop-in replacement for Computer,
        this allows `mock_backend.backend.method()` to work the same as
        `computer.backend.method()`.
        """
        return self

    async def act(
        self,
        instruction: str,
        verify: bool = False,
        wait: bool = True,
        context: dict = None,
        override_cache: bool = False,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        **_kwargs: Any,
    ) -> Any:
        """Mock implementation of `MagnitudeBackend.act` (signature-compatible).

        Notes:
        - The mock completes instantly; `wait` is accepted for signature compatibility.
        - We ignore `context`/`override_cache`/`verify` but accept them to match MagnitudeBackend.
        - For `wait=False`, we mimic MagnitudeBackend semantics by returning "Command queued."
        """

        _ = context
        _ = override_cache
        _ = verify
        _ = _clarification_up_q
        _ = _clarification_down_q
        _ = _kwargs
        self._seq += 1
        if not wait:
            return "Command queued."
        return self._act_response

    async def observe(
        self,
        query: str,
        response_format: Any = str,
        wait: bool = True,
        context: dict = None,
        bypass_dom_processing: bool = False,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        **_kwargs: Any,
    ) -> Any:
        """Mock implementation of `MagnitudeBackend.observe` (signature-compatible)."""

        _ = query
        _ = wait
        _ = context
        _ = bypass_dom_processing
        _ = _clarification_up_q
        _ = _clarification_down_q
        _ = _kwargs
        self._seq += 1
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            # If a Pydantic model is requested, try to create an instance with defaults
            try:
                return response_format()
            except Exception:
                return self._observe_response
        return self._observe_response

    async def query(
        self,
        query: str,
        response_format: Any = str,
        **_kwargs: Any,
    ) -> Any:
        """Returns the configured query response."""
        _ = _kwargs
        self._seq += 1
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            try:
                return response_format()
            except Exception:
                return self._query_response
        return self._query_response

    async def get_screenshot(self) -> str:
        """Returns the configured screenshot string."""
        return self._screenshot

    async def get_current_url(self) -> str:
        """Returns the configured URL."""
        return self._url

    async def navigate(
        self,
        url: str,
        wait: bool = True,
        context: dict = None,
        **_kwargs: Any,
    ) -> str:
        """Updates the internal URL and returns success."""
        _ = wait
        _ = context
        _ = _kwargs
        self._url = url
        self._seq += 1
        return "success"

    async def get_links(
        self,
        same_domain: bool = True,
        selector: str = None,
        **kwargs,
    ) -> dict:
        """Returns an empty links response."""
        return {
            "base_url": self._url,
            "current_url": self._url,
            "links": [],
            "total": 0,
        }

    async def get_content(self, format: str = "markdown", **kwargs) -> dict:
        """Returns a minimal content response."""
        return {
            "url": self._url,
            "title": "Mock Page",
            "content": "Mock page content",
            "format": format,
        }

    def stop(self):
        """No-op for mock backend."""

    # ─────────────────────────────────────────────────────────────────────────
    # Additional methods from MagnitudeBackend that tests may use
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def current_seq(self) -> int:
        """Returns the current command sequence number."""
        return self._seq

    async def barrier(self, *, up_to_seq: Optional[int] = None) -> None:
        """
        No-op barrier for mock backend.

        In MagnitudeBackend this waits for commands to complete.
        In the mock, all commands complete instantly, so this is a no-op.
        """

    async def interrupt_current_action(self) -> None:
        """
        No-op interrupt for mock backend.

        In MagnitudeBackend this interrupts the agent's action loop.
        In the mock, there's nothing to interrupt.
        """

    async def pause(self) -> None:
        """
        No-op pause for mock backend.

        In MagnitudeBackend this pauses the agent's action loop.
        In the mock, there's nothing to pause.
        """

    async def resume(self) -> None:
        """
        No-op resume for mock backend.

        In MagnitudeBackend this resumes a paused agent's action loop.
        In the mock, there's nothing to resume.
        """

    async def clear_pending_commands(self, run_id: int) -> None:
        """
        No-op clear for mock backend.

        In MagnitudeBackend this removes queued commands.
        In the mock, there are no queued commands.
        """

    async def execute_actions(self, actions: list[dict]) -> dict:
        """No-op execute_actions for mock backend."""
        self._seq += 1
        return {"status": "ok", "screenshot": self._screenshot}

    async def get_session(self, mode: str) -> "ComputerSession":
        """Return a mock session for the given mode."""
        return _MockSession(mode, self)

    async def create_session(
        self,
        mode: str,
        label: str | None = None,
    ) -> "ComputerSession":
        """Return a new mock session for the given mode."""
        if mode == "desktop":
            raise RuntimeError("Desktop mode is singleton")
        return _MockSession(mode, self)


class _MockSession(_LowLevelActionsMixin):
    """Lightweight mock that satisfies the ``ComputerSession`` interface."""

    _mock_id_counter = 0

    def __init__(self, mode: str, backend: MockComputerBackend):
        _MockSession._mock_id_counter += 1
        self._session_id = f"mock-{_MockSession._mock_id_counter}"
        self._mode = mode
        self._backend = backend

    async def act(self, instruction: str, verify: bool = False) -> ActResult:
        return self._backend._act_response

    async def observe(self, query: str, response_format: Any = str) -> Any:
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            try:
                return response_format()
            except Exception:
                pass
        return self._backend._observe_response

    async def query(self, query: str, response_format: Any = str) -> Any:
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            try:
                return response_format()
            except Exception:
                pass
        return self._backend._query_response

    async def navigate(self, url: str) -> str:
        self._backend._url = url
        return "success"

    async def get_screenshot(self) -> str:
        return self._backend._screenshot

    async def get_current_url(self) -> str:
        return self._backend._url

    async def get_content(self, format: str = "markdown") -> dict:
        return {
            "url": self._backend._url,
            "title": "Mock",
            "content": "Mock content",
            "format": format,
        }

    async def get_links(
        self,
        same_domain: bool = True,
        selector: str = None,
        **kwargs,
    ) -> dict:
        return {
            "base_url": self._backend._url,
            "current_url": self._backend._url,
            "links": [],
            "total": 0,
        }

    async def execute_actions(self, actions: list[dict]) -> dict:
        return {"status": "ok", "screenshot": self._backend._screenshot}

    async def stop(self) -> None:
        pass

    def sync_stop(self) -> None:
        pass


class ComputerSession(_LowLevelActionsMixin):
    """Handle for a single agent-service session (any mode).

    Each instance wraps its own ``sessionId`` and ``agent_base_url``, making
    it possible to have sessions that talk to different agent-service instances
    (e.g., container for desktop/web-vm, local process for web).

    All control is via HTTP to the agent-service; the session never touches
    the VM's mouse or keyboard directly (except desktop mode, which drives
    the VM display through noVNC inside the agent-service).
    """

    def __init__(self, session_id: str, mode: str, agent_base_url: str, ssl=None):
        self._session_id = session_id
        self._mode = mode
        self._agent_base_url = agent_base_url
        self._ssl = ssl
        self._last_cursor_position: tuple[int, int] | None = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        payload: dict | None = None,
    ) -> Any:
        import time as _rq_time

        _rq_t0 = _rq_time.perf_counter()
        url = f"{self._agent_base_url}{endpoint}"
        if payload is None:
            payload = {}
        payload["sessionId"] = self._session_id

        logger.debug(
            f"⏱️ [ComputerSession._request] {method} {endpoint} start "
            f"(session={self._session_id})",
        )

        auth_key = SESSION_DETAILS.unify_key
        headers = {"authorization": f"Bearer {auth_key}"}
        retries = 3
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method,
                        url,
                        json=payload,
                        headers=headers,
                        timeout=1000,
                        ssl=self._ssl,
                    ) as resp:
                        _rq_ms = (_rq_time.perf_counter() - _rq_t0) * 1000
                        if resp.status >= 400:
                            logger.debug(
                                f"⏱️ [ComputerSession._request] {method} {endpoint} "
                                f"HTTP {resp.status} ({_rq_ms:.0f}ms, attempt={attempt})",
                            )
                            try:
                                error_data = await resp.json()
                                raise ComputerAgentError(
                                    error_data.get("error", "unknown_http_error"),
                                    error_data.get("message", "No error message."),
                                )
                            except ComputerAgentError:
                                raise
                            except Exception:
                                raise ComputerAgentError(
                                    "http_error",
                                    f"HTTP {resp.status}: {await resp.text()}",
                                )
                        result = await resp.json()
                        _rq_ms = (_rq_time.perf_counter() - _rq_t0) * 1000
                        logger.debug(
                            f"⏱️ [ComputerSession._request] {method} {endpoint} "
                            f"OK ({_rq_ms:.0f}ms, attempt={attempt})",
                        )
                        return result
            except aiohttp.ClientConnectorError:
                if attempt < retries - 1:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise

    def _sync_request(
        self,
        method: str,
        endpoint: str,
        payload: dict | None = None,
    ) -> Any:
        url = f"{self._agent_base_url}{endpoint}"
        auth_key = SESSION_DETAILS.unify_key
        headers = {"authorization": f"Bearer {auth_key}"}
        if payload is None:
            payload = {}
        payload["sessionId"] = self._session_id
        result = http.request(
            method,
            url,
            json=payload,
            headers=headers,
            timeout=300,
            raise_for_status=False,
        )
        if result.status_code >= 400:
            raise RuntimeError(
                f"Failed to reach agent-service {endpoint}: {result.status_code} {result.text[:200]}",
            )
        if endpoint == "/start":
            try:
                return result.json()
            except Exception:
                return {}
        return result

    async def act(self, instruction: str, verify: bool = False) -> ActResult:
        """Perform an autonomous action on the current page or screen."""
        lineage = _get_current_lineage()
        payload: dict = {"task": instruction, "lineage": lineage}
        if verify:
            payload["verify"] = True
        response = await self._request("POST", "/act", payload)
        summary = response.get("summary", "")
        if not verify:
            import json as _json

            summary = _json.dumps({"thoughts": summary, "outcome": "completed"})
        return ActResult(
            summary=summary,
            screenshot=response.get("screenshot", ""),
        )

    async def observe(self, query: str, response_format: Any = str) -> Any:
        """Observe and extract information from the current page/screen."""
        payload = {"instructions": query}
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            try:
                schema = response_format.model_json_schema()
            except PydanticUserError:
                response_format.model_rebuild()
                schema = response_format.model_json_schema()
            payload["schema"] = schema
        response = await self._request("POST", "/extract", payload)
        data = response.get("data")
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            return response_format.model_validate(data)
        return data

    async def query(self, query: str, response_format: Any = str) -> Any:
        """Query the agent's memory and action history."""
        payload = {"query": query}
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            try:
                schema = response_format.model_json_schema()
            except PydanticUserError:
                response_format.model_rebuild()
                schema = response_format.model_json_schema()
            payload["schema"] = schema
        response = await self._request("POST", "/query", payload)
        data = response.get("data")
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            return response_format.model_validate(data)
        return data

    async def navigate(self, url: str) -> str:
        """Navigate to a specific URL."""
        if self._mode == "desktop":
            response = await self._request(
                "POST",
                "/act",
                {"task": f"Go to the page: {url}"},
            )
            return response.get("status", "success")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self._request("POST", "/nav", {"url": url})
                return response.get("status", "success")
            except ComputerAgentError as e:
                if "Target page" in str(e) and attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                raise

    async def get_screenshot(self) -> str:
        """Capture a screenshot (base64-encoded PNG)."""
        response = await self._request("POST", "/screenshot", {})
        cursor = response.get("cursorPosition")
        if cursor and isinstance(cursor, dict):
            self._last_cursor_position = (int(cursor["x"]), int(cursor["y"]))
        return response.get("screenshot", "")

    async def get_current_url(self) -> str:
        """Get the current URL or active window information."""
        try:
            response = await self._request("POST", "/state", {})
            return response.get("url", "")
        except Exception:
            return ""

    async def get_content(self, format: str = "markdown") -> dict:
        """Get raw page content without LLM processing."""
        return await self._request("POST", "/content", {"format": format})

    async def get_links(
        self,
        same_domain: bool = True,
        selector: str = None,
        **kwargs,
    ) -> dict:
        """Extract all links from the current page."""
        payload: dict[str, Any] = {"sameDomain": same_domain}
        if selector:
            payload["selector"] = selector
        return await self._request("POST", "/links", payload)

    async def execute_actions(self, actions: list[dict]) -> dict:
        """Execute low-level actions directly via the agent-service."""
        return await self._request("POST", "/execute-actions", {"actions": actions})

    async def stop(self) -> None:
        """Stop this session on the agent-service."""
        try:
            await self._request("POST", "/stop", {})
        except Exception:
            pass

    def sync_stop(self) -> None:
        """Synchronous stop for use in cleanup/shutdown paths."""
        try:
            self._sync_request("POST", "/stop", {})
        except Exception:
            pass


class MagnitudeBackend(ComputerBackend):
    """Multi-mode session factory backed by one or two agent-service instances.

    ``container_url`` serves desktop and web-vm sessions (inside the Docker
    container with Xvfb).  ``local_url`` serves headless web sessions (local
    process on the host).  Either may be ``None`` if that tier is unavailable.

    Sessions are created lazily on first access via ``get_session(mode)``.
    """

    _process = None

    # Start parameters sent to /start for each mode
    _MODE_START_PARAMS: dict[str, dict] = {
        "desktop": {"headless": True, "mode": "desktop"},
        "web-vm": {"headless": False, "mode": "web-vm"},
        "web": {"headless": True, "mode": "web"},
    }

    def __init__(
        self,
        container_url: str | None = None,
        local_url: str | None = None,
        *,
        # Legacy compat: if callers pass the old signature, translate it.
        agent_server_url: str | None = None,
        agent_mode: str | None = None,
        headless: bool = False,
        **kwargs,
    ):
        # Legacy translation: old callers pass (agent_server_url, agent_mode).
        if agent_server_url is not None and container_url is None and local_url is None:
            if agent_mode == "web":
                local_url = agent_server_url
            else:
                container_url = agent_server_url

        self._container_url = container_url
        self._local_url = local_url
        # Skip TLS verification for VM connections only.  Caddy may serve a
        # temporary self-signed cert during ACME; the connection is within
        # GCP's VPC where infrastructure-level encryption already applies.
        self._vm_ssl = (
            False if container_url and container_url.startswith("https://") else None
        )

        # Primary sessions: one per mode, created lazily
        self._sessions: dict[str, ComputerSession] = {}
        # Extra parallel sessions spawned via create_session()
        self._extra_sessions: list[ComputerSession] = []

        # Network-based logging infrastructure (ties to container agent-service)
        self._network_log_queue: Optional[asyncio.Queue] = None
        self._log_stream_task: Optional[asyncio.Task] = None
        self._current_capture_queue: Optional[asyncio.Queue] = None
        self._log_consumer_task: Optional[asyncio.Task] = None
        self._async_initialized: bool = False

        # Command queue infrastructure
        self._command_queue = asyncio.Queue()
        self._command_processor_task = None
        self._active_commands = {}
        self._seq: int = 0
        self._processed_seq: int = -1
        self._barrier_events: dict[int, asyncio.Event] = {}
        self._log_buffer: Dict[int, List[str]] = defaultdict(list)
        self._current_processing_seq: Optional[int] = None

        # The primary base URL for websocket log streaming
        self.agent_base_url = container_url or local_url or "http://localhost:3000"

        self._on_session_closed = None

        logger.info(
            f"🔗 MagnitudeBackend initialized (container={self._container_url}, local={self._local_url})",
        )

    def _url_for_mode(self, mode: str) -> str:
        if mode in ("desktop", "web-vm"):
            if self._container_url is None:
                raise RuntimeError(
                    f"No container agent-service URL configured (needed for {mode!r} mode)",
                )
            return self._container_url
        if mode == "web":
            if self._local_url is None:
                raise RuntimeError(
                    "No local agent-service URL configured (needed for web mode)",
                )
            return self._local_url
        raise ValueError(f"Unknown mode: {mode!r}")

    async def _create_session_async(
        self,
        mode: str,
        label: str | None = None,
    ) -> ComputerSession:
        """Create a session asynchronously."""
        import time as _cs_time

        _cs_t0 = _cs_time.perf_counter()
        url = self._url_for_mode(mode)
        params = dict(self._MODE_START_PARAMS[mode])
        if label is not None:
            params["label"] = label
        auth_key = SESSION_DETAILS.unify_key
        headers = {"authorization": f"Bearer {auth_key}"}
        use_ssl = self._vm_ssl if mode in ("desktop", "web-vm") else None
        logger.debug(
            f"⏱️ [MagnitudeBackend._create_session] POST /start ({mode}) begin",
        )
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"{url}/start",
                json=params,
                headers=headers,
                timeout=300,
                ssl=use_ssl,
            ) as resp:
                _cs_ms = (_cs_time.perf_counter() - _cs_t0) * 1000
                if resp.status >= 400:
                    logger.debug(
                        f"⏱️ [MagnitudeBackend._create_session] POST /start FAILED "
                        f"({_cs_ms:.0f}ms, status={resp.status})",
                    )
                    raise RuntimeError(
                        f"Failed to create {mode} session: {resp.status}",
                    )
                data = await resp.json()
        _cs_ms = (_cs_time.perf_counter() - _cs_t0) * 1000
        session_id = data.get("sessionId")
        if not session_id:
            raise RuntimeError(f"Failed to get sessionId for {mode} session")
        session = ComputerSession(session_id, mode, url, ssl=use_ssl)
        logger.info(f"✅ Created {mode} session {session_id} ({_cs_ms:.0f}ms)")
        return session

    async def get_session(self, mode: str) -> ComputerSession:
        """Get or lazily create the primary session for the given mode."""
        if mode in self._sessions:
            logger.debug(f"⏱️ [MagnitudeBackend.get_session] cache hit for {mode}")
            return self._sessions[mode]
        logger.debug(
            f"⏱️ [MagnitudeBackend.get_session] cache miss for {mode}, creating",
        )
        session = await self._create_session_async(mode)
        self._sessions[mode] = session
        return session

    def clear_session(self, mode: str) -> None:
        """Remove a cached session so the next get_session re-creates it."""
        self._sessions.pop(mode, None)

    async def create_session(
        self,
        mode: str,
        label: str | None = None,
    ) -> ComputerSession:
        """Spawn an additional parallel session (web/web-vm only).

        Desktop mode is singleton (one mouse, one keyboard) and cannot be
        duplicated.  Use ``get_session("desktop")`` for the single desktop session.
        """
        if mode == "desktop":
            raise RuntimeError(
                "Desktop mode is singleton -- cannot create additional sessions",
            )
        session = await self._create_session_async(mode, label=label)
        self._extra_sessions.append(session)
        return session

    async def _ensure_async_initialized(self):
        """
        Initialize async components when event loop is available.
        This is called lazily from async methods.
        """
        # Initialize async infra even if websockets are unavailable so commands always process
        if not self._async_initialized and websockets is not None:
            try:
                # Initialize the network log queue
                self._network_log_queue = asyncio.Queue()

                # Start the network log streaming and consumption tasks
                self._log_stream_task = asyncio.create_task(
                    self._start_log_stream_listener(),
                )
                self._log_consumer_task = asyncio.create_task(self._log_consumer())

                # Start the command processor task
                if not self._command_processor_task:
                    logger.info("⚙️ Starting command processor task")
                    self._command_processor_task = asyncio.create_task(
                        self._process_commands(),
                    )

                self._async_initialized = True
                logger.info("⚙️ Initialized async components")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize async components: {e}")
        elif websockets is None and not self._async_initialized:
            logger.warning("⚠️ Websockets not available, log streaming disabled")
            # Even without websockets, ensure command processor runs
            if not self._network_log_queue:
                self._network_log_queue = asyncio.Queue()
            if not self._command_processor_task:
                logger.info("⚙️ Starting command processor task (no websockets)")
                self._command_processor_task = asyncio.create_task(
                    self._process_commands(),
                )
            self._async_initialized = True

    async def _start_log_stream_listener(self):
        """
        Connects to the Magnitude WebSocket and streams logs into an async queue.
        Includes reconnection logic.
        """
        ws_url = self.agent_base_url.replace("http", "ws") + "/logs/stream"
        logger.info(f"🔌 Starting log stream listener for {ws_url}")

        # Prepare authentication headers
        auth_key = SESSION_DETAILS.unify_key
        headers = {
            "Authorization": f"Bearer {auth_key}",
        }

        while True:
            try:
                async with websockets.connect(
                    ws_url,
                    additional_headers=headers,
                ) as websocket:
                    logger.info(f"🔌 Connected to Magnitude log stream at {ws_url}")
                    async for message in websocket:
                        # The message from the WebSocket is put into our internal queue
                        if self._network_log_queue is not None:
                            await self._network_log_queue.put(str(message))
            except (
                websockets.exceptions.ConnectionClosedError,
                ConnectionRefusedError,
            ) as e:
                logger.warning(
                    f"⚠️ Log stream disconnected: {e}. Reconnecting in 5 seconds...",
                )
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(
                    f"🚨 An unexpected error occurred in the log streamer: {e}. Retrying in 10 seconds...",
                )
                await asyncio.sleep(10)

    async def _log_consumer(self):
        """
        Consumes logs from the internal network queue and directs them either
        to the actor's temporary capture queue or to Unity's LOGGER.

        Log lines arriving from the agent-service are pre-formatted with
        lineage labels (e.g. ``[CodeActActor.act(ab12)->desktop.act] 🛠️ ...``)
        so they integrate with Unity's hierarchical log output.

        Lines with the ``__MAG_DEBUG__`` prefix are structured debug payloads
        (screenshots, act traces) that get persisted to ``MAGNITUDE_LOG_DIR``
        and are **not** forwarded to the text log.
        """
        while True:
            try:
                if self._network_log_queue is None:
                    await asyncio.sleep(1)
                    continue

                log_line = await self._network_log_queue.get()

                if log_line.startswith(_MAG_DEBUG_PREFIX):
                    try:
                        _handle_magnitude_debug_payload(
                            log_line[len(_MAG_DEBUG_PREFIX) :],
                        )
                    except Exception as e:
                        _UNITY_LOGGER.warning(
                            f"[MagnitudeDebug] Failed to handle payload: {e}",
                        )
                    self._network_log_queue.task_done()
                    continue

                if log_line.startswith('{"__type":'):
                    try:
                        event = json.loads(log_line)
                        if event.get("__type") == "session:closed":
                            sid = event.get("sessionId", "")
                            reason = event.get("reason", "unknown")
                            _UNITY_LOGGER.info(
                                f"[SessionClosed] id={sid} reason={reason}",
                            )
                            if self._on_session_closed:
                                self._on_session_closed(sid)
                            self._network_log_queue.task_done()
                            continue
                    except (json.JSONDecodeError, KeyError):
                        pass

                if self._current_capture_queue is not None:
                    self._current_capture_queue.put_nowait(log_line)

                if self._current_processing_seq is not None:
                    self._log_buffer[self._current_processing_seq].append(log_line)

                if self._current_capture_queue is None:
                    _UNITY_LOGGER.info(log_line)

                self._network_log_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                _UNITY_LOGGER.error(f"[MagnitudeLogConsumerError] {e}")
                await asyncio.sleep(1)

    async def _process_commands(self):
        """A background worker that pulls commands, executes them in order, and handles barriers."""
        while True:
            try:
                seq, command_id, func, args, kwargs, future = (
                    await self._command_queue.get()
                )

                # Handle barrier commands
                if command_id.startswith("barrier_"):
                    self._processed_seq = seq
                    # Notify any waiting barriers
                    for barrier_seq, event in list(self._barrier_events.items()):
                        if self._processed_seq >= barrier_seq:
                            event.set()
                            del self._barrier_events[barrier_seq]
                    if future:
                        future.set_result("ok")
                    self._command_queue.task_done()
                    continue

                # Normal command execution
                if command_id not in self._active_commands:
                    self._command_queue.task_done()
                    continue
                try:
                    logger.info(f"▶️ Executing command seq={seq}, id={command_id}")
                    self._current_processing_seq = seq
                    result = await func(*args, **kwargs)
                    if future:
                        future.set_result(result)
                except Exception as e:
                    if future:
                        future.set_exception(e)
                finally:
                    self._current_processing_seq = None
                    logger.info(f"✅ Completed command seq={seq}, id={command_id}")
                    self._processed_seq = seq
                    # Notify any waiting barriers after a normal command completes
                    for barrier_seq, event in list(self._barrier_events.items()):
                        if self._processed_seq >= barrier_seq:
                            event.set()
                            del self._barrier_events[barrier_seq]
                    self._active_commands.pop(command_id, None)
                    self._command_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Command processor crashed: {e}", exc_info=True)
                await asyncio.sleep(1)

    @property
    def current_seq(self) -> int:
        """Returns the current command sequence number."""
        return self._seq

    async def barrier(self, *, up_to_seq: Optional[int] = None) -> None:
        """
        Waits until all commands up to a specific sequence number have been processed.
        """
        target_seq = up_to_seq if up_to_seq is not None else self._seq

        # If no commands have been issued yet, there's nothing to wait for
        if target_seq == 0 and self._processed_seq == -1:
            return

        if self._processed_seq >= target_seq:
            return  # All relevant commands are already done

        # Check if an event for this sequence already exists or create one
        if target_seq not in self._barrier_events:
            self._barrier_events[target_seq] = asyncio.Event()

        await self._barrier_events[target_seq].wait()

    async def _request(
        self,
        method: str,
        endpoint: str,
        payload: dict | None = None,
    ) -> Any:
        """Backend-level async HTTP helper (no session ID injection)."""
        url = f"{self.agent_base_url}{endpoint}"
        if payload is None:
            payload = {}
        use_ssl = self._vm_ssl
        retries = 3
        for attempt in range(retries):
            try:
                auth_key = SESSION_DETAILS.unify_key
                headers = {"authorization": f"Bearer {auth_key}"}
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method,
                        url,
                        json=payload,
                        headers=headers,
                        timeout=1000,
                        ssl=use_ssl,
                    ) as resp:
                        if resp.status >= 400:
                            try:
                                error_data = await resp.json()
                                raise ComputerAgentError(
                                    error_data.get("error", "unknown_http_error"),
                                    error_data.get("message", "No error message."),
                                )
                            except ComputerAgentError:
                                raise
                            except Exception:
                                raise ComputerAgentError(
                                    "service_error",
                                    f"Server error: {resp.status} - {await resp.text()}",
                                )
                        return await resp.json()
            except aiohttp.ClientConnectorError:
                if attempt < retries - 1:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise

    # ── ABC-required methods (delegate to first available session) ──────

    async def _default_session(self) -> ComputerSession:
        """Return the first available primary session (for ABC backward compat)."""
        for mode in ("web-vm", "desktop", "web"):
            if mode in self._sessions:
                return self._sessions[mode]
        raise RuntimeError(
            "No sessions created yet. Use get_session(mode) to create one.",
        )

    async def act(self, instruction: str, verify: bool = False, **kwargs) -> ActResult:
        s = await self._default_session()
        try:
            return await s.act(instruction, verify=verify)
        except asyncio.CancelledError:
            await self.interrupt_current_action()
            raise

    async def observe(self, query: str, response_format: Any = str, **kwargs) -> Any:
        s = await self._default_session()
        return await s.observe(query, response_format)

    async def query(self, query: str, response_format: Any = str, **kwargs) -> Any:
        s = await self._default_session()
        return await s.query(query, response_format)

    async def get_screenshot(self) -> str:
        s = await self._default_session()
        return await s.get_screenshot()

    async def get_current_url(self) -> str:
        s = await self._default_session()
        return await s.get_current_url()

    async def navigate(self, url: str, **kwargs) -> str:
        s = await self._default_session()
        return await s.navigate(url)

    async def get_links(
        self,
        same_domain: bool = True,
        selector: str = None,
        **kwargs,
    ) -> dict:
        s = await self._default_session()
        return await s.get_links(same_domain, selector)

    async def get_content(self, format: str = "markdown", **kwargs) -> dict:
        s = await self._default_session()
        return await s.get_content(format)

    async def execute_actions(self, actions: list[dict]) -> dict:
        s = await self._default_session()
        return await s.execute_actions(actions)

    async def interrupt_current_action(self):
        try:
            await self._request("POST", "/interrupt_action")
        except Exception as e:
            logger.info(f"Warning: Failed to send interrupt request: {e}")

    async def pause(self) -> None:
        try:
            await self._request("POST", "/pause")
        except Exception as e:
            logger.info(f"Warning: Failed to send pause request: {e}")

    async def resume(self) -> None:
        try:
            await self._request("POST", "/resume")
        except Exception as e:
            logger.info(f"Warning: Failed to send resume request: {e}")

    async def clear_pending_commands(self, run_id: int):
        kept_items = []
        removed_count = 0
        original_size = self._command_queue.qsize()
        while not self._command_queue.empty():
            seq, command_id, func, args, kwargs, future = (
                self._command_queue.get_nowait()
            )
            context = self._active_commands.get(command_id, [None, {}])[1]
            if context.get("run_id") != run_id:
                kept_items.append((seq, command_id, func, args, kwargs, future))
            else:
                if future:
                    future.cancel()
                self._active_commands.pop(command_id, None)
                removed_count += 1
        for item in kept_items:
            self._command_queue.put_nowait(item)
        logger.info(
            f"Cleared {removed_count}/{original_size} queued commands for run_id={run_id}.",
        )

    def stop(self):
        """Stops all sessions and cancels background tasks."""
        if self._log_stream_task and not self._log_stream_task.done():
            self._log_stream_task.cancel()
        if self._log_consumer_task and not self._log_consumer_task.done():
            self._log_consumer_task.cancel()
        if self._command_processor_task and not self._command_processor_task.done():
            self._command_processor_task.cancel()

        for session in list(self._sessions.values()) + self._extra_sessions:
            session.sync_stop()
        self._sessions.clear()
        self._extra_sessions.clear()

        if MagnitudeBackend._process:
            logger.info(
                f"Stopping Magnitude agent service (PID: {MagnitudeBackend._process.pid})...",
            )
            MagnitudeBackend._process.terminate()
            try:
                MagnitudeBackend._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                MagnitudeBackend._process.kill()
            MagnitudeBackend._process = None

    async def await_sequence_logs(self, seq: int) -> list[str]:
        while self._processed_seq < seq:
            await asyncio.sleep(0.1)
        return self._log_buffer.get(seq, [])
