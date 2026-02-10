import inspect
import subprocess
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any
from typing import Optional, List, Dict
import logging
import aiohttp
from unify.utils import http
from pydantic import BaseModel, PydanticUserError
import asyncio
import functools
import websockets
from unity.session_details import SESSION_DETAILS
from unity.image_manager.utils import make_solid_png_base64

logger = logging.getLogger("websockets")


class ComputerAgentError(Exception):
    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(f"[{error_type}] {message}")


class ComputerBackend(ABC):
    """
    Abstract Base Class defining the interface for any computer use backend.

    Supports both web automation (agent_mode="web") and general
    desktop/computer control (agent_mode="desktop") via vision-based agents.
    """

    @abstractmethod
    async def act(self, instruction: str) -> str:
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

        Parameters
        ----------
        instruction : str
            High-level, natural language description of the desired outcome.
            The agent autonomously determines the steps needed.

        Returns
        -------
        str
            Confirmation message describing what action was performed, or an
            error message if the action could not be completed.
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
    def stop(self):
        """Cleanly shut down the backend."""


# A valid 32x32 white PNG image encoded as base64 - used as default mock screenshot
# This ensures screenshot values don't cause "invalid image format" errors when sent to LLMs
VALID_MOCK_SCREENSHOT_PNG = make_solid_png_base64(32, 32, (255, 255, 255))


class MockComputerBackend(ComputerBackend):
    """
    A lightweight mock backend for testing Actor logic without external services.

    This backend requires no Redis, no Playwright, no Magnitude service. It returns
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
        act_response: str = "done",
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
        self._act_response = act_response
        self._observe_response = observe_response
        self._query_response = query_response

        # Sequence tracking (for barrier compatibility)
        self._seq = 0

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
        - We ignore `context`/`override_cache` but accept them to match MagnitudeBackend.
        - For `wait=False`, we mimic MagnitudeBackend semantics by returning "Command queued."
        """

        _ = context
        _ = override_cache
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

    async def clear_pending_commands(self, run_id: int) -> None:
        """
        No-op clear for mock backend.

        In MagnitudeBackend this removes queued commands.
        In the mock, there are no queued commands.
        """


class MagnitudeBackend(ComputerBackend):
    _agent_base_url = "http://localhost:3000"
    _process = None  # Keep for process management if needed

    def __init__(
        self,
        agent_server_url: str = "http://localhost:3000",
        headless: bool = False,
        agent_mode: str = "web",
        **kwargs,
    ):
        self.agent_mode = agent_mode

        # Network-based logging infrastructure
        self._network_log_queue: Optional[asyncio.Queue] = None
        self._log_stream_task: Optional[asyncio.Task] = None
        self._current_capture_queue: Optional[asyncio.Queue] = None
        self._log_consumer_task: Optional[asyncio.Task] = None
        self._async_initialized: bool = False

        # Command queue infrastructure
        self._command_queue = asyncio.Queue()
        self._command_processor_task = None
        self._active_commands = {}  # id -> (instruction, context)
        self._seq: int = 0
        self._processed_seq: int = -1
        self._barrier_events: dict[int, asyncio.Event] = (
            {}
        )  # For barrier synchronization
        self._log_buffer: Dict[int, List[str]] = defaultdict(list)
        self._current_processing_seq: Optional[int] = None

        # Keep the simpler initialization from HEAD but add logging support
        MagnitudeBackend._agent_base_url = agent_server_url
        self.agent_base_url = agent_server_url

        # Session ID for this backend instance
        self._session_id: Optional[str] = None

        logger.info(
            f"🔗 Connecting to Magnitude service at {self.agent_base_url} (Mode: {self.agent_mode})",
        )

        try:
            response = self._sync_request(
                "POST",
                "/start",
                {"headless": headless, "mode": self.agent_mode},
            )
            # Extract sessionId from response
            if isinstance(response, dict):
                self._session_id = response.get("sessionId")
            else:
                # Fallback: try to parse as JSON if it's a response object
                try:
                    json_data = response.json() if hasattr(response, "json") else {}
                    self._session_id = json_data.get("sessionId")
                except Exception:
                    pass

            if not self._session_id:
                raise RuntimeError("Failed to get sessionId from /start endpoint")

            logger.info(f"✅ Session created: {self._session_id}")
            self._check_service_ready()

            # Initialize the network log queue - defer creation until event loop is available
            self._network_log_queue = None

            # Mark that async initialization is needed
            self._async_initialized = False

        except Exception as e:
            logger.info(f"❌ Failed to initialize MagnitudeBackend: {e}")
            self.stop()
            raise

    def _check_service_ready(self):
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                r = self._sync_request("POST", "/screenshot")
                if hasattr(r, "status_code") and r.status_code < 500:
                    logger.info(
                        f"✅ Magnitude service is ready on {self.agent_base_url}",
                    )
                    break
            except Exception:
                time.sleep(0.5)
        else:
            self.stop()
            raise RuntimeError(
                f"Magnitude agent failed to become ready within 30 seconds on {self.agent_base_url}",
            )

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
        to the actor's temporary capture queue or to stdout.
        """
        while True:
            try:
                if self._network_log_queue is None:
                    await asyncio.sleep(1)
                    continue

                log_line = await self._network_log_queue.get()

                # If the actor is currently capturing, put it in its queue
                if self._current_capture_queue is not None:
                    logger.debug(f"📥 Capturing magnitude log: {log_line[:100]}...")
                    self._current_capture_queue.put_nowait(log_line)

                # Buffer logs if we are processing a command
                if self._current_processing_seq is not None:
                    self._log_buffer[self._current_processing_seq].append(log_line)

                if self._current_capture_queue is None:
                    # Otherwise, log to console (this will show up as regular logs)
                    logger.info(f"🔍 Magnitude: {log_line}")

                self._network_log_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Log consumer task cancelled.")
                break
            except Exception as e:
                logger.error(f"[MagnitudeLogConsumerError] {e}")
                await asyncio.sleep(1)  # Prevent rapid-fire error loops

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
        url = f"{MagnitudeBackend._agent_base_url}{endpoint}"

        if payload is None:
            payload = {}
        if self._session_id and endpoint != "/start":  # /start doesn't need sessionId
            payload["sessionId"] = self._session_id

        retries = 3
        for attempt in range(retries):
            try:
                auth_key = SESSION_DETAILS.unify_key
                headers = {
                    "authorization": f"Bearer {auth_key}",
                }
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method,
                        url,
                        json=payload,
                        headers=headers,
                        timeout=1000,
                    ) as resp:
                        if resp.status >= 400:
                            try:
                                error_data = await resp.json()
                                error_type = error_data.get(
                                    "error",
                                    "unknown_http_error",
                                )
                                message = error_data.get("message", "No error message.")
                                raise ComputerAgentError(error_type, message)
                            except Exception as e:
                                raise ComputerAgentError(
                                    "service_error",
                                    f"Server error: {resp.status} - {await resp.text()}",
                                ) from e
                        return await resp.json()
            except aiohttp.ClientConnectorError as e:
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
        try:
            url = f"{MagnitudeBackend._agent_base_url}{endpoint}"
            auth_key = SESSION_DETAILS.unify_key
            headers = {
                "authorization": f"Bearer {auth_key}",
            }

            if payload is None:
                payload = {}
            if (
                hasattr(self, "_session_id")
                and self._session_id
                and endpoint != "/start"
            ):
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
            # Return JSON for /start to extract sessionId, otherwise return result object
            if endpoint == "/start":
                try:
                    return result.json()
                except Exception:
                    return {}
            return result
        except Exception as e:
            raise RuntimeError(f"Could not reach agent-service {endpoint}: {e}")

    async def act(
        self,
        instruction: str,
        wait: bool = True,
        context: dict = None,
        override_cache: bool = False,
    ) -> Any:
        """
        Executes a high-level computer task using the Magnitude agent.

        This tool is **autonomous and can perform multiple steps** (e.g., typing, clicking, scrolling) to achieve the goal described in the instruction. It operates based on a visual understanding of the current web view.

        Args:
            instruction (str): A high-level, natural-language command describing the desired outcome.
            wait (bool): If True (default), the function will block and wait for the action to
                        complete before returning. If False,
                        the command is added to a queue for background execution, and
                        the function returns immediately.
            context (dict): Internal metadata for command tracking.
            override_cache (bool): If True, deletes any matching cache entries before execution,
                                  allowing the action to populate a fresh cache entry. Useful
                                  when cache entries are corrupted or inefficient.

        ### Non-Blocking Example (`wait=False`)
        This is ideal for sequences of actions where the plan doesn't need immediate feedback.
        ```python
        # The plan queues up all actions and continues its own execution
        # without waiting for the agent to finish each one.
        await computer_primitives.act("Type 'testuser' into the username field", wait=False)
        await computer_primitives.act("Type 'password123' into the password field", wait=False)
        await computer_primitives.act("Click the 'Login' button", wait=False)
        ```

        ### Blocking Example (`wait=True`, Default)
        Use this when the outcome of an action is required for a subsequent decision in the plan.
        ```python
        # The plan pauses until the button click is complete and the new page has loaded.
        await computer_primitives.act("Click the 'Proceed to Checkout' button", wait=True)
        # Now that we've waited, we can safely observe the new page state.
        cart_total = await computer_primitives.observe("What is the final total?")
        ```

        Examples:
            # ✅ Good Example (Multi-Step Task)
            - instruction: "Log into the account using username 'testuser' and password 'password123'."
            # The agent will find the fields, type, and click the login button.

            # ✅ Good Example (Vague Goal, Agent figures it out)
            - instruction: "Find the cheapest blue t-shirt on the page and add it to the cart."
            # The agent will visually scan, find the item, and click the corresponding 'Add to Cart' button.

            # ✅ Good Example (Clear Action)
            - instruction: "Click the 'Promotions' link in the navigation bar."

            # ❌ Bad Example (Too low-level)
            # Avoid breaking down simple actions. Let the agent handle it.
            - instruction: "Move the mouse to coordinate 250, 400, then click."
        """
        await self._ensure_async_initialized()
        context = context or {}
        self._seq += 1
        seq = self._seq
        command_id = f"{context.get('function_name', 'unknown')}_{seq}"

        bound_func = functools.partial(
            self._request,
            "POST",
            "/act",
            {"task": instruction, "override_cache": override_cache},
        )

        future = asyncio.get_event_loop().create_future() if wait else None
        self._active_commands[command_id] = (instruction, context)
        logger.info(
            f"🧾 Queueing command seq={seq}, id={command_id}, wait={wait}, task='{instruction[:120]}'",
        )
        await self._command_queue.put((seq, command_id, bound_func, [], {}, future))

        if wait:
            response = await future
            return response.get("status", "success")
        else:
            return "Command queued."

    async def interrupt_current_action(self):
        """Sends a non-destructive request to interrupt the agent's current action loop."""
        try:
            await self._request("POST", "/interrupt_action")
        except Exception as e:
            logger.info(
                f"⚠️ Warning: Failed to send interrupt request. The action may continue in the background. Error: {e}",
            )

    async def observe(
        self,
        query: str,
        response_format: Any = str,
        wait: bool = True,
        context: dict = None,
        bypass_dom_processing: bool = False,
    ) -> Any:
        """
        Extracts structured information from the current page/screen using the Magnitude agent.

        This is your primary tool for perception. The agent uses a vision-language model to
        analyze the page, so its success depends entirely on the quality and clarity of your `query`.

        This is a perception tool for what is *currently visible* in the web view or
        on-screen desktop. It is NOT a general-purpose data access tool.

        **Key Principles for an Effective Query:**

        1.  **Be Specific and Descriptive**: Don't just ask "what's on the page." Guide the agent.
            Instead of "get the product details," prefer "Extract the product name from the top, the price
            listed in bold, and the author's name below the title."

        2.  **Provide a Strategy for Non-Textual Elements**: For visual elements like star
            ratings, progress bars, or icons, you MUST provide a method for interpretation.
            - **Good (Star Rating):** "For the 'star_rating', visually count the number of filled yellow
              stars and provide it as a number (e.g., 4.0). If you see a half-filled star, add 0.5."
            - **Good (Active Icon):** "Determine which navigation link is active by identifying the one
              that is underlined or has a different text color."
            - **Bad:** "Get the star rating." (This will fail if the rating is not plain text).

        3.  **Request Specific Data Types**: Guide the model to return the correct data type to
            ensure successful validation against your Pydantic schema.
            - **Good:** "Extract the number of reviews as an integer."
            - **Good:** "Get the price as a floating-point number, without the currency symbol."

        4.  **Leverage Pydantic for Structure**: For any non-trivial extraction (more than a single
            string), always use a Pydantic model. This forces the agent to return clean,
            structured, and validated data.

        5.  **Embrace Optional Fields for Robustness**: Web pages are unpredictable; an element
            might be missing. Define fields that might not always be present as `Optional` in
            your Pydantic model (e.g., `rating: Optional[float]`) to prevent failures.

        6.  **Resolve Visual Ambiguity**: If the page presents conflicting information,
            your query MUST instruct the model on how to resolve the conflict. Prioritize the
            element that reflects the true state of the page.
            - **Scenario:** A recipe page has a "2X" serving size button selected, but nearby static
              text says "Original recipe (1X) yields 6 servings".
            - **Bad Query:** "Get the number of servings." (The model may incorrectly read the static text).
            - **Good Query:** "Determine the active serving size multiplier. CRITICAL: Identify which
              multiplier button ('1/2X', '1X', '2X') is visually selected (e.g., has a checkmark or
              filled background). IGNORE any nearby static text like 'Original recipe yields...'."

        **✅ Good Queries (Following the Principles):**
        - **(Principles 1, 4, 5):** "List all user comments. For each comment, extract the author's
          name and the comment text. Also, extract the date it was posted, but note that the date
          may be missing for some older comments."
        - **(Principles 1, 2, 3, 4, 6):** "For every product on the page, extract the product name,
          the price as a float, and the star rating. For the rating, visually count the filled stars
          and return it as a number (e.g., 4.5). For the active sorting option, identify which one
          is visually highlighted in blue."

        **❌ Bad Queries (HTML/DOM Specific):**
        - "Get the href attribute of the 'About Us' link."
        # Instead, ask: "What is the destination URL of the 'About Us' link?"

        Args:
            query: The natural-language instruction for what to extract and, if necessary, a
                   strategy for visual interpretation.
            response_format: Optional. A Pydantic model to structure the output.
                             **Highly recommended for reliable extraction.**
            bypass_dom_processing: Optional. If True, skips DOM manipulation and uses
                                   screenshot-only extraction. This preserves the original
                                   page state but may be less accurate for text-heavy content.
        """
        await self._ensure_async_initialized()

        await self.barrier()

        def _safe_model_json_schema(model: type[BaseModel]):
            try:
                return model.model_json_schema()
            except PydanticUserError:
                model.model_rebuild()
                return model.model_json_schema()

        payload = {"instructions": query}
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            payload["schema"] = _safe_model_json_schema(response_format)
        if bypass_dom_processing:
            payload["bypassDomProcessing"] = True

        response = await self._request("POST", "/extract", payload)
        data = response.get("data")

        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            return response_format.model_validate(data)
        return data

    async def clear_pending_commands(self, run_id: int):
        """
        Removes all queued and active commands that were issued by a specific run_id.
        This is used to prevent stale commands from an old execution run from executing
        after the run has been cancelled and a new one has started.
        """
        # Drain existing queue items and requeue only those not from the cancelled run.
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
                logger.info(
                    f"Cancelling queued command from cancelled run_id={run_id}: {self._active_commands.get(command_id, ['unknown'])[0]}",
                )
                if future:
                    future.cancel()
                self._active_commands.pop(command_id, None)
                removed_count += 1

        # Requeue the kept items back onto the same queue to avoid swapping the queue object
        for item in kept_items:
            self._command_queue.put_nowait(item)

        logger.info(
            f"🧹 Cleared {removed_count}/{original_size} queued commands for run_id={run_id}.",
        )

    async def query(self, query: str, response_format: Any = str) -> Any:
        """
        Asks questions about the agent's action history and memory context.

        This method allows you to query the agent's understanding of what it has done and observed.
        It does not interact with the live webpage but rather introspects the agent's memory.

        **Key characteristics**:
        - **Memory-focused**: Uses the agent's accumulated memory and context from past actions.
        - **Historical analysis**: Analyzes what happened during previous `act()` calls.
        - **Context-aware**: Includes full agent memory context in the query.
        - **No fresh content**: Doesn't capture new page content, works with existing observations.

        Args:
            query: The natural-language question to ask about the agent's history.
            response_format: Optional. A Pydantic model to structure the output.

        **✅ Good Queries (What the agent has done):**
        - "Did the login attempt succeed?"
        - "What were the steps you took to add the item to the cart?"
        - "Summarize the actions you have performed so far."

        **❌ Bad Queries (Requires live page content):**
        - "What is the current price of the item on the page?" (Use `observe` for this)
        - "Click the 'Submit' button." (Use `act` for this)
        """
        await self._ensure_async_initialized()

        def _safe_model_json_schema(model: type[BaseModel]):
            try:
                return model.model_json_schema()
            except PydanticUserError:
                model.model_rebuild()
                return model.model_json_schema()

        payload = {"query": query}
        if inspect.isclass(response_format) and issubclass(
            response_format,
            BaseModel,
        ):
            payload["schema"] = _safe_model_json_schema(response_format)

        response = await self._request("POST", "/query", payload)
        data = response.get("data")

        if inspect.isclass(response_format) and issubclass(
            response_format,
            BaseModel,
        ):
            return response_format.model_validate(data)
        return data

    async def get_screenshot(self) -> str:
        await self._ensure_async_initialized()
        response = await self._request("POST", "/screenshot")
        return response.get("screenshot")

    async def get_current_url(self) -> str:
        try:
            response = await self._request("POST", "/state")
            return response.get("url", "")
        except Exception as e:
            return ""

    async def get_links(
        self,
        same_domain: bool = True,
        selector: str = None,
        **kwargs,
    ) -> dict:
        """
        Extract all links from the current page.

        Args:
            same_domain: If True, only return links from the same domain.
            selector: Optional CSS selector (default: 'a[href]').

        Returns:
            dict with keys:
            - base_url: Origin of current page
            - current_url: Full URL of current page
            - links: List of {href, text} objects
            - total: Number of links found
        """
        await self._ensure_async_initialized()
        payload = {"sameDomain": same_domain}
        if selector:
            payload["selector"] = selector
        return await self._request("POST", "/links", payload)

    async def get_content(self, format: str = "markdown", **kwargs) -> dict:
        """
        Get raw page content without LLM processing.

        Args:
            format: 'markdown' (default), 'text', or 'html'

        Returns:
            dict with keys:
            - url: Current page URL
            - title: Page title
            - content: Extracted content in requested format
            - format: The format used
        """
        await self._ensure_async_initialized()
        return await self._request("POST", "/content", {"format": format})

    async def navigate(self, url: str, wait: bool = True, context: dict = None) -> str:
        """Navigates to a given URL."""
        await self._ensure_async_initialized()
        logger.info(f"🐍 PYTHON: Navigating to URL: {url}")

        if self.agent_mode == "desktop":
            # Controlling virtual desktop
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
                    logger.info(
                        f"⚠️ Navigation failed due to closed page, retrying (attempt {attempt + 1}/{max_retries})...",
                    )
                    await asyncio.sleep(2)
                    continue
                raise

    def stop(self):
        """Stops the Node.js service subprocess and cancels background tasks."""
        # Cancel the new asyncio tasks
        if self._log_stream_task and not self._log_stream_task.done():
            self._log_stream_task.cancel()
        if self._log_consumer_task and not self._log_consumer_task.done():
            self._log_consumer_task.cancel()
        if self._command_processor_task and not self._command_processor_task.done():
            self._command_processor_task.cancel()

        try:
            if self._session_id:
                self._sync_request("POST", "/stop", {"sessionId": self._session_id})
                logger.info(f"✅ Stopped session {self._session_id}")
        except Exception as e:
            # Don't fail stop() if the request fails
            logger.info(f"Warning: Failed to send stop request: {e}")

        # If the backend started the process, terminate it
        if MagnitudeBackend._process:
            logger.info(
                f"🛑 Stopping Magnitude agent service (PID: {MagnitudeBackend._process.pid})...",
            )
            MagnitudeBackend._process.terminate()
            try:
                MagnitudeBackend._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                MagnitudeBackend._process.kill()
            MagnitudeBackend._process = None

    async def await_sequence_logs(self, seq: int) -> list[str]:
        """
        Waits until the command with the given sequence number has been processed,
        and returns the logs generated during its execution.
        """
        # Wait until the command is processed
        while self._processed_seq < seq:
            # If the command is not even in the queue or active map, and we are past it, it might be lost/skipped
            # But here we just wait for processed_seq to advance.
            # We can add a timeout or check if the command exists if needed.
            await asyncio.sleep(0.1)

        return self._log_buffer.get(seq, [])
