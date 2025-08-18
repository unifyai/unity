import inspect
import os
import subprocess
import sys
import time
import atexit
import threading
from abc import ABC, abstractmethod
from typing import Any
import socket
import contextlib

import aiohttp
import requests
from pydantic import BaseModel, PydanticUserError
import asyncio

from .controller import Controller


class BrowserAgentError(Exception):
    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(f"[{error_type}] {message}")


class BrowserBackend(ABC):
    """
    Abstract Base Class defining the interface for any browser backend.
    """

    @abstractmethod
    async def act(self, instruction: str, expectation: str = "") -> str:
        """Perform an action in the browser."""

    @abstractmethod
    async def observe(self, query: str, response_format: Any = str) -> Any:
        """Observe the state of the browser page."""

    @abstractmethod
    async def get_screenshot(self) -> str:
        """Get a base64 encoded screenshot of the current page."""

    @abstractmethod
    async def get_current_url(self) -> str:
        """Get the current URL of the browser."""

    @abstractmethod
    async def navigate(self, url: str) -> str:
        """Navigate the browser to a specific URL."""

    @abstractmethod
    def stop(self):
        """Cleanly shut down the backend."""


class LegacyBrowserBackend(BrowserBackend):
    """
    An implementation that uses the original, Controller-based browser stack.
    """

    def __init__(self, controller_mode: str = "hybrid", **kwargs):
        self.controller = Controller(mode=controller_mode, **kwargs)
        if not self.controller.is_alive():
            self.controller.start()

    async def act(self, instruction: str, expectation: str = "") -> str:
        """
        Performs a **single, high-level action** in the browser and verifies its outcome.

        This tool functions by looking at the screen; it **does not have access to the underlying HTML or DOM**. Therefore, instructions must describe elements based on their **visible text or position**, not by HTML attributes like `id`, `class`, or `aria-label`.

        Args:
            instruction (str): A single, natural-language command. Describe the element to interact with
                            based on its visible properties.
            expectation (str): A clear, verifiable description of what the page should look like *after*
                            the action is successfully completed.

        Examples:
            # ✅ Good Example (Using Visible Text)
            - instruction: "Click the 'Login' button"
            expectation: "The page should now show a password field."

            # ✅ Good Example (Using Visible Text)
            - instruction: "Type 'hello world' into the search bar"
            expectation: "The search bar should contain the text 'hello world'."

            # ❌ Bad Example (Using HTML Attributes)
            - instruction: "Click the button with id 'submit-btn'"
            # This will fail because the tool cannot see HTML IDs.

            # ❌ Bad Example (Using ARIA Labels)
            - instruction: "Click the image with 'logo' in the aria-label"
            # This will fail because the tool cannot see aria-labels.

            # ❌ Bad Example (Chained Actions)
            - instruction: "Click the login button and then enter 'my_user' into the username field."
        """
        return await self.controller.act(
            instruction,
            expectation=expectation,
            multi_step_mode=True,
        )

    async def observe(self, query: str, response_format: Any = str) -> Any:
        """
        Analyzes a screenshot of the current browser page to answer a question.

        This tool functions like a person looking at the screen; it **does not have access to the underlying HTML or DOM structure**. It can only answer questions about what is currently visible. Use it for read-only operations to gather information without changing the page state.

        **✅ Good Queries (What you can see):**
        - "What is the title of the page?"
        - "List the text on all visible buttons."
        - "Is the text 'Welcome back, user!' visible on the screen?"
        - "Transcribe the text from the paragraph under the 'About Us' heading."
        - "What is the phone number displayed at the top of the page?"

        **❌ Bad Queries (Requires HTML/DOM access):**
        - Avoid asking for non-visible information.
        - **Do not ask for HTML attributes** like `href`, `src`, or `alt` text (e.g., "What is the URL of the main product image?" or "Get the alt text for the logo.").
        - **Do not ask about HTML tags** (e.g., "Find all the `<h1>` tags.").
        - Avoid asking the tool to interpret meaning. Instead of "Does this image look professional?", ask "Describe the image in the center of the page."
        - Avoid multi-step queries. Instead of "Find the contact link and tell me the email," break it into separate steps.

        Args:
            query: The natural-language question to ask about what is visible on the page.
            response_format: Optional. A Pydantic model to structure the output. The LLM will return a JSON object matching the model.
        """
        return await self.controller.observe(query, response_format)

    async def get_screenshot(self) -> str:
        return self.controller._last_shot

    async def get_current_url(self) -> str:
        try:
            return self.controller.state.url
        except Exception as e:
            return ""

    async def navigate(self, url: str) -> str:
        return await self.controller.act(
            f"Navigate to {url}",
            expectation=f"The browser is on the page with URL '{url}'",
        )

    def stop(self):
        self.controller.stop()


class MagnitudeBrowserBackend(BrowserBackend):
    _process = None
    _agent_base_url = "http://localhost:3000"
    _lock = threading.Lock()

    @staticmethod
    def _find_free_port() -> int:
        """Find and return a free port on the system."""
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def __init__(
        self,
        agent_server_url: str = "http://localhost:3000",
        headless: bool = False,
        **kwargs,
    ):
        with MagnitudeBrowserBackend._lock:
            if MagnitudeBrowserBackend._process is None:
                self.agent_base_url = agent_server_url
                self._start_service(headless)
            else:
                print(
                    "✅ Magnitude service already running. Attaching to existing process.",
                )
                self.agent_base_url = MagnitudeBrowserBackend._agent_base_url

    def _start_service(self, headless: bool):
        port = self._find_free_port()
        MagnitudeBrowserBackend._agent_base_url = f"http://localhost:{port}"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        service_path = os.path.abspath(
            os.path.join(current_dir, "..", "..", "agent-service"),
        )
        script_path = os.path.join(service_path, "src", "index.ts")

        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"Could not find agent service script at expected path: {script_path}",
            )

        command = ["npx", "ts-node", script_path]
        if headless:
            command.append("--headless")

        env = os.environ.copy()
        env["PORT"] = str(port)

        MagnitudeBrowserBackend._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=service_path,
            env=env,
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )

        print(
            f"🚀 Starting Magnitude BrowserAgent service (PID: {MagnitudeBrowserBackend._process.pid}) on port {port}...",
        )

        self._start_output_readers()
        atexit.register(self.stop)

        deadline = time.time() + 30
        url = f"{MagnitudeBrowserBackend._agent_base_url}/screenshot"

        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=1)
                if r.status_code < 500:
                    print(f"✅ Magnitude service is ready on port {port}")
                    break
            except Exception:
                time.sleep(0.5)
        else:
            self.stop()
            raise RuntimeError(
                f"Magnitude BrowserAgent failed to become ready within 30 seconds on port {port}",
            )

    def _start_output_readers(self):
        """Start threads to read stdout/stderr to prevent buffer blocking."""

        def read_output(pipe, prefix):
            for line in iter(pipe.readline, ""):
                if line:
                    print(f"[{prefix}] {line.strip()}")
                    if "listening on http://localhost:" in line:
                        import re

                        match = re.search(r"http://localhost:(\d+)", line)
                        if match:
                            MagnitudeBrowserBackend._agent_base_url = (
                                f"http://localhost:{match.group(1)}"
                            )
                            print(
                                f"✨ Detected service running on {MagnitudeBrowserBackend._agent_base_url}",
                            )
            pipe.close()

        stdout_thread = threading.Thread(
            target=read_output,
            args=(MagnitudeBrowserBackend._process.stdout, "Magnitude"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=read_output,
            args=(MagnitudeBrowserBackend._process.stderr, "Magnitude-ERR"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

    async def _request(
        self,
        method: str,
        endpoint: str,
        payload: dict | None = None,
    ) -> Any:
        url = f"{MagnitudeBrowserBackend._agent_base_url}{endpoint}"

        retries = 3
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method,
                        url,
                        json=payload,
                        timeout=300,
                    ) as resp:
                        if resp.status >= 400:
                            try:
                                from ..planner.hierarchical_planner import (
                                    ReplanFromParentException,
                                )

                                error_data = await resp.json()
                                error_type = error_data.get(
                                    "error",
                                    "unknown_http_error",
                                )
                                message = error_data.get("message", "No error message.")
                                if error_type == "misalignment":
                                    raise ReplanFromParentException(message)
                                raise BrowserAgentError(error_type, message)
                            except Exception as e:
                                raise BrowserAgentError(
                                    "service_error",
                                    f"Server error: {resp.status} - {await resp.text()}",
                                ) from e
                        return await resp.json()
            except aiohttp.ClientConnectorError as e:
                if attempt < retries - 1:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise

    async def act(self, instruction: str, expectation: str = "") -> str:
        """
        Executes a high-level browser task using the Magnitude BrowserAgent.

        This tool is **autonomous and can perform multiple steps** (e.g., typing, clicking, scrolling) to achieve the goal described in the instruction. It operates based on a visual understanding of the page. The agent will return successfully only if it believes the task is complete.

        Args:
            instruction (str): A high-level, natural-language command describing the desired outcome.
            expectation (str): (Optional) A description of the expected state after the action, which helps the agent confirm success.

        Examples:
            # ✅ Good Example (Multi-Step Task)
            - instruction: "Log into the account using username 'testuser' and password 'password123'."
            # The agent will find the fields, type, and click the login button.

            # ✅ Good Example (Vague Goal, Agent figures it out)
            - instruction: "Find the cheapest blue t-shirt on the page and add it to the cart."
            # The agent will visually scan, find the item, and click the corresponding 'Add to Cart' button.

            # ✅ Good Example (Combining Action and Verification)
            - instruction: "Click the 'Promotions' link in the navigation bar."
            - expectation: "The page should show a heading titled 'Current Promotions'."

            # ❌ Bad Example (Too low-level)
            # Avoid breaking down simple actions. Let the agent handle it.
            - instruction: "Move the mouse to coordinate 250, 400, then click."
        """
        task_desc = f"{instruction}. {expectation}".strip()
        response = await self._request("POST", "/act", {"task": task_desc})
        return response.get("status", "success")

    async def observe(self, query: str, response_format: Any = str) -> Any:
        """
        Extracts structured information from the current page using the Magnitude BrowserAgent.

        The agent uses a vision-language model to analyze the page content and screenshot, allowing it to understand context and structure. It can return complex, nested data if a Pydantic model is provided.

         **Key Principles for Effective Observation:**

        1.  **Be Specific and Descriptive**: Don't just ask "what's on the page." Guide the agent. Instead of "get the product details," prefer "Extract the product name from the top, the price listed in bold, and the author's name below the title."

        2.  **Provide a Strategy for Non-Textual Elements**: For visual elements like star ratings, progress bars, or icons, you MUST provide a method for interpretation, as the model cannot infer it.
            * **Good (Star Rating):** "For the 'star_rating', visually count the number of filled yellow stars and provide it as a number (e.g., 4.0). If you see a half-filled star, add 0.5. If you cannot determine the rating, approximate the value to the nearest half-star."
            * **Good (Active Icon):** "Determine which navigation link is active by identifying the one that is underlined or has a different text color."
            * **Bad:** "Get the star rating." (This will fail if the rating is not plain text).

        3.  **Request Specific Data Types**: Guide the model to return the correct data type to ensure successful validation against your Pydantic schema.
            * **Good:** "Extract the number of reviews as an integer."
            * **Good:** "Get the price as a floating-point number, without the currency symbol."

        4.  **Leverage Pydantic for Structure**: For any non-trivial extraction (more than a single string), always use a Pydantic model. This forces the agent to return clean, structured, and validated data.

        5.  **Embrace Optional Fields for Robustness**: Web pages are unpredictable; an element might be missing for one item but not another. Define fields that might not always be present as `Optional` in your Pydantic model (e.g., `rating: Optional[float]`). This prevents the entire extraction from failing if a single piece of data is missing.

        **✅ Good Queries (Following the 5 Principles):**
        - **(Principles 1, 4, 5):** "List all user comments. For each comment, extract the author's name and the comment text. Also, extract the date it was posted, but note that the date may be missing for some older comments."
        - **(Principles 1, 2, 3, 4, 5):** "For every product card on the page, extract the product name, the price as a float, and the star rating. For the rating, visually count the number of filled stars and return it as a number (e.g., 4.0 or 4.5). If an exact value cannot be determined, approximate the value to the nearest half-star."
        - **(Principles 1, 2, 4, 5):** "From the user data table, extract a list of users. For each user, get their full name and email. Also, check their 'Status' icon: a green checkmark means 'Active', and a red 'X' means 'Inactive'. Extract the status as the corresponding string. The email may be missing for some users."

        **❌ Bad Queries (HTML/DOM Specific):**
        - "Get the href attribute of the 'About Us' link."
        # Instead, ask: "What is the destination URL of the 'About Us' link?" The agent can often infer this by navigating and checking the URL.

        Args:
            query: The natural-language instruction for what to extract and, if necessary, a strategy for visual interpretation.
            response_format: Optional. A Pydantic model to structure the output. **Highly recommended for reliable extraction.**
        """

        def _safe_model_json_schema(model: type[BaseModel]):
            try:
                return model.model_json_schema()
            except PydanticUserError:
                model.model_rebuild()
                return model.model_json_schema()

        payload = {"instructions": query}
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            payload["schema"] = _safe_model_json_schema(response_format)

        response = await self._request("POST", "/extract", payload)
        data = response.get("data")

        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            return response_format.model_validate(data)
        return data

    async def get_screenshot(self) -> str:
        response = await self._request("GET", "/screenshot")
        return response.get("screenshot")

    async def get_current_url(self) -> str:
        try:
            # Get the current URL through the browser state
            response = await self._request("GET", "/state")
            return response.get("url", "")
        except Exception as e:
            return ""

    async def navigate(self, url: str) -> str:
        """Navigates the browser using the dedicated /nav endpoint."""
        print(f"🐍 PYTHON: Navigating to URL: {url}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self._request("POST", "/nav", {"url": url})
                return response.get("status", "success")
            except BrowserAgentError as e:
                if "Target page" in str(e) and attempt < max_retries - 1:
                    print(
                        f"⚠️ Navigation failed due to closed page, retrying (attempt {attempt + 1}/{max_retries})...",
                    )
                    await asyncio.sleep(2)
                    continue
                raise

    def stop(self):
        """Stops the Node.js service subprocess."""
        with MagnitudeBrowserBackend._lock:
            if (
                MagnitudeBrowserBackend._process
                and MagnitudeBrowserBackend._process.poll() is None
            ):
                print(
                    f"🐍 PYTHON: Explicitly calling stop() on MagnitudeBrowserBackend. PID: {MagnitudeBrowserBackend._process.pid}",
                )
                print(
                    f"🛑 Stopping Magnitude BrowserAgent service (PID: {MagnitudeBrowserBackend._process.pid})...",
                )
                if sys.platform != "win32":
                    import signal

                    try:
                        os.killpg(
                            os.getpgid(MagnitudeBrowserBackend._process.pid),
                            signal.SIGTERM,
                        )
                    except ProcessLookupError:
                        pass
                else:
                    MagnitudeBrowserBackend._process.terminate()

                try:
                    MagnitudeBrowserBackend._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    MagnitudeBrowserBackend._process.kill()
                MagnitudeBrowserBackend._process = None


# ---------------------------------------------------------------------------
#  Desktop backend (xdotool/x11 driven) via the same Magnitude service
# ---------------------------------------------------------------------------


class MagnitudeDesktopBackend(BrowserBackend):
    """Thin wrapper that calls the `/linux/*` endpoints exposed by the Node
    BrowserAgent service. It now manages its own service lifecycle instead of
    piggy-backing on `MagnitudeBrowserBackend`.
    """

    _process = None
    _agent_base_url = "http://localhost:3000"
    _lock = threading.Lock()

    @staticmethod
    def _find_free_port() -> int:
        """Find and return a free port on the system."""
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def __init__(self, **kwargs):
        # Headless service is fine – we only need its Linux endpoints
        with MagnitudeDesktopBackend._lock:
            if MagnitudeDesktopBackend._process is None:
                self.agent_base_url = MagnitudeDesktopBackend._agent_base_url
                self._start_service(headless=True)
            else:
                print(
                    "✅ Magnitude service already running. Attaching to existing process.",
                )
                self.agent_base_url = MagnitudeDesktopBackend._agent_base_url
        # Initialize per-act loop state
        self._last_exists: dict[str, bool] = {}
        self._last_windows: list[dict] = []
        self._typed_history: list[str] = []
        self._enter_after_last_type: bool = False
        self._last_tool_name: str | None = None
        self._last_focus_title: str | None = None
        self._consecutive_repeat_count: int = 0
        self._target_cmd: str | None = None
        # Track last two full desktop screenshots captured by the loop
        self._prev_full_shot_b64: str = ""
        self._curr_full_shot_b64: str = ""

    def _start_service(self, headless: bool):
        port = self._find_free_port()
        MagnitudeDesktopBackend._agent_base_url = f"http://localhost:{port}"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        service_path = os.path.abspath(
            os.path.join(current_dir, "..", "..", "agent-service"),
        )
        script_path = os.path.join(service_path, "src", "index.ts")

        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"Could not find agent service script at expected path: {script_path}",
            )

        command = ["npx", "ts-node", script_path]
        if headless:
            command.append("--headless")

        env = os.environ.copy()
        env["PORT"] = str(port)

        MagnitudeDesktopBackend._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=service_path,
            env=env,
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )

        print(
            f"🚀 Starting Magnitude BrowserAgent service (PID: {MagnitudeDesktopBackend._process.pid}) on port {port}...",
        )

        self._start_output_readers()
        atexit.register(self.stop)

        deadline = time.time() + 30
        url = f"{MagnitudeDesktopBackend._agent_base_url}/screenshot"

        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=1)
                if r.status_code < 500:
                    print(f"✅ Magnitude service is ready on port {port}")
                    break
            except Exception:
                time.sleep(0.5)
        else:
            self.stop()
            raise RuntimeError(
                f"Magnitude BrowserAgent failed to become ready within 30 seconds on port {port}",
            )

    def _start_output_readers(self):
        """Start threads to read stdout/stderr to prevent buffer blocking."""

        def read_output(pipe, prefix):
            for line in iter(pipe.readline, ""):
                if line:
                    print(f"[{prefix}] {line.strip()}")
                    if "listening on http://localhost:" in line:
                        import re

                        match = re.search(r"http://localhost:(\d+)", line)
                        if match:
                            MagnitudeDesktopBackend._agent_base_url = (
                                f"http://localhost:{match.group(1)}"
                            )
                            print(
                                f"✨ Detected service running on {MagnitudeDesktopBackend._agent_base_url}",
                            )
            pipe.close()

        stdout_thread = threading.Thread(
            target=read_output,
            args=(MagnitudeDesktopBackend._process.stdout, "Magnitude"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=read_output,
            args=(MagnitudeDesktopBackend._process.stderr, "Magnitude-ERR"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

    async def _request(
        self,
        method: str,
        endpoint: str,
        payload: dict | None = None,
    ) -> Any:
        url = f"{MagnitudeDesktopBackend._agent_base_url}{endpoint}"

        retries = 3
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method,
                        url,
                        json=payload,
                        timeout=300,
                    ) as resp:
                        if resp.status >= 400:
                            try:
                                from ..planner.hierarchical_planner import (
                                    ReplanFromParentException,
                                )

                                error_data = await resp.json()
                                error_type = error_data.get(
                                    "error",
                                    "unknown_http_error",
                                )
                                message = error_data.get("message", "No error message.")
                                if error_type == "misalignment":
                                    raise ReplanFromParentException(message)
                                raise BrowserAgentError(error_type, message)
                            except Exception as e:
                                raise BrowserAgentError(
                                    "service_error",
                                    f"Server error: {resp.status} - {await resp.text()}",
                                ) from e
                        return await resp.json()
            except aiohttp.ClientConnectorError as e:
                if attempt < retries - 1:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise

    async def _pointer_context_line(self) -> str:
        """Return 'Mouse/Pointer: x=.. y=.. screen=.. window=..' or empty string on failure."""
        try:
            mouse = await self._request("GET", "/linux/mouse/position")
            if isinstance(mouse, dict):
                px = mouse.get("x")
                py = mouse.get("y")
                scr = mouse.get("screen")
                win = mouse.get("window")
                return f"Mouse/Pointer: x={px} y={py} screen={scr} window={win}"
        except Exception:
            pass
        return ""

    async def _get_focused_window_geometry(self) -> dict | None:
        """Best-effort: return geometry of the currently focused window.

        Strategy:
        - Read window id from /linux/mouse/position
        - Match against /linux/window list (id,x,y,w,h)
        - Fallback to the first window if no exact match
        """
        try:
            pos = await self._request("GET", "/linux/mouse/position")
            wid = str((pos or {}).get("window") or "").strip().lower()
        except Exception:
            wid = ""
        wins = []
        try:
            res = await self._request("GET", "/linux/window")
            wins = res.get("windows", []) or []
        except Exception:
            wins = []

        def _norm(v):
            return str(v or "").strip().lower()

        if wid:
            for w in wins:
                if _norm(w.get("id")) == wid:
                    return {
                        "x": int(w.get("x", 0) or 0),
                        "y": int(w.get("y", 0) or 0),
                        "w": int(w.get("w", 0) or 0),
                        "h": int(w.get("h", 0) or 0),
                    }
        if wins:
            w = wins[0]
            return {
                "x": int(w.get("x", 0) or 0),
                "y": int(w.get("y", 0) or 0),
                "w": int(w.get("w", 0) or 0),
                "h": int(w.get("h", 0) or 0),
            }
        return None

    # Private helper to execute a selected tool step and update context
    async def _execute_tool(self, tool: str, args: dict, dims: tuple[int, int]) -> bool:

        if tool == "exists_window":
            title = str(
                args.get("title") or args.get("windowTitle") or args.get("name") or "",
            ).strip()
            if not title:
                return False
            try:
                res = await self._request(
                    "GET",
                    f"/linux/window/exist?windowTitle={requests.utils.quote(title)}",
                )
                self._last_exists[title] = bool(res.get("exists", False))
            except Exception:
                return False
            return True

        if tool == "focus_window":
            title = str(
                args.get("title") or args.get("windowTitle") or args.get("name") or "",
            ).strip()
            if not title:
                return False
            if title == self._last_focus_title and self._consecutive_repeat_count >= 1:
                # nudge progress if focusing repeats
                try:
                    await self._request(
                        "POST",
                        "/linux/type",
                        payload={"keys": ["Enter"]},
                    )
                    await asyncio.sleep(0.2)
                except Exception:
                    return False
                return True
            await self._request("POST", "/linux/window/focus", payload={"title": title})
            self._last_focus_title = title
            self._enter_after_last_type = False
            return True

        if tool == "list_windows":
            try:
                res = await self._request("GET", "/linux/window")
                self._last_windows = res.get("windows", []) or []
            except Exception:
                return False
            return True

        if tool == "list_windows_extended":
            try:
                res = await self._request("GET", "/linux/window?extended=1")
                self._last_windows = res.get("windows", []) or []
            except Exception:
                return False
            return True

        if tool == "move_window":
            id_val = args.get("id")
            title_val = args.get("title")
            try:
                x_val = int(args.get("x"))
                y_val = int(args.get("y"))
            except Exception:
                return False
            await self._request(
                "POST",
                "/linux/window/move",
                payload={"id": id_val, "title": title_val, "x": x_val, "y": y_val},
            )
            return True

        if tool == "resize_window":
            id_val = args.get("id")
            title_val = args.get("title")
            try:
                w_val = int(args.get("w"))
                h_val = int(args.get("h"))
            except Exception:
                return False
            await self._request(
                "POST",
                "/linux/window/resize",
                payload={"id": id_val, "title": title_val, "w": w_val, "h": h_val},
            )
            return True

        if tool == "set_window_state":
            id_val = args.get("id")
            title_val = args.get("title")
            action_val = str(args.get("action") or "").strip().lower()
            if action_val not in {"minimize", "maximize", "restore", "close"}:
                return False
            await self._request(
                "POST",
                "/linux/window/state",
                payload={"id": id_val, "title": title_val, "action": action_val},
            )
            return True

        if tool == "app_open":
            cmd = args.get("cmd")
            if not cmd:
                return False
            payload = {
                "cmd": cmd,
                "args": args.get("args") or [],
                "cwd": args.get("cwd"),
                "wait": bool(args.get("wait", False)),
            }
            # Optional env omitted for safety unless explicitly passed
            if isinstance(args.get("env"), dict):
                payload["env"] = args.get("env")
            await self._request("POST", "/linux/app/open", payload=payload)
            # Refresh window context to nudge LLM toward focus_window rather than re-open
            try:
                await asyncio.sleep(0.5)
                res = await self._request("GET", "/linux/window?extended=1")
                self._last_windows = res.get("windows", []) or []
            except Exception:
                pass
            return True

        if tool == "app_focus":
            title_val = args.get("title")
            class_val = args.get("class")
            if not (title_val or class_val):
                return False
            await self._request(
                "POST",
                "/linux/window/focus",
                payload={"title": title_val, "class": class_val},
            )
            self._last_focus_title = str(title_val or self._last_focus_title or "")
            self._enter_after_last_type = False
            return True

        if tool == "app_close":
            title_val = args.get("title")
            class_val = args.get("class")
            if not (title_val or class_val):
                return False
            await self._request(
                "POST",
                "/linux/window/state",
                payload={"title": title_val, "class": class_val, "action": "close"},
            )
            # Refresh window context after closing
            try:
                await asyncio.sleep(0.3)
                res = await self._request("GET", "/linux/window?extended=1")
                self._last_windows = res.get("windows", []) or []
            except Exception:
                pass
            return True

        if tool == "click_at":
            # Support absolute pixels or window-relative percents
            if args.get("xPercent") is not None and args.get("yPercent") is not None:
                try:
                    win = await self._get_focused_window_geometry()
                    if not win or win.get("w", 0) <= 0 or win.get("h", 0) <= 0:
                        return False
                    xp = max(0.0, min(1.0, float(args.get("xPercent"))))
                    yp = max(0.0, min(1.0, float(args.get("yPercent"))))
                    x_val = int(win["x"] + xp * win["w"])
                    y_val = int(win["y"] + yp * win["h"])
                except Exception:
                    return False
            else:
                try:
                    x_val = int(args.get("x"))
                    y_val = int(args.get("y"))
                except Exception:
                    return False
            button = int(args.get("button", 1) or 1)
            # optional enhancements
            repeat_val = None
            delay_ms = None
            modifiers_val = None
            try:
                if args.get("repeat") is not None:
                    repeat_val = max(1, int(args.get("repeat")))
            except Exception:
                repeat_val = None
            try:
                if args.get("delayMs") is not None:
                    delay_ms = max(0, int(args.get("delayMs")))
            except Exception:
                delay_ms = None
            try:
                if isinstance(args.get("modifiers"), list):
                    modifiers_val = [str(m) for m in args.get("modifiers")]
            except Exception:
                modifiers_val = None
            if dims[0] > 0 and dims[1] > 0:
                x_val = max(0, min(dims[0] - 1, x_val))
                y_val = max(0, min(dims[1] - 1, y_val))
            mouse_before = None
            try:
                mouse_before = await self._request("GET", "/linux/mouse/position")
            except Exception:
                mouse_before = None
            await self._request(
                "POST",
                "/linux/click",
                payload={
                    "clicks": [
                        {
                            "x": x_val,
                            "y": y_val,
                            "button": button,
                            **(
                                {"repeat": repeat_val} if repeat_val is not None else {}
                            ),
                            **({"delayMs": delay_ms} if delay_ms is not None else {}),
                            **(
                                {"modifiers": modifiers_val}
                                if modifiers_val is not None
                                else {}
                            ),
                        },
                    ],
                },
            )
            await asyncio.sleep(0.2)
            mouse_after = None
            try:
                mouse_after = await self._request("GET", "/linux/mouse/position")
            except Exception:
                mouse_after = None

            def _dist(a, b):
                try:
                    return abs((a or {}).get("x", -1) - (b or {}).get("x", -1)) + abs(
                        (a or {}).get("y", -1) - (b or {}).get("y", -1),
                    )
                except Exception:
                    return 0

            if _dist(mouse_after, mouse_before) < 2:
                self._consecutive_repeat_count = max(self._consecutive_repeat_count, 1)
            return True

        if tool == "type_text":
            # Support either plain text, or structured keys/combos
            if args.get("keys") is not None:
                keys_payload = args.get("keys")
                try:
                    await self._request(
                        "POST",
                        "/linux/type",
                        payload={"keys": keys_payload},
                    )
                except Exception:
                    return False
                # best-effort typed history update
                try:
                    if isinstance(keys_payload, list):
                        flat_parts: list[str] = []
                        for item in keys_payload:
                            if isinstance(item, list):
                                if item:
                                    flat_parts.append("+".join([str(x) for x in item]))
                            else:
                                flat_parts.append(str(item))
                        if flat_parts:
                            self._typed_history.append(" ".join(flat_parts))
                    else:
                        self._typed_history.append(str(keys_payload))
                except Exception:
                    pass
                self._enter_after_last_type = False
                return True
            text = str(args.get("text") or "")
            if not text:
                return False
            if text.endswith("\n"):
                text = text[:-1]
                await self._request("POST", "/linux/type", payload={"keys": [text]})
                await self._request("POST", "/linux/type", payload={"keys": ["Enter"]})
                self._typed_history.append(text)
                self._enter_after_last_type = True
                await asyncio.sleep(0.4)
            else:
                await self._request("POST", "/linux/type", payload={"keys": [text]})
                self._typed_history.append(text)
                self._enter_after_last_type = False
            return True

        if tool == "mouse_move":
            try:
                x_val = int(args.get("x"))
                y_val = int(args.get("y"))
            except Exception:
                return False
            if dims[0] > 0 and dims[1] > 0:
                x_val = max(0, min(dims[0] - 1, x_val))
                y_val = max(0, min(dims[1] - 1, y_val))
            await self._request(
                "POST",
                "/linux/mouse/move",
                payload={"x": x_val, "y": y_val},
            )
            return True

        if tool == "drag":
            # fromX/fromY optional; if absent service uses current pointer
            def _opt(name: str):
                try:
                    val = args.get(name)
                    return int(val) if val is not None else None
                except Exception:
                    return None

            # Support percent inputs relative to focused window
            use_pct = (
                args.get("toXPercent") is not None
                and args.get("toYPercent") is not None
            ) or (
                args.get("fromXPercent") is not None
                and args.get("fromYPercent") is not None
            )

            if use_pct:
                try:
                    win = await self._get_focused_window_geometry()
                    if not win or win.get("w", 0) <= 0 or win.get("h", 0) <= 0:
                        return False

                    def _pct(v):
                        return max(0.0, min(1.0, float(v)))

                    from_x = (
                        None
                        if args.get("fromXPercent") is None
                        else int(
                            win["x"] + _pct(args.get("fromXPercent")) * win["w"],
                        )
                    )
                    from_y = (
                        None
                        if args.get("fromYPercent") is None
                        else int(
                            win["y"] + _pct(args.get("fromYPercent")) * win["h"],
                        )
                    )
                    to_x = (
                        int(win["x"] + _pct(args.get("toXPercent")) * win["w"])
                        if args.get("toXPercent") is not None
                        else _opt("toX")
                    )
                    to_y = (
                        int(win["y"] + _pct(args.get("toYPercent")) * win["h"])
                        if args.get("toYPercent") is not None
                        else _opt("toY")
                    )
                except Exception:
                    return False
            else:
                from_x = _opt("fromX")
                from_y = _opt("fromY")
                to_x = _opt("toX")
                to_y = _opt("toY")
            if to_x is None or to_y is None:
                return False
            if dims[0] > 0 and dims[1] > 0:
                if from_x is not None:
                    from_x = max(0, min(dims[0] - 1, from_x))
                if from_y is not None:
                    from_y = max(0, min(dims[1] - 1, from_y))
                to_x = max(0, min(dims[0] - 1, to_x))
                to_y = max(0, min(dims[1] - 1, to_y))
            payload = {
                **({"fromX": from_x} if from_x is not None else {}),
                **({"fromY": from_y} if from_y is not None else {}),
                "toX": to_x,
                "toY": to_y,
            }
            try:
                if args.get("button") is not None:
                    payload["button"] = int(args.get("button"))
            except Exception:
                pass
            try:
                if args.get("steps") is not None:
                    payload["steps"] = max(1, int(args.get("steps")))
            except Exception:
                pass
            try:
                if args.get("delayMs") is not None:
                    payload["delayMs"] = max(0, int(args.get("delayMs")))
            except Exception:
                pass
            # optional modifiers for drag operations
            try:
                if isinstance(args.get("modifiers"), list):
                    payload["modifiers"] = [str(m) for m in args.get("modifiers")]
            except Exception:
                pass
            await self._request("POST", "/linux/drag", payload=payload)
            return True

        if tool == "scroll":
            direction = str(args.get("direction") or "").lower().strip()
            if direction not in {"up", "down", "left", "right"}:
                return False
            payload = {"direction": direction}
            try:
                if args.get("amount") is not None:
                    payload["amount"] = max(1, int(args.get("amount")))
            except Exception:
                pass
            try:
                if args.get("delayMs") is not None:
                    payload["delayMs"] = max(0, int(args.get("delayMs")))
            except Exception:
                pass
            await self._request("POST", "/linux/scroll", payload=payload)
            return True

        if tool == "screenshot":
            try:
                out = await self._request("GET", "/linux/screenshot")
                b64 = (out or {}).get("screenshot")
                if b64:
                    self._last_exists["_last_full_screenshot"] = True
            except Exception:
                return False
            return True

        if tool == "screenshot_region":
            try:
                x = int(args.get("x"))
                y = int(args.get("y"))
                w = int(args.get("w"))
                h = int(args.get("h"))
            except Exception:
                return False
            # clamp to dims
            if dims[0] > 0 and dims[1] > 0:
                x = max(0, min(dims[0] - 1, x))
                y = max(0, min(dims[1] - 1, y))
                w = max(1, min(dims[0] - x, w))
                h = max(1, min(dims[1] - y, h))
            out = await self._request(
                "GET",
                f"/linux/screenshot/region?x={x}&y={y}&w={w}&h={h}",
            )
            # store last region bytes for potential region_changed
            try:
                b64 = out.get("screenshot")
                if b64:
                    # store as before if empty; otherwise after
                    if not self._last_region_before_b64:
                        self._last_region_before_b64 = b64
                    else:
                        self._last_region_after_b64 = b64
                    self._last_region_rect = {"x": x, "y": y, "w": w, "h": h}
            except Exception:
                pass
            return True

        if tool == "image_locate":
            payload: dict = {}
            if args.get("templatePath"):
                payload["templatePath"] = str(args.get("templatePath"))
            elif args.get("templateB64"):
                payload["templateB64"] = str(args.get("templateB64"))
            else:
                return False
            for k in ("x", "y", "w", "h"):
                if args.get(k) is not None:
                    try:
                        payload[k] = int(args.get(k))
                    except Exception:
                        pass
            if args.get("threshold") is not None:
                try:
                    payload["threshold"] = float(args.get("threshold"))
                except Exception:
                    pass
            out = await self._request("POST", "/linux/image/locate", payload=payload)
            try:
                self._last_image_locate = out
            except Exception:
                pass
            return True

        if tool == "ocr_locate_text":
            # args: { query, x?, y?, w?, h?, caseSensitive?, exact? }
            q = str(args.get("query") or args.get("text") or "").strip()
            if not q:
                return False
            payload: dict = {"query": q}
            for k in ("x", "y", "w", "h"):
                if args.get(k) is not None:
                    try:
                        payload[k] = int(args.get(k))
                    except Exception:
                        pass
            if args.get("caseSensitive") is not None:
                payload["caseSensitive"] = bool(args.get("caseSensitive"))
            if args.get("exact") is not None:
                payload["exact"] = bool(args.get("exact"))
            out = await self._request("POST", "/linux/ocr/locate_text", payload=payload)
            try:
                self._last_ocr_bbox = out if out and out.get("found") else None
            except Exception:
                self._last_ocr_bbox = None
            return True

        if tool == "click_bbox_center":
            # Prefer provided bbox in args; else use last OCR bbox
            bbox = None
            if all(k in args for k in ("x", "y", "w", "h")):
                try:
                    bbox = {
                        "x": int(args.get("x")),
                        "y": int(args.get("y")),
                        "w": int(args.get("w")),
                        "h": int(args.get("h")),
                    }
                except Exception:
                    bbox = None
            if bbox is None:
                bbox = getattr(self, "_last_ocr_bbox", None)
            if not bbox or not bbox.get("w") or not bbox.get("h"):
                return False
            cx = int(bbox["x"] + bbox["w"] / 2)
            cy = int(bbox["y"] + bbox["h"] / 2)
            await self._request(
                "POST",
                "/linux/click",
                payload={"x": cx, "y": cy, "button": int(args.get("button", 1) or 1)},
            )
            return True

        if tool == "region_changed":
            need = ("beforeB64", "afterB64", "x", "y", "w", "h")
            if not all(k in args for k in need):
                return False
            payload = {
                "beforeB64": str(args.get("beforeB64")),
                "afterB64": str(args.get("afterB64")),
                "x": int(args.get("x")),
                "y": int(args.get("y")),
                "w": int(args.get("w")),
                "h": int(args.get("h")),
            }
            if args.get("metric"):
                payload["metric"] = str(args.get("metric"))
            if args.get("threshold") is not None:
                try:
                    payload["threshold"] = float(args.get("threshold"))
                except Exception:
                    pass
            out = await self._request("POST", "/linux/region/changed", payload=payload)
            try:
                self._last_exists["_last_region_changed"] = bool(out.get("changed"))
            except Exception:
                pass
            return True

        if tool == "type_and_enter":
            text = str(args.get("text") or "")
            if not text:
                return False
            await self._request(
                "POST",
                "/linux/type",
                payload={"keys": [text, "Enter"]},
            )
            self._typed_history.append(text)
            self._enter_after_last_type = True
            await asyncio.sleep(0.4)
            return True

        if tool == "press_enter":
            await self._request("POST", "/linux/type", payload={"keys": ["Enter"]})
            if self._typed_history:
                self._enter_after_last_type = True
                await asyncio.sleep(0.4)
            return True

        # --- Recording and File System Tools ---
        if tool == "record_start":
            # Start screen recording at given fps and optional region
            params: dict = {}
            if args.get("fps") is not None:
                try:
                    params["fps"] = int(args.get("fps"))
                except:
                    pass
            if isinstance(args.get("region"), dict):
                params["region"] = args.get("region")
            await self._request("POST", "/linux/record/start", payload=params)
            return True

        if tool == "record_stop":
            # Stop screen recording by id
            rec_id = args.get("id")
            if not rec_id:
                return False
            await self._request("POST", "/linux/record/stop", payload={"id": rec_id})
            return True

        if tool == "fs_list":
            # List files in a directory
            path_val = str(args.get("path") or "")
            if not path_val:
                return False
            # URL-encode path
            encoded = requests.utils.quote(path_val)
            await self._request("GET", f"/linux/fs/list?path={encoded}")
            return True

        if tool == "fs_write":
            # Write a file with base64 content
            p = args.get("path")
            content = args.get("contentBase64")
            if not p or not isinstance(content, str):
                return False
            await self._request(
                "POST",
                "/linux/fs/write",
                payload={"path": str(p), "contentBase64": content},
            )
            return True

        # Unknown tool
        return False

    # ------------------------------------------------------------------
    #  Core actions
    # ------------------------------------------------------------------

    async def act(self, instruction: str, expectation: str = "") -> str:
        """
        Execute a high‑level, natural‑language desktop command by iteratively
        selecting tools until the instruction is achieved.

        How it completes:
        - The loop executes one tool per step and then observes the screen.
        - Completion is decided by observing that the instruction (or the explicit
          expectation) is satisfied.

        Good instruction examples (high‑level):
        - "Focus the 'xterm' window, type 'echo READY' and press Enter. Finish when 'READY' is visible."
        - "Install 'cowsay' using apt-get -y and run 'cowsay READY'; finish when the cowsay output shows READY."
        - "Move the 'xterm' window to around (50,50) and resize it to about 900x700; finish when that looks true."
        - "Maximize the 'xterm' window; finish when it fills most of the screen."
        - "Take a full desktop screenshot; finish when done."

        Good expectation examples:
        - "The 'xterm' window looks focused (title bar highlighted or on top)."
        - "The terminal shows the word READY."
        - "The 'xterm' window appears near the top-left and roughly 900x700 in size (±15%)."

        Bad instruction examples (overly low‑level/ambiguous):
        - "Move to (250,400), click, then move to (300,420) and click again." (prefer a goal like
          "Click the OK button" or describe the visible target)
        - "Press the key with code 38." (use human‑readable key names or plain text typing)
        - "Close the hidden window with PID 1234." (decide from visible state; use title/class)
        - "Drag from somewhere to somewhere right." (describe what to drag or provide approximate
          start/end in context of the screenshot)
        - "Click the exact pixel (123,456) twice." (prefer describing the visible target or use
          window‑relative percents in the focused window)

        Notes:
        - Coordinates are clamped to the screenshot bounds.
        - Prefer window‑relative percents for pointing when a position must be used
          (xPercent/yPercent in [0,1] of the focused window).
        - Modifiers for input are supported: Shift, Ctrl, Alt, Super (Command/Win).
        - App install is supported via typing in a focused terminal or via app_open with
          bash -lc and non‑interactive flags (e.g., -y).
        """
        import json

        # High-level NL instruction path: LLM tool loop
        try:
            import unify  # type: ignore
            from pydantic import BaseModel, Field
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "High-level desktop actions require the 'unify' client to be installed and configured.",
            ) from e

        class NextAction(BaseModel):
            tool: str = Field(
                ...,
                description=(
                    "One of: exists_window, focus_window, list_windows, list_windows_extended, "
                    "move_window, resize_window, set_window_state, app_open, app_focus, app_close, "
                    "type_text, type_and_enter, click_at, mouse_move, drag, scroll, screenshot, screenshot_region, image_locate, region_changed, press_enter"
                ),
            )
            args: dict = Field(default_factory=dict)
            reason: str | None = None
            verify: str | None = None

        client = unify.Unify(endpoint="claude-4-sonnet@anthropic")
        client.set_response_format(NextAction)
        client.set_system_message(
            "You are a desktop control assistant. Decide the next atomic tool to apply to achieve the user's goal.\n"
            "Return ONLY JSON matching the response schema.\n"
            "For each selected tool, include a short 'verify' statement that should be TRUE if the tool succeeded (a visual check the observer can answer).\n"
            "Available tools (tool field):\n"
            "- exists_window: args {title} — check if a window with given title exists.\n"
            "- focus_window: args {title|class} — bring a window to front by title or WM_CLASS.\n"
            "- list_windows: args {} — list windows; you will receive id, geometry and title.\n"
            "- list_windows_extended: args {} — list windows with pid/class using wmctrl -lx -p.\n"
            "- move_window: args {id|title, x, y} — move a window.\n"
            "- resize_window: args {id|title, w, h} — resize a window.\n"
            "- set_window_state: args {id|title, action} — action in {minimize|maximize|restore|close}.\n"
            "- app_open: args {cmd, args?, cwd?, wait?} — launch an application/command.\n"
            "- app_focus: args {title|class} — focus a running app window (alias of focus_window).\n"
            "- app_close: args {title|class} — close a running app by its window(s).\n"
            "- type_text: args {text | keys} — either plain text, or keys (supports combos as nested arrays, e.g., ['ctrl','c'] or ['ctrl','shift','t']).\n"
            "- type_and_enter: args {text} — type the given text and then press Enter.\n"
            "- click_at: args {x, y, button?, repeat?, delayMs?, modifiers?[]} — click at pixel coordinates. Prefer window-relative percents: {xPercent, yPercent} in [0,1] referencing the currently focused window.\n"
            "- mouse_move: args {x, y} — move/hover the pointer without clicking.\n"
            "- drag: args {fromX?, fromY?, toX, toY, button?, steps?, delayMs?} — click-drag between points. Prefer percents: {fromXPercent?, fromYPercent?, toXPercent, toYPercent} in [0,1] of the focused window.\n"
            "- scroll: args {direction:'up'|'down'|'left'|'right', amount?, delayMs?} — scroll wheel events.\n"
            "- ocr_locate_text: args {query, x?, y?, w?, h?, caseSensitive?, exact?} — run OCR on the current screenshot/region to get a text bounding box.\n"
            "- click_bbox_center: args {x?, y?, w?, h?, button?} — click the center of a provided bbox, or of the last OCR-located bbox if omitted.\n"
            "- screenshot: args {} — capture the full desktop screenshot (base64).\n"
            "- screenshot_region: args {x,y,w,h} — returns a cropped screenshot region (base64).\n"
            "- image_locate: args {templatePath|templateB64, x?,y?,w?,h?, threshold?} — locate template on screen/region.\n"
            "- region_changed: args {beforeB64, afterB64, x,y,w,h, metric?, threshold?} — compare before/after regions.\n"
            "- press_enter: args {} — press the Enter key.\n"
            "You can install apps/packages by running shell commands. If no terminal is open, prefer app_open with cmd='bash' and args=['-lc', '<install command>'] (e.g., 'sudo apt-get update && sudo apt-get install -y <pkg>'). If a terminal (e.g., xterm) is already open, focus_window it and use click_at/type_text/press_enter to run commands. Use non-interactive flags (-y, -q, --no-install-recommends) to avoid prompts.\n"
            "You can use key combos by passing a list of keys to type_text. For example, ['ctrl', 'c'] will copy the current selection, and ['ctrl', 'shift', 't'] will open a new terminal window.\n"
            "You will ALWAYS receive a current desktop screenshot. Base your decisions primarily on that visual context.\n"
            "You will also receive screen width/height (pixels) and pointer coordinates. Prefer relative percent coordinates in the focused window (xPercent/yPercent) when you must target by position. When targeting specific text, first call ocr_locate_text('<text>') then click_bbox_center; for selection, prefer double-/triple-click and Shift-extend rather than raw drags.\n"
            "Avoid repeating the same tool more than once in a row; after focusing a window, prefer typing or clicking next.\n"
            "For terminal commands, typically: focus_window → click_at (if needed) → type_text (the exact command) → press_enter.\n",
        )

        max_steps = 30
        step = 0
        # Reset per-instruction state
        self._last_exists = {}
        self._last_windows = []
        self._typed_history = []
        self._enter_after_last_type = False
        self._last_tool_name = None
        self._last_focus_title = None
        self._consecutive_repeat_count = 0

        def _context_blob() -> dict:
            return {
                "known_windows": self._last_exists,
                "windows": self._last_windows,
            }

        async def _must_get_screenshot(attempts: int = 5, delay_s: float = 0.3) -> str:
            for _ in range(attempts):
                try:
                    shot = await self._request("GET", "/linux/screenshot")
                    b64 = (shot or {}).get("screenshot", "") or ""
                    if b64:
                        return b64
                except Exception:
                    pass
                await asyncio.sleep(delay_s)
            raise BrowserAgentError(
                "screenshot_unavailable",
                "Could not capture desktop screenshot",
            )

        # --- Simple completion heuristic for command typing ---
        import re as _re
        import base64 as _b64
        import struct as _struct

        # target command extracted from instruction
        # Try to extract a quoted command from the instruction (e.g., 'echo ready')
        target_cmd_match = _re.search(r"'([^']+)'|\"([^\"]+)\"", instruction)
        self._target_cmd = None
        if target_cmd_match:
            self._target_cmd = next(g for g in target_cmd_match.groups() if g)

        # Track step history and discourage loops
        recent_steps: list[str] = []

        def _png_size_from_b64(b64: str) -> tuple[int, int] | None:
            try:
                data = _b64.b64decode(b64)
                if data[:8] != b"\x89PNG\r\n\x1a\n":
                    return None
                width = _struct.unpack(">I", data[16:20])[0]
                height = _struct.unpack(">I", data[20:24])[0]
                return width, height
            except Exception:
                return None

        while step < max_steps:
            step += 1
            # Mandatory screenshot for each step
            # roll previous/current
            self._prev_full_shot_b64 = getattr(self, "_curr_full_shot_b64", "")
            screenshot_b64 = await _must_get_screenshot()
            self._curr_full_shot_b64 = screenshot_b64
            dims = _png_size_from_b64(screenshot_b64) or (0, 0)

            history_txt = " | ".join(recent_steps[-5:]) if recent_steps else "(none)"
            # Include pointer context
            ptr = await self._pointer_context_line()
            header = (
                f"Goal: {instruction}\n"
                f"Screen: width={dims[0]} height={dims[1]} (pixels)\n"
                f"{ptr}\n"
                if ptr
                else f"Goal: {instruction}\nScreen: width={dims[0]} height={dims[1]} (pixels)\n"
            )
            header += (
                f"Recent steps: {history_txt}\nContext: {json.dumps(_context_blob())}"
            )
            content = [
                {
                    "type": "text",
                    "text": header,
                },
            ]
            if screenshot_b64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                )

            result_json = client.generate(
                messages=client.messages + [{"role": "user", "content": content}],
            )
            try:
                next_action = NextAction.model_validate_json(result_json)
            except Exception:
                # fallback: treat as done to avoid infinite loop
                break

            tool = (next_action.tool or "").strip().lower()
            args = next_action.args or {}
            verify_stmt = next_action.verify or ""

            # Loop-avoidance guard: detect consecutive repeats
            if self._last_tool_name == tool:
                self._consecutive_repeat_count += 1
            self._last_tool_name = tool

            # Record step intent for the next iteration's guidance
            if tool in ("focus_window", "exists_window"):
                t = str(
                    args.get("title")
                    or args.get("windowTitle")
                    or args.get("name")
                    or "",
                ).strip()
                recent_steps.append(f"{tool}('{t}')")
            elif tool in ("type_text", "type_and_enter"):
                shown = str(args.get("text") or args.get("keys") or "").strip()
                recent_steps.append(
                    (
                        f"{tool}('{shown[:24]}…')"
                        if len(shown) > 24
                        else f"{tool}('{shown}')"
                    ),
                )
            elif tool == "click_at":
                x_dbg = args.get("x")
                y_dbg = args.get("y")
                recent_steps.append(f"click_at({x_dbg},{y_dbg})")
            elif tool in (
                "list_windows",
                "list_windows_extended",
                "move_window",
                "resize_window",
                "set_window_state",
                "app_open",
                "app_focus",
                "app_close",
            ):
                recent_steps.append(tool)
            else:
                recent_steps.append(tool)

            print(f"🐍 PYTHON: Executing tool: {tool} with args: {args}")
            ok = await self._execute_tool(tool, args, dims)

            # Observe for completion each step only when the tool provides 'verify'
            verify_stmt = (next_action.verify or "").strip()
            if verify_stmt:
                try:
                    obs = await self.observe(verify_stmt)
                    if isinstance(obs, dict) and obs.get("matches"):
                        return "success"
                except Exception:
                    # non-fatal; continue planning
                    pass

        return "incomplete"

    async def observe(self, query: str, response_format: Any = str) -> Any:
        # LLM verification loop: get screenshot, ask the model if query is true/false
        try:
            import unify  # type: ignore
            from pydantic import BaseModel, Field
        except Exception as e:
            raise RuntimeError(
                "Observation requires the 'unify' client to be installed and configured.",
            ) from e

        class Verify(BaseModel):
            matches: bool = Field(
                ...,
                description="True if the statement matches the current desktop state",
            )
            reason: str | None = None

        client = unify.Unify(endpoint="claude-4-sonnet@anthropic")
        client.set_response_format(Verify)
        client.set_system_message(
            "You are a strict visual verifier. You may receive one or two screenshots: a previous ('before') and a current ('after').\n"
            "Judge the statement against the current ('after') screenshot. If the statement implies a change (e.g., resized, moved), compare before vs after.\n"
            "Return only JSON with {matches: boolean, reason?: string}. Do not infer beyond visible evidence.\n",
        )

        # capture and track current/previous screenshots
        self._prev_full_shot_b64 = getattr(self, "_curr_full_shot_b64", "")
        screenshot_b64 = await self.get_screenshot()
        self._curr_full_shot_b64 = screenshot_b64

        # Derive screen width/height from the PNG header, if available
        screen_w = screen_h = None
        if screenshot_b64:
            try:
                import base64 as _b64, struct as _struct

                data = _b64.b64decode(screenshot_b64)
                # PNG signature + IHDR width/height positions
                if data[:8] == b"\x89PNG\r\n\x1a\n":
                    screen_w = _struct.unpack(">I", data[16:20])[0]
                    screen_h = _struct.unpack(">I", data[20:24])[0]
            except Exception:
                pass

        statement_txt = f"Statement: {query}"
        if screen_w is not None and screen_h is not None:
            statement_txt += f"\nScreen: width={screen_w} height={screen_h} (pixels)"

        # Include current mouse/pointer position for better spatial grounding (helper)
        try:
            ptr = await self._pointer_context_line()
            if ptr:
                statement_txt += f"\n{ptr}"
        except Exception:
            pass

        content = [{"type": "text", "text": statement_txt}]
        # Include BEFORE then AFTER images if available
        if self._prev_full_shot_b64:
            content.append({"type": "text", "text": "Previous (before):"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._prev_full_shot_b64}",
                    },
                },
            )
        if screenshot_b64:
            content.append({"type": "text", "text": "Current (after):"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                },
            )

        raw = client.generate(
            messages=client.messages + [{"role": "user", "content": content}],
        )
        out = Verify.model_validate_json(raw)
        return {
            "matches": out.matches,
            # "screenshot": screenshot_b64,
            "reason": out.reason,
        }

    async def get_screenshot(self, *, save: bool = False, return_meta: bool = False):
        """Capture a full desktop screenshot.

        Parameters:
            save: when True, the service writes the PNG to a temp filepath and includes it in the response.
            return_meta: when True, return the full JSON (including filepath and saved flags). Otherwise, return base64 PNG string.
        """
        params = "?save=true" if save else ""
        data = await self._request("GET", f"/linux/screenshot{params}")
        if return_meta:
            return data
        return data.get("screenshot", "")

    async def get_current_url(self) -> str:
        # Desktop context has no concept of URL
        return ""

    async def navigate(self, url: str) -> str:
        raise NotImplementedError("Desktop backend cannot navigate to URLs.")

    def stop(self):
        with MagnitudeDesktopBackend._lock:
            if (
                MagnitudeDesktopBackend._process
                and MagnitudeDesktopBackend._process.poll() is None
            ):
                print(
                    f"🐍 PYTHON: Explicitly calling stop() on MagnitudeDesktopBackend. PID: {MagnitudeDesktopBackend._process.pid}",
                )
                print(
                    f"🛑 Stopping Magnitude BrowserAgent service (PID: {MagnitudeDesktopBackend._process.pid})...",
                )
                if sys.platform != "win32":
                    import signal

                    try:
                        os.killpg(
                            os.getpgid(MagnitudeDesktopBackend._process.pid),
                            signal.SIGTERM,
                        )
                    except ProcessLookupError:
                        pass
                else:
                    MagnitudeDesktopBackend._process.terminate()

                try:
                    MagnitudeDesktopBackend._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    MagnitudeDesktopBackend._process.kill()
                MagnitudeDesktopBackend._process = None
