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

    # Private helper to execute a selected tool step and update context
    async def _execute_tool(self, tool: str, args: dict, dims: tuple[int, int]) -> bool:
        # done/finish
        if tool in ("done", "finish", "stop"):
            return True

        if tool == "exists_window":
            title = str(
                args.get("title") or args.get("windowTitle") or args.get("name") or "",
            ).strip()
            if title:
                try:
                    res = await self._request(
                        "GET",
                        f"/linux/window/exist?windowTitle={requests.utils.quote(title)}",
                    )
                    self._last_exists[title] = bool(res.get("exists", False))
                except Exception:
                    self._last_exists[title] = False
            return False

        if tool == "focus_window":
            title = str(
                args.get("title") or args.get("windowTitle") or args.get("name") or "",
            ).strip()
            if not title:
                return False
            if title == self._last_focus_title and self._consecutive_repeat_count >= 1:
                # nudge progress if focusing repeats
                await self._request("POST", "/linux/type", payload={"keys": ["Enter"]})
                await asyncio.sleep(0.2)
                return False
            await self._request("POST", "/linux/window/focus", payload={"title": title})
            self._last_focus_title = title
            self._enter_after_last_type = False
            return False

        if tool == "list_windows":
            try:
                res = await self._request("GET", "/linux/window")
                self._last_windows = res.get("windows", []) or []
            except Exception:
                self._last_windows = []
            return False

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
            return False

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
            return False

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
            return False

        if tool == "click_at":
            try:
                x_val = int(args.get("x"))
                y_val = int(args.get("y"))
            except Exception:
                return False
            button = int(args.get("button", 1) or 1)
            if dims[0] > 0 and dims[1] > 0:
                x_val = max(0, min(dims[0] - 1, x_val))
                y_val = max(0, min(dims[1] - 1, y_val))
            mouse_before = None
            try:
                mouse_before = await self._request("GET", "/linux/mouse")
            except Exception:
                mouse_before = None
            await self._request(
                "POST",
                "/linux/click",
                payload={"clicks": [{"x": x_val, "y": y_val, "button": button}]},
            )
            await asyncio.sleep(0.2)
            mouse_after = None
            try:
                mouse_after = await self._request("GET", "/linux/mouse")
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
            return False

        if tool == "type_text":
            text = str(args.get("text") or args.get("keys") or "")
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
            if (
                self._target_cmd
                and self._target_cmd in " ".join(self._typed_history)
                and self._enter_after_last_type
            ):
                return True
            return False

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
            if self._target_cmd and self._target_cmd in " ".join(self._typed_history):
                return True
            return False

        if tool == "press_enter":
            await self._request("POST", "/linux/type", payload={"keys": ["Enter"]})
            if self._typed_history:
                self._enter_after_last_type = True
                await asyncio.sleep(0.4)
                if self._target_cmd and self._target_cmd in " ".join(
                    self._typed_history,
                ):
                    return True
            return False

        # Unknown tool
        return False

    # ------------------------------------------------------------------
    #  Core actions
    # ------------------------------------------------------------------

    async def act(self, instruction: str, expectation: str = "") -> str:
        """`instruction` should be a JSON string specifying clicks / keys.

        Example:
            instruction = json.dumps({
                "focusWindowTitle": "xterm",  # optional: focus a window by title before typing/clicking
                "clicks": [{"x": 100, "y": 200, "button": 1}],
                "keys": ["Hello", "Enter"]
            })

        If a non-JSON natural-language instruction is provided, this method
        will enter a tool-selection loop (LLM-guided) using the available /linux
        endpoints (exists/focus/type/press/click/screenshot) to pursue the goal.
        """
        import json

        # If the instruction is a JSON payload, preserve existing behavior
        try:
            payload = json.loads(instruction)
            if isinstance(payload, dict):
                # Back-compat: map legacy fields to new modular endpoints
                focus_title = payload.get("focusWindowTitle")
                if isinstance(focus_title, str) and focus_title.strip():
                    await self._request(
                        "POST",
                        "/linux/window/focus",
                        payload={"title": focus_title},
                    )
                clicks = payload.get("clicks") or []
                if isinstance(clicks, list) and clicks:
                    await self._request(
                        "POST",
                        "/linux/click",
                        payload={"clicks": clicks},
                    )
                keys = payload.get("keys") or []
                if isinstance(keys, list) and keys:
                    await self._request("POST", "/linux/type", payload={"keys": keys})
                return expectation or "ok"
        except json.JSONDecodeError:
            pass

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
                description="One of: exists_window, focus_window, list_windows, move_window, resize_window, set_window_state, type_text, type_and_enter, click_at, press_enter, done",
            )
            args: dict = Field(default_factory=dict)
            reason: str | None = None

        client = unify.Unify(endpoint="claude-4-sonnet@anthropic")
        client.set_response_format(NextAction)
        client.set_system_message(
            "You are a desktop control assistant. Decide the next atomic tool to apply to achieve the user's goal.\n"
            "Return ONLY JSON matching the response schema.\n"
            "Available tools (tool field):\n"
            "- exists_window: args {title} — check if a window with given title exists.\n"
            "- focus_window: args {title} — bring the window to front.\n"
            "- list_windows: args {} — list all windows; you will receive their id, geometry and title.\n"
            "- move_window: args {id|title, x, y} — move a window.\n"
            "- resize_window: args {id|title, w, h} — resize a window.\n"
            "- set_window_state: args {id|title, action} — action in {minimize|maximize|restore|close}.\n"
            "- type_text: args {text} — type the given text into the active window.\n"
            "- type_and_enter: args {text} — type the given text and then press Enter.\n"
            "- click_at: args {x, y, button?} — click at pixel coordinates (0,0 top-left). Button defaults to 1.\n"
            "- press_enter: args {} — press the Enter key.\n"
            "- done: args {} — when the goal is achieved.\n"
            "You will ALWAYS receive a current desktop screenshot. Base your decisions primarily on that visual context.\n"
            "You will also receive screen width/height (pixels). Ensure coordinates are within bounds.\n"
            "Avoid repeating the same tool more than once in a row; after focusing a window, prefer typing or clicking next.\n"
            "For terminal commands, typically: focus_window → click_at (if needed) → type_text (the exact command) → press_enter → done.\n",
        )

        max_steps = 12
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
            screenshot_b64 = await _must_get_screenshot()
            dims = _png_size_from_b64(screenshot_b64) or (0, 0)

            history_txt = " | ".join(recent_steps[-5:]) if recent_steps else "(none)"
            content = [
                {
                    "type": "text",
                    "text": (
                        f"Goal: {instruction}\n"
                        f"Screen: width={dims[0]} height={dims[1]} (pixels)\n"
                        f"Recent steps: {history_txt}\n"
                        f"Context: {json.dumps(_context_blob())}"
                    ),
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

            # Loop-avoidance guard: detect consecutive repeats
            if self._last_tool_name == tool:
                self._consecutive_repeat_count += 1
            else:
                self._consecutive_repeat_count = 0
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
                "move_window",
                "resize_window",
                "set_window_state",
            ):
                recent_steps.append(tool)
            else:
                recent_steps.append(tool)

            finished = await self._execute_tool(tool, args, dims)
            if finished:
                return expectation or "success"

        # Final heuristic: if we typed something and pressed Enter at least once, consider success
        if self._typed_history and self._enter_after_last_type:
            return expectation or "success"

        return expectation or "incomplete"

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
            "You are a strict visual verifier. Determine if the user's statement is true for the current desktop screenshot.\n"
            "Return only JSON with {matches: boolean, reason?: string}. Do not infer beyond visible evidence.\n",
        )

        screenshot_b64 = await self.get_screenshot()
        content = [
            {"type": "text", "text": f"Statement: {query}"},
        ]
        if screenshot_b64:
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
            "screenshot": screenshot_b64,
            "reason": out.reason,
        }

    async def get_screenshot(self) -> str:
        data = await self._request("GET", "/linux/screenshot")
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
