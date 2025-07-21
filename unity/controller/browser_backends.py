import inspect
import os
import subprocess
import sys
import time
import atexit
import threading
from abc import ABC, abstractmethod
from typing import Any

import aiohttp
import requests
from pydantic import BaseModel

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

    def stop(self):
        self.controller.stop()


class MagnitudeBrowserBackend(BrowserBackend):
    def __init__(
        self,
        agent_server_url: str = "http://localhost:3000",
        headless: bool = False,
        **kwargs,
    ):
        self.agent_base_url = agent_server_url
        self.process = None
        self._detected_port = None

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

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=service_path,
            env=env,
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )

        print(
            f"🚀 Starting Magnitude BrowserAgent service (PID: {self.process.pid})...",
        )

        self._start_output_readers()
        atexit.register(self.stop)
        self._wait_for_service()

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
                            self._detected_port = match.group(1)
                            self.agent_base_url = (
                                f"http://localhost:{self._detected_port}"
                            )
                            print(
                                f"✨ Detected service running on port {self._detected_port}",
                            )
            pipe.close()

        stdout_thread = threading.Thread(
            target=read_output,
            args=(self.process.stdout, "Magnitude"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=read_output,
            args=(self.process.stderr, "Magnitude-ERR"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

    def _wait_for_service(self, timeout: int = 30):
        """Pings the /health endpoint until the service is ready or timeout."""
        start_time = time.time()
        last_error = None

        time.sleep(2)

        while self._detected_port is None and time.time() - start_time < 5:
            time.sleep(0.5)

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.agent_base_url}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Magnitude service is healthy and ready.")
                    return
                elif response.status_code == 503:
                    print("⏳ Service is initializing...")
                    time.sleep(1)
                    continue
            except requests.exceptions.ConnectionError as e:
                last_error = str(e)
                time.sleep(1)
            except Exception as e:
                last_error = str(e)
                print(f"Health check error: {e}")
                time.sleep(1)

        if self.process.poll() is not None:
            error_message = (
                f"Magnitude service process terminated with code {self.process.poll()}.\n"
                f"Last error: {last_error}\n"
            )
        else:
            error_message = (
                f"Magnitude service failed to become healthy within {timeout} seconds.\n"
                f"Process is still running but not responding.\n"
                f"Last error: {last_error}\n"
                f"Detected port: {self._detected_port}\n"
                f"Using URL: {self.agent_base_url}\n"
            )

        raise RuntimeError(error_message)

    async def _request(
        self,
        method: str,
        endpoint: str,
        payload: dict | None = None,
    ) -> Any:
        url = f"{self.agent_base_url}{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, json=payload, timeout=300) as resp:
                if resp.status >= 400:
                    try:
                        from ..planner.hierarchical_planner import (
                            ReplanFromParentException,
                        )

                        error_data = await resp.json()
                        error_type = error_data.get("error", "unknown_http_error")
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

        **✅ Good Queries (Leveraging Structure and Vision):**
        - "List all the user comments on the page, including the author and the comment text." (Requires a Pydantic model for the response format).
        - "Extract the product name, price, and rating for every item shown."
        - "What is the shipping address displayed in the order summary?"

        **❌ Bad Queries (HTML/DOM Specific):**
        - "Get the href attribute of the 'About Us' link."
        # Instead, ask: "What is the destination URL of the 'About Us' link?" The agent can often infer this by navigating and checking the URL.

        Args:
            query: The natural-language instruction for what to extract.
            response_format: Optional. A Pydantic model to structure the output.
        """
        payload = {"instructions": query}
        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            payload["schema"] = response_format.model_json_schema()

        response = await self._request("POST", "/extract", payload)
        data = response.get("data")

        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            return response_format.model_validate(data)
        return data

    async def get_screenshot(self) -> str:
        response = await self._request("GET", "/screenshot")
        return response.get("screenshot")

    def stop(self):
        """Stops the Node.js service subprocess."""
        if self.process and self.process.poll() is None:
            print(
                f"🛑 Stopping Magnitude BrowserAgent service (PID: {self.process.pid})...",
            )
            if sys.platform != "win32":
                import signal

                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                self.process.terminate()

            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
