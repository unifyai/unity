import inspect
import subprocess
import time
import atexit
from abc import ABC, abstractmethod
from typing import Any

import aiohttp
import requests
from pydantic import BaseModel

from .controller import Controller
from ..planner.hierarchical_planner import ReplanFromParentException


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

    def __init__(self, **kwargs):
        self.controller = Controller(**kwargs)
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

        service_path = "../agent-service"
        command = ["npx", "ts-node", f"{service_path}/src/index.ts"]
        if headless:
            command.append("--headless")

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(
            f"🚀 Starting Magnitude BrowserAgent service (PID: {self.process.pid})...",
        )

        atexit.register(self.stop)

        self._wait_for_service()

    def _wait_for_service(self, timeout: int = 30):
        """Pings the /health endpoint until the service is ready or timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.agent_base_url}/health")
                if response.status_code == 200:
                    print("✅ Magnitude service is healthy and ready.")
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(1)

        stderr_output = self.process.stderr.read() if self.process.stderr else "N/A"
        raise RuntimeError(
            f"Magnitude service failed to start within {timeout} seconds. Error: {stderr_output}",
        )

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
        The agent can handle multi-step sequences and autonomously decides on the
        necessary clicks, scrolls, and typing to achieve the goal.
        """
        task_desc = f"{instruction}. {expectation}".strip()
        response = await self._request("POST", "/act", {"task": task_desc})
        return response.get("status", "success")

    async def observe(self, query: str, response_format: Any = str) -> Any:
        """
        Extracts structured information from the page using the Magnitude BrowserAgent.
        The agent uses a vision-language model to analyze the page content and screenshot.
        Can return structured data if a Pydantic model is provided in `response_format`.
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
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None
