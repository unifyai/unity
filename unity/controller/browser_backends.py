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

        self._load_persistent_data()

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
                                from ..actor.hierarchical_actor import (
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

    def _load_persistent_data(self):
        """
        Load all files and folders in the assistant's data directory from a remote endpoint.
        """
        # list all files in /home/install through the endpoint, then for each file, save in local /home/install
        try:
            orchestra_url = os.getenv("UNIFY_BASE_URL")
            endpoint = f"{orchestra_url}/admin/file"

            user_id = os.environ.get("USER_ID", "default")
            assistant_name = os.environ.get("ASSISTANT_NAME", "assistant")
            project = "Assistants"

            headers = {
                "Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY', '')}",
            }

            # 1) Get all files from Orchestra (path -> content)
            resp = requests.get(
                endpoint,
                params={
                    "user_id": user_id,
                    "project": project,
                    "staging": "staging" in orchestra_url,
                },
                headers=headers,
                timeout=60,
            )
            if resp.status_code >= 400:
                print(
                    f"Warning: Orchestra list failed: {resp.status_code} {resp.text[:200]}",
                )
                return
            files_map = resp.json() or {}
            print("files_map:", files_map)

            # install_root = "/home/install"
            # os.makedirs(install_root, exist_ok=True)

            # # 2) Write each file for this assistant under /home/install
            # prefix = f"{user_id}/{project}/{assistant_name}/"
            # for path_key, content in files_map.items():
            #     try:
            #         if not isinstance(path_key, str) or not path_key.startswith(prefix):
            #             continue
            #         rel = path_key[len(prefix) :]
            #         local_path = os.path.join(install_root, rel)
            #         os.makedirs(os.path.dirname(local_path), exist_ok=True)
            #         data = content if isinstance(content, str) else str(content)
            #         with open(local_path, "wb") as f:
            #             f.write(data.encode("utf-8", errors="ignore"))
            #     except Exception as e:
            #         print(f"Warning: Could not restore {path_key}: {e}")

            # 2) For each file entry, download its bytes and write to local /home/install
            # for entry in listing:
            #     try:
            #         if entry.get("type") != "file":
            #             continue
            #         rel_name = entry.get("name") or entry.get("path") or entry.get("filename")
            #         if not rel_name:
            #             continue
            #         # Request the file as binary
            #         fr = requests.get(
            #             endpoint,
            #             params={
            #                 "user_id": user_id,
            #                 "project": project,
            #                 "filename": rel_name,
            #                 "as": "binary",
            #             },
            #             timeout=60,
            #         )
            #         if fr.status_code >= 400:
            #             print(f"Warning: Failed to fetch {rel_name}: {fr.status_code} {fr.text[:200]}")
            #             continue
            #         local_path = os.path.join(install_root, rel_name)
            #         os.makedirs(os.path.dirname(local_path), exist_ok=True)
            #         with open(local_path, "wb") as f:
            #             f.write(fr.content)
            #     except Exception as e:
            #         print(f"Warning: Could not restore file '{entry}': {e}")

        except Exception as e:
            print(f"Warning: Could not query remote files for persistence: {e}")

        # Optionally install packages recorded in apt-manual.txt if present
        # try:
        #     if os.path.exists("/home/install/apt-manual.txt"):
        #         subprocess.run(
        #             ["xargs", "-a", "/home/install/apt-manual.txt", "apt-get", "install", "-y"],
        #             check=True,
        #         )
        # except Exception as e:
        #     print(f"Warning: Could not execute apt-get install from apt-manual.txt: {e}")

    def _save_persistent_data(self):
        """
        Save all files and folders in the assistant's data directory by sending them
        to a remote endpoint for persistence.
        """
        try:
            subprocess.run(
                ["apt-mark", "showmanual"],
                check=True,
                stdout=open("/home/install/apt-manual.txt", "w"),
            )
            # Now sort the file in place
            with open("/home/install/apt-manual.txt", "r") as f:
                lines = f.readlines()
            lines.sort()
            with open("/home/install/apt-manual.txt", "w") as f:
                f.writelines(lines)
        except Exception as e:
            print(f"Warning: Could not save apt manual package list: {e}")

        # save files in /home/install folder with the endpoint
        try:
            install_root = "/home/install"

            # Build files mapping: relative_path -> base64 content
            files_map: dict[str, str] = {}
            for root, _, files in os.walk(install_root):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    rel_path = os.path.join(
                        os.getenv("ASSISTANT_NAME", "assistant"),
                        fpath.lstrip("/"),
                    )
                    try:
                        with open(fpath, "rb") as fp:
                            import base64

                            b64 = base64.b64encode(fp.read()).decode("utf-8")
                        files_map[rel_path] = b64
                    except Exception as e:
                        print(
                            f"Warning: Could not read {rel_path} for persistence: {e}",
                        )

            print("files_map:", files_map)

            if files_map:
                orchestra_url = os.getenv("UNIFY_BASE_URL")
                endpoint = f"{orchestra_url}/admin/file"

                user_id = os.environ.get("USER_ID", "default")
                project = f"Assistants"
                headers = {
                    "Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY', '')}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "user_id": user_id,
                    "project": project,
                    "files": files_map,
                    "staging": "staging" in orchestra_url,
                }

                try:
                    r = requests.post(
                        endpoint,
                        json=payload,
                        headers=headers,
                        timeout=120,
                    )
                    if r.status_code >= 400:
                        print(
                            f"Warning: Orchestra persist failed: {r.status_code} {r.text[:200]}",
                        )
                except Exception as e:
                    print(f"Warning: Could not reach Orchestra to persist files: {e}")
        except Exception as e:
            print(f"Warning: Could not enumerate /home/install for persistence: {e}")

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

        response = await self._request(
            "POST",
            "/act",
            {"task": f"Go to the page: {url}"},
        )
        return response.get("status", "success")

        # max_retries = 3
        # for attempt in range(max_retries):
        #     try:
        #         response = await self._request("POST", "/nav", {"url": url})
        #         return response.get("status", "success")
        #     except BrowserAgentError as e:
        #         if "Target page" in str(e) and attempt < max_retries - 1:
        #             print(
        #                 f"⚠️ Navigation failed due to closed page, retrying (attempt {attempt + 1}/{max_retries})...",
        #             )
        #             await asyncio.sleep(2)
        #             continue
        #         raise

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

                self._save_persistent_data()

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
