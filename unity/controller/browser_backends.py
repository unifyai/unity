import inspect
import os
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any
from typing import Optional
import logging
import aiohttp
import requests
from pydantic import BaseModel, PydanticUserError
import asyncio
import functools
import websockets
from .controller import Controller

logger = logging.getLogger("websockets")


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
    async def act(self, instruction: str) -> str:
        """Perform an action in the browser."""

    @abstractmethod
    async def observe(self, query: str, response_format: Any = str) -> Any:
        """Observe the state of the browser page."""

    @abstractmethod
    async def query(self, query: str, response_format: Any = str) -> Any:
        """Query the agent's memory and action history."""

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

    async def act(self, instruction: str) -> str | BrowserAgentError:
        """
        Performs a **single, high-level action** in the browser.

        This tool functions by looking at the screen; it **does not have access to the underlying HTML or DOM**. Therefore, instructions must describe elements based on their **visible text or position**, not by HTML attributes like `id`, `class`, or `aria-label`.

        Args:
            instruction (str): A single, natural-language command. Describe the element to interact with
                            based on its visible properties.

        Return value:
            str: A single string 'done' if the action was successful.
            BrowserAgentError: If the action failed, the error message explaining the reason for the failure.

        Examples:
            # ✅ Good Example (Using Visible Text)
            - instruction: "Click the 'Login' button"

            # ✅ Good Example (Using Visible Text)
            - instruction: "Type 'hello world' into the search bar"

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
            expectation="",
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

    async def query(self, query: str, response_format: Any = str) -> Any:
        """
        Query the agent's memory and action history (not supported in LegacyBrowserBackend).

        This method is not supported by the legacy backend as it doesn't have access to
        agent memory. Consider using MagnitudeBrowserBackend for this functionality.
        """
        raise NotImplementedError(
            "Query method is not supported by LegacyBrowserBackend. Use MagnitudeBrowserBackend instead.",
        )

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
    _agent_base_url = "http://localhost:3000"
    _process = None  # Keep for process management if needed

    def __init__(
        self,
        agent_server_url: str = "http://localhost:3000",
        headless: bool = False,
        agent_mode: str = "browser",
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
        self._active_commands = {}  # Tracks commands for cancellation
        self._seq: int = 0  # Add sequence number counter
        self._processed_seq: int = -1  # Track the last completed sequence number
        self._barrier_events: dict[int, asyncio.Event] = (
            {}
        )  # For barrier synchronization

        # Keep the simpler initialization from HEAD but add logging support
        MagnitudeBrowserBackend._agent_base_url = agent_server_url
        self.agent_base_url = agent_server_url

        print(
            f"🔗 Connecting to Magnitude service at {self.agent_base_url} (Mode: {self.agent_mode})",
        )

        try:
            self._sync_request(
                "POST",
                "/start",
                {"headless": headless, "mode": self.agent_mode},
            )
            self._check_service_ready()

            # Initialize the network log queue - defer creation until event loop is available
            self._network_log_queue = None

            # Mark that async initialization is needed
            self._async_initialized = False

        except Exception as e:
            print(f"❌ Failed to initialize MagnitudeBrowserBackend: {e}")
            self.stop()
            raise

    def _check_service_ready(self):
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                r = self._sync_request("GET", "/screenshot")
                if r.status_code < 500:
                    print(f"✅ Magnitude service is ready on {self.agent_base_url}")
                    break
            except Exception:
                time.sleep(0.5)
        else:
            self.stop()
            raise RuntimeError(
                f"Magnitude BrowserAgent failed to become ready within 30 seconds on {self.agent_base_url}",
            )

    async def _ensure_async_initialized(self):
        """
        Initialize async components when event loop is available.
        This is called lazily from async methods.
        """
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
                    self._command_processor_task = asyncio.create_task(
                        self._process_commands(),
                    )

                self._async_initialized = True
                logger.info("⚙️ Initialized async components")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize async components: {e}")
        elif websockets is None and not self._async_initialized:
            logger.warning("⚠️ Websockets not available, log streaming disabled")
            self._async_initialized = True

    async def _start_log_stream_listener(self):
        """
        Connects to the Magnitude WebSocket and streams logs into an async queue.
        Includes reconnection logic.
        """
        ws_url = self.agent_base_url.replace("http", "ws") + "/logs/stream"
        logger.info(f"🔌 Starting log stream listener for {ws_url}")

        # Prepare authentication headers
        auth_key = os.getenv("UNIFY_KEY", "")
        assistant_email = os.getenv("ASSISTANT_EMAIL", "")
        headers = {
            "Authorization": f"Bearer {auth_key} {assistant_email}".strip(),
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
                else:
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
                    result = await func(*args, **kwargs)
                    if future:
                        future.set_result(result)
                except Exception as e:
                    if future:
                        future.set_exception(e)
                finally:
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
        url = f"{MagnitudeBrowserBackend._agent_base_url}{endpoint}"

        retries = 3
        for attempt in range(retries):
            try:
                # Build auth header: "authorization: Bearer <UNIFY_KEY> <ASSISTANT_EMAIL>"
                auth_key = os.getenv("UNIFY_KEY", "")
                assistant_email = os.getenv("ASSISTANT_EMAIL", "")
                headers = {
                    "authorization": f"Bearer {auth_key} {assistant_email}".strip(),
                }
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method,
                        url,
                        json=payload,
                        headers=headers,
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

    def _sync_request(
        self,
        method: str,
        endpoint: str,
        payload: dict | None = None,
    ) -> Any:
        try:
            url = f"{MagnitudeBrowserBackend._agent_base_url}{endpoint}"
            auth_key = os.getenv("UNIFY_KEY", "")
            assistant_email = os.getenv("ASSISTANT_EMAIL", "")
            headers = {
                "authorization": f"Bearer {auth_key} {assistant_email}".strip(),
            }
            from ..common.http import request as http_request

            result = http_request(
                method,
                url,
                json=payload,
                headers=headers,
                timeout=300,
            )
            if result.status_code >= 400:
                raise RuntimeError(
                    f"Failed to reach agent-service {endpoint}: {result.status_code} {result.text[:200]}",
                )
            return result
        except Exception as e:
            raise RuntimeError(f"Could not reach agent-service {endpoint}: {e}")

    def _load_persistent_data(self):
        """
        Load all files and folders in the assistant's data directory from a remote endpoint.
        """
        # list all files in /tmp/unify/assistant/install through the endpoint, then for each file, save in local /tmp/unify/assistant/install
        print("🐍 PYTHON: Loading persistent installs...")
        try:
            orchestra_url = os.getenv("UNIFY_BASE_URL")
            dl_endpoint = f"{orchestra_url}/admin/file/download_url"

            user_id = os.environ.get("USER_ID", "default")
            assistant_name = os.environ.get("ASSISTANT_NAME", "assistant")
            project = "Assistants"

            headers = {
                "Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY', '')}",
            }

            os.makedirs("/tmp/unify/assistant/install", exist_ok=True)
            os.makedirs("/tmp/unify/assistant/deb", exist_ok=True)

            # Download folders via prefix (assistant-scoped)
            for prefix_folder in ["home/install", "home/deb"]:
                try:
                    # Request signed URLs for all files under the prefix
                    from ..common.http import get as http_get

                    dl_resp = http_get(
                        dl_endpoint,
                        params={
                            "user_id": user_id,
                            "project": project,
                            "path": f"{assistant_name}/{prefix_folder}",
                            "staging": "staging" in orchestra_url,
                            "expires_in": 5 * 60,
                            "as_prefix": True,
                        },
                        headers=headers,
                        timeout=60,
                    )
                    if dl_resp.status_code >= 400:
                        print(
                            f"Warning: download_url (prefix) failed for {prefix_folder}: {dl_resp.status_code} {dl_resp.text[:200]}",
                        )
                        continue
                    payload = dl_resp.json() or {}
                    items = payload.get("items", [])
                    for it in items:
                        try:
                            full_path = it.get(
                                "path",
                                "",
                            )  # e.g., user/project/assistant/tmp/unify/assistant/install/file
                            url = it.get("download_url")
                            if not full_path or not url:
                                continue
                            # Derive local absolute path by stripping up to '/<assistant_name>/'
                            marker = f"/{assistant_name}/"
                            idx = full_path.find(marker)
                            if idx == -1:
                                continue
                            rel_from_assistant = full_path[idx + len(marker) :]
                            local_path = (
                                "/" + rel_from_assistant
                            )  # starts with home/install or home/deb
                            os.makedirs(os.path.dirname(local_path), exist_ok=True)
                            bin_resp = http_get(url, timeout=300)
                            if bin_resp.status_code >= 400:
                                print(
                                    f"Warning: download content failed for {full_path}: {bin_resp.status_code}",
                                )
                                continue
                            with open(local_path, "wb") as f:
                                f.write(bin_resp.content)
                        except Exception as e:
                            print(
                                f"Warning: Could not restore item under {prefix_folder}: {e}",
                            )
                except Exception as e:
                    print(f"Warning: Could not list prefix {prefix_folder}: {e}")

        except Exception as e:
            print(f"Warning: Could not query remote files for persistence: {e}")

        # Install downloaded/custom deb files
        if os.path.exists("/tmp/unify/assistant/deb"):
            for deb_file in os.listdir("/tmp/unify/assistant/deb"):
                try:
                    subprocess.run(
                        [
                            "dpkg",
                            "-i",
                            os.path.join("/tmp/unify/assistant/deb", deb_file),
                        ],
                        check=True,
                    )
                except Exception as e:
                    print(f"Warning: Could not install {deb_file}: {e}")

        # Optionally install packages recorded in apt-manual.txt if present
        try:
            if os.path.exists("/tmp/unify/assistant/install/apt-manual.txt"):
                subprocess.run(
                    [
                        "xargs",
                        "-a",
                        "/tmp/unify/assistant/install/apt-manual.txt",
                        "apt-get",
                        "install",
                        "-y",
                    ],
                    check=True,
                )
        except Exception as e:
            print(
                f"Warning: Could not execute apt-get install from apt-manual.txt: {e}",
            )

    def _save_persistent_data(self):
        """
        Save all files and folders in the assistant's data directory by sending them
        to a remote endpoint for persistence.
        """
        print("🐍 PYTHON: Saving persistent installs...")
        try:
            subprocess.run(
                ["apt-mark", "showmanual"],
                check=True,
                stdout=open("/tmp/unify/assistant/install/apt-manual.txt", "w"),
            )
            # Now sort the file in place
            with open("/tmp/unify/assistant/install/apt-manual.txt", "r") as f:
                lines = f.readlines()
            lines.sort()
            with open("/tmp/unify/assistant/install/apt-manual.txt", "w") as f:
                f.writelines(lines)
        except Exception as e:
            print(f"Warning: Could not save apt manual package list: {e}")

        # save files in /tmp/unify/assistant/install folder with the endpoint
        try:
            # Iterate local files and upload each via signed upload URL
            orchestra_url = os.getenv("UNIFY_BASE_URL")
            up_endpoint = f"{orchestra_url}/admin/file/upload_url"
            user_id = os.environ.get("USER_ID", "default")
            project = f"Assistants"
            headers = {
                "Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY', '')}",
                "Content-Type": "application/json",
            }

            def _iter_local_files(root_dir: str):
                assistant_name = os.getenv("ASSISTANT_NAME", "assistant")
                for r, _, files in os.walk(root_dir):
                    for fn in files:
                        ap = os.path.join(r, fn)
                        rel = os.path.relpath(ap, root_dir)
                        base = root_dir.lstrip("/")  # e.g., home/install or home/deb
                        key = os.path.join(assistant_name, base, rel)
                        yield key, ap

            for base_dir in [
                "/tmp/unify/assistant/install",
                "/tmp/unify/assistant/deb",
            ]:
                if not os.path.exists(base_dir):
                    continue
                for key, abs_path in _iter_local_files(base_dir):
                    try:
                        # 1) Request upload URL for this file
                        req = {
                            "user_id": user_id,
                            "project": project,
                            "path": key,
                            "staging": "staging" in orchestra_url,
                            "content_type": "application/octet-stream",
                        }
                        up_resp = requests.post(
                            up_endpoint,
                            json=req,
                            headers=headers,
                            timeout=60,
                        )
                        if up_resp.status_code >= 400:
                            print(
                                f"Warning: upload_url failed for {key}: {up_resp.status_code} {up_resp.text[:200]}",
                            )
                            continue
                        upload_url = up_resp.json().get("upload_url")
                        if not upload_url:
                            continue
                        # 2) Upload bytes to signed URL (single-shot resumable PUT)
                        with open(abs_path, "rb") as fp:
                            data = fp.read()
                        total = len(data)
                        put_headers = {
                            "Content-Type": "application/octet-stream",
                            "Content-Length": str(total),
                            "Content-Range": f"bytes 0-{total-1}/{total}",
                        }
                        put_resp = requests.put(
                            upload_url,
                            data=data,
                            headers=put_headers,
                            timeout=600,
                        )
                        if put_resp.status_code not in (200, 201, 204):
                            print(
                                f"Warning: upload failed for {key}: {put_resp.status_code} {put_resp.text[:200]}",
                            )
                    except Exception as e:
                        print(f"Warning: Could not upload {key}: {e}")
        except Exception as e:
            print(
                f"Warning: Could not enumerate /tmp/unify/assistant/install for persistence: {e}",
            )

    async def act(
        self,
        instruction: str,
        wait: bool = False,
        context: dict = None,
    ) -> Any:
        """
        Executes a high-level browser task using the Magnitude BrowserAgent.

        This tool is **autonomous and can perform multiple steps** (e.g., typing, clicking, scrolling) to achieve the goal described in the instruction. It operates based on a visual understanding of the page.

        Args:
            instruction (str): A high-level, natural-language command describing the desired outcome.
            wait (bool): If True, the function will block and wait for the action to
                        complete in the browser before returning. If False (default),
                        the command is added to a queue for background execution, and
                        the function returns immediately.
            context (dict): Internal metadata for command tracking.

        ### Non-Blocking Example (`wait=False`, Default)
        This is ideal for sequences of actions where the plan doesn't need immediate feedback.
        ```python
        # The plan queues up all actions and continues its own execution
        # without waiting for the browser to finish each one.
        await action_provider.act("Type 'testuser' into the username field", wait=False)
        await action_provider.act("Type 'password123' into the password field", wait=False)
        await action_provider.act("Click the 'Login' button", wait=False)
        ```

        ### Blocking Example (`wait=True`)
        Use this when the outcome of an action is required for a subsequent decision in the plan.
        ```python
        # The plan pauses until the button click is complete and the new page has loaded.
        await action_provider.act("Click the 'Proceed to Checkout' button", wait=True)
        # Now that we've waited, we can safely observe the new page state.
        cart_total = await action_provider.observe("What is the final total?")
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
            {"task": instruction},
        )

        future = asyncio.get_event_loop().create_future() if wait else None
        self._active_commands[command_id] = (instruction, context)
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
            print(
                f"⚠️ Warning: Failed to send interrupt request. The browser action may continue in the background. Error: {e}",
            )

    async def observe(
        self,
        query: str,
        response_format: Any = str,
        wait: bool = True,
        context: dict = None,
    ) -> Any:
        """
        Extracts structured information from the current page using the Magnitude BrowserAgent.

        This is your primary tool for perception. The agent uses a vision-language model to
        analyze the page, so its success depends entirely on the quality and clarity of your `query`.

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

        response = await self._request("POST", "/extract", payload)
        data = response.get("data")

        if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
            return response_format.model_validate(data)
        return data

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
        await self._ensure_async_initialized()
        print(f"🐍 PYTHON: Navigating to URL: {url}")

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
            except BrowserAgentError as e:
                if "Target page" in str(e) and attempt < max_retries - 1:
                    print(
                        f"⚠️ Navigation failed due to closed page, retrying (attempt {attempt + 1}/{max_retries})...",
                    )
                    await asyncio.sleep(2)
                    continue
                raise

    def stop(self):
        """Stops the Node.js service subprocess and cancels background tasks."""
        # if "localhost:3000" in self.agent_base_url:
        #     self._save_persistent_data()

        # Cancel the new asyncio tasks
        if self._log_stream_task and not self._log_stream_task.done():
            self._log_stream_task.cancel()
        if self._log_consumer_task and not self._log_consumer_task.done():
            self._log_consumer_task.cancel()

        try:
            self._sync_request("POST", "/stop")
        except Exception as e:
            # Don't fail stop() if the request fails
            print(f"Warning: Failed to send stop request: {e}")

        # If the backend started the process, terminate it
        if MagnitudeBrowserBackend._process:
            print(
                f"🛑 Stopping Magnitude BrowserAgent service (PID: {MagnitudeBrowserBackend._process.pid})...",
            )
            MagnitudeBrowserBackend._process.terminate()
            try:
                MagnitudeBrowserBackend._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                MagnitudeBrowserBackend._process.kill()
            MagnitudeBrowserBackend._process = None
