import threading
import asyncio
from datetime import datetime, timezone
from typing import Any, Type, Optional
import json

import redis

from .playwright_utils.worker import BrowserWorker
from .agent import InvalidActionError, ask_llm, text_to_browser_action
from ..constants import LOGGER


class Controller(threading.Thread):
    def __init__(
        self,
        *,
        daemon: bool = True,
        session_connect_url: str | None = None,
        headless: bool = False,
        mode: str = "heuristic",
        debug: bool = False,
    ) -> None:
        super().__init__(daemon=daemon)
        self._redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self._pubsub_text_action = self._redis_client.pubsub()
        self._pubsub_text_action.subscribe("text_action")
        self._pubsub_browser_state = self._redis_client.pubsub()
        self._pubsub_browser_state.subscribe("browser_state")
        self.session_connect_url = session_connect_url

        self._headless = headless
        self._mode = mode
        self._debug = debug
        self._browser_worker = None
        self._browser_open = False
        self._stop_event = threading.Event()

        # Cached data for LLM observation queries
        self._observe_ctx: dict[str, Any] = {}
        self._last_shot: bytes = b""

    def run(self) -> None:
        """
        Background loop: listen for browser_state messages and update cached context.
        """
        if self._browser_worker is None:
            self._browser_worker = BrowserWorker(
                start_url="https://www.google.com/",
                refresh_interval=0.4,
                session_connect_url=self.session_connect_url,
                headless=self._headless,
                mode=self._mode,
                debug=self._debug,
            )
        if not self._browser_open:
            self._browser_worker.start()
            self._browser_open = True

        try:
            for msg in self._pubsub_browser_state.listen():
                if self._stop_event.is_set():
                    break
                if msg.get("type") != "message":
                    continue
                data = msg.get("data")
                try:
                    payload = (
                        data.decode() if isinstance(data, (bytes, bytearray)) else data
                    )
                    import json, ast

                    try:
                        browser_state = json.loads(payload)
                    except Exception:
                        browser_state = ast.literal_eval(payload)
                except Exception:
                    browser_state = {}
                if isinstance(browser_state, dict):
                    self._observe_ctx["ts"] = browser_state.get("ts", 0.0)
                    raw_elements = browser_state.get("elements", [])
                    elements: list[tuple[Any, Any]] = []
                    for item in raw_elements:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            elements.append((item[0], item[1]))
                    self._observe_ctx.update(
                        {
                            "state": browser_state.get("state", {}),
                            "elements": elements,
                            "tabs": browser_state.get("tabs", []),
                            "history": browser_state.get("history", []),
                        },
                    )
                    self._last_shot = browser_state.get("screenshot", b"")
        except (redis.ConnectionError, ValueError) as e:
            # Redis connection closed or file operation on closed file
            if not self._stop_event.is_set():
                LOGGER.warning(f"Redis connection error in Controller: {e}")
        except Exception as e:
            # Catch any other unexpected errors
            if not self._stop_event.is_set():
                LOGGER.error(f"Unexpected error in Controller: {e}")

    def stop(self) -> None:
        """Signal the controller thread to stop."""
        self._stop_event.set()

        # Stop the browser worker first if it's running
        if self._browser_worker and self._browser_open:
            try:
                self._browser_worker.stop()
                self._browser_worker.join(timeout=2)
                self._browser_open = False
            except Exception as e:
                LOGGER.warning(f"Error stopping browser worker: {e}")

        # Close pubsub connections to break out of the listen loop
        try:
            self._pubsub_browser_state.close()
            self._pubsub_text_action.close()
        except Exception:
            pass

        # Close the Redis client
        try:
            self._redis_client.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helper – perform one or more low-level primitives (in order)
    # ------------------------------------------------------------------
    def _perform_action(self, actions: str | list[str], request_id: str) -> None:
        # if given a list of commands, execute each in sequence
        for action in actions if isinstance(actions, list) else [actions]:
            payload = json.dumps({"action": action, "request_id": request_id})
            # action is a single string
            if action.startswith("click_button_"):
                try:
                    idx = action.split("_")[2]
                    if idx.isdigit():
                        action = f"click {idx}"
                except IndexError:
                    pass

            # ── browser life-cycle primitives ────────────────────────────
            if action == "open_browser" and not self._browser_open:
                self._browser_worker.start()
                self._browser_open = True

            elif action == "close_browser" and self._browser_open:
                self._browser_worker.stop()
                self._browser_worker.join(timeout=2)
                self._browser_open = False

            # other primitives ------------------------------------------------
            elif not self._browser_open:
                # lazily (auto) start the worker if it isn't running
                self._browser_worker.start()
                self._browser_open = True
                self._redis_client.publish("browser_command", payload)
            else:
                self._redis_client.publish("browser_command", payload)

            # notify listeners that the action finished (optimistic)
            self._redis_client.publish("action_completion", payload)

            t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
            LOGGER.info(f"\n🕹️ Performed Action: {action} [⏱️ {t}]\n")

    # ------------------------------------------------------------------
    #  Public helper – high-level "observe" question-answering
    # ------------------------------------------------------------------
    async def observe(
        self,
        request: str,
        response_format: Type = str,
    ) -> Any:  # noqa: ANN401
        """
        Ask a question about the current browser session.

        Args:
            request: The natural-language question to ask about the browser state.
            response_format: The Python or Pydantic type to coerce the answer into.

        Returns:
            Any: The answer returned by the LLM, coerced to the specified response_format.
        """

        # strip the history from the context to save tokens
        current_context = {
            "state": self._observe_ctx.get("state", {}),
            "elements": self._observe_ctx.get("elements", []),
            "tabs": self._observe_ctx.get("tabs", []),
        }
        # call LLM to answer based on refreshed context
        result = await asyncio.to_thread(
            ask_llm,
            request,
            response_format=response_format,
            context=current_context,
            screenshot=self._last_shot,
        )
        return result

    # ------------------------------------------------------------------
    #  Public helper – synchronous one-shot action
    # ------------------------------------------------------------------
    async def act(self, action: str, timeout: float = 7.0) -> Optional[str]:
        """
        Convert natural-language text to a browser action and execute it,
        waiting for the browser state to confirm the update.
        """
        try:
            cmd = await asyncio.to_thread(
                text_to_browser_action,
                text=action,
                screenshot=self._last_shot,
                tabs=self._observe_ctx.get("tabs", []),
                buttons=self._observe_ctx.get("elements", []),
                history=self._observe_ctx.get("history", []),
                state=self._observe_ctx.get("state", {}),
            )
        except InvalidActionError as e:
            LOGGER.error(f"Error converting action to browser command: {e}")
            raise e

        assert cmd is not None, f"requested action {action} returned empty command"
        actions = cmd.get("action")

        if not actions:
            LOGGER.warning(f"LLM returned no action for instruction: '{action}'")
            return None

        # --- MODIFICATION: Wait for browser state to update (Corrected Version) ---
        before_ts = self._observe_ctx.get("ts", 0.0)

        # This future will bridge the asyncio event loop with the redis thread
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        # Create a new pubsub instance for this temporary, precise listener
        ps = self._redis_client.pubsub(ignore_subscribe_messages=True)

        def _waiter(msg):
            """The handler that runs in the listener thread."""
            if msg["type"] != "message":
                return
            try:
                data = json.loads(msg["data"])
                if data.get("ts", 0.0) > before_ts:
                    # Safely set the future's result from the background thread
                    loop.call_soon_threadsafe(fut.set_result, None)
            except (json.JSONDecodeError, KeyError):
                pass

        # Subscribe and start the listener in a background thread
        await asyncio.to_thread(ps.subscribe, **{"browser_state": _waiter})
        listener_thread = ps.run_in_thread(daemon=True)

        # Now, perform the action. The listener is already running.
        await asyncio.to_thread(self._perform_action, actions)

        try:
            # Wait for the future to be resolved by the _waiter
            await asyncio.wait_for(fut, timeout)
            LOGGER.info(f"✅ State confirmation received for action: {actions}")
        except asyncio.TimeoutError:
            LOGGER.warning(f"Timed out waiting for DOM refresh after action: {actions}")
        finally:
            # Crucially, stop the listener thread to clean up resources
            listener_thread.stop()

        return actions

    async def get_action_history(self) -> list[dict]:
        """
        Retrieves a lightweight summary of the executed browser actions,
        including the command and timestamp for each.
        """
        full_history = self._observe_ctx.get("history", [])
        # Return only the command and timestamp to save tokens
        return [
            {"command": record.get("command"), "timestamp": record.get("timestamp")}
            for record in full_history
        ]

    async def get_screenshots_for_action(self, timestamp: float) -> dict:
        """
        Retrieves the before and after screenshots for a specific action,
        identified by its unique timestamp.
        """
        full_history = self._observe_ctx.get("history", [])
        for record in full_history:
            # Use a small tolerance for float comparison
            if abs(record.get("timestamp", 0) - timestamp) < 0.001:
                return {
                    "command": record.get("command"),
                    "before_screenshot_b64": record.get("before_screenshot_b64"),
                    "after_screenshot_b64": record.get("after_screenshot_b64"),
                }
        return {"error": "Action with the specified timestamp not found."}
