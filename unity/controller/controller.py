import threading
import asyncio
from datetime import datetime, timezone
from typing import Any, Type, Optional
import json
import uuid
import random
from pydantic import BaseModel, Field
import redis

from .playwright_utils.worker import BrowserWorker
from .agent import InvalidActionError, ask_llm, text_to_browser_action
from .states import BrowserState
from ..constants import LOGGER


class ActionFailedError(Exception):
    """Custom exception raised when a browser action fails to meet its expectation."""

    def __init__(
        self,
        message: str,
        action: str = None,
        expectation: str = None,
        reason: str = None,
    ):
        super().__init__(message)
        self.action = action
        self.expectation = expectation
        self.reason = reason


class VerificationResult(BaseModel):
    is_satisfied: bool = Field(..., description="Whether the expectation is met.")
    reason: str = Field(
        ...,
        description="A brief explanation for why the expectation was or was not met.",
    )


class Controller(threading.Thread):
    def __init__(
        self,
        *,
        daemon: bool = True,
        session_connect_url: str | None = None,
        headless: bool = False,
        mode: str = "heuristic",
        debug: bool = False,
        redis_db: int = 0,
    ) -> None:
        super().__init__(daemon=daemon)
        self._redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self._pubsub_text_action = self._redis_client.pubsub()
        self._pubsub_text_action.subscribe(f"text_action_{redis_db}")
        self._pubsub_browser_state = self._redis_client.pubsub()
        self._pubsub_browser_state.subscribe(f"browser_state_{redis_db}")
        self.session_connect_url = session_connect_url
        self._redis_db = redis_db

        self._headless = headless
        self._mode = mode
        self._debug = debug
        self._browser_worker = None
        self._browser_open = False
        self._stop_event = threading.Event()

        # Cached data for LLM observation queries
        self._observe_ctx: dict[str, Any] = {}
        self._last_shot: str = ""  # Changed from bytes to str for base64

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
                redis_db=self._redis_db,
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
                    self._last_shot = browser_state.get("screenshot", "")
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
    #  Public helper – evaluate arbitrary JS in the active page context
    # ------------------------------------------------------------------
    async def eval_js(self, script: str, timeout: float = 10.0) -> Any:  # noqa: ANN401
        """Evaluate JavaScript in the active page and return the result.

        Uses a lightweight Redis RPC to ask the BrowserWorker to run
        `page.evaluate(script)` and publishes the result on a reply channel.
        """
        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        ps = self._redis_client.pubsub(ignore_subscribe_messages=True)

        def _on_result(msg):
            try:
                data = json.loads(msg["data"])  # {request_id, result|error}
                if data.get("request_id") == request_id:
                    loop.call_soon_threadsafe(fut.set_result, data)
            except Exception:
                pass

        # listen for our result
        await asyncio.to_thread(
            ps.subscribe,
            **{f"browser_eval_result_{self._redis_db}": _on_result},
        )
        listener_thread = ps.run_in_thread(daemon=True)

        # dispatch eval request
        payload = json.dumps({"eval_js": script, "request_id": request_id})
        await asyncio.to_thread(
            self._redis_client.publish,
            f"browser_command_{self._redis_db}",
            payload,
        )

        try:
            data = await asyncio.wait_for(fut, timeout)
        finally:
            listener_thread.stop()

        if data.get("error"):
            raise Exception(f"eval_js error: {data['error']}")
        return data.get("result")

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
                        payload = json.dumps(
                            {"action": action, "request_id": request_id},
                        )
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
                self._redis_client.publish(f"browser_command_{self._redis_db}", payload)
            else:
                self._redis_client.publish(f"browser_command_{self._redis_db}", payload)

            # notify listeners that the action finished (optimistic)
            self._redis_client.publish(f"action_completion_{self._redis_db}", payload)

            t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
            LOGGER.info(f"\n🕹️ Performed Action: {action} [⏱️ {t}]\n")

    # ------------------------------------------------------------------
    #  Public helper – high-level "observe" question-answering
    # ------------------------------------------------------------------
    async def observe(
        self,
        request: str,
        response_format: Type = str,
        screenshots: dict[str, bytes] | None = None,
    ) -> Any:  # noqa: ANN401
        """
        Ask a question about the current browser session.

        Args:
            request: The natural-language question to ask about the browser state.
            response_format: The Python or Pydantic type to coerce the answer into.

        Returns:
            Any: The answer returned by the LLM, coerced to the specified response_format.
        """
        if screenshots is None:
            screenshots = {"current_view": self._last_shot}
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
            screenshots=screenshots,
        )
        return result

    # ------------------------------------------------------------------
    #  Public helper – synchronous one-shot action
    # ------------------------------------------------------------------
    async def act(
        self,
        action: str,
        expectation: Optional[str] = None,
        multi_step_mode: bool = False,
        timeout: float = 60.0,
    ) -> str:
        """
        Converts a natural-language instruction into a browser action, executes it,
        waits for state confirmation, and optionally verifies that the result matches the expectation.
        """
        # 1. GET THE LOW-LEVEL ACTION COMMAND (with retry logic)
        MAX_PARSE_RETRIES = 3
        BASE_DELAY_S = 0.1  # 100 ms
        actions = None
        last_error = None

        for attempt in range(1, MAX_PARSE_RETRIES + 1):
            # Construct the text with error feedback from previous attempt
            if attempt > 1 and last_error:
                action_with_feedback = (
                    f"{action}\n\n"
                    f"[Previous attempt failed with: {last_error}. "
                    f"Please avoid this error and try a different approach.]"
                )
            else:
                action_with_feedback = action

            try:
                # Convert state dict to BrowserState object
                state_data = self._observe_ctx.get("state", {})
                state = (
                    BrowserState(**state_data)
                    if isinstance(state_data, dict)
                    else state_data
                )

                cmd_payload = await asyncio.to_thread(
                    text_to_browser_action,
                    text=action_with_feedback,
                    screenshot=self._last_shot,
                    tabs=self._observe_ctx.get("tabs", []),
                    buttons=self._observe_ctx.get("elements", []),
                    history=self._observe_ctx.get("history", []),
                    state=state,
                    multi_step_mode=multi_step_mode,
                )
                actions = cmd_payload.get("action")
                if not actions:
                    raise ActionFailedError(
                        f"Attempt {attempt}: LLM returned no action for '{action}'",
                    )
                break  # ✅ parsed OK → stop retry loop

            except InvalidActionError as e:
                last_error = str(e)
                LOGGER.warning(
                    "Parse failure %d/%d for '%s': %s",
                    attempt,
                    MAX_PARSE_RETRIES,
                    action,
                    e,
                )
                if attempt == MAX_PARSE_RETRIES:
                    raise  # bubble up after last try
                # Exponential backoff with small random jitter
                await asyncio.sleep(
                    BASE_DELAY_S * (2 ** (attempt - 1)) + random.uniform(0, 0.05),
                )

        # 2. WAIT FOR SPECIFIC STATE ACKNOWLEDGEMENT
        request_id = str(
            uuid.uuid4(),
        )  # Create a unique ID for this specific action request
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        ps = self._redis_client.pubsub(ignore_subscribe_messages=True)

        def _waiter(msg):
            try:
                data = json.loads(msg["data"])
                # The waiter now checks for its specific request_id in the state update!
                if data.get("ack_request_id") == request_id:
                    loop.call_soon_threadsafe(fut.set_result, None)
            except (json.JSONDecodeError, KeyError):
                pass

        await asyncio.to_thread(
            ps.subscribe,
            **{f"browser_state_{self._redis_db}": _waiter},
        )
        listener_thread = ps.run_in_thread(daemon=True)

        # Dispatch the action with its unique ID
        await asyncio.to_thread(self._perform_action, actions, request_id)

        try:
            await asyncio.wait_for(fut, timeout)
            LOGGER.info(f"✅ ACK received for action: {actions} (ID: {request_id})")
        except asyncio.TimeoutError:
            # This timeout is now much more meaningful. It means the worker
            # never acknowledged completing our specific action.
            LOGGER.warning(f"Timed out waiting for ACK for action ID: {request_id}")
            raise ActionFailedError(
                f"Browser worker did not acknowledge action '{action}' in time.",
            )
        finally:
            listener_thread.stop()

        # 2.1 WAIT FOR DOM TO SETTLE
        if expectation:  # only do it when we are about to verify
            loop2 = asyncio.get_running_loop()
            post_ack_fut = loop2.create_future()

            def _settle(msg):
                try:
                    data = json.loads(msg["data"])
                    # ignore the ACK message itself, look for *any* later tick
                    if data.get("ack_request_id") != request_id:
                        loop2.call_soon_threadsafe(post_ack_fut.set_result, None)
                except Exception:
                    pass

            ps2 = self._redis_client.pubsub(ignore_subscribe_messages=True)
            await asyncio.to_thread(
                ps2.subscribe,
                **{f"browser_state_{self._redis_db}": _settle},
            )
            t2 = ps2.run_in_thread(daemon=True)
            try:
                await asyncio.wait_for(post_ack_fut, timeout)
                LOGGER.info("✅ DOM settled - received post-ACK browser state")
            except asyncio.TimeoutError:
                LOGGER.warning("Timed out waiting for post-ACK browser state")
            finally:
                t2.stop()

        # 3. VERIFY EXPECTATION (if provided)
        if expectation:
            MAX_VERIFY_TRIES = 3
            delay = 0.4  # start at 400 ms
            last_reason = ""

            for attempt in range(1, MAX_VERIFY_TRIES + 1):
                try:
                    action_screenshots = {}
                    history = self._observe_ctx.get("history", [])
                    if history:
                        last_action_record = history[-1]  # Get the most recent action
                        before_b64 = last_action_record.get("before_screenshot_b64")
                        after_b64 = last_action_record.get("after_screenshot_b64")
                        if before_b64 and after_b64:
                            action_screenshots = {
                                "before_action": before_b64,
                                "after_action": after_b64,
                            }
                    verification_prompt = (
                        "You are a meticulous QA engineer. Your task is to determine if a browser action was successful by "
                        "analyzing the 'Before' and 'After' screenshots.\n\n"
                        "## CONTEXT\n"
                        f"- **Action Performed**: '{action}'\n"
                        f"- **Expected Outcome**: '{expectation}'\n\n"
                        "## VERIFICATION TASK\n"
                        "Compare the 'before_action' and 'after_action' screenshots. Does the visual change between them "
                        "logically correspond to the 'Action Performed' and satisfy the core intent of the 'Expected Outcome'? "
                        "Do not infer the intent. Your judgment must be based on evidence from the browser context and screenshots."
                    )
                    verification = await self.observe(
                        verification_prompt,
                        response_format=VerificationResult,
                        screenshots=action_screenshots,
                    )

                    if verification.is_satisfied:
                        LOGGER.info(
                            f"✅ Action '{action}' SUCCEEDED and met expectation '{expectation}'. Reason: {verification.reason}",
                        )
                        return actions

                    last_reason = verification.reason
                    LOGGER.warning(
                        f"Action '{action}' did not meet expectation (attempt {attempt}/{MAX_VERIFY_TRIES}). Reason: {verification.reason}",
                    )

                    if attempt < MAX_VERIFY_TRIES:
                        await asyncio.sleep(delay)
                        delay *= 2  # exponential back-off

                except Exception as e:
                    LOGGER.error(
                        f"Error during verification of action '{action}' (attempt {attempt}/{MAX_VERIFY_TRIES}): {e}",
                    )
                    if attempt == MAX_VERIFY_TRIES:
                        raise e

            # If all retries fail, raise the exception with the rich reason
            message = f"Action '{action}' failed. Reason: '{last_reason}'"
            raise ActionFailedError(
                message=message,
                action=action,
                expectation=expectation,
                reason=last_reason,
            )
        else:
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
