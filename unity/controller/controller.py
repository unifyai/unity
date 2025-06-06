import threading
import asyncio
from datetime import datetime, timezone
from typing import Any, Type, Optional

import redis

from .playwright_utils.worker import BrowserWorker
from .agent import text_to_browser_action, ask_llm
from ..constants import LOGGER


class Controller(threading.Thread):
    def __init__(
        self,
        *,
        daemon: bool = True,
        session_connect_url: str | None = None,
        headless: bool = False,
    ) -> None:
        super().__init__(daemon=daemon)
        self._redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self._pubsub_text_action = self._redis_client.pubsub()
        self._pubsub_text_action.subscribe("text_action")
        self._pubsub_browser_state = self._redis_client.pubsub()
        self._pubsub_browser_state.subscribe("browser_state")
        self.session_connect_url = session_connect_url

        self._browser_worker = BrowserWorker(
            start_url="https://www.google.com/",
            refresh_interval=0.4,
            session_connect_url=self.session_connect_url,
            headless=headless,
        )
        self._browser_open = False
        self._stop_event = threading.Event()

        # Cached data for LLM observation queries
        self._observe_ctx: dict[str, Any] = {}
        self._last_shot: bytes = b""

    def run(self) -> None:
        """
        Background loop: listen for browser_state messages and update cached context.
        """
        if not self._browser_open:
            self._browser_worker.start()
            self._browser_open = True

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
                raw_elements = browser_state.get("elements", [])
                elements: list[tuple[Any, Any]] = []
                for item in raw_elements:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        elements.append((item[0], item[1]))
                self._observe_ctx = {
                    "state": browser_state.get("state", {}),
                    "elements": elements,
                    "tabs": browser_state.get("tabs", []),
                    "history": browser_state.get("history", []),
                }
                self._last_shot = browser_state.get("screenshot", b"")

    def stop(self) -> None:
        """Signal the controller thread to stop."""
        self._stop_event.set()
        # Close pubsub connections to break out of the listen loop
        self._pubsub_browser_state.close()
        self._pubsub_text_action.close()

    # ------------------------------------------------------------------
    # Internal helper – perform one or more low-level primitives (in order)
    # ------------------------------------------------------------------
    def _perform_action(self, actions: str | list[str]) -> None:
        # if given a list of commands, execute each in sequence
        for action in actions:
            # action is a single string
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
                self._redis_client.publish("browser_command", action)
            else:
                self._redis_client.publish("browser_command", action)

            # notify listeners that the action finished (optimistic)
            self._redis_client.publish("action_completion", action)

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

        # call LLM to answer based on refreshed context
        result = await asyncio.to_thread(
            ask_llm,
            request,
            response_format=response_format,
            context=self._observe_ctx,
            screenshot=self._last_shot,
        )
        return result

    # ------------------------------------------------------------------
    #  Public helper – synchronous one-shot action
    # ------------------------------------------------------------------
    async def act(self, action: str) -> Optional[str]:
        """
        Convert natural-language text to a browser action and execute it.

        Args:
            action: The natural-language instruction to convert into a browser primitive.

        Returns:
            Optional[str]: The executed action string, or None if no action was selected by the LLM.
        """

        cmd = await asyncio.to_thread(
            text_to_browser_action,
            text=action,
            screenshot=self._last_shot,
            tabs=self._observe_ctx.get("tabs", []),
            buttons=self._observe_ctx.get("elements", []),
            history=self._observe_ctx.get("history", []),
            state=self._observe_ctx.get("state", {}),
        )

        assert cmd is not None, f"requested action {action} returned empty command"
        actions = cmd.get("action")
        # execute the primitive action
        await asyncio.to_thread(self._perform_action, actions)
        return actions
