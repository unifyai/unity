import queue
import threading
from datetime import datetime, timezone
from typing import List
import logging

logger = logging.getLogger(__name__)

import unify
from controller.worker import BrowserWorker
from controller.agent import text_to_browser_action
from constants import SESSION_ID


class Controller(threading.Thread):

    def __init__(
        self,
        text_command_q: "queue.Queue[List[str]]",
        browser_state_q: "queue.Queue[List[str]]",
        browser_command_q: "queue.Queue[List[str]]",
        action_completion_q: "queue.Queue[List[str]]",
        *,
        daemon: bool = True,
    ) -> None:
        super().__init__(daemon=daemon)
        self._text_action_q = text_command_q
        self._browser_state_q = browser_state_q
        self._browser_command_q = browser_command_q
        self._action_completion_q = action_completion_q

        self._browser_worker = BrowserWorker(
            command_q=self._browser_command_q,
            update_q=self._browser_state_q,
            start_url="https://www.google.com/",
            refresh_interval=0.4,
        )
        self._browser_open = False

    def run(self) -> None:
        while True:
            text_action = self._text_action_q.get()
            if text_action is None:
                break

            if self._browser_open:
                browser_state = self._browser_state_q.get()
            else:
                browser_state = {}
            with unify.Context("LLM Traces"), unify.Log(
                session_id=SESSION_ID,
                name="text_to_browser_action",
            ):
                cmd = text_to_browser_action(
                    text=text_action,
                    screenshot=browser_state.get("screenshot", b""),
                    tabs=browser_state.get("tabs", []),
                    buttons=browser_state.get("elements", []),
                    history=browser_state.get("history", []),
                    state=browser_state.get("state", {}),
                )
            assert cmd is not None, f"text_command {text_action} returned empty command"
            action = cmd["action"]

            if action == "open browser" and not self._browser_open:
                # ToDo: implement this action
                self._browser_worker.start()
                self._browser_open = True
            elif action == "close browser" and self._browser_open:
                # ToDo: implement this action
                self._browser_worker.stop()
                self._browser_worker.join(timeout=2)
                self._browser_open = False
            elif not self._browser_open:
                self._browser_worker.start()
                self._browser_open = True
                self._browser_command_q.put(action)
            else:
                self._browser_command_q.put(action)

            # ToDo: only send this once we KNOW the browser action has completed successfully
            self._action_completion_q.put(action)
            t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
            logger.info(f"\n🕹️ Performed Action: {action} [⏱️ {t}]\n")
