import threading
from datetime import datetime, timezone

import redis
import unify

from .playwright.worker import BrowserWorker
from .agent import text_to_browser_action
from ..constants import SESSION_ID, LOGGER


class Controller(threading.Thread):

    def __init__(self, *, daemon: bool = True) -> None:
        super().__init__(daemon=daemon)
        self._redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self._pubsub_text_action = self._redis_client.pubsub()
        self._pubsub_text_action.subscribe("text_action")
        self._pubsub_browser_state = self._redis_client.pubsub()
        self._pubsub_browser_state.subscribe("browser_state")

        self._browser_worker = BrowserWorker(
            start_url="https://www.google.com/",
            refresh_interval=0.4,
        )
        self._browser_open = False

    def run(self) -> None:
        for text_action_ in self._pubsub_text_action.listen():
            if text_action_["type"] != "message":
                continue
            text_action = text_action_["data"]
            if self._browser_open:
                browser_state = self._pubsub_browser_state.get_message()
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
                self._redis_client.publish("browser_state", action)
            else:
                self._redis_client.publish("browser_state", action)

            # ToDo: only send this once we KNOW the browser action has completed successfully
            self._redis_client.publish("action_completion", action)
            t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
            LOGGER.info(f"\n🕹️ Performed Action: {action} [⏱️ {t}]\n")
