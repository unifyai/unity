import queue
import asyncio
import threading
from typing import List

from controller.worker import BrowserWorker
from controller.agent import text_to_browser_action


class Controller(threading.Thread):

    def __init__(
        self,
        text_command_q: "queue.Queue[List[str]]",
        browser_state_q: "queue.Queue[List[str]]",
        browser_command_q: "queue.Queue[List[str]]",
        *,
        daemon: bool = True,
    ) -> None:
        super().__init__(daemon=daemon)
        self._text_command_q = text_command_q
        self._browser_state_q = browser_state_q
        self._browser_command_q = browser_command_q

        self._browser_worker = BrowserWorker(
            self._browser_command_q,
            self._browser_state_q,
            start_url="https://www.google.com/",
            refresh_interval=0.4,
        )
        self._browser_open = False

    def run(self) -> None:
        while True:
            text_command = self._text_command_q.get()
            if text_command is None:
                break

            if self._browser_open:
                browser_state = self._browser_state_q.get()
            else:
                browser_state = {}
            cmd = text_to_browser_action(
                text=text_command,
                screenshot=browser_state.get("screenshot", b""),
                tabs=browser_state.get("tabs", []),
                buttons=browser_state.get("elements", []),
                history=browser_state.get("history", []),
                state=browser_state.get("state", {}),
            )

            if cmd == "open browser" and not self._browser_open:
                # ToDo: implement this action
                self._browser_worker.start()
            elif cmd == "close browser" and self._browser_open:
                # ToDo: implement this action
                self._browser_worker.stop()
                self._browser_worker.join(timeout=2)
            elif not self._browser_open:
                self._browser_worker.start()
                self._browser_command_q(cmd)
            else:
                self._browser_command_q(cmd)

    # Properties #
    # -----------#

    @property
    def user_request_q(self) -> queue.Queue[list[str]]:
        return self._user_request_q

    @property
    def speech_interupt_q(self) -> asyncio.Queue[str]:
        return self._speech_interupt_q
