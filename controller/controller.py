import queue
import asyncio
import threading
from typing import List

from controller.worker import BrowserWorker


class Controller(threading.Thread):

    def __init__(
        self,
        text_command_q: "queue.Queue[List[str]]",
        browser_screenshot_q: "queue.Queue[List[str]]",
        browser_command_q: "queue.Queue[List[str]]",
        *,
        daemon: bool = True,
    ) -> None:
        super().__init__(daemon=daemon)
        self._text_command_q = text_command_q
        self._browser_screenshot_q = browser_screenshot_q
        self._browser_command_q = browser_command_q

        self._browser_worker = BrowserWorker(
            self._browser_command_q,
            self._browser_screenshot_q,
            start_url="https://www.google.com/",
            refresh_interval=0.4,
        )

    def start(self):
        self._browser_worker.start()

    def run(self) -> None:
        while True:
            text_command = self._text_command_q.get()
            if text_command is None:
                break

            screenshot = self._browser_screenshot_q.get()

            # ToDo: implement LLM logic
            cmd = None

            self._browser_command_q(cmd)

    # Properties #
    # -----------#

    @property
    def user_request_q(self) -> queue.Queue[list[str]]:
        return self._user_request_q

    @property
    def speech_interupt_q(self) -> asyncio.Queue[str]:
        return self._speech_interupt_q
