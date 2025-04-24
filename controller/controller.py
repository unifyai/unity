import queue
import asyncio
from task_managers.task_organizer import TaskOrganizer
from controller.worker import BrowserWorker
from controller.controller import Controller


class Controller:

    def __init__(self) -> None:

        # Queues
        self._user_request_q: queue.Queue[list[str]] = queue.Queue()
        self._browser_command_q: queue.Queue[str] = queue.Queue()
        self._browser_screenshot_q: queue.Queue[str] = queue.Queue()
        self._speech_interupt_q: asyncio.Queue[str] = asyncio.Queue()

        # Managers
        self._task_organizer = TaskOrganizer(
            self._user_request_q,
            self._browser_command_q,
        )

        self._controller = Controller(
            self._,
        )

        self._browser_worker = BrowserWorker(
            self._browser_command_q,
            self._browser_screenshot_q,
            start_url="https://www.google.com/",
            refresh_interval=0.4,
        )

    def start(self):
        self._task_organizer.start()
        self._browser_worker.start()

    # Properties #
    # -----------#

    @property
    def user_request_q(self) -> queue.Queue[list[str]]:
        return self._user_request_q

    @property
    def speech_interupt_q(self) -> asyncio.Queue[str]:
        return self._speech_interupt_q
