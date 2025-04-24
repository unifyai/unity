import queue
import asyncio
from task_managers.task_organizer import TaskOrganizer
from controller.controller import Controller


class BusManager:

    def __init__(self) -> None:

        # Queues
        self._transcript_q: queue.Queue[list[str]] = queue.Queue()
        self._browser_command_q: queue.Queue[str] = queue.Queue()
        self._browser_screenshot_q: queue.Queue[str] = queue.Queue()
        self._speech_interupt_q: asyncio.Queue[str] = asyncio.Queue()

        # Managers
        self._task_organizer = TaskOrganizer(
            self._transcript_q,
            self._browser_command_q,
        )
        self._controller = Controller(
            self._transcript_q,
            self._browser_screenshot_q,
            self._browser_command_q,
        )

    def start(self):
        self._task_organizer.start()
        self._controller.start()

    # Properties #
    # -----------#

    @property
    def transcript_q(self) -> queue.Queue[list[str]]:
        return self._transcript_q

    @property
    def speech_interupt_q(self) -> asyncio.Queue[str]:
        return self._speech_interupt_q
