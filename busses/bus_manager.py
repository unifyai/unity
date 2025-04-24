import queue
import asyncio
from task_managers.task_manager import TaskManager
from controller.controller import Controller


class BusManager:

    def __init__(self) -> None:

        # Queues #
        # -------#

        # the latest (windowed) user-agent transcript, updated after every new exchange
        self._transcript_q: queue.Queue[list[str]] = queue.Queue()

        # user task requests, in text form
        self._text_command_q: queue.Queue[str] = queue.Queue()

        # lower-level browser commands
        self._browser_command_q: queue.Queue[str] = queue.Queue()

        # playwright browser state
        self._browser_state_q: queue.Queue[str] = queue.Queue()

        # tasks which have been completed, by their title
        self._task_completion_q: asyncio.Queue[str] = asyncio.Queue()

        # Managers #
        # ---------#

        # re-organizes and schedules task, based on transcripts
        self._task_manager = TaskManager(
            self._transcript_q,
            self._text_command_q,
        )

        # handles hierarchical task planning + decomposition
        # ToDo
        # include self._task_completion_q,
        # self._planner = ...

        # handles text -> low-level browser commands
        self._controller = Controller(
            self._text_command_q,
            self._browser_state_q,
            self._browser_command_q,
        )

    def start(self):
        self._task_manager.start()
        self._controller.start()

    # Properties #
    # -----------#

    @property
    def transcript_q(self) -> queue.Queue[list[str]]:
        return self._transcript_q

    @property
    def task_completion_q(self) -> asyncio.Queue[str]:
        return self._task_completion_q
