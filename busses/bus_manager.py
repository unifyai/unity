import queue
import asyncio
from task_managers.task_manager import TaskManager
from controller.controller import Controller
from planner.planner import Planner


class BusManager:

    def __init__(self) -> None:

        # Queues #
        # -------#

        # the latest (windowed) user-agent transcript, updated after every new exchange
        self._transcript_q: queue.Queue[list[str]] = queue.Queue()

        # low-level browser actions, in text form
        self._text_action_q: queue.Queue[str] = queue.Queue()

        # user task requests, in text form
        self._text_task_q: queue.Queue[str] = queue.Queue()

        # lower-level browser commands
        self._browser_command_q: queue.Queue[str] = queue.Queue()

        # playwright browser state
        self._browser_state_q: queue.Queue[str] = queue.Queue()

        # actions which have been completed, referenced by their title
        self._action_completion_q: queue.Queue[str] = queue.Queue()

        # tasks which have been completed, referenced by their title
        self._task_completion_q: asyncio.Queue[str] = asyncio.Queue()

        # Managers #
        # ---------#

        # re-organizes and schedules task, based on transcripts
        self._task_manager = TaskManager(
            # [reads from]: detect when a task trigger + change is requested from transcript
            self._transcript_q,
            # [writes to]: parses intent from transcript + sends clear text commands
            self._text_task_q,
        )

        # handles hierarchical task planning + decomposition
        self._planner = Planner(
            # [read from]: take high-level text-based tasks and decomposes into low-level text-based actions
            self._text_task_q,
            # [writes to]: send these low-level text-based actions to the controller
            self._text_action_q,
            # [reads from]: determines when the low-level actions are completed
            self._action_completion_q,
            # [writes to]: writes incremental task progress, so the user-facing assistant stays updated
            self._task_completion_q,
        )

        # handles text -> low-level browser commands
        self._controller = Controller(
            # [reads from]: take low-level text-based actions and convert to browser actions
            self._text_action_q,
            # [reads from]: use the browser state as context for text->action controller
            self._browser_state_q,
            # [writes to]: send the browser commands for the browser worker to execute
            self._browser_command_q,
            # [writes to]: sends the name of the completed action, once it is completed
            self._action_completion_q,
        )

    def start(self):
        self._task_manager.start()
        self._planner.start()
        self._controller.start()

    # Properties #
    # -----------#

    @property
    def transcript_q(self) -> queue.Queue[list[str]]:
        return self._transcript_q

    @property
    def task_completion_q(self) -> asyncio.Queue[str]:
        return self._task_completion_q
