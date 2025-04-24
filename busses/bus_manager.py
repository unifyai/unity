import queue
import asyncio
from busses.task_request_listener import TaskRequestListener


class BusManager:

    def __init__(self) -> None:
        self._create_queues()
        self._create_listeners()

    def _create_queues(self) -> None:
        self._task_q: queue.Queue[list[str]] = queue.Queue()
        self._async_task_q: asyncio.Queue[str] = asyncio.Queue()

    def _create_listeners(self) -> None:
        self._task_request_listener = TaskRequestListener(self._task_q)

    def start(self):
        self._task_request_listener.start()

    # Properties #
    # -----------#

    @property
    def task_q(self) -> queue.Queue[list[str]]:
        return self._task_q

    @property
    def async_task_q(self) -> asyncio.Queue[str]:
        return self._async_task_q
