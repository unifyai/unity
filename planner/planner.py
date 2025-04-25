import queue
import asyncio
import threading
from typing import List
from asyncio import AbstractEventLoop


class Planner(threading.Thread):

    def __init__(
        self,
        text_task_q: "queue.Queue[List[str]]",
        text_action_q: "queue.Queue[List[str]]",
        action_completion_q: "queue.Queue[List[str]]",
        task_completion_q: asyncio.Queue[str],
        coms_asyncio_loop: AbstractEventLoop,
        *,
        daemon: bool = True,
    ) -> None:
        super().__init__(daemon=daemon)
        self._text_task_q = text_task_q
        self._text_action_q = text_action_q
        self._action_completion_q = action_completion_q
        self._task_completion_q = task_completion_q
        self._coms_asyncio_loop = coms_asyncio_loop

    def run(self) -> None:
        while True:
            text_task = self._text_task_q.get()
            if text_task is None:
                break

            # ToDo: implement task decomposition, instead of this trivial pass-through
            text_action = text_task
            # end ToDO

            # there will typically be several actions per task, currently just one
            self._text_action_q.put(text_action)
            text_action_completed = self._action_completion_q.get()
            assert text_action == text_action_completed

            self._coms_asyncio_loop.call_soon_threadsafe(
                self._task_completion_q.put_nowait,
                text_task,
            )
