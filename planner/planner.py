import queue
import asyncio
import threading
from typing import List
from asyncio import AbstractEventLoop

import unify


class Planner(threading.Thread):

    def __init__(
        self,
        task_update_queue: "queue.Queue[List[str]]",
        text_action_q: "queue.Queue[List[str]]",
        action_completion_q: "queue.Queue[List[str]]",
        task_completion_q: asyncio.Queue[str],
        coms_asyncio_loop: AbstractEventLoop,
        *,
        daemon: bool = True,
    ) -> None:
        """
        Receives a stream of user inputs related to this task (can either be high-level or low-level guidance), and must stream a series of low-level actions to the controller, as quickly and efficiently as possible, in order to complete the task.

        Args:
            task_update_queue (queue.Queue[List[str]]): Where the text-based user updates for the task come from, as well as the initial task request.
            text_action_q (queue.Queue[List[str]]): Where the low-level text actions are sent.
            action_completion_q (queue.Queue[List[str]]): Where the completion status of the low-level text actions come from.
            task_completion_q (asyncio.Queue[str]): Where we inform the user that the *overall* task is complete.
            coms_asyncio_loop (AbstractEventLoop): The asyncio loop for the user-facing agent. Need for task_completion_q.
        """
        super().__init__(daemon=daemon)
        self._text_task_q = task_update_queue
        self._text_action_q = text_action_q
        self._action_completion_q = action_completion_q
        self._task_completion_q = task_completion_q
        self._coms_asyncio_loop = coms_asyncio_loop

    def run(self) -> None:
        while True:
            ret = self._text_task_q.get()
            if ret is None:
                break
            text_task_log, task_description = ret
            text_task_log = unify.Log.from_json(text_task_log)

            # ToDo: implement task decomposition, instead of this trivial pass-through
            text_action = task_description
            # end ToDO

            # there will typically be several actions per task, currently just one
            self._text_action_q.put(text_action)

            # ToDo: work out why this is not working
            # text_task_log.update_entries(status="completed")
            self._coms_asyncio_loop.call_soon_threadsafe(
                self._task_completion_q.put_nowait,
                task_description,
            )
