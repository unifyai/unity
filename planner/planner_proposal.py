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
        Takes high level tasks as input, and orchestrates their execution by sending low-level commands to the controller, and simultaneously listening for new guidance and requests from the user. The plan takes the form of python source code, where each lowest-level function is a simple english language command which should map directly to a single playwright command. At any point in time, the plan is in progress, with a call stack showing the full nest of functions which are "in progress".

        Args:
            task_update_queue (queue.Queue[List[str]]): Where the text-based user updates for the task come from, as well as the initial task request.
            text_action_q (queue.Queue[List[str]]): Where the low-level text actions are sent.
            action_completion_q (queue.Queue[List[str]]): Where the completion status of the low-level text actions come from.
            task_completion_q (asyncio.Queue[str]): Where we inform the user that the *overall* task is complete.
            coms_asyncio_loop (AbstractEventLoop): The asyncio loop for the user-facing agent. Need for task_completion_q.
        """
        super().__init__(daemon=daemon)
        self.task_update_queue = task_update_queue
        self._text_action_q = text_action_q
        self._action_completion_q = action_completion_q
        self._task_completion_q = task_completion_q
        self._coms_asyncio_loop = coms_asyncio_loop

    def _first_pass(self, task_description: str) -> None:
        """
        Create an inital implementation in a zero shot manner, making use of **any** functions which are already cached in the backend. This could include a 1-for-1 replacement if the text description coming from the user matches an existing function docstirng exactly (this is how tasks are "memorized").

        Args:
            task_description (str): The description of the task to be executed.
        """
        raise NotImplemented

    def _verify_completes() -> None:
        """
        Decorator which wraps **all** functions in the task's Python implementation, at all levels in the call stack hierarchy. At runtime, it ensures that the function has completed successfully, with help from an LLM. When an Exception is raised during execution, then a reimplementation is necessary (no checking needed), whereas if the function completed without errors, then the full call stack (below this functon, not above) and the set of primitve actions and set of screenshots should be parsed, and the agent should validate whether the task described in the function docstring has been executed successfully. If not, then a re-implementation should be made. The total task is complete once this decorator passes for the uppermost python function, for the original task description.
        """
        raise NotImplemented

    def redirect() -> None:
        """
        Constantly monitor the text task queue, and if a new command is given, this function must immediately parse the full stack trace, with all of the pending function states up the stack, and a clear representation of where the assistant currently is in the plan, at the lowest leaf in the call stack, and this function should create a new state, modifying the location in the call stack and the python implementation of functions as needed. The function should actually produce **three** ouputs. Firstly, a new task implementation in python source code, secondly, a new call-stack state the agent should resume from within this task, and finally, a `course_correct` task implementation which takes the agent from it's current state to the desired resuming state.
        """
        raise NotImplemented

    def run(self) -> None:
        while True:
            ret = self.task_update_queue.get()
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
