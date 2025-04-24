import json
import queue
import threading
from typing import List, Dict, Optional

import unify
from task_managers.sys_msgs import DETECT_TASK_REQUEST, FIRST_TASK, REORGANISE_TASKS
from pydantic import BaseModel, Field


class TaskRequested(BaseModel):
    """
    Whether or not a task was requested from the user.
    """

    reasoning: str = Field(
        ...,
        description="The reasoning behind your decision as to whether or not a task was requested.",
    )
    task_was_requested: bool = Field(
        ...,
        description="You believe a task was requested.",
    )


class FirstTask(BaseModel):
    """
    Details for the first task to be assigned.
    """

    should_create: bool = Field(
        ...,
        description="Whether or not this task should be created.",
    )
    title: Optional[str] = Field(
        ...,
        description="If `should_create is True`, then a suitable title for the task, a few words.",
    )
    description: str = Field(
        ...,
        description="If `should_create is True`, then a suitable task description, one or two sentences.",
    )
    start_at: Optional[str] = Field(
        ...,
        description="Optional starting timestamp (YYYY-MM-DDTHH:MM:SS). If unspecified, then the task is started as soon as possible.",
    )
    recurring: Optional[List[str]] = Field(
        ...,
        description="Optional recurring schedule, as a list of days and times (['Monday:HH:MM:SS', 'Wednesday:HH:MM:SS'])",
    )


class TaskManager(threading.Thread):

    def __init__(
        self,
        transcript_q: "queue.Queue[List[str]]",
        text_command_q: "queue.Queue[List[str]]",
        *,
        daemon: bool = True,
    ) -> None:
        super().__init__(daemon=daemon)
        self._transcript_q = transcript_q
        self._text_command_q = text_command_q

        self._task_request_client = unify.Unify("gpt-4o-mini@openai")
        self._task_request_client.set_system_message(DETECT_TASK_REQUEST)
        self._task_request_client.set_response_format(TaskRequested)

        self._task_organizer_client = unify.Unify("o3-mini@openai")

    def _detect_task_request(self, messages: List[Dict[str, str]]) -> bool:
        raw = self._task_request_client.copy().generate(json.dumps(messages, indent=4))
        parsed = TaskRequested.model_validate_json(raw)
        return parsed.task_was_requested

    def _update_tasks(self, messages: List[Dict[str, str]]):
        # Debug code ----
        if "Tasks" in unify.get_contexts():
            unify.delete_context("Tasks")
            unify.create_context("Tasks")
        # End Debug code ----
        if "Tasks" not in unify.get_contexts():
            unify.create_context("Tasks")
        current_tasks = unify.get_logs(
            context="Tasks",
        )
        if current_tasks:
            self._task_organizer_client.set_system_message(REORGANISE_TASKS)
        else:
            self._task_organizer_client.set_system_message(FIRST_TASK)
            self._task_organizer_client.set_response_format(FirstTask)
            first_task = self._task_organizer_client.generate(
                json.dumps(messages, indent=4),
            )
            first_task = FirstTask.model_validate_json(first_task)
            if first_task.should_create:
                print(
                    f"\n\nStarting task: {first_task.title} - {first_task.description}\n",
                )
                self._text_command_q.put(first_task.description)
            unify.log(
                context="Tasks",
                title=first_task.title,
                description=first_task.description,
                in_progress=False,
                start_at=first_task.start_at,
                recurring=first_task.recurring,
            )

    def run(self) -> None:
        while True:
            messages = self._transcript_q.get()
            if messages is None:
                break

            task_was_requested = self._detect_task_request(messages)
            if not task_was_requested:
                continue

            self._update_tasks(messages)
