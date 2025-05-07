import json
import time
import threading
from datetime import datetime, timezone
from typing import List, Dict, Optional

import redis
import unify
from constants import SESSION_ID, LOGGER
from task_managers.sys_msgs import FIRST_TASK
from pydantic import BaseModel, Field


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
        description="If `should_create is True`, then a suitable description of the task, written in third person, ignoring the chat context (no quotes etc.)",
    )
    start_at: Optional[str] = Field(
        ...,
        description="Optional starting timestamp (YYYY-MM-DDTHH:MM:SS). If unspecified, then the task is started as soon as possible.",
    )
    recurring: Optional[List[str]] = Field(
        ...,
        description="Optional recurring schedule, as a list of days and times (['Monday:HH:MM:SS', 'Wednesday:HH:MM:SS'])",
    )


class FirstTaskResponse(BaseModel):
    """
    Whether or not a task was requested from the user. If so, also return a full description of the requested task.
    """

    reasoning: str = Field(
        ...,
        description="The reasoning behind your decision as to whether or not a task was requested.",
    )
    task_was_requested: bool = Field(
        ...,
        description="You believe a task was requested.",
    )
    first_task: Optional[FirstTask] = Field(
        ...,
        description="Full breakdown of the task specified by the user.",
    )


class TaskManager(threading.Thread):

    def __init__(self, *, daemon: bool = True) -> None:
        super().__init__(daemon=daemon)
        self._redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self._pubsub = self._redis_client.pubsub()
        self._pubsub.subscribe("transcript")
        self._task_organizer_client = unify.Unify("o4-mini@openai", cache=True, traced=True)
        self._current_task = None

    def _maybe_update_tasks(self, messages: List[Dict[str, str]]):
        if "Tasks" not in unify.get_contexts():
            unify.create_context("Tasks")
        current_tasks = unify.get_logs(context="Tasks", filter="status != 'completed'")
        # if current_tasks:
        #     # ToDo: implement this properly
        #     current_tasks = {
        #         log.entries["title"]: {
        #             k: v
        #             for k, v in log.entries.items()
        #             if k in ("description", "in_progress", "start_at", "recurring")
        #         }
        #         for log in current_tasks
        #     }
        #     self._task_organizer_client.set_system_message(
        #         REORGANISE_TASKS.replace(
        #             "{current_tasks}",
        #             json.dumps(current_tasks, indent=4),
        #         ),
        #     )
        # else:
        self._task_organizer_client.set_system_message(FIRST_TASK)
        self._task_organizer_client.set_response_format(FirstTaskResponse)
        t0 = time.perf_counter()
        t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
        LOGGER.info(
            f"\n🤖 Task Manager: transcript to possible task updates... ⏳ [⏱️ {t}]\n",
        )
        messages = {
            "latest_message": messages[-1],
            "message_history": messages[:-1],
        }
        resp = self._task_organizer_client.generate(
            json.dumps(messages, indent=4),
        )
        t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
        LOGGER.info(
            f"\n🤖 Task Manager: transcript to possible task updates ✅ [⏱️ {t}] [⏩{(time.perf_counter() - t0):.3g}s]\n",
        )
        resp = FirstTaskResponse.model_validate_json(resp)
        if not resp.task_was_requested:
            return
        first_task = resp.first_task
        self._task_log = unify.log(
            context="Tasks",
            session_id=SESSION_ID,
            title=first_task.title,
            description=first_task.description,
            status="in progress",
            start_at=first_task.start_at,
            recurring=first_task.recurring,
            new=True,
        )
        if first_task.should_create:
            self._redis_client.publish("text_task", json.dumps([
                self._task_log.to_json(), first_task.description
            ]))

    def run(self) -> None:
        for transcript in self._pubsub.listen():
            if transcript["type"] != "message":
                continue
            messages = json.loads(transcript["data"])
            with unify.Context("LLM Traces"), unify.Log(
                session_id=SESSION_ID,
                name="_update_tasks",
            ):
                self._maybe_update_tasks(messages)
