import asyncio
import json
import queue
import threading
from typing import List
from asyncio import AbstractEventLoop

import redis
import unify


class Planner(threading.Thread):

    def __init__(self, *, daemon: bool = True) -> None:
        """
        Receives a stream of user inputs related to this task (can either be high-level
        or low-level guidance), and must stream a series of low-level actions to the
        controller, as quickly and efficiently as possible, in order to complete the
        task.
        """
        super().__init__(daemon=daemon)
        self._redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self._pubsub = self._redis_client.pubsub()
        self._pubsub.subscribe("text_task")

    def run(self) -> None:
        for text_task in self._pubsub.listen():
            if text_task["type"] != "message":
                continue
            text_task_log, task_description = json.loads(text_task["data"])
            text_task_log = unify.Log.from_json(text_task_log)

            # ToDo: implement task decomposition, instead of this trivial pass-through
            text_action = task_description
            # end ToDO

            # there will typically be several actions per task, currently just one
            self._redis_client.publish("text_action", text_action)

            # ToDo: work out why this is not working
            # text_task_log.update_entries(status="completed")
            self._redis_client.publish("task_completion", task_description)
