"""Centralised task-state wrapper around Redis.

Both the VoiceAssistant process and the BrowserWorker process share the same
Redis instance.  This dataclass provides a simple, process-local façade that
persists every change to Redis and always reads the latest value back when a
property is accessed.
"""

from dataclasses import dataclass, field
import json
import redis


@dataclass
class State:
    """Local helper that transparently proxies state to Redis."""

    _redis_client = redis.Redis(host="localhost", port=6379, db=0)

    def set_task_running(self, running: bool):
        self._redis_client.set("task_running", json.dumps(running))

    def set_task_paused(self, paused: bool):
        self._redis_client.set("task_paused", json.dumps(paused))

    def set_last_task_result(self, result: str | None):
        self._redis_client.set("last_task_result", json.dumps(result))

    def set_last_step_results(self, result: list[dict]):
        self._redis_client.set("last_step_results", json.dumps(result))

    @property
    def task_running(self) -> bool:
        return json.loads(self._redis_client.get("task_running").decode())

    @property
    def task_paused(self) -> bool:
        return json.loads(self._redis_client.get("task_paused").decode())

    @property
    def last_task_result(self) -> str:
        return self._redis_client.get("last_task_result").decode()

    @property
    def last_step_results(self) -> str:
        return self._redis.get("last_step_results").decode()
