import json
import queue
import threading
from typing import List, Optional

import unify
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
#  Pydantic response model
# --------------------------------------------------------------------------- #
class Response(BaseModel):
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


# --------------------------------------------------------------------------- #
#  Thin wrapper around Unify – thread-safe because we create one per thread.
# --------------------------------------------------------------------------- #
def _detect_task(messages: List[str]) -> bool:
    """
    Run the LLM and return True/False for 'task requested'.
    """
    client = unify.Unify("gpt-4o-mini@openai")
    client.set_system_message(
        "Your task is to determine whether or not the provided messages contain a "
        "request from the user, asking the assistant to perform a task. You should "
        "state the reasoning for your decision clearly in the reasoning field, "
        "before responding with your final answer, `True` if a task was requested, "
        "and `False` if not.",
    )
    client.set_response_format(Response)

    raw = client.generate(json.dumps(messages, indent=4))
    parsed = Response.model_validate_json(raw)
    return parsed.task_was_requested


# --------------------------------------------------------------------------- #
#  Listener thread
# --------------------------------------------------------------------------- #
class TaskRequestListener(threading.Thread):
    """
    Background thread that blocks on a Queue[List[str]].  Each item is a full
    message history; when one arrives we ask the LLM whether a task was
    requested and optionally publish the Boolean to an output queue or callback.
    """

    def __init__(
        self,
        in_q: "queue.Queue[List[str]]",
        *,
        out_q: Optional["queue.Queue[bool]"] = None,
        callback: Optional[callable] = None,
        daemon: bool = True,
    ) -> None:
        super().__init__(daemon=daemon)
        self._in_q = in_q
        self._out_q = out_q
        self._callback = callback

    def run(self) -> None:
        while True:
            messages = self._in_q.get()  # blocks until something arrives
            if messages is None:  # -- sentinel → shut down
                break

            try:
                result = _detect_task(messages)
            except Exception as exc:  # never let the thread die silently
                print(f"[TaskRequestListener] error: {exc}")
                continue

            # Fan the result back out if requested
            if self._out_q is not None:
                self._out_q.put(result)
            if self._callback is not None:
                try:
                    self._callback(result, messages)
                except Exception as exc:  # keep going even if callback errs
                    print(f"[TaskRequestListener] callback error: {exc}")

        print("[TaskRequestListener] stopped.")
