from __future__ import annotations
import pytest

from tests.helpers import _handle_project
from unity.events.event_bus import EVENT_BUS
from unity.events.manager_event_logging import wrap_handle_with_logging, new_call_id
from unity.common.async_tool_loop import SteerableToolHandle


class _TupleAnswerHandle(SteerableToolHandle):  # returns [answer, steps]
    def __init__(self) -> None:
        self._done = False
        self._paused = False

    # SteerableHandle API
    async def ask(
        self,
        question: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ) -> "SteerableToolHandle":
        return self

    def interject(
        self,
        message: str,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ):
        return "ack"

    # SteerableToolHandle API
    def stop(
        self,
        reason: str | None = None,
        *,
        parent_chat_context_cont: list[dict] | None = None,
    ):
        self._done = True
        return "Stopped"

    def pause(self):
        self._paused = True
        return "Paused"

    def resume(self):
        self._paused = False
        return "Resumed"

    def done(self):
        return self._done

    async def result(self):
        self._done = True
        return ["OK", [{"role": "system", "content": "..."}]]

    async def next_clarification(self) -> dict:
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        return None


@pytest.mark.asyncio
@_handle_project
async def test_manager_logging_sanitizes_iterable_answer_to_string():
    inner = _TupleAnswerHandle()
    call_id = new_call_id()
    logged = wrap_handle_with_logging(inner, call_id, "UnitTestManager", "ask")

    # Invoke result() → wrapper publishes a ManagerMethod event with phase="outgoing"
    out = await logged.result()
    assert isinstance(out, list) and out[0] == "OK"

    # Ensure logs are flushed to backend before searching
    EVENT_BUS.join_published()

    # Fetch the newest ManagerMethod event for our manager/method with outgoing phase
    events = await EVENT_BUS.search(
        filter=(
            'type == "ManagerMethod" and '
            'payload["manager"] == "UnitTestManager" and '
            'payload["method"] == "ask" and '
            'payload["phase"] == "outgoing"'
        ),
        limit=1,
    )

    assert len(events) == 1
    evt = events[0]
    # The logged answer must be a string (sanitized from a list/tuple return)
    assert isinstance(evt.payload.get("answer"), str)
    assert evt.payload.get("answer") == "OK"
