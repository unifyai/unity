from __future__ import annotations

from typing import Callable, Dict, Optional

from ..events.event_bus import EVENT_BUS, Event
from .llm_helpers import make_request_clarification_tool, make_send_notification_tool


def add_clarification_tool_with_events(
    tools: Dict[str, Callable],
    up_q,
    down_q,
    *,
    manager: str,
    method: str,
    call_id: Optional[str],
) -> None:
    """
    Add a `request_clarification` tool to the tools dict that publishes
    ManagerMethod events for requests/answers.
    """

    async def _on_request(q: str):
        try:
            await EVENT_BUS.publish(
                Event(
                    type="ManagerMethod",
                    calling_id=call_id,
                    payload={
                        "manager": manager,
                        "method": method,
                        "action": "clarification_request",
                        "question": q,
                    },
                ),
            )
        except Exception:
            pass

    async def _on_answer(ans: str):
        try:
            await EVENT_BUS.publish(
                Event(
                    type="ManagerMethod",
                    calling_id=call_id,
                    payload={
                        "manager": manager,
                        "method": method,
                        "action": "clarification_answer",
                        "answer": ans,
                    },
                ),
            )
        except Exception:
            pass

    tools["request_clarification"] = make_request_clarification_tool(
        up_q,
        down_q,
        on_request=_on_request,
        on_answer=_on_answer,
    )


def add_notification_tool_with_events(
    tools: Dict[str, Callable],
    *,
    manager: str,
    method: str,
    call_id: Optional[str],
) -> None:
    """
    Add a `send_notification` tool to the tools dict that publishes
    ManagerMethod events for each notification.
    """

    async def _on_notify(message: str):
        try:
            await EVENT_BUS.publish(
                Event(
                    type="ManagerMethod",
                    calling_id=call_id,
                    payload={
                        "manager": manager,
                        "method": method,
                        "action": "notification",
                        "message": message,
                    },
                ),
            )
        except Exception:
            pass

    tools["send_notification"] = make_send_notification_tool(
        on_notify=_on_notify,
    )
