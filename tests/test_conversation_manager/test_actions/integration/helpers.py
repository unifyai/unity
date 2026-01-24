"""
Shared utilities for ConversationManager → CodeActActor integration tests.

These helpers are intentionally lightweight and deterministic:
- No fixed sleeps.
- All waits are condition/poll based with explicit timeouts.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import unify

from tests.test_conversation_manager.cm_helpers import filter_events_by_type
from unity.conversation_manager.events import (
    ActorHandleStarted,
    ActorResult,
    Event,
    Error,
    SMSSent,
    EmailSent,
    UnifyMessageSent,
    PhoneCallSent,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Generic waits (no sleeps)
# ---------------------------------------------------------------------------


async def wait_for_condition(
    predicate: Callable[[], bool | Awaitable[bool]],
    *,
    timeout: float = 30.0,
    poll: float = 0.05,
    timeout_message: str | None = None,
) -> None:
    """Poll predicate until it returns True, or raise TimeoutError."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        ret = predicate()
        ok = await ret if asyncio.iscoroutine(ret) else ret
        if ok:
            return
        await asyncio.sleep(poll)
    raise TimeoutError(
        timeout_message or "Timed out waiting for condition to become true.",
    )


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def get_actor_started_event(result: Any) -> ActorHandleStarted:
    """Return the first ActorHandleStarted event from a StepResult."""
    matches = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert matches, "Expected at least one ActorHandleStarted event"
    return matches[0]


def extract_actor_handle(cm: Any, handle_id: int) -> Any:
    """Extract the SteerableToolHandle stored in ConversationManager.in_flight_actions."""
    handle_data = cm.cm.in_flight_actions.get(handle_id)
    assert (
        handle_data is not None
    ), f"No in-flight action found for handle_id={handle_id}"
    handle = handle_data.get("handle")
    assert (
        handle is not None
    ), f"In-flight action missing handle for handle_id={handle_id}"
    return handle


async def wait_for_actor_result_event(
    cm: Any,
    handle_id: int,
    *,
    timeout: float = 30.0,
) -> ActorResult:
    """Wait for ActorResult on the in-memory broker (published by actor_watch_result)."""
    broker = cm.cm.event_broker
    async with broker.pubsub() as pubsub:
        await pubsub.subscribe("app:actor:result")
        msg = None
        # Use repeated get_message timeouts so we can enforce a hard overall timeout.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            msg = await pubsub.get_message(timeout=0.5, ignore_subscribe_messages=True)
            if not msg:
                continue
            try:
                evt = Event.from_json(msg["data"])
            except Exception:
                continue
            if isinstance(evt, ActorResult) and int(evt.handle_id) == int(handle_id):
                return evt
        raise TimeoutError(
            f"Timed out waiting for ActorResult for handle_id={handle_id}",
        )


async def wait_for_actor_completion(
    cm: Any,
    handle_id: int,
    *,
    timeout: float = 90.0,
) -> str:
    """Wait for the actor handle result (primary completion signal)."""
    handle = extract_actor_handle(cm, handle_id)
    return await asyncio.wait_for(handle.result(), timeout=timeout)


def assert_no_errors(result: Any) -> None:
    """Assert there are no Error events emitted during the step."""
    errors = filter_events_by_type(result.output_events, Error)
    assert not errors, f"Unexpected Error events: {[e.to_dict() for e in errors]}"


async def inject_actor_result(
    cm_driver: Any,
    *,
    handle_id: int,
    result: str,
    success: bool = True,
) -> None:
    """
    Deterministically apply an ActorResult event to CM state.

    Why this exists:
    - In these integration tests, we drive CM via `CMStepDriver.step_until_wait()`,
      which patches the event broker and does not always forward background events.
    - Some smoke flows need the CM brain to observe the actor's completion result
      before it can take the next step (e.g., "find phone → send SMS").
    """
    from unity.conversation_manager.domains.event_handlers import EventHandler

    cm = cm_driver.cm
    evt = ActorResult(handle_id=handle_id, success=success, result=result)
    await EventHandler.handle_event(
        evt,
        cm,
        is_voice_call=cm.call_manager.uses_realtime_api,
    )


# ---------------------------------------------------------------------------
# CM "continue brain" helper (no new input event)
# ---------------------------------------------------------------------------


async def run_cm_until_wait(
    cm_driver: Any,
    *,
    max_steps: int = 5,
) -> list[Event]:
    """
    Run the ConversationManager's LLM loop until it calls `wait` (or max_steps).

    This is used to deterministically advance the conversation after background
    events (e.g., ActorResult) request another LLM turn.

    Returns:
        Output events emitted during these LLM steps (e.g., SMSSent).
    """
    cm = cm_driver.cm
    output_events: list[Event] = []

    original_publish = cm.event_broker.publish

    async def publish_wrapper(channel: str, message: str) -> int:
        try:
            evt = Event.from_json(message)
        except Exception:
            evt = None
        if evt is not None and isinstance(
            evt,
            (
                SMSSent,
                EmailSent,
                UnifyMessageSent,
                PhoneCallSent,
            ),
        ):
            output_events.append(evt)
        # Handle locally for deterministic state updates; do not forward to broker.
        if evt is not None:
            from unity.conversation_manager.domains.event_handlers import EventHandler

            await EventHandler.handle_event(
                evt,
                cm,
                is_voice_call=cm.call_manager.uses_realtime_api,
            )
        return 0

    # Patch request_llm_run so "requested turns" don't escape into background debouncers.
    original_request = cm.request_llm_run
    requested: list[tuple[float, bool]] = []

    async def patched_request(delay=0, cancel_running=False) -> None:
        requested.append((delay, cancel_running))
        return

    try:
        cm.event_broker.publish = publish_wrapper
        cm.request_llm_run = patched_request

        # Run until `wait`.
        for _ in range(max_steps):
            tool_name = await cm._run_llm()
            if tool_name == "wait" or tool_name is None:
                break
        return output_events
    finally:
        cm.event_broker.publish = original_publish
        cm.request_llm_run = original_request


# ---------------------------------------------------------------------------
# Clarification helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Clarification:
    call_id: str
    question: str
    raw: dict[str, Any]


async def wait_for_clarification(
    handle: Any,
    *,
    timeout: float = 30.0,
) -> Clarification:
    """Wait for a clarification request on the handle."""
    clar = await asyncio.wait_for(handle.next_clarification(), timeout=timeout)
    if not isinstance(clar, dict):
        raise AssertionError(
            f"Expected dict clarification, got: {type(clar).__name__}: {clar!r}",
        )
    # Some handles (e.g., ActorHandle) intentionally omit call_id; they treat it as opaque/ignored.
    call_id = str(clar.get("call_id") or "")
    question = str(clar.get("question") or "")
    assert question, f"Clarification missing question: {clar}"
    return Clarification(call_id=call_id, question=question, raw=clar)


async def answer_clarification_and_continue(
    handle: Any,
    *,
    call_id: str,
    answer: str,
    timeout: float = 60.0,
) -> None:
    """
    Answer a clarification and wait for the handle to progress.

    We avoid sleeps by waiting for either:
    - the handle to complete, or
    - the next notification (common after answering), or
    - the handle to stop requiring clarifications (best-effort).
    """
    await handle.answer_clarification(call_id, answer)

    async def _progressed() -> bool:
        # If the handle finished, we definitely progressed.
        if handle.done():
            return True
        # If a notification arrives, we progressed (most implementations emit one).
        try:
            _ = await asyncio.wait_for(handle.next_notification(), timeout=0.1)
            return True
        except asyncio.TimeoutError:
            return False
        except RuntimeError:
            # Some handles disable notifications entirely.
            return False

    await wait_for_condition(
        _progressed,
        timeout=timeout,
        poll=0.05,
        timeout_message="Timed out waiting for handle to progress after answering clarification.",
    )


# ---------------------------------------------------------------------------
# Database verification helpers
# ---------------------------------------------------------------------------


def verify_contact_in_db(
    cm: Any,
    contact_id: int,
    expected_fields: dict[str, Any],
) -> dict[str, Any]:
    """
    Verify ContactManager has the expected fields for contact_id.

    Note: ContactManager.get_contact_info() returns dict[int, dict], keyed by contact_id.
    """
    mgr = cm.cm.contact_manager
    assert mgr is not None, "ConversationManager has no ContactManager"
    contact_dict = mgr.get_contact_info(contact_id)
    contact = (contact_dict or {}).get(contact_id)
    assert contact is not None, f"Contact {contact_id} not found in ContactManager"
    for k, v in expected_fields.items():
        assert (
            contact.get(k) == v
        ), f"Contact {contact_id} field {k!r}: expected {v!r}, got {contact.get(k)!r}"
    return contact


def verify_task_in_db(
    cm: Any,
    task_id: int,
    expected_fields: dict[str, Any],
) -> dict[str, Any]:
    """
    Verify TaskScheduler has a row for task_id with the expected fields.

    TaskScheduler stores tasks in a Unify "Tasks" context. We read a minimal row
    projection via its internal TasksStore for deterministic verification.
    """
    scheduler = cm.cm.task_scheduler
    assert scheduler is not None, "ConversationManager has no TaskScheduler"
    store = getattr(scheduler, "_store", None)
    assert store is not None, "TaskScheduler missing _store (storage not provisioned?)"

    logs = store.get_minimal_rows_by_task_ids(
        task_ids=int(task_id),
        fields=list(expected_fields.keys()),
    )
    assert (
        logs and len(logs) == 1
    ), f"Expected exactly 1 task row for task_id={task_id}, got {len(logs) if logs else 0}"
    row = logs[0].entries or {}
    for k, v in expected_fields.items():
        assert (
            row.get(k) == v
        ), f"Task {task_id} field {k!r}: expected {v!r}, got {row.get(k)!r}"
    return row


def verify_transcript_logged(
    cm: Any,
    *,
    expected_substring: str,
    contact_id: int | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    """
    Verify TranscriptManager logged a message containing expected_substring.

    We query the underlying transcripts context directly for robustness.
    """
    tm = cm.cm.transcript_manager
    assert tm is not None, "ConversationManager has no TranscriptManager"
    ctx = getattr(tm, "_transcripts_ctx", None)
    assert isinstance(ctx, str) and ctx, "TranscriptManager missing _transcripts_ctx"

    fields = [
        "message_id",
        "timestamp",
        "content",
        "sender_id",
        "receiver_ids",
        "exchange_id",
    ]
    logs = unify.get_logs(
        context=ctx,
        limit=limit,
        sorting={"timestamp": "descending"},
        from_fields=fields,
    )
    expected_lower = expected_substring.lower()
    for lg in logs or []:
        content = str((lg.entries or {}).get("content") or "")
        if expected_lower in content.lower():
            if contact_id is None:
                return dict(lg.entries or {})
            sender = (lg.entries or {}).get("sender_id")
            receivers = (lg.entries or {}).get("receiver_ids") or []
            if int(contact_id) == int(sender) or int(contact_id) in [
                int(x) for x in receivers
            ]:
                return dict(lg.entries or {})

    raise AssertionError(
        f"Did not find transcript message containing {expected_substring!r} "
        f"(contact_id={contact_id}) in last {limit} messages.",
    )


# ---------------------------------------------------------------------------
# File fixtures helper
# ---------------------------------------------------------------------------


def get_fixture_file_path(filename: str) -> str:
    """Return absolute path to a file in integration/fixtures/."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    path = fixtures_dir / filename
    assert path.exists(), f"Fixture file not found: {path}"
    return str(path)
