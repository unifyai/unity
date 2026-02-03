"""Multi-handle coordination for async tool loops.

This module provides the coordinator and per-request handle classes that enable
a single async tool loop to serve multiple concurrent requests with shared context.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from .request_state import RequestRegistry
from .tagging import (
    tag_message_with_request,
    format_request_cancelled_notice,
    format_request_paused_notice,
    format_request_resumed_notice,
)


class MultiHandleCoordinator:
    """Coordinates multiple requests within a single async tool loop.

    This class manages the shared state between multiple request handles,
    routing messages and results to the appropriate request based on ID.
    """

    def __init__(
        self,
        *,
        interject_queue: asyncio.Queue,
        clarification_channels: dict,
        persist: bool = False,
    ) -> None:
        """Initialize the coordinator.

        Parameters
        ----------
        interject_queue : asyncio.Queue
            The shared interjection queue for the loop.
        clarification_channels : dict
            The shared clarification channels mapping (call_id -> queues).
        persist : bool
            Whether the loop should persist after all requests complete.
        """
        self._registry = RequestRegistry()
        self._interject_queue = interject_queue
        self._clarification_channels = clarification_channels
        self._persist = persist
        # Per-request clarification queues (request_id -> Queue)
        self._request_clarification_queues: dict[int, asyncio.Queue] = {}
        # Per-request notification queues (request_id -> Queue)
        self._request_notification_queues: dict[int, asyncio.Queue] = {}

    @property
    def registry(self) -> RequestRegistry:
        """Return the underlying request registry."""
        return self._registry

    def register_request(self, handle_ref: Any = None) -> int:
        """Register a new request and return its ID.

        Parameters
        ----------
        handle_ref : Any, optional
            Reference to the handle for this request.

        Returns
        -------
        int
            The assigned request ID.
        """
        request_id = self._registry.register(handle_ref)
        # Create per-request event queues
        self._request_clarification_queues[request_id] = asyncio.Queue()
        self._request_notification_queues[request_id] = asyncio.Queue()
        return request_id

    def complete_request(self, request_id: int, result: str) -> bool:
        """Mark a request as completed with the given result.

        Parameters
        ----------
        request_id : int
            The request ID to complete.
        result : str
            The final answer for this request.

        Returns
        -------
        bool
            True if successful, False if request_id invalid or already done.
        """
        return self._registry.complete(request_id, result)

    def cancel_request(self, request_id: int, reason: str | None = None) -> bool:
        """Mark a request as cancelled.

        Parameters
        ----------
        request_id : int
            The request ID to cancel.
        reason : str | None
            Optional cancellation reason.

        Returns
        -------
        bool
            True if successful, False if request_id invalid or already done.
        """
        return self._registry.cancel(request_id, reason)

    def inject_interjection(self, request_id: int, message: str) -> None:
        """Inject a tagged interjection into the shared queue.

        Parameters
        ----------
        request_id : int
            The request ID this interjection belongs to.
        message : str
            The interjection message.
        """
        tagged = tag_message_with_request(message, request_id)
        self._interject_queue.put_nowait(tagged)

    def inject_cancellation_notice(
        self,
        request_id: int,
        reason: str | None = None,
    ) -> None:
        """Inject a cancellation notice into the loop.

        Parameters
        ----------
        request_id : int
            The request ID being cancelled.
        reason : str | None
            Optional cancellation reason.
        """
        notice = format_request_cancelled_notice(request_id, reason)
        self._interject_queue.put_nowait(notice)

    def inject_pause_notice(self, request_id: int) -> None:
        """Inject a pause notice into the loop.

        Parameters
        ----------
        request_id : int
            The request ID being paused.
        """
        notice = format_request_paused_notice(request_id)
        self._interject_queue.put_nowait(notice)

    def inject_resume_notice(self, request_id: int) -> None:
        """Inject a resume notice into the loop.

        Parameters
        ----------
        request_id : int
            The request ID being resumed.
        """
        notice = format_request_resumed_notice(request_id)
        self._interject_queue.put_nowait(notice)

    def route_clarification_to_request(
        self,
        request_id: int,
        clarification: dict,
    ) -> None:
        """Route a clarification question to the appropriate request's queue.

        Parameters
        ----------
        request_id : int
            The target request ID.
        clarification : dict
            The clarification event dict.
        """
        q = self._request_clarification_queues.get(request_id)
        if q is not None:
            q.put_nowait(clarification)

    def route_notification_to_request(
        self,
        request_id: int,
        notification: dict,
    ) -> None:
        """Route a notification to the appropriate request's queue.

        Parameters
        ----------
        request_id : int
            The target request ID.
        notification : dict
            The notification event dict.
        """
        q = self._request_notification_queues.get(request_id)
        if q is not None:
            q.put_nowait(notification)

    def get_clarification_queue(self, request_id: int) -> asyncio.Queue | None:
        """Get the clarification queue for a request."""
        return self._request_clarification_queues.get(request_id)

    def get_notification_queue(self, request_id: int) -> asyncio.Queue | None:
        """Get the notification queue for a request."""
        return self._request_notification_queues.get(request_id)

    def should_terminate(self) -> bool:
        """Check if the loop should terminate.

        Returns
        -------
        bool
            True if all requests are done and persist is False.
        """
        if self._persist:
            return False
        return self._registry.is_empty()

    def is_closed(self) -> bool:
        """Check if the coordinator is closed."""
        return self._registry.is_closed()

    def close(self) -> None:
        """Close the coordinator, preventing new requests."""
        self._registry.close()

    def get_request_future(self, request_id: int) -> asyncio.Future | None:
        """Get the result future for a request.

        Parameters
        ----------
        request_id : int
            The request ID.

        Returns
        -------
        asyncio.Future | None
            The future, or None if request not found.
        """
        state = self._registry.get(request_id)
        return state.result_future if state else None

    def is_request_done(self, request_id: int) -> bool:
        """Check if a specific request is done.

        Parameters
        ----------
        request_id : int
            The request ID to check.

        Returns
        -------
        bool
            True if the request is completed or cancelled.
        """
        state = self._registry.get(request_id)
        return state.is_done if state else True

    def validate_request_id(self, request_id: int) -> str | None:
        """Validate a request ID and return an error message if invalid.

        Parameters
        ----------
        request_id : int
            The request ID to validate.

        Returns
        -------
        str | None
            Error message if invalid, None if valid.
        """
        state = self._registry.get(request_id)
        if state is None:
            return f"Invalid request_id {request_id}: no such request exists."
        if state.is_done:
            return f"Invalid request_id {request_id}: request is already {state.status.value}."
        return None


class MultiRequestHandle:
    """Per-request handle for multi-handle async tool loops.

    This handle wraps a specific request_id and routes all steering
    operations through the shared MultiHandleCoordinator.
    """

    def __init__(
        self,
        request_id: int,
        coordinator: MultiHandleCoordinator,
        *,
        loop_id: str = "",
    ) -> None:
        """Initialize the per-request handle.

        Parameters
        ----------
        request_id : int
            The request ID this handle represents.
        coordinator : MultiHandleCoordinator
            The shared coordinator managing all requests.
        loop_id : str
            The loop identifier for logging.
        """
        self._request_id = request_id
        self._coordinator = coordinator
        self._loop_id = loop_id
        self._log_label = (
            f"{loop_id}[req:{request_id}]" if loop_id else f"req:{request_id}"
        )

    @property
    def request_id(self) -> int:
        """Return the request ID for this handle."""
        return self._request_id

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
    ) -> "MultiRequestHandle":
        """Ask about this request's status.

        In multi-handle mode, this injects a tagged question and returns self.
        """
        self._coordinator.inject_interjection(self._request_id, question)
        return self

    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
    ) -> None:
        """Inject additional context for this request."""
        self._coordinator.inject_interjection(self._request_id, message)

    async def stop(
        self,
        reason: Optional[str] = None,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
        **kwargs,
    ) -> None:
        """Stop/cancel this specific request.

        This marks the request as cancelled and notifies the LLM.
        Does NOT stop the entire loop unless this is the last pending request.
        """
        state = self._coordinator.registry.get(self._request_id)
        if state is None or state.is_done:
            return

        # Notify the LLM that this request is cancelled
        self._coordinator.inject_cancellation_notice(self._request_id, reason)
        # Mark the request as cancelled
        self._coordinator.cancel_request(self._request_id, reason)

    async def pause(self) -> None:
        """Pause this specific request (soft pause via LLM notification)."""
        state = self._coordinator.registry.get(self._request_id)
        if state is None or state.is_done:
            return
        self._coordinator.inject_pause_notice(self._request_id)

    async def resume(self) -> None:
        """Resume this specific request (soft resume via LLM notification)."""
        state = self._coordinator.registry.get(self._request_id)
        if state is None or state.is_done:
            return
        self._coordinator.inject_resume_notice(self._request_id)

    def done(self) -> bool:
        """Check if this request has completed."""
        return self._coordinator.is_request_done(self._request_id)

    async def result(self) -> str:
        """Wait for this request's final answer."""
        future = self._coordinator.get_request_future(self._request_id)
        if future is None:
            return f"Request {self._request_id} not found"
        return await future

    async def next_clarification(self) -> dict:
        """Await the next clarification for this request."""
        q = self._coordinator.get_clarification_queue(self._request_id)
        if q is None:
            # Block forever if no queue (request doesn't exist)
            await asyncio.Future()
        return await q.get()

    async def next_notification(self) -> dict:
        """Await the next notification for this request."""
        q = self._coordinator.get_notification_queue(self._request_id)
        if q is None:
            await asyncio.Future()
        return await q.get()

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        """Answer a clarification question.

        Routes the answer through the shared clarification channels.
        """
        channels = self._coordinator._clarification_channels
        if call_id in channels:
            _, down_q = channels[call_id]
            await down_q.put(answer)

    def add_request(self, message: str) -> "MultiRequestHandle":
        """Add a new request to the same loop.

        Parameters
        ----------
        message : str
            The initial message for the new request.

        Returns
        -------
        MultiRequestHandle
            A new handle for the added request.

        Raises
        ------
        RuntimeError
            If the loop has been closed.
        """
        if self._coordinator.is_closed():
            raise RuntimeError(
                "Loop has terminated. Start a new loop via start_async_tool_loop().",
            )

        # Register the new request
        new_request_id = self._coordinator.register_request()

        # Inject the tagged message into the loop
        self._coordinator.inject_interjection(new_request_id, message)

        # Create and return a new handle
        new_handle = MultiRequestHandle(
            new_request_id,
            self._coordinator,
            loop_id=self._loop_id,
        )

        # Store handle reference in registry
        state = self._coordinator.registry.get(new_request_id)
        if state:
            state.handle_ref = new_handle

        return new_handle

    def get_history(self) -> list[dict]:
        """Returns empty list - full history is on the shared loop."""
        return []
