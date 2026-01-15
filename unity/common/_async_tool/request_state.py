"""Request state management for multi-handle async tool loops.

This module provides the foundational dataclasses and registry for tracking
multiple concurrent requests within a single async tool loop instance.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class RequestStatus(Enum):
    """Status of a request within a multi-handle tool loop."""

    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class RequestState:
    """State for a single request in a multi-handle tool loop.

    Attributes
    ----------
    request_id : int
        Auto-assigned integer identifier for this request (0-indexed).
    status : RequestStatus
        Current lifecycle status of the request.
    result_future : asyncio.Future | None
        Future that resolves when the request completes or is cancelled.
        Created lazily on first access to avoid event loop issues.
    handle_ref : SteerableToolHandle | None
        Reference to the handle associated with this request.
    """

    request_id: int
    status: RequestStatus = RequestStatus.PENDING
    _result_future: asyncio.Future | None = field(default=None, repr=False)
    handle_ref: Any = None  # Typed as Any to avoid circular import issues

    @property
    def result_future(self) -> asyncio.Future:
        """Get the result future, creating it lazily if needed."""
        if self._result_future is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            self._result_future = loop.create_future()
        return self._result_future

    @property
    def is_pending(self) -> bool:
        """Return True if this request is still pending."""
        return self.status == RequestStatus.PENDING

    @property
    def is_completed(self) -> bool:
        """Return True if this request has completed."""
        return self.status == RequestStatus.COMPLETED

    @property
    def is_cancelled(self) -> bool:
        """Return True if this request was cancelled."""
        return self.status == RequestStatus.CANCELLED

    @property
    def is_done(self) -> bool:
        """Return True if this request is no longer pending."""
        return self.status != RequestStatus.PENDING


class RequestRegistry:
    """Registry for managing multiple requests within a single tool loop.

    Handles auto-incrementing request IDs, lifecycle transitions, and
    provides query methods for checking loop termination conditions.
    """

    def __init__(self) -> None:
        self._requests: dict[int, RequestState] = {}
        self._next_request_id: int = 0
        self._closed: bool = False

    def next_id(self) -> int:
        """Return the next request ID that would be assigned (without registering)."""
        return self._next_request_id

    def register(self, handle_ref: Any = None) -> int:
        """Register a new request and return its assigned ID.

        Parameters
        ----------
        handle_ref : Any, optional
            Reference to the handle associated with this request.

        Returns
        -------
        int
            The auto-assigned request ID (0-indexed, auto-incrementing).

        Raises
        ------
        RuntimeError
            If the registry has been closed.
        """
        if self._closed:
            raise RuntimeError(
                "Request registry is closed. Start a new loop via start_async_tool_loop().",
            )

        request_id = self._next_request_id
        self._next_request_id += 1

        state = RequestState(
            request_id=request_id,
            handle_ref=handle_ref,
        )
        self._requests[request_id] = state
        return request_id

    def get(self, request_id: int) -> RequestState | None:
        """Get the state for a specific request ID.

        Parameters
        ----------
        request_id : int
            The request ID to look up.

        Returns
        -------
        RequestState | None
            The request state, or None if not found.
        """
        return self._requests.get(request_id)

    def complete(self, request_id: int, result: str) -> bool:
        """Mark a request as completed and resolve its future.

        Parameters
        ----------
        request_id : int
            The request ID to complete.
        result : str
            The final answer/result for this request.

        Returns
        -------
        bool
            True if the request was successfully completed, False if the
            request_id was invalid or the request was already done.
        """
        state = self._requests.get(request_id)
        if state is None:
            return False
        if state.is_done:
            return False

        state.status = RequestStatus.COMPLETED
        if not state.result_future.done():
            state.result_future.set_result(result)
        return True

    def cancel(self, request_id: int, reason: str | None = None) -> bool:
        """Mark a request as cancelled and resolve its future.

        Parameters
        ----------
        request_id : int
            The request ID to cancel.
        reason : str | None, optional
            Optional reason for cancellation.

        Returns
        -------
        bool
            True if the request was successfully cancelled, False if the
            request_id was invalid or the request was already done.
        """
        state = self._requests.get(request_id)
        if state is None:
            return False
        if state.is_done:
            return False

        state.status = RequestStatus.CANCELLED
        if not state.result_future.done():
            cancel_msg = f"Request {request_id} cancelled"
            if reason:
                cancel_msg += f": {reason}"
            state.result_future.set_result(cancel_msg)
        return True

    def pending_count(self) -> int:
        """Return the number of requests still pending."""
        return sum(1 for s in self._requests.values() if s.is_pending)

    def is_empty(self) -> bool:
        """Return True if there are no pending requests."""
        return self.pending_count() == 0

    def is_closed(self) -> bool:
        """Return True if the registry has been closed."""
        return self._closed

    def close(self) -> None:
        """Close the registry, preventing new registrations."""
        self._closed = True

    def all_request_ids(self) -> list[int]:
        """Return a list of all request IDs (pending and done)."""
        return list(self._requests.keys())

    def pending_request_ids(self) -> list[int]:
        """Return a list of all pending request IDs."""
        return [rid for rid, s in self._requests.items() if s.is_pending]
