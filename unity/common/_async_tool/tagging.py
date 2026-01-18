"""Message tagging utilities for multi-handle async tool loops.

Provides simple, isolated functions for tagging and parsing request IDs
in messages flowing through multi-handle tool loops.
"""

from __future__ import annotations

import re
from typing import Tuple

# Pattern for request tags: [Request 0], [Request 1], etc.
REQUEST_TAG_PATTERN = re.compile(r"^\[Request (\d+)\]\s*")


def tag_message_with_request(message: str, request_id: int) -> str:
    """Add a request ID tag prefix to a message.

    Parameters
    ----------
    message : str
        The message to tag.
    request_id : int
        The request ID to tag with.

    Returns
    -------
    str
        The tagged message with format: "[Request {id}] {message}"
    """
    return f"[Request {request_id}] {message}"


def parse_request_tag(message: str) -> Tuple[int | None, str]:
    """Extract request ID tag from a message if present.

    Parameters
    ----------
    message : str
        The message to parse.

    Returns
    -------
    Tuple[int | None, str]
        A tuple of (request_id, remaining_message). If no tag is found,
        request_id is None and remaining_message is the original message.
    """
    if not isinstance(message, str):
        return None, str(message) if message else ""

    match = REQUEST_TAG_PATTERN.match(message)
    if match:
        request_id = int(match.group(1))
        remaining = message[match.end() :]
        return request_id, remaining

    return None, message


def format_request_cancelled_notice(request_id: int, reason: str | None = None) -> str:
    """Format a cancellation notice for the LLM.

    Parameters
    ----------
    request_id : int
        The request ID that was cancelled.
    reason : str | None
        Optional reason for cancellation.

    Returns
    -------
    str
        A formatted notice string for the LLM.
    """
    notice = f"[System] Request {request_id} has been cancelled by the user"
    if reason:
        notice += f": {reason}"
    notice += ". Stop working on this request."
    return notice


def format_request_paused_notice(request_id: int) -> str:
    """Format a pause notice for the LLM.

    Parameters
    ----------
    request_id : int
        The request ID that was paused.

    Returns
    -------
    str
        A formatted notice string for the LLM.
    """
    return f"[System] Request {request_id} has been paused by the user. Deprioritize work on this request until resumed."


def format_request_resumed_notice(request_id: int) -> str:
    """Format a resume notice for the LLM.

    Parameters
    ----------
    request_id : int
        The request ID that was resumed.

    Returns
    -------
    str
        A formatted notice string for the LLM.
    """
    return f"[System] Request {request_id} has been resumed by the user. You may continue work on this request."
