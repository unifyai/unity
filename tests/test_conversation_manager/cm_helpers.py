"""
tests/test_conversation_manager/cm_helpers.py
=============================================

Shared helper functions for ConversationManager tests.

These helpers are specific to the event-driven testing paradigm used by
ConversationManager tests, where we step through events and filter output
events by type. Other managers (ContactManager, TranscriptManager) use
a different paradigm (direct async tool loop with single return values)
and don't need these helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from tests.test_conversation_manager.cm_test_driver import CMStepDriver, StepResult

T = TypeVar("T")


# =============================================================================
# Event Filtering Helpers
# =============================================================================


def filter_events_by_type(events: list, typ: type[T]) -> list[T]:
    """Filter a list of events by type.

    Args:
        events: List of events (typically StepResult.output_events)
        typ: Event type to filter by (e.g., SMSSent, EmailSent)

    Returns:
        List of events matching the given type.
    """
    return [e for e in events if isinstance(e, typ)]


def get_exactly_one(events: list, typ: type[T]) -> T:
    """Get exactly one event of the given type, asserting there is exactly one.

    Args:
        events: List of events (typically StepResult.output_events)
        typ: Event type to find (e.g., SMSSent, EmailSent)

    Returns:
        The single event of the given type.

    Raises:
        AssertionError: If there is not exactly one event of the given type.
    """
    matches = filter_events_by_type(events, typ)
    assert len(matches) == 1, f"Expected 1 {typ.__name__}, got {len(matches)}"
    return matches[0]


def assert_has_one(events: list, typ: type) -> bool:
    """Assert exactly one event of the given type exists.

    Args:
        events: List of events (typically StepResult.output_events)
        typ: Event type to check for (e.g., SMSSent, EmailSent)

    Returns:
        True if exactly one event exists.

    Raises:
        AssertionError: If there is not exactly one event of the given type.
    """
    matches = filter_events_by_type(events, typ)
    count = len(matches)
    assert count == 1, f"Expected exactly 1 {typ.__name__}, got {count}"
    return True


# =============================================================================
# Efficiency Assertion Helpers
# =============================================================================

# Maximum LLM steps for efficient tool calling
# - Ideal: 2 steps (action + acknowledge concurrent, then wait)
# - Acceptable: 3 steps (action, acknowledge, wait - or action + query + wait)
MAX_EFFICIENT_STEPS = 3


def assert_efficient(result: "StepResult", context: str = "") -> None:
    """Assert that the LLM completed efficiently (≤ MAX_EFFICIENT_STEPS steps).

    The LLM should call tools concurrently in one step, then wait.

    Args:
        result: StepResult from CMStepDriver.step_until_wait()
        context: Optional context string for error messages.

    Raises:
        AssertionError: If the LLM took more than MAX_EFFICIENT_STEPS steps.
    """
    assert result.llm_step_count <= MAX_EFFICIENT_STEPS, (
        f"Expected efficient concurrent tool calling (<= {MAX_EFFICIENT_STEPS} steps), "
        f"but took {result.llm_step_count} steps. "
        f"LLM should call tools concurrently in one step, then wait. {context}"
    )


# =============================================================================
# Task and Steering Helpers
# =============================================================================


def get_active_task_count(cm: "CMStepDriver") -> int:
    """Get the number of active tasks from the ConversationManager.

    Args:
        cm: CMStepDriver instance wrapping the ConversationManager.

    Returns:
        Number of active tasks.
    """
    return len(cm.cm.active_tasks or {})


def has_steering_tool_call(cm: "CMStepDriver", operation_prefix: str) -> bool:
    """Check if any steering tool with the given operation prefix was called.

    Checks all tool calls made during the test (tracked by CMStepDriver),
    not just active tasks, since completed/stopped tasks are removed from
    active_tasks.

    Args:
        cm: CMStepDriver instance wrapping the ConversationManager.
        operation_prefix: Prefix to match (e.g., "pause_", "stop_", "interject_").

    Returns:
        True if a steering tool with the given prefix was called.
    """
    for tool_name in cm.all_tool_calls:
        if tool_name.startswith(operation_prefix):
            return True
    return False


def get_steering_action(result: "StepResult", operation_prefix: str) -> str | None:
    """Check if the LLM called a steering tool with the given operation prefix.

    The action name pattern is: {operation}_{short_name}__{handle_id}

    Args:
        result: StepResult from CMStepDriver.step() or step_until_wait().
        operation_prefix: Prefix to match (e.g., "pause_", "stop_", "interject_").

    Returns:
        The action name if found, None otherwise.
    """
    if hasattr(result, "last_action") and result.last_action:
        action = result.last_action
        if action.startswith(operation_prefix):
            return action
    return None
