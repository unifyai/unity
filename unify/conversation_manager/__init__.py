"""
ConversationManager service management.

Provides in-process async operation of ConversationManager:
- Call `start_async()` to run ConversationManager in the current process
- Uses in-memory event broker
- Direct access to ConversationManager instance
- Simple testing with direct monkey-patching

Example:
    async def main():
        cm = await start_async()
        try:
            # Interact with cm directly
            await cm.event_broker.publish("app:comms:test", "hello")
        finally:
            await stop_async()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from unify.session_details import SESSION_DETAILS

LOGGER = logging.getLogger("unify")

if TYPE_CHECKING:
    from unify.conversation_manager.conversation_manager import ConversationManager

# =============================================================================
# Global state
# =============================================================================
_conversation_manager: Optional["ConversationManager"] = None
_shutdown_reason: Optional[str] = None


# =============================================================================
# Public API
# =============================================================================


async def start_async(
    *,
    project_name: str = "Assistants",
    enable_comms_manager: bool | None = None,
    apply_test_mocks: bool | None = None,
) -> "ConversationManager":
    """
    Start ConversationManager in-process (async entry point).

    Runs the ConversationManager in the same process using asyncio,
    with in-memory event passing.

    Args:
        project_name: Project name for logging
        enable_comms_manager: Whether to start CommsManager for external
            communications (GCP PubSub). If None, defaults to True unless
            TEST env is set.
        apply_test_mocks: Whether to apply test mocks. If None, defaults to
            True if TEST env var is set.

    Returns:
        The running ConversationManager instance.

    Example:
        async def test_something():
            cm = await start_async()
            try:
                # Interact with cm directly
                await cm.event_broker.publish("app:comms:test", "hello")
            finally:
                await stop_async()
    """
    global _conversation_manager

    if _conversation_manager is not None:
        from unify.common.hierarchical_logger import ICONS

        LOGGER.debug(f"{ICONS['lifecycle']} ConversationManager is already running")
        return _conversation_manager

    # Import here to avoid circular imports
    from unify.conversation_manager.main import run_conversation_manager

    _conversation_manager = await run_conversation_manager(
        project_name=project_name,
        enable_comms_manager=enable_comms_manager,
        apply_test_mocks=apply_test_mocks,
    )

    return _conversation_manager


async def stop_async(reason: str = "manual_stop") -> None:
    """
    Stop the ConversationManager.

    An explicit stop is a retirement, not an idle timeout: it goes through
    the same ``_request_shutdown`` sequence the inactivity route uses (record
    the reason, log ``session_end``, set ``stop``, close the event broker),
    then runs ``cleanup()`` — which discards in-flight actions rather than
    waiting on them — and flushes buffered EventBus writes. The whole
    sequence completes in seconds so an in-process successor can boot over
    the same durable world immediately.

    Args:
        reason: Reason for stopping (recorded as the shutdown reason)
    """
    global _conversation_manager, _shutdown_reason

    if _conversation_manager is None:
        return

    from unify.common.hierarchical_logger import ICONS

    LOGGER.debug(
        f"{ICONS['lifecycle']} Stopping ConversationManager (reason: {reason})...",
    )

    try:
        if _conversation_manager.shutdown_reason is None:
            await _conversation_manager._request_shutdown(
                reason,
                f"Explicit shutdown requested ({reason})",
            )
        else:
            # An internal exit (idle timeout, drain, …) already ran the
            # retirement sequence; don't overwrite its recorded reason.
            _conversation_manager.stop.set()

        await _conversation_manager.cleanup()

        # Mirror the subprocess exit path: buffered EventBus writes must not
        # die with the session.
        from unify.events.event_bus import EVENT_BUS

        if EVENT_BUS:
            EVENT_BUS.flush()

        LOGGER.debug(f"{ICONS['lifecycle']} ConversationManager stopped")
        _shutdown_reason = reason
    except Exception as e:
        LOGGER.error(f"{ICONS['lifecycle']} Error stopping ConversationManager: {e}")
        _shutdown_reason = f"stop_error: {e}"
    finally:
        # A successor must not inherit the retired session's machinery: the
        # ConversationManager is a registry singleton (handing it back gives
        # the next boot a session whose stop event is already set), and the
        # broker singleton was just closed by the retirement sequence (a next
        # boot that received it would publish into a void).
        from unify.conversation_manager.event_broker import reset_event_broker
        from unify.manager_registry import ManagerRegistry

        ManagerRegistry.deregister_instance(type(_conversation_manager))
        reset_event_broker()
        _conversation_manager = None


def get_conversation_manager() -> Optional["ConversationManager"]:
    """
    Get the ConversationManager instance, if running.

    Returns:
        The ConversationManager instance or None if not running.
    """
    return _conversation_manager


def is_running() -> bool:
    """
    Check if the ConversationManager is currently running.

    Returns:
        bool: True if running, False otherwise
    """
    return _conversation_manager is not None


def get_status() -> Dict[str, Any]:
    """
    Get detailed status of the ConversationManager.

    Returns:
        dict: Status information including running state, assistant ID, etc.
    """
    global _shutdown_reason

    if _conversation_manager is not None:
        return {
            "running": True,
            "assistant_id": SESSION_DETAILS.assistant.agent_id,
            "shutdown_reason": _shutdown_reason,
        }

    return {
        "running": False,
        "assistant_id": SESSION_DETAILS.assistant.agent_id,
        "shutdown_reason": _shutdown_reason,
    }


async def cleanup() -> None:
    """
    Clean up the ConversationManager state.

    Alias for stop_async("cleanup") for convenience.
    """
    await stop_async("cleanup")


def reset() -> None:
    """
    Reset the global state without cleanup.

    Useful for testing when you need to reset state without
    going through the full cleanup process.
    """
    global _conversation_manager, _shutdown_reason
    _conversation_manager = None
    _shutdown_reason = None
