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

from unity.session_details import SESSION_DETAILS

LOGGER = logging.getLogger("unity")

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager

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
        from unity.common.hierarchical_logger import ICONS

        LOGGER.debug(f"{ICONS['lifecycle']} ConversationManager is already running")
        return _conversation_manager

    # Import here to avoid circular imports
    from unity.conversation_manager.main import run_conversation_manager

    _conversation_manager = await run_conversation_manager(
        project_name=project_name,
        enable_comms_manager=enable_comms_manager,
        apply_test_mocks=apply_test_mocks,
    )

    return _conversation_manager


async def stop_async(reason: str = "manual_stop") -> None:
    """
    Stop the ConversationManager.

    Args:
        reason: Reason for stopping (for logging)
    """
    global _conversation_manager, _shutdown_reason

    if _conversation_manager is None:
        return

    from unity.common.hierarchical_logger import ICONS

    LOGGER.debug(
        f"{ICONS['lifecycle']} Stopping ConversationManager (reason: {reason})...",
    )

    try:
        # Signal shutdown
        _conversation_manager.stop.set()

        # Clean up
        await _conversation_manager.cleanup()

        LOGGER.debug(f"{ICONS['lifecycle']} ConversationManager stopped")
        _shutdown_reason = reason
    except Exception as e:
        LOGGER.error(f"{ICONS['lifecycle']} Error stopping ConversationManager: {e}")
        _shutdown_reason = f"stop_error: {e}"
    finally:
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
