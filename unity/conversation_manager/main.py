"""
ConversationManager main entry point.

Supports two modes:
1. Subprocess mode (default): Run via `python -m unity.conversation_manager.main`
2. In-process mode: Call `run_conversation_manager()` directly from async code

The mode is determined by how this module is invoked:
- As __main__: Subprocess mode with signal handling and sys.exit
- Via run_conversation_manager(): In-process mode, returns ConversationManager instance
"""

from __future__ import annotations

from datetime import datetime
import os
import signal
import sys
from typing import TYPE_CHECKING

from dotenv import load_dotenv

load_dotenv()
import asyncio

from unity.settings import SETTINGS
from unity.session_details import DEFAULT_ASSISTANT_ID, SESSION_DETAILS
from unity.conversation_manager import debug_logger
from unity.conversation_manager.comms_manager import CommsManager
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.domains import comms_utils, managers_utils
from unity.conversation_manager.domains.event_handlers import EventHandler
from unity.conversation_manager.domains.utils import log_task_exc
from unity.conversation_manager.conversation_manager import ConversationManager
from unity.conversation_manager.events import SummarizeContext
from unity.helpers import cleanup_dangling_call_processes

if TYPE_CHECKING:
    from unity.conversation_manager.event_broker import EventBroker


# Global state for subprocess mode
_stop: asyncio.Event | None = None
_conversation_manager: ConversationManager | None = None
_signal_shutdown: bool = False


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully (subprocess mode only)"""
    global _signal_shutdown

    print(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        + " - [MAIN.PY] Received signal "
        + str(signum)
        + ", shutting down gracefully...",
    )
    _signal_shutdown = True
    if _stop:
        _stop.set()


def _apply_test_mocks(cm: ConversationManager) -> None:
    """Apply test mocks when TEST env var is set."""

    def _sync_mock_success(*args, **kwargs):
        return {"success": True}

    async def _async_mock_success(*args, **kwargs):
        return {"success": True}

    comms_utils.send_sms_message_via_number = _async_mock_success
    comms_utils.send_unify_message = _async_mock_success
    comms_utils.send_email_via_address = _async_mock_success
    comms_utils.start_call = _async_mock_success
    cm.call_manager.start_call = _async_mock_success
    cm.call_manager.start_unify_meet = _async_mock_success
    cm.schedule_proactive_speech = _async_mock_success
    debug_logger.log_job_startup = _sync_mock_success
    debug_logger.mark_job_done = _sync_mock_success
    managers_utils.log_message = _async_mock_success
    managers_utils.publish_bus_events = _async_mock_success
    EventHandler._registry[SummarizeContext] = _async_mock_success


def _populate_session_details_from_env() -> None:
    """Populate SESSION_DETAILS from environment variables."""
    SESSION_DETAILS.populate_from_env()


def create_conversation_manager(
    event_broker: "EventBroker",
    stop_event: asyncio.Event,
    project_name: str = "Assistants",
) -> ConversationManager:
    """
    Create a ConversationManager instance.

    This is the factory function for creating a ConversationManager with
    the current session details. Can be used in both subprocess and
    in-process modes.

    Args:
        event_broker: The event broker (Redis or in-memory)
        stop_event: Event to signal shutdown
        project_name: Project name for logging

    Returns:
        Configured ConversationManager instance
    """
    return ConversationManager(
        event_broker,
        SETTINGS.conversation.JOB_NAME,
        SESSION_DETAILS.user.id,
        SESSION_DETAILS.assistant.id,
        SESSION_DETAILS.user.name,
        SESSION_DETAILS.assistant.name,
        SESSION_DETAILS.assistant.age,
        SESSION_DETAILS.assistant.nationality,
        SESSION_DETAILS.assistant.about,
        SESSION_DETAILS.assistant.number,
        SESSION_DETAILS.assistant.email,
        SESSION_DETAILS.user.number,
        SESSION_DETAILS.user.email,
        SESSION_DETAILS.voice.provider,
        SESSION_DETAILS.voice.id or None,
        SESSION_DETAILS.voice.mode,
        project_name=project_name,
        stop=stop_event,
    )


async def run_conversation_manager(
    *,
    project_name: str = "Assistants",
    event_broker: "EventBroker | None" = None,
    stop_event: asyncio.Event | None = None,
    enable_comms_manager: bool | None = None,
    apply_test_mocks: bool | None = None,
    cleanup_call_processes: bool = True,
) -> ConversationManager:
    """
    Run ConversationManager in-process (async entry point).

    This is the preferred way to run ConversationManager when you want it
    to share the same process as other components. It sets up all the
    background tasks and returns the ConversationManager instance.

    Args:
        project_name: Project name for logging
        event_broker: Optional event broker. If None, uses get_event_broker()
        stop_event: Optional stop event. If None, creates a new one
        enable_comms_manager: Whether to start CommsManager. If None, defaults
            to True unless TEST env var is set
        apply_test_mocks: Whether to apply test mocks. If None, defaults to
            True if TEST env var is set
        cleanup_call_processes: Whether to clean up dangling call processes

    Returns:
        The running ConversationManager instance. Call cm.stop.set() to
        trigger shutdown, or await cm.cleanup() when done.

    Example:
        async def my_app():
            cm = await run_conversation_manager()
            try:
                # Do stuff with cm
                await some_task()
            finally:
                cm.stop.set()
                await cm.cleanup()
    """
    # Populate session details from environment
    _populate_session_details_from_env()

    # Set the process working directory to the local file root so that relative
    # file paths in CodeActActor-generated code (e.g. "Downloads/report.pdf")
    # resolve against the same root used by LocalFileSystemAdapter.  This must
    # happen after settings/env are loaded but before any concurrent tasks are
    # created, since os.chdir() is process-global.
    from pathlib import Path as _P
    from unity.file_manager.settings import get_local_root

    _local_root = _P(get_local_root())
    _local_root.mkdir(parents=True, exist_ok=True)
    os.chdir(_local_root)

    # Clean up dangling call processes
    if cleanup_call_processes:
        print("Checking for dangling call processes from previous runs...")
        cleanup_dangling_call_processes()

    # Create event broker and stop event
    if event_broker is None:
        event_broker = get_event_broker()
    if stop_event is None:
        stop_event = asyncio.Event()

    # Create conversation manager
    cm = create_conversation_manager(event_broker, stop_event, project_name)

    # Apply test mocks if requested
    should_apply_mocks = (
        apply_test_mocks if apply_test_mocks is not None else SETTINGS.TEST
    )
    if should_apply_mocks:
        _apply_test_mocks(cm)

    # Start background tasks
    asyncio.create_task(cm.wait_for_events()).add_done_callback(log_task_exc)
    asyncio.create_task(cm.check_inactivity())

    # For local development (non-idle containers), trigger initialization directly.
    # In cloud deployment, initialization is triggered by StartupEvent from CommsManager.
    # But for local dev, the assistant ID is already set from .env, so no StartupEvent arrives.
    # Skip this in test mode - tests initialize managers explicitly with custom actors.
    if SESSION_DETAILS.assistant.id != DEFAULT_ASSISTANT_ID and not should_apply_mocks:
        # No _startup_sequence in local dev, so unblock the VM readiness gate
        # directly (the VM is assumed reachable if configured via .env).
        from unity.function_manager.primitives.runtime import _vm_ready

        _vm_ready.set()
        asyncio.create_task(managers_utils.init_conv_manager(cm))
        asyncio.create_task(managers_utils.listen_to_operations(cm))

    # Start CommsManager if enabled
    should_enable_comms = (
        enable_comms_manager if enable_comms_manager is not None else not SETTINGS.TEST
    )
    if should_enable_comms:
        comms_manager = CommsManager(event_broker=event_broker)
        asyncio.create_task(comms_manager.start())

    print("ConversationManager is running...")
    return cm


async def main(project_name: str = "Assistants"):
    """
    Main entry point for subprocess mode.

    Sets up signal handlers, runs the ConversationManager, and handles
    graceful shutdown with appropriate exit codes.
    """
    global _conversation_manager, _stop, _signal_shutdown

    # Set up signal handlers (subprocess mode only)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Create stop event
    _stop = asyncio.Event()

    # Run conversation manager
    _conversation_manager = await run_conversation_manager(
        project_name=project_name,
        stop_event=_stop,
    )

    print("Server is Running...")
    await _stop.wait()

    print("Cleaning up conversation manager...")
    await _conversation_manager.cleanup()
    print("Cleanup finished")

    print("Shutdown finished")

    # Exit with special code 42 if:
    # - Shutdown was triggered by external signal (i.e. not inactivity timeout)
    # - AND assistant_id is the default (i.e. it's an idle container)
    # This signals to start.py to exit immediately to trigger restart
    # within the backoff limit
    if _signal_shutdown and _conversation_manager.assistant_id == DEFAULT_ASSISTANT_ID:
        sys.exit(42)


if __name__ == "__main__":
    asyncio.run(main())
