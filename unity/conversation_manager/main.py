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
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from dotenv import load_dotenv

load_dotenv()
import asyncio

from unity.logger import LOGGER
from unity.common.hierarchical_logger import ICONS
from unity.settings import SETTINGS
from unity.session_details import SESSION_DETAILS
from unity.conversation_manager import assistant_jobs
from unity.conversation_manager.comms_manager import CommsManager
from unity.conversation_manager.metrics import container_spinup
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.domains import comms_utils, managers_utils
from unity.conversation_manager.domains.utils import log_task_exc
from unity.conversation_manager.conversation_manager import ConversationManager
from unity.conversation_manager.metrics_push import init_metrics, shutdown_metrics
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

    LOGGER.info(
        f"{ICONS['lifecycle']} "
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
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
    assistant_jobs.log_job_startup = _sync_mock_success
    assistant_jobs.mark_job_done = _sync_mock_success
    managers_utils.log_message = _async_mock_success
    managers_utils.publish_bus_events = _async_mock_success


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
        event_broker: The event broker
        stop_event: Event to signal shutdown
        project_name: Project name for logging

    Returns:
        Configured ConversationManager instance
    """
    return ConversationManager(
        event_broker,
        SETTINGS.conversation.JOB_NAME,
        SESSION_DETAILS.user.id,
        SESSION_DETAILS.assistant.agent_id,
        SESSION_DETAILS.user.first_name,
        SESSION_DETAILS.user.surname,
        SESSION_DETAILS.assistant.first_name,
        SESSION_DETAILS.assistant.surname,
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
        SESSION_DETAILS.assistant.whatsapp_number,
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

    # Initialise OTel metrics export to GCP Managed Prometheus.
    # Only for cloud-deployed containers where the assistant is assigned via
    # StartupEvent.  Pre-specified assistants (local dev / default) have no
    # startup job or assistant-job logging, so metrics are skipped — all
    # metric instruments remain harmless no-ops.
    if SESSION_DETAILS.assistant.agent_id is None:
        init_metrics()

    # Set the process working directory to the local file root so that relative
    # file paths in CodeActActor-generated code (e.g. "Attachments/abc123_report.pdf")
    # resolve against the same root used by LocalFileSystemAdapter.  This must
    # happen after settings/env are loaded but before any concurrent tasks are
    # created, since os.chdir() is process-global.
    from pathlib import Path as _P
    from unity.file_manager.settings import get_local_root

    _local_root = _P(get_local_root())
    _local_root.mkdir(parents=True, exist_ok=True)
    os.chdir(_local_root)

    # Ensure standard workspace directories exist.
    (_local_root / "Attachments").mkdir(exist_ok=True)

    import shutil as _shutil

    # Clear Outputs/ between sessions so generated files don't accumulate.
    _outputs = _local_root / "Outputs"
    if _outputs.exists():
        asyncio.create_task(asyncio.to_thread(_shutil.rmtree, _outputs))
    else:
        _outputs.mkdir(exist_ok=True)

    # Clear Screenshots/ between sessions (ephemeral visual context).
    _screenshots = _local_root / "Screenshots"
    if _screenshots.exists():
        asyncio.create_task(asyncio.to_thread(_shutil.rmtree, _screenshots))
    else:
        (_screenshots / "User").mkdir(parents=True, exist_ok=True)
        (_screenshots / "Assistant").mkdir(parents=True, exist_ok=True)
        (_screenshots / "Webcam").mkdir(parents=True, exist_ok=True)

    # Clean up dangling call processes
    if cleanup_call_processes:
        LOGGER.info(
            f"{ICONS['process_cleanup']} Checking for dangling call processes from previous runs...",
        )
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
    asyncio.create_task(cm.check_inactivity()).add_done_callback(log_task_exc)

    # For local development (non-idle containers), trigger initialization directly.
    # In cloud deployment, initialization is triggered by StartupEvent from CommsManager.
    # But for local dev, the assistant ID is already set from .env, so no StartupEvent arrives.
    # Skip this in test mode - tests initialize managers explicitly with custom actors.
    if SESSION_DETAILS.assistant.agent_id is not None and not should_apply_mocks:
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
        # U1: Record container spin-up time for idle containers
        # (entrypoint.sh → CommsManager start)
        if SESSION_DETAILS.assistant.agent_id is None:
            _container_start_ms = os.environ.get("CONTAINER_START_TIME_MS")
            if _container_start_ms:
                _spinup_s = (time.time() * 1000 - int(_container_start_ms)) / 1000.0
                container_spinup.record(_spinup_s)
                LOGGER.debug(
                    f"{ICONS['metrics']} [metrics] Container spin-up: {_spinup_s:.2f}s",
                )

        comms_manager = CommsManager(event_broker=event_broker)
        asyncio.create_task(comms_manager.start())

    LOGGER.debug(f"{ICONS['lifecycle']} ConversationManager is running...")
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

    LOGGER.debug(f"{ICONS['lifecycle']} Server is Running...")
    await _stop.wait()

    _oom_prevention = os.path.isfile("/tmp/oom_prevention_shutdown")
    if _oom_prevention:
        LOGGER.warning(
            f"{ICONS['lifecycle']} Shutdown triggered by memory pressure (OOM prevention)",
        )
        comms_utils.publish_system_error(
            "The assistant ran out of memory. Please wait a moment and try again.",
            error_type="oom",
        )
        try:
            from unity.conversation_manager.memory_dump import write_oom_memory_dump

            dump_path = write_oom_memory_dump()
            if dump_path:
                LOGGER.warning(
                    f"{ICONS['lifecycle']} OOM memory dump written to {dump_path}",
                )
        except Exception as exc:
            LOGGER.warning(
                f"{ICONS['lifecycle']} OOM memory dump failed: {exc}",
            )

    LOGGER.debug(f"{ICONS['lifecycle']} Cleaning up conversation manager...")
    await _conversation_manager.cleanup()
    LOGGER.debug(f"{ICONS['lifecycle']} Cleanup finished")

    # Flush buffered EventBus writes to the backend before exit.
    from unity.events.event_bus import EVENT_BUS

    if EVENT_BUS:
        LOGGER.info(f"{ICONS['lifecycle']} Final EventBus flush...")
        EVENT_BUS.flush()

    # Shut down the metrics exporter (flushes remaining data internally).
    LOGGER.info(f"{ICONS['lifecycle']} Shutting down metrics...")
    await shutdown_metrics()

    LOGGER.debug(f"{ICONS['lifecycle']} Shutdown finished")

    # Upload pod logs to GCS so they survive pod termination.
    # This runs here (not in entrypoint.sh) so it executes on all exit paths:
    # both SIGTERM from Kubernetes and self-initiated inactivity shutdown.
    _upload_script = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "scripts",
            "upload_pod_logs.py",
        ),
    )
    if os.path.isfile(_upload_script):
        LOGGER.info(f"{ICONS['lifecycle']} Uploading pod logs to GCS...")
        try:
            subprocess.run([sys.executable, _upload_script], timeout=120)
        except Exception as e:
            LOGGER.warning(
                f"{ICONS['lifecycle']} Pod log upload failed (non-fatal): {e}",
            )

    # Final hard exit to ensure the pod is deallocated.
    # sys.exit() can hang if there are non-daemon threads (e.g. from OTel or PubSub)
    # that haven't closed. os._exit() forces the process to terminate immediately.
    if _signal_shutdown and _conversation_manager.assistant_id is None:
        os._exit(42)
    os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
