"""
ConversationManager service management.

Supports two modes of operation:

1. **Subprocess mode** (default for production):
   - Call `start()` to launch ConversationManager in a subprocess
   - Uses Redis for inter-process communication
   - Full process isolation for voice/call handling

2. **In-process mode** (for testing and local development):
   - Call `start_async()` to run ConversationManager in the current process
   - Uses in-memory event broker (no Redis required)
   - Direct access to ConversationManager instance
   - Simpler testing with direct monkey-patching

The mode is selected automatically based on UNITY_EVENT_BROKER setting:
- "redis" (default): Subprocess mode
- "in_memory": In-process mode
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from unity.session_details import SESSION_DETAILS

if TYPE_CHECKING:
    from unity.conversation_manager.conversation_manager import ConversationManager


# =============================================================================
# Global state for subprocess mode
# =============================================================================
_process: Optional[subprocess.Popen] = None
_start_time: Optional[float] = None
_shutdown_reason: Optional[str] = None
_monitor_thread: Optional[threading.Thread] = None
_monitoring: bool = False

# =============================================================================
# Global state for in-process mode
# =============================================================================
_in_process_cm: Optional["ConversationManager"] = None
_in_process_task: Optional[asyncio.Task] = None


# =============================================================================
# Subprocess mode functions
# =============================================================================


def terminate_process(proc: subprocess.Popen) -> Optional[int]:
    """
    Terminate a subprocess gracefully, falling back to force kill if needed.
    Handles both Windows and Unix-like systems.

    Args:
        proc: The subprocess.Popen object to terminate

    Returns:
        The exit code of the process (0 for normal exit, 42 for immediate exit signal),
        or None if no process was provided
    """
    if proc is None:
        return None

    try:
        # Send SIGTERM to the process group
        if sys.platform.startswith("win"):
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

        # Wait for process to terminate
        try:
            exit_code = proc.wait(timeout=60)
            print(f"Process terminated gracefully with exit code {exit_code}")
            return exit_code
        except subprocess.TimeoutExpired:
            # If process doesn't terminate gracefully, force kill
            print("Process did not terminate gracefully, force killing...")
            if sys.platform.startswith("win"):
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            exit_code = proc.wait()
            return exit_code
    except Exception as e:
        print(f"Error during process termination: {e}")
        return None


def _start_monitoring() -> None:
    """Start background monitoring of the Unity process"""
    global _monitoring, _monitor_thread

    if not _monitoring:
        _monitoring = True
        _monitor_thread = threading.Thread(
            target=_monitor_process,
            daemon=True,
        )
        _monitor_thread.start()


def _stop_monitoring() -> None:
    """Stop background monitoring"""
    global _monitoring
    _monitoring = False


def _monitor_process() -> None:
    """Background thread to monitor process health"""
    global _monitoring, _process, _shutdown_reason

    while _monitoring and _process:
        try:
            # Check if process is still running
            if _process.poll() is not None:
                # Process has exited
                exit_code = _process.poll()
                if exit_code == 0 and not _shutdown_reason:
                    # Clean exit without explicit reason - likely inactivity timeout
                    _shutdown_reason = "inactivity_timeout"
                    print(
                        "Unity service exited cleanly - likely due to inactivity timeout",
                    )
                elif exit_code != 0 and not _shutdown_reason:
                    _shutdown_reason = f"process_crashed (exit_code: {exit_code})"
                    print(f"Unity service crashed with exit code: {exit_code}")

                _monitoring = False
                break

            # Check every 10 seconds
            time.sleep(10)

        except Exception as e:
            print(f"Error in process monitoring: {e}")
            _monitoring = False
            break


def wait_for_service_ready(timeout: int = 30) -> bool:
    """Wait for the service to be ready by checking the event manager server"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Try to connect to the event manager server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(("127.0.0.1", 8090))
            sock.close()
            if result == 0:
                print("Unity service is ready and accepting connections")
                return True
        except Exception as e:
            print(f"Waiting for service to be ready... ({e})")
        time.sleep(1)

    print(f"Unity service failed to become ready within {timeout} seconds")
    return False


def start(
    start_local: bool = False,
    enabled_tools: list | str | None = "conductor",
    project_name: str = "Assistants",
) -> bool:
    """
    Start the Unity service as a subprocess and wait for it to be ready.

    This is the subprocess mode entry point, used for production deployments
    where ConversationManager runs in isolation with Redis communication.

    Returns:
        bool: True if service started and is ready, False otherwise
    """
    global _process, _start_time, _shutdown_reason

    if _process and _process.poll() is None:
        print("Unity service is already running")
        return True  # Already running

    try:
        # Start main.py using subprocess
        print(
            f"Starting Unity service (main.py) for assistant {SESSION_DETAILS.assistant.id}",
        )

        cmd = [sys.executable, "unity/conversation_manager/main.py"]
        cmd.append("--project-name")
        cmd.append(project_name)

        _process = subprocess.Popen(
            cmd,
            start_new_session=True,
        )

        _start_time = time.time()

        # Give it a moment to start
        time.sleep(2)

        # Check if process is still running (didn't crash immediately)
        if _process.poll() is None:
            print("Unity service started successfully")
            _shutdown_reason = None  # Clear any previous shutdown reason
            _start_monitoring()
            return True
        else:
            print("Unity service failed to start (process exited)")
            _shutdown_reason = "startup_failure"
            return False

    except Exception as e:
        print(f"Failed to start Unity service: {e}")
        return False


def stop(reason: str = "manual_stop") -> Optional[int]:
    """
    Stop the Unity service and all its child processes.

    Args:
        reason: Reason for stopping the service

    Returns:
        The exit code of the terminated process, or None if no process was running
    """
    global _process, _shutdown_reason

    _stop_monitoring()  # Stop monitoring first

    exit_code: Optional[int] = None
    if _process:
        try:
            print("Stopping Unity service and all child processes...")
            # Use the terminate_process function which handles process groups properly
            exit_code = terminate_process(_process)
            print("Unity service and child processes stopped")
            _shutdown_reason = reason
        except Exception as e:
            print(f"Error stopping Unity service: {e}")
            _shutdown_reason = f"stop_error: {e}"

        _process = None
        return exit_code
    return None


# =============================================================================
# In-process mode functions
# =============================================================================


async def start_async(
    *,
    project_name: str = "Assistants",
    enable_comms_manager: bool | None = None,
    apply_test_mocks: bool | None = None,
) -> "ConversationManager":
    """
    Start ConversationManager in-process (async entry point).

    This is the in-process mode entry point, ideal for testing and local
    development. The ConversationManager runs in the same process using
    asyncio, with in-memory event passing (no Redis required).

    Args:
        project_name: Project name for logging
        enable_comms_manager: Whether to start CommsManager for external
            communications. If None, defaults to True unless TEST env is set.
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
    global _in_process_cm

    if _in_process_cm is not None:
        print("ConversationManager is already running in-process")
        return _in_process_cm

    # Import here to avoid circular imports
    from unity.conversation_manager.main import run_conversation_manager

    _in_process_cm = await run_conversation_manager(
        project_name=project_name,
        enable_comms_manager=enable_comms_manager,
        apply_test_mocks=apply_test_mocks,
    )

    return _in_process_cm


async def stop_async(reason: str = "manual_stop") -> None:
    """
    Stop the in-process ConversationManager.

    Args:
        reason: Reason for stopping (for logging)
    """
    global _in_process_cm, _in_process_task, _shutdown_reason

    if _in_process_cm is None:
        return

    print(f"Stopping ConversationManager in-process (reason: {reason})...")

    try:
        # Signal shutdown
        _in_process_cm.stop.set()

        # Clean up
        await _in_process_cm.cleanup()
        print("ConversationManager stopped")
        _shutdown_reason = reason
    except Exception as e:
        print(f"Error stopping ConversationManager: {e}")
        _shutdown_reason = f"stop_error: {e}"
    finally:
        _in_process_cm = None


def get_in_process_cm() -> Optional["ConversationManager"]:
    """
    Get the in-process ConversationManager instance, if running.

    Returns:
        The ConversationManager instance or None if not running in-process.
    """
    return _in_process_cm


# =============================================================================
# Common functions (work for both modes)
# =============================================================================


def is_running() -> bool:
    """
    Check if the Unity service is currently running (either mode).

    Returns:
        bool: True if service is running, False otherwise
    """
    # Check in-process mode first
    if _in_process_cm is not None:
        return True

    # Check subprocess mode
    return _process is not None and _process.poll() is None


def get_status() -> Dict[str, Any]:
    """
    Get detailed status of the Unity service.

    Returns:
        dict: Status information including running state, uptime, process ID, etc.
    """
    global _process, _start_time, _shutdown_reason

    # Check in-process mode
    if _in_process_cm is not None:
        return {
            "running": True,
            "mode": "in_process",
            "assistant_id": SESSION_DETAILS.assistant.id,
            "shutdown_reason": _shutdown_reason,
            "message": "Running in-process (no subprocess)",
        }

    # Subprocess mode status
    running = _process is not None and _process.poll() is None
    uptime = time.time() - _start_time if _start_time and running else 0

    status = {
        "running": running,
        "mode": "subprocess",
        "uptime_seconds": uptime,
        "process_id": _process.pid if _process else None,
        "assistant_id": SESSION_DETAILS.assistant.id,
        "shutdown_reason": _shutdown_reason,
        "inactivity_timeout_minutes": 6,
    }

    # Add additional context based on shutdown reason
    if _shutdown_reason == "inactivity_timeout":
        status["message"] = "Service shut down due to 6 minutes of inactivity"
    elif _shutdown_reason == "manual_stop":
        status["message"] = "Service stopped manually via API"
    elif _shutdown_reason and "process_crashed" in _shutdown_reason:
        status["message"] = "Service process crashed unexpectedly"

    return status


def get_process() -> Optional[subprocess.Popen]:
    """
    Get the current subprocess object (for advanced usage).

    Returns:
        Optional[subprocess.Popen]: The current process object or None
    """
    return _process


def cleanup() -> None:
    """
    Clean up the service manager state (subprocess mode only).
    Useful for testing or when you want to reset the global state.

    For in-process mode, use `await stop_async()` instead.
    """
    global _process, _start_time, _shutdown_reason, _monitor_thread, _monitoring

    if _process:
        stop("cleanup")

    _process = None
    _start_time = None
    _shutdown_reason = None
    _monitor_thread = None
    _monitoring = False
