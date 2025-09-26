import os
import subprocess
import sys
import time
import threading
import signal
import socket
from typing import Optional, Dict, Any


# Global state for the service manager
_process: Optional[subprocess.Popen] = None
_start_time: Optional[float] = None
_shutdown_reason: Optional[str] = None
_monitor_thread: Optional[threading.Thread] = None
_monitoring: bool = False


def terminate_process(proc: subprocess.Popen) -> None:
    """
    Terminate a subprocess gracefully, falling back to force kill if needed.
    Handles both Windows and Unix-like systems.

    Args:
        proc: The subprocess.Popen object to terminate
    """
    if proc is None:
        return

    try:
        # Send SIGTERM to the process group
        if sys.platform.startswith("win"):
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

        # Wait for process to terminate
        try:
            proc.wait(timeout=6)
            print("Process terminated gracefully")
        except subprocess.TimeoutExpired:
            # If process doesn't terminate gracefully, force kill
            print("Process did not terminate gracefully, force killing...")
            if sys.platform.startswith("win"):
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
    except Exception as e:
        print(f"Error during process termination: {e}")


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

    Returns:
        bool: True if service started and is ready, False otherwise
    """
    global _process, _start_time, _shutdown_reason

    if _process and _process.poll() is None:
        print("Unity service is already running")
        return True  # Already running

    try:
        # Get environment variables for this assistant (set by Cloud Run)
        assistant_id = os.environ.get("ASSISTANT_ID", "default")

        # Start main.py using subprocess
        print(f"Starting Unity service (main.py) for assistant {assistant_id}")

        cmd = [sys.executable, "unity/conversation_manager_2/main.py"]

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


def stop(reason: str = "manual_stop") -> bool:
    """
    Stop the Unity service and all its child processes.

    Args:
        reason: Reason for stopping the service

    Returns:
        bool: True if service was stopped successfully
    """
    global _process, _shutdown_reason

    _stop_monitoring()  # Stop monitoring first

    if _process:
        try:
            print("Stopping Unity service and all child processes...")
            # Use the terminate_process function which handles process groups properly
            terminate_process(_process)
            print("Unity service and child processes stopped")
            _shutdown_reason = reason
        except Exception as e:
            print(f"Error stopping Unity service: {e}")
            _shutdown_reason = f"stop_error: {e}"

        _process = None
        return True
    return True


def is_running() -> bool:
    """
    Check if the Unity service is currently running.

    Returns:
        bool: True if service is running, False otherwise
    """
    global _process
    return _process and _process.poll() is None


def get_status() -> Dict[str, Any]:
    """
    Get detailed status of the Unity service.

    Returns:
        dict: Status information including running state, uptime, process ID, etc.
    """
    global _process, _start_time, _shutdown_reason

    running = is_running()
    uptime = time.time() - _start_time if _start_time and running else 0

    status = {
        "running": running,
        "uptime_seconds": uptime,
        "process_id": _process.pid if _process else None,
        "assistant_id": os.environ.get("ASSISTANT_ID", "default"),
        "shutdown_reason": _shutdown_reason,
        "inactivity_timeout_minutes": 6,  # Document the timeout setting
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
    Get the current process object (for advanced usage).

    Returns:
        Optional[subprocess.Popen]: The current process object or None
    """
    global _process
    return _process


def cleanup() -> None:
    """
    Clean up the service manager state.
    Useful for testing or when you want to reset the global state.
    """
    global _process, _start_time, _shutdown_reason, _monitor_thread, _monitoring

    if _process:
        stop("cleanup")

    _process = None
    _start_time = None
    _shutdown_reason = None
    _monitor_thread = None
    _monitoring = False
