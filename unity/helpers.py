"""
unity/helpers.py
=================

Subprocess and process management utilities.

Provides cross-platform helpers for:
  - Launching Python scripts in new terminals or background processes
  - Graceful/forceful process termination
  - Cleaning up orphaned call processes from crashed sessions

These are low-level utilities used by the Actor and call infrastructure,
not typically called directly by state managers.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Union

import psutil

from unity.logger import LOGGER
from unity.common.hierarchical_logger import DEFAULT_ICON, ICONS


def _find_unix_terminal() -> str | None:
    for term in (
        "gnome-terminal",
        "konsole",
        "xfce4-terminal",
        "xterm",
        "lxterminal",
        "mate-terminal",
        "tilix",
        "alacritty",
        "kitty",
    ):
        if shutil.which(term):
            return term
    return None


def run_script(
    script: Union[str, os.PathLike],
    *script_args: str,
    terminal: bool = False,
) -> subprocess.Popen:
    """
    Launch *script* and return a `subprocess.Popen` representing the actual
    Python process.

    Parameters
    ----------
    script : Path-like
        The .py file to run.
    *script_args : str
        Extra args forwarded to the script.
    terminal : bool, default False
        • False – run invisibly (shares the parent console / no window).
        • True  – open a new terminal window and start Python **-i**.

    Returns
    -------
    subprocess.Popen
        Handle to the Python process (not any wrapper shell).
    """
    script_path = Path(script).expanduser().resolve()
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    # Build the python command
    py_cmd = [sys.executable]
    if terminal:
        py_cmd.append("-i")  # interactive prompt *only* in a terminal
    py_cmd += [str(script_path), *script_args]

    if sys.platform.startswith("win"):
        # ───────────────────────── Windows ─────────────────────────
        if terminal:
            creationflags = (
                subprocess.CREATE_NEW_CONSOLE
                | subprocess.CREATE_NEW_PROCESS_GROUP  # lets us send CTRL_BREAK_EVENT
            )
        else:
            creationflags = 0  # inherit caller’s console
        return subprocess.Popen(py_cmd, creationflags=creationflags)

    elif sys.platform == "darwin":
        # ───────────────────────── macOS ───────────────────────────
        if not terminal:
            return subprocess.Popen(py_cmd, start_new_session=True)

        # Create a unique PID-file so we can discover the real python PID
        process_id = f"{script_path.stem}_{int(time.time())}"
        pid_file = Path(f"/tmp/{process_id}.pid")

        shell = f"""
            echo $$ > {pid_file};
            trap 'rm -f {pid_file}' EXIT;
            {shlex.join(py_cmd)}
        """

        # Use AppleScript to activate Terminal and run the command in a new window/tab
        osa = f"""
            tell application "Terminal"
                activate
                do script "{shell}"
            end tell
        """
        subprocess.run(["osascript", "-e", osa], check=True)

        # Wait (max 5 s) for the child to write its PID
        start = time.time()
        while time.time() - start < 5:
            try:
                pid = int(pid_file.read_text())
                return psutil.Process(pid)
            except (FileNotFoundError, ValueError, psutil.NoSuchProcess):
                time.sleep(0.1)

        raise RuntimeError("Timed out waiting for python process in Terminal")

    else:
        # ───────────────────────── Linux / BSD / WSL ───────────────
        if not terminal:
            return subprocess.Popen(py_cmd, start_new_session=True)

        term = _find_unix_terminal()  # your helper that finds gnome-terminal / xterm …
        if not term:
            raise RuntimeError("No terminal emulator found (gnome-terminal, xterm …)")

        # Start python first so we know its PID
        proc = subprocess.Popen(py_cmd, start_new_session=True)
        # Point the new terminal at *that* interpreter
        subprocess.Popen(
            [
                term,
                "--",
                "bash",
                "-c",
                f"exec {' '.join(map(shlex.quote, py_cmd))}",
            ],
        )
        return proc


def terminate_process(proc: subprocess.Popen, timeout: int = 5) -> None:
    """
    Terminate a subprocess gracefully, falling back to force kill if needed.
    Handles both Windows and Unix-like systems.
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
            proc.wait(timeout=timeout)
            LOGGER.info(f"{DEFAULT_ICON} Process terminated gracefully")
        except subprocess.TimeoutExpired:
            # If process doesn't terminate gracefully, force kill
            LOGGER.warning(
                f"{DEFAULT_ICON} Process did not terminate gracefully, force killing...",
            )
            if sys.platform.startswith("win"):
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
    except Exception as e:
        LOGGER.error(f"{DEFAULT_ICON} Error during process termination: {e}")


def cleanup_dangling_call_processes() -> int:
    """
    Find and force-kill all dangling call processes from previous runs.

    This searches for Python processes running call/unify_meet/sts_call scripts
    and immediately terminates them with SIGKILL. Since these are leftover processes
    from crashed/ungraceful shutdowns, there's no need for graceful termination.

    Returns
    -------
    int
        Number of processes terminated
    """
    if sys.platform.startswith("win"):
        # Windows implementation would use tasklist/taskkill
        LOGGER.warning(
            f"{ICONS['process_cleanup']} cleanup_dangling_call_processes not yet implemented for Windows",
        )
        return 0

    try:
        # Find all Python processes running call scripts
        output = subprocess.getoutput(
            "ps -eo pid,command | grep -E 'medium_scripts/(call|unify_meet|sts_call)\\.py'",
        )

        # Parse PIDs and commands, excluding the grep process itself
        processes = {
            int(line.strip().split(" ")[0]): line.strip().split(" ")[1]
            for line in output.split("\n")
            if line.strip() and "grep" not in line
        }

        if not processes:
            LOGGER.info(f"{ICONS['process_cleanup']} No dangling call processes found")
            return 0

        terminated_count = 0
        for pid, command in processes.items():
            try:
                LOGGER.info(
                    f"{ICONS['process_cleanup']} Force killing dangling call process PID {pid} with command {command}",
                )
                os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
                LOGGER.info(f"{ICONS['process_cleanup']} Killed process {pid}")
            except ProcessLookupError:
                LOGGER.debug(
                    f"{ICONS['process_cleanup']} Process {pid} -> {command} not found",
                )
            except PermissionError:
                LOGGER.error(
                    f"{ICONS['process_cleanup']} Permission denied to kill process {pid} -> {command}",
                )
            except ValueError:
                LOGGER.error(
                    f"{ICONS['process_cleanup']} Invalid PID: {pid} -> {command}",
                )
            except Exception as e:
                LOGGER.error(
                    f"{ICONS['process_cleanup']} Error terminating process {pid} -> {command}: {e}",
                )
                continue

        LOGGER.info(
            f"{ICONS['process_cleanup']} Terminated {terminated_count} dangling call process(es)",
        )
        return terminated_count

    except Exception as e:
        LOGGER.error(
            f"{ICONS['process_cleanup']} Error during dangling process cleanup: {e}",
        )
        return 0
