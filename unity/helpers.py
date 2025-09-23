from __future__ import annotations
import os
import requests
import signal
import sys
import shutil
import shlex
import subprocess
from pathlib import Path
import time
from typing import Union
import psutil
from unity.constants import PROJECT_ROOT, VENV_DIR


def _find_project_frame(start):
    """Return first frame in our project tree but *not* in the venv dir."""
    frame = start
    while frame is not None:
        p = Path(frame.f_code.co_filename).resolve()

        # True if p is inside PROJECT_ROOT (handles Py 3.8–3.10 gracefully)
        in_project = (
            p.is_relative_to(PROJECT_ROOT)
            if hasattr(p, "is_relative_to")
            else str(p).startswith(str(PROJECT_ROOT))
        )

        # Treat it as external if it lives in the venv folder
        in_venv = VENV_DIR in p.parents

        if in_project and not in_venv:
            return frame  # ← first “real” project frame
        frame = frame.f_back
    return None


def _handle_exceptions(response):
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}") from e


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


def terminate_process(proc: subprocess.Popen) -> None:
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
            proc.wait(timeout=5)
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
