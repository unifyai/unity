"""
run_in_new_terminal.py  –  launches a script in its *own* window and
returns a handle so you can later stop it.

Usage
-----
proc = run_in_new_terminal("my_script.py", "arg1", "arg2")
# ... do stuff ...
proc.terminate()       # send polite SIGTERM / CTRL_BREAK_EVENT
# or
proc.kill()            # force-kill
"""

from __future__ import annotations
import os
import signal
import sys
import shutil
import shlex
import subprocess
from pathlib import Path
import time
from typing import Union
import psutil


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
            return subprocess.Popen(py_cmd)

        # Create a unique PID-file so we can discover the real python PID
        process_id = f"{script_path.stem}_{int(time.time())}"
        pid_file = Path(f"/tmp/{process_id}.pid")

        shell = f"""
            echo $$ > {pid_file};
            trap 'rm -f {pid_file}' EXIT;
            {shlex.join(py_cmd)}
        """

        osa = f"""
            tell application "Terminal"
                do script "{shell}" in selected tab of front window
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
            # Headless: run in its own session so we can kill the whole group later
            return subprocess.Popen(py_cmd, start_new_session=True)

        term = _find_unix_terminal()           # gnome-terminal, xterm, kitty, …
        if not term:
            raise RuntimeError(
                "No terminal emulator found (gnome-terminal, xterm, kitty, …)"
            )

        # The *only* process we launch is the terminal.  Inside it we `exec` Python,
        # so the interpreter replaces the shell and becomes the leader of the new
        # process-group created by start_new_session=True.
        #
        # ──> `proc` is therefore in the *same* PGID as the interpreter and any
        #     children (e.g. the LiveKit binary), so os.killpg(proc.pid, …) works.
        full_cmd = shlex.join(py_cmd)

        proc = subprocess.Popen(
            [term, "--", "bash", "-c", f"exec {full_cmd}"],
            start_new_session=True,            # give the group its own PGID
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


# DEMO ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_in_new_terminal.py <your_script.py> [args …]")
        sys.exit(1)

    child = run_script(sys.argv[1], *sys.argv[2:])
    print(f"Started {child.pid=}.  Press Enter to stop it.")
    input()
    # Try a graceful shutdown first
    if sys.platform.startswith("win"):
        # Windows: send CTRL+BREAK to the whole group
        child.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        child.terminate()

    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("Graceful exit failed; killing...")
        child.kill()
