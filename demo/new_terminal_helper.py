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
import os, sys, shutil, shlex, subprocess, signal
from pathlib import Path
import time
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


def run_in_new_terminal(
    script: str | os.PathLike,
    *script_args: str,
) -> subprocess.Popen:
    """
    Launch *script* in a brand-new terminal/console window and **return**
    the `subprocess.Popen` instance representing the *Python process*
    itself (not a transient wrapper).

    Call `proc.terminate()` / `proc.kill()` / `proc.send_signal(sig)` later
    to close it from the parent process.
    """
    script_path = Path(script).expanduser().resolve()
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    py_cmd = [sys.executable, str(script_path), *script_args]

    if sys.platform.startswith("win"):
        # ─ Windows: run the script in a *new* console and create its own process-group
        creationflags = (
            subprocess.CREATE_NEW_CONSOLE
            | subprocess.CREATE_NEW_PROCESS_GROUP  # lets us send CTRL_BREAK_EVENT
        )
        return subprocess.Popen(
            py_cmd,
            creationflags=creationflags,
        )

    elif sys.platform == "darwin":
        # shell line run inside Terminal
        shell = f"echo $$ > /tmp/{script_path.stem}.pid; exec {shlex.join(py_cmd)}"

        osa = f"""
            tell application "Terminal"
                activate
                do script "{shell}"
            end tell
        """

        subprocess.Popen(["osascript", "-e", osa])

        # wait a moment for the PID file to appear
        pid_file = Path(f"/tmp/{script_path.stem}.pid")
        for _ in range(50):
            if pid_file.exists():
                pid = int(pid_file.read_text())
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("couldn’t obtain PID from Terminal")

        return psutil.Process(pid)

    else:  # Linux / BSD / WSL
        term = _find_unix_terminal()
        if not term:
            raise RuntimeError("No terminal emulator found (gnome-terminal, xterm …)")
        # Start python first so we know its PID, then point the terminal at it
        proc = subprocess.Popen(py_cmd, start_new_session=True)
        subprocess.Popen(
            [term, "--", "bash", "-c", f"exec {' '.join(map(shlex.quote, py_cmd))}"],
        )
        return proc


# DEMO ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_in_new_terminal.py <your_script.py> [args …]")
        sys.exit(1)

    child = run_in_new_terminal(sys.argv[1], *sys.argv[2:])
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
