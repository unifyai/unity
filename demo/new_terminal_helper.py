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
from threading import Thread


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

    py_cmd = [sys.executable, "-i", str(script_path), *script_args]

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
        # Create a unique identifier for this process
        process_id = f"{script_path.stem}_{int(time.time())}"
        pid_file = Path(f"/tmp/{process_id}.pid")

        # Clean up any existing terminal windows with call.py
        try:
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "Terminal" to close (every window whose name contains "call.py")',
                ]
            )
        except Exception as e:
            print(f"Warning: Could not clean up existing terminal windows: {e}")

        # shell line run inside Terminal with better error handling
        shell = f"""
            echo $$ > {pid_file};
            trap 'rm -f {pid_file}' EXIT;
            {shlex.join(py_cmd)}
        """

        # Use osascript to create the terminal and run the command
        osa = f"""
            tell application "Terminal"
                do script "{shell}" in selected tab of front window
            end tell
        """

        # Run osascript and wait for it to complete
        subprocess.run(["osascript", "-e", osa], check=True)

        # Wait for the PID file with better timeout and error handling
        start_time = time.time()
        pid = None
        while time.time() - start_time < 5:  # 5 second timeout
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    # Verify the process is actually running
                    if psutil.pid_exists(pid):
                        # Wait a bit to ensure the process is fully started
                        time.sleep(0.5)
                        if psutil.pid_exists(pid):  # Double check it's still running
                            return psutil.Process(pid)
                except (ValueError, psutil.NoSuchProcess):
                    pass
            time.sleep(0.1)

        if pid is None:
            raise RuntimeError("Failed to start process in new terminal window")
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


def run_as_subprocess(
    script: str | os.PathLike,
    *script_args: str,
) -> subprocess.Popen:
    """
    Launch *script* as a background subprocess and return the Popen instance.
    Output is both shown in the terminal and logged to files.
    """
    script_path = Path(script).expanduser().resolve()
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    py_cmd = [sys.executable, str(script_path), *script_args]

    # Create log files
    out_log = open(f"{script_path.stem}.out.log", "a")
    err_log = open(f"{script_path.stem}.err.log", "a")

    # Start background tasks to handle output
    def handle_output(pipe, log_file):
        for line in pipe:
            print(line, end="")  # Show in terminal
            log_file.write(line)  # Write to log file
            log_file.flush()

    # Create a new process group for better process management
    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            py_cmd,
            creationflags=creationflags,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    # On Unix-like systems, create a new process group
    else:
        proc = subprocess.Popen(
            py_cmd,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    # Start output handling threads
    Thread(target=handle_output, args=(proc.stdout, out_log), daemon=True).start()
    Thread(target=handle_output, args=(proc.stderr, err_log), daemon=True).start()

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
