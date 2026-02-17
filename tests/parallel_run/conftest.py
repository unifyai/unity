"""
Pytest configuration for parallel_run.sh tests.

Provides fixtures for:
- Managing tmux sessions
- Cleaning up test artifacts
- Running the parallel script
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pytest

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
TESTS_DIR = REPO_ROOT / "tests"
SCRIPT_PATH = TESTS_DIR / "parallel_run.sh"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
PYTEST_LOGS_DIR = REPO_ROOT / "logs" / "pytest"


def get_unity_sockets() -> List[str]:
    """Get all unity* tmux sockets for the current user."""
    socket_dir = Path(f"/tmp/tmux-{os.getuid()}")
    if not socket_dir.exists():
        return []
    return [s.name for s in socket_dir.glob("unity*")]


@dataclass
class TmuxSession:
    """Represents a tmux session."""

    name: str
    windows: int = 1
    created: str = ""
    socket: str = ""  # The tmux socket this session belongs to

    @property
    def is_pending(self) -> bool:
        return self.name.startswith("r")

    @property
    def is_passed(self) -> bool:
        return self.name.startswith("p")

    @property
    def is_failed(self) -> bool:
        return self.name.startswith("f")

    @property
    def base_name(self) -> str:
        """Strip the status prefix from the session name."""
        for prefix in ["p ✅ ", "f ❌ ", "r ⏳ "]:
            if self.name.startswith(prefix):
                return self.name[len(prefix) :]
        return self.name


@dataclass
class RunResult:
    """Result of running parallel_run.sh."""

    exit_code: int
    stdout: str
    stderr: str
    sessions_created: List[str] = field(default_factory=list)
    log_files: List[Path] = field(default_factory=list)
    socket: str = ""  # The tmux socket used for this run

    @property
    def success(self) -> bool:
        return self.exit_code == 0


def list_tmux_sessions(socket: Optional[str] = None) -> List[TmuxSession]:
    """List tmux sessions.

    Args:
        socket: If provided, list sessions from this specific socket.
                If None, list from all unity* sockets.
    """
    if socket:
        sockets = [socket]
    else:
        sockets = get_unity_sockets()
        if not sockets:
            return []

    sessions = []
    for sock in sockets:
        try:
            result = subprocess.run(
                ["tmux", "-L", sock, "ls"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                continue

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                # Format: "session_name: N windows (created ...)"
                match = re.match(
                    r"^(.+?):\s+(\d+)\s+windows?\s+\(created\s+(.+?)\)",
                    line,
                )
                if match:
                    sessions.append(
                        TmuxSession(
                            name=match.group(1),
                            windows=int(match.group(2)),
                            created=match.group(3),
                            socket=sock,
                        ),
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    return sessions


def kill_tmux_session(name: str, socket: Optional[str] = None) -> bool:
    """Kill a tmux session by name.

    Args:
        name: Session name to kill.
        socket: If provided, use this specific socket.
                If None, search all unity* sockets for the session.
    """
    if socket:
        sockets = [socket]
    else:
        sockets = get_unity_sockets()

    for sock in sockets:
        try:
            result = subprocess.run(
                ["tmux", "-L", sock, "kill-session", "-t", name],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    return False


def kill_sessions_matching(pattern: str, socket: Optional[str] = None) -> int:
    """Kill all sessions matching a pattern. Returns count killed."""
    sessions = list_tmux_sessions(socket=socket)
    killed = 0
    for session in sessions:
        if re.search(pattern, session.name) or re.search(pattern, session.base_name):
            if kill_tmux_session(session.name, socket=session.socket):
                killed += 1
    return killed


def wait_for_sessions_to_complete(
    session_patterns: List[str],
    timeout: float = 300,
    poll_interval: float = 0.5,
    socket: Optional[str] = None,
) -> List[TmuxSession]:
    """Wait for sessions matching patterns to complete (no longer pending)."""
    start = time.time()
    while time.time() - start < timeout:
        sessions = list_tmux_sessions(socket=socket)
        matching = []
        all_done = True

        for session in sessions:
            for pattern in session_patterns:
                if re.search(pattern, session.name) or re.search(
                    pattern,
                    session.base_name,
                ):
                    matching.append(session)
                    if session.is_pending:
                        all_done = False
                    break

        if all_done and matching:
            return matching

        time.sleep(poll_interval)

    return list_tmux_sessions(socket=socket)


def wait_for_sessions_adaptive(
    session_base_names: List[str],
    socket: str,
    no_progress_timeout: float = 300,
    poll_interval: float = 0.5,
) -> tuple[List[TmuxSession], bool]:
    """Wait for sessions with adaptive timeout based on progress.

    Progress is counted whenever a session transitions away from pending,
    including disappearing after a successful auto-close. When no progress
    is observed for ``no_progress_timeout`` seconds, the wait ends.

    Returns: (final matching sessions, all_completed_successfully)
    """
    last_progress_time = time.time()
    last_completed_count = 0
    seen_sessions = False  # Track if we've ever seen the sessions
    polls_without_seeing = (
        0  # Count consecutive polls where we don't see expected sessions
    )
    base_names_set = set(session_base_names)

    def _completion_state(
        pending: list[TmuxSession],
        matching: list[TmuxSession],
        seen: bool,
        polls_without: int,
    ) -> tuple[bool, bool]:
        """Return (done, success) without mutating outer scope."""
        if pending:
            return False, False
        if matching:
            return True, not any(s.is_failed for s in matching)
        if seen:
            return True, True
        if polls_without >= 3:
            # Sessions were created and auto-closed before we observed them
            return True, True
        return False, False

    while True:
        sessions = list_tmux_sessions(socket=socket)

        # Find sessions matching our base names
        matching = [s for s in sessions if s.base_name in base_names_set]
        pending = [s for s in matching if s.is_pending]
        completed = [s for s in matching if not s.is_pending]

        # Track if we've seen any of our sessions
        if matching:
            seen_sessions = True
            polls_without_seeing = 0
        else:
            polls_without_seeing += 1

        done, success = _completion_state(
            pending,
            matching,
            seen_sessions,
            polls_without_seeing,
        )
        if done:
            return matching, success

        # Check for progress (more sessions completed/gone than before)
        current_done_count = len(completed) + (
            len(base_names_set) - len(matching) if seen_sessions else 0
        )
        if current_done_count > last_completed_count:
            last_progress_time = time.time()
            last_completed_count = current_done_count

        # Check for timeout (no progress for too long)
        if time.time() - last_progress_time > no_progress_timeout:
            return matching, False

        time.sleep(poll_interval)


def _kill_tmux_server(socket: str) -> None:
    """Kill a tmux server and remove its socket file (best-effort)."""
    try:
        subprocess.run(
            ["tmux", "-L", socket, "kill-server"],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass
    try:
        sock_path = Path(f"/tmp/tmux-{os.getuid()}") / socket
        sock_path.unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
def clean_tmux_sessions():
    """Fixture that provides socket-scoped cleanup for parallel test isolation.

    Each ParallelRunner instance has its own unique socket (based on PID), so
    cleanup is handled by ParallelRunner.cleanup() which only kills sessions
    from its own socket. This fixture is now a no-op but kept for API compatibility.

    Previously this fixture killed sessions across ALL sockets, which caused
    cross-test interference when running in parallel with -t.
    """
    # No-op: cleanup is handled by ParallelRunner.cleanup() which is socket-scoped
    yield


class ParallelRunner:
    """Helper class to run parallel_run.sh with various arguments."""

    def __init__(self):
        self.script_path = SCRIPT_PATH
        self.fixtures_dir = FIXTURES_DIR
        self.repo_root = REPO_ROOT
        self._created_sessions: List[tuple[str, str]] = []  # (socket, session_name)
        # Generate a unique socket name for this runner instance so all runs
        # within the same test use the same socket (enables collision detection)
        self._socket_name = f"unity_test_{os.getpid()}"

    def run(
        self,
        *args: str,
        timeout: float = 600,
        wait_for_completion: bool = False,
        completion_timeout: float = 300,
        env: Optional[dict] = None,
    ) -> RunResult:
        """Run parallel_run.sh with the given arguments.

        Args:
            *args: Arguments to pass to the script
            timeout: Subprocess timeout (default 600s to handle stress test scenarios)
            wait_for_completion: If True, use adaptive wait for sessions (script blocks by default)
            completion_timeout: No-progress timeout for session completion (default 300s)
            env: Additional environment variables

        Returns:
            RunResult with exit code, output, created sessions, etc.
        """
        # Build command
        cmd = [str(self.script_path)] + list(args)

        # Set up environment
        run_env = os.environ.copy()
        # Ensure UTF-8 locale for proper emoji handling in tmux session names
        run_env["LC_ALL"] = "en_US.UTF-8"
        run_env["LANG"] = "en_US.UTF-8"
        # Clear UNITY_LOG_SUBDIR so the nested script derives its own datetime-prefixed subdir
        # (otherwise it inherits the outer parallel_run.sh's log subdir)
        run_env.pop("UNITY_LOG_SUBDIR", None)
        # Use a consistent socket name for all runs within this runner instance
        # This enables collision detection between sequential runs in the same test
        run_env["UNITY_TEST_SOCKET"] = self._socket_name
        # Skip the heavyweight shared project preparation for nested parallel_run.sh calls.
        # The fixture tests don't need the real UnityTests project, and the outer test runner
        # has already prepared it. This dramatically speeds up nested invocations.
        run_env["UNITY_SKIP_SHARED_PROJECT_PREP"] = "1"
        if env:
            run_env.update(env)

        # Determine the actual socket name (user override takes precedence)
        actual_socket = run_env.get("UNITY_TEST_SOCKET", self._socket_name)

        # Record existing sessions (log subdir will be parsed from script output)
        existing_sessions = {(s.socket, s.name) for s in list_tmux_sessions()}

        # Run the script
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.repo_root),
                env=run_env,
            )
            exit_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired as e:
            exit_code = -1
            stdout = e.stdout.decode() if e.stdout else ""
            stderr = e.stderr.decode() if e.stderr else ""

        # Use the actual socket name (respects user overrides via env parameter)
        socket_name = actual_socket

        # Find new sessions - filter by our specific socket to avoid cross-test interference
        time.sleep(0.3)  # Brief pause for sessions to register
        current_sessions = {
            (s.socket, s.name) for s in list_tmux_sessions(socket=socket_name)
        }
        filtered_existing = {
            (sock, name) for sock, name in existing_sessions if sock == socket_name
        }
        new_session_tuples = list(current_sessions - filtered_existing)
        new_sessions = [name for _, name in new_session_tuples]
        self._created_sessions.extend(new_session_tuples)

        # If wait_for_completion requested, wait for sessions using adaptive timeout
        if wait_for_completion and new_sessions:
            # Extract base names for adaptive waiting
            base_names = []
            for session_name in new_sessions:
                base = session_name
                for prefix in ["p ✅ ", "f ❌ ", "r ⏳ "]:
                    if session_name.startswith(prefix):
                        base = session_name[len(prefix) :]
                        break
                base_names.append(base)

            wait_for_sessions_adaptive(
                base_names,
                socket=socket_name,
                no_progress_timeout=completion_timeout,
            )

        # Parse log subdir from script output (format: "📁 Test logs for THIS run: logs/pytest/{subdir}/")
        # This is more robust than trying to predict the datetime-prefixed name
        log_subdir = None
        log_subdir_match = re.search(
            r"Test logs for THIS run: logs/pytest/([^/]+)/",
            stdout,
        )
        if log_subdir_match:
            log_subdir = log_subdir_match.group(1)

        # Find new log files in the parsed log directory
        new_logs = []
        if log_subdir:
            logs_dir = PYTEST_LOGS_DIR / log_subdir
            if logs_dir.exists():
                new_logs = list(logs_dir.glob("*.txt"))

        return RunResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            sessions_created=new_sessions,
            log_files=new_logs,
            socket=socket_name,
        )

    def fixture_path(self, *parts: str) -> str:
        """Get the path to a fixture file relative to repo root."""
        path = self.fixtures_dir.joinpath(*parts)
        return str(path.relative_to(self.repo_root))

    def cleanup(self):
        """Kill all sessions created by this runner.

        Session names change during their lifecycle (r ⏳ → p ✅ or f ❌),
        so we match by base_name to find sessions regardless of current status.
        """
        if not self._created_sessions:
            _kill_tmux_server(self._socket_name)
            return

        # Build set of base names we need to clean up
        base_names_to_kill: set[str] = set()
        for _, session_name in self._created_sessions:
            # Extract base name (strip status prefix)
            base = session_name
            for prefix in ["p ✅ ", "f ❌ ", "r ⏳ "]:
                if session_name.startswith(prefix):
                    base = session_name[len(prefix) :]
                    break
            base_names_to_kill.add(base)

        # Find current sessions in our socket and kill any matching base names
        current_sessions = list_tmux_sessions(socket=self._socket_name)
        for session in current_sessions:
            if session.base_name in base_names_to_kill:
                kill_tmux_session(session.name, socket=session.socket)

        self._created_sessions.clear()
        # After sessions are gone, tear down the dedicated tmux server to free ptys
        _kill_tmux_server(self._socket_name)


@pytest.fixture
def runner(clean_tmux_sessions):
    """Fixture providing a ParallelRunner instance."""
    r = ParallelRunner()
    yield r
    r.cleanup()


@pytest.fixture
def fixtures_dir():
    """Path to the fixtures directory relative to repo root."""
    return str(FIXTURES_DIR.relative_to(REPO_ROOT))
