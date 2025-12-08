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
PYTEST_LOGS_DIR = REPO_ROOT / ".pytest_logs"


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
    timeout: float = 60,
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


@pytest.fixture
def clean_tmux_sessions():
    """Fixture that cleans up any test-related tmux sessions before and after."""
    # Pattern to match ONLY the fixture test sessions created by tests, NOT parent sessions.
    # Parent sessions are named like "test_parallel_run-test_isolation" (from the test file path).
    # Fixture sessions are named like "r ⏳ test_parallel_run-fixtures-test_always_pass"
    # (from the fixtures/ directory path: tests/test_parallel_run/fixtures/).
    # We only want to clean up the fixture sessions, not kill our parent session!
    pattern = r"^[pfr]\s*[✅❌⏳]\s*test_parallel_run-fixtures-"

    # Clean before
    kill_sessions_matching(pattern)

    yield

    # Clean after - give sessions time to auto-close
    time.sleep(0.5)
    kill_sessions_matching(pattern)


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
        timeout: float = 120,
        wait_for_completion: bool = False,
        completion_timeout: float = 60,
        env: Optional[dict] = None,
    ) -> RunResult:
        """Run parallel_run.sh with the given arguments.

        Args:
            *args: Arguments to pass to the script
            timeout: Subprocess timeout
            wait_for_completion: If True, wait for sessions to complete even if --wait not passed
            completion_timeout: How long to wait for session completion
            env: Additional environment variables

        Returns:
            RunResult with exit code, output, created sessions, etc.
        """
        # Build command
        cmd = [str(self.script_path)] + list(args)

        # Set up environment
        run_env = os.environ.copy()
        # Ensure we use random projects mode to avoid interfering with the shared UnityTests project
        run_env["UNIFY_TESTS_RAND_PROJ"] = "True"
        run_env["UNIFY_TESTS_DELETE_PROJ_ON_EXIT"] = "True"
        # Clear UNIFY_SKIP_SESSION_SETUP - random projects mode needs full session setup
        # (inherited True from outer parallel_run.sh would conflict with random project creation)
        run_env["UNIFY_SKIP_SESSION_SETUP"] = "False"
        # Use a consistent socket name for all runs within this runner instance
        # This enables collision detection between sequential runs in the same test
        run_env["UNITY_TEST_SOCKET"] = self._socket_name
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

        # If wait_for_completion requested, wait for sessions
        if wait_for_completion and new_sessions:
            patterns = [re.escape(s) for s in new_sessions]
            wait_for_sessions_to_complete(
                patterns,
                timeout=completion_timeout,
                socket=socket_name,
            )

        # Parse log subdir from script output (format: "📁 Test logs for THIS run: .pytest_logs/{subdir}/")
        # This is more robust than trying to predict the datetime-prefixed name
        log_subdir = None
        log_subdir_match = re.search(
            r"Test logs for THIS run: \.pytest_logs/([^/]+)/",
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
        """Kill all sessions created by this runner."""
        for socket, session in self._created_sessions:
            kill_tmux_session(session, socket=socket)
        self._created_sessions.clear()


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
