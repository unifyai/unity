"""
Pytest configuration for .parallel_run.sh tests.

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
SCRIPT_PATH = TESTS_DIR / ".parallel_run.sh"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
PYTEST_LOGS_DIR = REPO_ROOT / ".pytest_logs"


@dataclass
class TmuxSession:
    """Represents a tmux session."""

    name: str
    windows: int = 1
    created: str = ""

    @property
    def is_pending(self) -> bool:
        return self.name.startswith("?")

    @property
    def is_passed(self) -> bool:
        return self.name.startswith("o")

    @property
    def is_failed(self) -> bool:
        return self.name.startswith("x")

    @property
    def base_name(self) -> str:
        """Strip the status prefix from the session name."""
        # Remove status prefixes like "? ⏳ ", "o ✅ ", "x ❌ "
        for prefix in ["? ⏳ ", "o ✅ ", "x ❌ ", "⏳ ", "✅ ", "❌ "]:
            if self.name.startswith(prefix):
                return self.name[len(prefix) :]
        return self.name


@dataclass
class RunResult:
    """Result of running .parallel_run.sh."""

    exit_code: int
    stdout: str
    stderr: str
    sessions_created: List[str] = field(default_factory=list)
    log_files: List[Path] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.exit_code == 0


def list_tmux_sessions() -> List[TmuxSession]:
    """List all current tmux sessions."""
    try:
        result = subprocess.run(
            ["tmux", "ls"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        sessions = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            # Format: "session_name: N windows (created ...)"
            match = re.match(r"^(.+?):\s+(\d+)\s+windows?\s+\(created\s+(.+?)\)", line)
            if match:
                sessions.append(
                    TmuxSession(
                        name=match.group(1),
                        windows=int(match.group(2)),
                        created=match.group(3),
                    ),
                )
        return sessions
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def kill_tmux_session(name: str) -> bool:
    """Kill a tmux session by name."""
    try:
        result = subprocess.run(
            ["tmux", "kill-session", "-t", name],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def kill_sessions_matching(pattern: str) -> int:
    """Kill all sessions matching a pattern. Returns count killed."""
    sessions = list_tmux_sessions()
    killed = 0
    for session in sessions:
        if re.search(pattern, session.name) or re.search(pattern, session.base_name):
            if kill_tmux_session(session.name):
                killed += 1
    return killed


def wait_for_sessions_to_complete(
    session_patterns: List[str],
    timeout: float = 60,
    poll_interval: float = 0.5,
) -> List[TmuxSession]:
    """Wait for sessions matching patterns to complete (no longer pending)."""
    start = time.time()
    while time.time() - start < timeout:
        sessions = list_tmux_sessions()
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

    return list_tmux_sessions()


@pytest.fixture
def clean_tmux_sessions():
    """Fixture that cleans up any test-related tmux sessions before and after."""
    # Pattern to match our fixture test sessions
    pattern = r"test_parallel_run|fixtures"

    # Clean before
    kill_sessions_matching(pattern)

    yield

    # Clean after - give sessions time to auto-close
    time.sleep(0.5)
    kill_sessions_matching(pattern)


@pytest.fixture
def clean_pytest_logs():
    """Fixture that tracks and cleans up pytest log files."""
    # Record existing log files
    existing_logs = set()
    if PYTEST_LOGS_DIR.exists():
        existing_logs = set(PYTEST_LOGS_DIR.glob("*.txt"))

    yield

    # Clean up new log files created during test
    if PYTEST_LOGS_DIR.exists():
        new_logs = set(PYTEST_LOGS_DIR.glob("*.txt")) - existing_logs
        for log in new_logs:
            if "fixtures" in log.name or "test_parallel_run" in log.name:
                try:
                    log.unlink()
                except OSError:
                    pass


class ParallelRunner:
    """Helper class to run .parallel_run.sh with various arguments."""

    def __init__(self):
        self.script_path = SCRIPT_PATH
        self.fixtures_dir = FIXTURES_DIR
        self.repo_root = REPO_ROOT
        self._created_sessions: List[str] = []

    def run(
        self,
        *args: str,
        timeout: float = 120,
        wait_for_completion: bool = False,
        completion_timeout: float = 60,
        env: Optional[dict] = None,
    ) -> RunResult:
        """Run .parallel_run.sh with the given arguments.

        Args:
            *args: Arguments to pass to the script
            timeout: Subprocess timeout
            wait_for_completion: If True, wait for sessions to complete even if --wait not passed
            completion_timeout: How long to wait for session completion
            env: Additional environment variables

        Returns:
            RunResult with exit code, output, created sessions, etc.
        """
        # Record existing sessions
        existing_sessions = {s.name for s in list_tmux_sessions()}
        existing_logs = set()
        if PYTEST_LOGS_DIR.exists():
            existing_logs = set(PYTEST_LOGS_DIR.glob("*.txt"))

        # Build command
        cmd = [str(self.script_path)] + list(args)

        # Set up environment
        run_env = os.environ.copy()
        # Ensure we use random projects mode to avoid interfering with the shared UnityTests project
        run_env["UNIFY_TESTS_RAND_PROJ"] = "True"
        run_env["UNIFY_TESTS_DELETE_PROJ_ON_EXIT"] = "True"
        if env:
            run_env.update(env)

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

        # Find new sessions
        time.sleep(0.3)  # Brief pause for sessions to register
        current_sessions = {s.name for s in list_tmux_sessions()}
        new_sessions = list(current_sessions - existing_sessions)
        self._created_sessions.extend(new_sessions)

        # If wait_for_completion requested, wait for sessions
        if wait_for_completion and new_sessions:
            patterns = [re.escape(s) for s in new_sessions]
            wait_for_sessions_to_complete(patterns, timeout=completion_timeout)

        # Find new log files
        new_logs = []
        if PYTEST_LOGS_DIR.exists():
            new_logs = list(set(PYTEST_LOGS_DIR.glob("*.txt")) - existing_logs)

        return RunResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            sessions_created=new_sessions,
            log_files=new_logs,
        )

    def fixture_path(self, *parts: str) -> str:
        """Get the path to a fixture file relative to repo root."""
        path = self.fixtures_dir.joinpath(*parts)
        return str(path.relative_to(self.repo_root))

    def cleanup(self):
        """Kill all sessions created by this runner."""
        for session in self._created_sessions:
            kill_tmux_session(session)
        self._created_sessions.clear()


@pytest.fixture
def runner(clean_tmux_sessions, clean_pytest_logs):
    """Fixture providing a ParallelRunner instance."""
    r = ParallelRunner()
    yield r
    r.cleanup()


@pytest.fixture
def fixtures_dir():
    """Path to the fixtures directory relative to repo root."""
    return str(FIXTURES_DIR.relative_to(REPO_ROOT))
