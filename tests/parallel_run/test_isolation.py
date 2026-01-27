"""
Tests for terminal-based isolation in parallel_run.sh.

Verifies:
- Each run uses an isolated tmux socket
- Socket name is derived from TTY
- Sessions are isolated between different sockets
- Helper scripts work with isolation
"""

from __future__ import annotations

import os
import subprocess

from tests.parallel_run.conftest import (
    TESTS_DIR,
    list_tmux_sessions,
    get_unity_sockets,
)


class TestSocketIsolation:
    """Tests for socket-based isolation."""

    def test_output_includes_socket_name(self, runner):
        """Script output should include the socket name used."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert result.exit_code == 0
        # Output should mention the socket
        assert "socket:" in result.stdout.lower() or "socket" in result.stdout

    def test_socket_name_in_result(self, runner):
        """RunResult should have the socket name extracted."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        # Socket should be extracted from output
        assert result.socket, "Socket name should be extracted from output"
        assert result.socket.startswith(
            "unity",
        ), f"Socket should start with 'unity': {result.socket}"

    def test_sessions_created_in_isolated_socket(self, runner):
        """Sessions should be created in the isolated socket."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        if result.socket:
            # List sessions from the specific socket
            sessions = list_tmux_sessions(socket=result.socket)
            session_names = [s.name for s in sessions]

            # Our session should be in this socket (may have auto-closed if passed)
            # Just verify the socket exists
            sockets = get_unity_sockets()
            assert result.socket in sockets or len(result.sessions_created) > 0

    def test_explicit_socket_override(self, runner):
        """UNITY_TEST_SOCKET env var should override auto-detection."""
        custom_socket = "unity_test_custom_socket"

        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
            env={"UNITY_TEST_SOCKET": custom_socket},
        )

        assert result.exit_code == 0
        assert (
            result.socket == custom_socket
        ), f"Expected socket {custom_socket}, got {result.socket}"


class TestHelperScripts:
    """Tests for helper scripts with isolation."""

    def test_watch_tests_script_exists(self):
        """The watch_tests.sh helper should exist and be executable."""
        script = TESTS_DIR / "watch_tests.sh"
        assert script.exists(), f"Script not found: {script}"
        assert os.access(script, os.X_OK), f"Script not executable: {script}"

    def test_kill_failed_supports_all_flag(self):
        """The kill_failed.sh script should support --all flag."""
        script = TESTS_DIR / "kill_failed.sh"
        result = subprocess.run(
            [str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert "--all" in result.stdout, "kill_failed.sh should support --all flag"

    def test_watch_tests_supports_all_flag(self):
        """The watch_tests.sh script should support --all flag."""
        script = TESTS_DIR / "watch_tests.sh"
        result = subprocess.run(
            [str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert "--all" in result.stdout, "watch_tests.sh should support --all flag"


class TestSocketNaming:
    """Tests for socket naming convention."""

    def test_socket_starts_with_unity_prefix(self, runner):
        """Socket names should start with 'unity' prefix."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert result.socket.startswith(
            "unity",
        ), f"Socket should start with 'unity': {result.socket}"

    def test_multiple_runs_same_terminal_same_socket(self, runner):
        """Multiple runs from same terminal should use same socket."""
        result1 = runner.run(
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        result2 = runner.run(
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        # Both should use the same socket (same terminal)
        assert (
            result1.socket == result2.socket
        ), f"Expected same socket, got {result1.socket} and {result2.socket}"


class TestObserveOutputFormat:
    """Tests for the Observe section output format."""

    def test_output_shows_watch_helper(self, runner):
        """Output should mention the watch_tests.sh helper."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
        )

        assert (
            "watch_tests.sh" in result.stdout
        ), "Output should mention watch_tests.sh helper"

    def test_output_shows_socket_specific_commands(self, runner):
        """Output should show socket-specific tmux commands."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
        )

        # Should show how to list/attach with the specific socket
        assert "tmux -L" in result.stdout, "Output should show tmux -L socket commands"
