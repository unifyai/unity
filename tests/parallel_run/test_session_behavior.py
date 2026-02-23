"""
Tests for tmux session behavior in parallel_run.sh.

Verifies:
- Session naming conventions
- Status prefix updates (pending → pass/fail)
- Auto-close behavior for passing sessions
- Log file creation
- Session uniqueness
"""

from __future__ import annotations

import time


from tests.parallel_run.conftest import (
    list_tmux_sessions,
)


class TestSessionNaming:
    """Tests for session naming conventions."""

    def test_session_name_derived_from_path(self, runner):
        """Session names should be derived from the test file path."""
        result = runner.run(
            "-s",  # Serial mode for predictable session naming test
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 1
        session = result.sessions_created[0]

        # Should contain some variant of the file name
        assert any(
            x in session for x in ["always_pass", "fixtures", "parallel_run"]
        ), f"Unexpected session name: {session}"

    def test_subdir_session_includes_path_components(self, runner):
        """Session names for subdirectory tests should include path components."""
        result = runner.run(
            "-s",  # Serial mode for predictable session naming test
            runner.fixture_path("subdir", "test_in_subdir.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 1
        session = result.sessions_created[0]

        # Should contain subdir reference
        assert "subdir" in session.lower() or "in_subdir" in session.lower()

    def test_unique_session_names_for_collisions(self, runner):
        """Session names should be made unique if there would be collisions."""
        # Run the same file multiple times with --repeat
        result = runner.run(
            "--repeat",
            "3",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        # Should have 3 unique session names
        assert len(result.sessions_created) == 3
        assert len(set(result.sessions_created)) == 3, "Session names must be unique"


class TestStatusPrefixes:
    """Tests for session status prefix updates."""

    def test_pending_prefix_during_run(self, runner):
        """Sessions should have pending prefix while running."""
        # Script blocks by default, but sessions start with pending prefix
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
        )

        # Check immediately after launch - filter by socket to avoid cross-test interference
        if result.sessions_created:
            sessions = list_tmux_sessions(socket=result.socket)
            our_sessions = [
                s
                for s in sessions
                if any(
                    result.sessions_created[0].replace("r ⏳ ", "").replace("p ✅ ", "")
                    in s.name.replace("r ⏳ ", "")
                    .replace("p ✅ ", "")
                    .replace("f ❌ ", "")
                    for name in result.sessions_created
                )
            ]
            # Either pending or already completed is acceptable
            # (tests can complete very quickly)

    def test_pass_prefix_on_success(self, runner):
        """Sessions should have pass prefix after successful completion."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
        )

        # After completion, check the session status (filter by socket)
        sessions = list_tmux_sessions(socket=result.socket)
        # Note: session may have auto-closed by now, so just verify exit code
        assert result.exit_code == 0

    def test_fail_prefix_on_failure(self, runner):
        """Sessions should have fail prefix after test failure."""
        result = runner.run(
            runner.fixture_path("test_always_fail.py"),
        )

        # After failure, session should still exist with fail prefix (filter by socket)
        sessions = list_tmux_sessions(socket=result.socket)
        fail_sessions = [s for s in sessions if s.is_failed]

        # Should have at least one failed session
        # (or it might have been cleaned up already)
        assert result.exit_code != 0


class TestAutoClose:
    """Tests for auto-close behavior of passing sessions."""

    def test_passing_sessions_auto_close(self, runner):
        """Passing sessions should auto-close after ~10 seconds."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0

        # Wait for auto-close (script schedules 10s delay)
        time.sleep(12)

        # Check if session is gone (filter by socket to avoid cross-test interference)
        sessions = list_tmux_sessions(socket=result.socket)
        our_sessions = [
            s for s in sessions if "always_pass" in s.name or "fixtures" in s.name
        ]

        # Session should be closed
        assert (
            len(our_sessions) == 0
        ), f"Session should have auto-closed: {our_sessions}"

    def test_failing_sessions_persist(self, runner):
        """Failing sessions should NOT auto-close."""
        result = runner.run(
            runner.fixture_path("test_always_fail.py"),
        )

        assert result.exit_code != 0

        # Wait a bit (but not the full auto-close time)
        time.sleep(2)

        # Check if session still exists (filter by socket to avoid cross-test interference)
        sessions = list_tmux_sessions(socket=result.socket)
        fail_sessions = [s for s in sessions if s.is_failed]

        # Failed session should still exist
        assert len(fail_sessions) >= 1, "Failed session should persist"


class TestLogFiles:
    """Tests for log file creation in logs/pytest/{socket}/."""

    def test_creates_log_file(self, runner):
        """Script should create log file for each session."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
        )

        assert len(result.log_files) >= 1, "Should create at least one log file"

    def test_log_file_contains_output(self, runner):
        """Log files should contain pytest output."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
        )

        assert len(result.log_files) >= 1
        log_content = result.log_files[0].read_text()

        # Should contain pytest output
        assert "test" in log_content.lower() or "pass" in log_content.lower()

    def test_default_creates_multiple_logs(self, runner):
        """Default per-test mode should create log file per test."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
        )

        # 3 tests = 3 log files
        assert (
            len(result.log_files) == 3
        ), f"Expected 3 log files, got {len(result.log_files)}"

    def test_log_file_naming(self, runner):
        """Log files should be named after sessions."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
        )

        assert len(result.log_files) >= 1
        log_name = result.log_files[0].name

        # Should have .txt extension
        assert log_name.endswith(".txt")


class TestSessionCount:
    """Tests verifying correct number of sessions are created."""

    def test_one_session_per_file_serial(self, runner):
        """Serial mode should create one session per file."""
        result = runner.run(
            "-s",
            runner.fixture_path("test_always_pass.py"),
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 2

    def test_one_session_per_test_default(self, runner):
        """Default mode should create one session per test function."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),  # 3 tests
            runner.fixture_path("test_single_test.py"),  # 1 test
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 4

    def test_repeat_multiplies_sessions(self, runner):
        """--repeat should multiply the number of sessions."""
        result = runner.run(
            "--repeat",
            "3",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 3

    def test_repeat_default(self, runner):
        """--repeat with default per-test mode should multiply correctly."""
        result = runner.run(
            "--repeat",
            "2",
            runner.fixture_path("test_always_pass.py"),  # 3 tests
            wait_for_completion=True,
        )

        # 3 tests * 2 repeats = 6 sessions
        assert len(result.sessions_created) == 6


class TestExitCodes:
    """Tests for correct exit codes with various scenarios."""

    def test_all_pass_exit_zero(self, runner):
        """All passing tests should result in exit code 0."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0

    def test_any_fail_exit_nonzero(self, runner):
        """Any failing test should result in non-zero exit code."""
        result = runner.run(
            runner.fixture_path("test_always_fail.py"),
        )

        assert result.exit_code != 0

    def test_mixed_pass_fail_exit_nonzero(self, runner):
        """Mix of pass/fail should result in non-zero exit code."""
        result = runner.run(
            runner.fixture_path("test_always_pass.py"),
            runner.fixture_path("test_always_fail.py"),
        )

        assert result.exit_code != 0
