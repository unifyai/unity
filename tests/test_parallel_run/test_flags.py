"""
Individual flag tests for parallel_run.sh.

Tests each flag in isolation:
- --wait / -w
- --per-test / -t
- --match / -m
- --env / -e
- --eval-only
- --symbolic-only
- --repeat
- --tags
"""

from __future__ import annotations

import time


class TestWaitFlag:
    """Tests for --wait / -w flag."""

    def test_wait_blocks_until_completion(self, runner):
        """--wait should block until all tests complete."""
        start = time.time()
        result = runner.run(
            "--wait",
            runner.fixture_path("test_always_pass.py"),
        )
        elapsed = time.time() - start

        # Should have blocked for some time (tests take at least a moment)
        # but not forever
        assert elapsed > 0.5, "Should have waited for tests"
        assert elapsed < 60, "Should not wait forever"

    def test_wait_returns_zero_on_all_pass(self, runner):
        """--wait should return 0 when all tests pass."""
        result = runner.run(
            "--wait",
            runner.fixture_path("test_always_pass.py"),
        )

        assert (
            result.exit_code == 0
        ), f"Should return 0 on success, got {result.exit_code}. stderr: {result.stderr}"

    def test_wait_returns_nonzero_on_failure(self, runner):
        """--wait should return non-zero when any test fails."""
        result = runner.run(
            "--wait",
            runner.fixture_path("test_always_fail.py"),
        )

        assert (
            result.exit_code != 0
        ), f"Should return non-zero on failure, got {result.exit_code}"

    def test_wait_returns_nonzero_on_mixed_results(self, runner):
        """--wait should return non-zero when some tests fail."""
        result = runner.run(
            "--wait",
            runner.fixture_path("test_mixed_results.py"),
        )

        assert result.exit_code != 0, "Should return non-zero on partial failure"

    def test_wait_creates_log_files(self, runner):
        """--wait should create log files in .pytest_logs/{socket}/."""
        result = runner.run(
            "--wait",
            runner.fixture_path("test_always_pass.py"),
        )

        # Should have created at least one log file
        assert len(result.log_files) >= 1, "Should create log files with --wait"

    def test_short_wait_flag(self, runner):
        """-w should work the same as --wait."""
        result = runner.run(
            "-w",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0, "Short -w flag should work"


class TestWaitWithTimeout:
    """Tests for --wait N timeout functionality."""

    def test_wait_with_timeout_completes_normally(self, runner):
        """--wait N should complete normally if tests finish before timeout."""
        import time

        start = time.time()
        result = runner.run(
            "--wait",
            "120",  # 2 minute timeout - plenty of time
            runner.fixture_path("test_always_pass.py"),
        )
        elapsed = time.time() - start

        # Should complete successfully (not timeout)
        assert result.exit_code == 0, f"Should pass within timeout: {result.stderr}"
        # Should have taken some time but not the full 120s
        assert elapsed < 100, f"Should complete quickly, took {elapsed:.1f}s"

    def test_wait_with_timeout_times_out(self, runner):
        """--wait N should timeout and exit with code 2 if tests don't complete in time."""
        # Use a very short timeout (2 seconds) - tests won't complete in time
        result = runner.run(
            "--wait",
            "2",  # 2 second timeout - too short
            runner.fixture_path("test_always_pass.py"),
        )

        # Should timeout with exit code 2
        assert (
            result.exit_code == 2
        ), f"Should timeout with code 2, got {result.exit_code}"
        # Should mention timeout in output
        assert "timeout" in result.stdout.lower() or "Timeout" in result.stdout

    def test_wait_with_short_flag_and_timeout(self, runner):
        """-w N should work the same as --wait N."""
        result = runner.run(
            "-w",
            "120",
            runner.fixture_path("test_always_pass.py"),
        )

        assert (
            result.exit_code == 0
        ), f"Short -w with timeout should work: {result.stderr}"

    def test_wait_timeout_with_per_test(self, runner):
        """--wait N with -t should work correctly."""
        result = runner.run(
            "-t",
            "--wait",
            "120",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0, f"Should pass: {result.stderr}"


class TestPerTestFlag:
    """Tests for --per-test / -t flag."""

    def test_per_test_creates_session_per_function(self, runner):
        """--per-test should create one session per test function."""
        result = runner.run(
            "-t",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        # test_always_pass.py has 3 test functions
        assert (
            len(result.sessions_created) == 3
        ), f"Expected 3 sessions (one per test), got {len(result.sessions_created)}: {result.sessions_created}"

    def test_per_test_session_names_include_test_name(self, runner):
        """Per-test session names should include the test function name."""
        result = runner.run(
            "-t",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        # Session names should contain test function identifiers
        session_names_joined = " ".join(result.sessions_created)
        assert (
            "test_pass" in session_names_joined
            or "pass" in session_names_joined.lower()
        ), f"Session names should reference test names: {result.sessions_created}"

    def test_per_test_long_flag(self, runner):
        """--per-test should work same as -t."""
        result = runner.run(
            "--per-test",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 3

    def test_per_test_with_directory(self, runner, fixtures_dir):
        """--per-test with directory should create session per test function."""
        # Just test with subdirectory to keep it manageable
        result = runner.run(
            "-t",
            runner.fixture_path("subdir"),
            wait_for_completion=True,
        )

        # subdir/test_in_subdir.py has 2 tests
        assert (
            len(result.sessions_created) == 2
        ), f"Expected 2 sessions, got {len(result.sessions_created)}"

    def test_per_test_with_wait(self, runner):
        """--per-test with --wait should wait for all test sessions."""
        result = runner.run(
            "-t",
            "--wait",
            runner.fixture_path("test_always_pass.py"),
        )

        # Should block and return success
        assert result.exit_code == 0


class TestMatchFlag:
    """Tests for --match / -m flag."""

    def test_match_filters_by_filename(self, runner, fixtures_dir):
        """--match should filter files by name pattern."""
        result = runner.run(
            "--match",
            "*docstring*",
            fixtures_dir,
            wait_for_completion=True,
        )

        # Should only find test_docstring_pattern.py
        assert (
            len(result.sessions_created) == 1
        ), f"Expected 1 matching file, got {len(result.sessions_created)}"

    def test_match_with_wildcard(self, runner, fixtures_dir):
        """--match should support wildcard patterns."""
        result = runner.run(
            "-m",
            "*always*",
            fixtures_dir,
            wait_for_completion=True,
        )

        # Should find test_always_pass.py and test_always_fail.py
        assert (
            len(result.sessions_created) == 2
        ), f"Expected 2 matching files, got {len(result.sessions_created)}"

    def test_match_no_matches(self, runner, fixtures_dir):
        """--match with no matches should find no tests."""
        result = runner.run(
            "-m",
            "*nonexistent_pattern*",
            fixtures_dir,
        )

        # Should report no tests found
        assert "No tests" in result.stdout or len(result.sessions_created) == 0


class TestEnvFlag:
    """Tests for --env / -e flag."""

    def test_env_single_value(self, runner):
        """--env should pass environment variable to pytest sessions."""
        result = runner.run(
            "--env",
            "TEST_VAR=test_value",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert result.exit_code == 0

    def test_env_multiple_values(self, runner):
        """Multiple --env flags should all be passed."""
        result = runner.run(
            "-e",
            "VAR1=value1",
            "-e",
            "VAR2=value2",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert result.exit_code == 0

    def test_env_requires_value(self, runner):
        """--env without a value should error."""
        result = runner.run(
            "--env",
        )

        assert result.exit_code != 0 or "Error" in result.stderr


class TestEvalOnlyFlag:
    """Tests for --eval-only flag."""

    def test_eval_only_runs_only_eval_tests(self, runner, fixtures_dir):
        """--eval-only should only run tests marked with pytest.mark.eval."""
        result = runner.run(
            "--eval-only",
            fixtures_dir,
            wait_for_completion=True,
        )

        # Should only find eval-marked tests
        # Only test_eval_marked.py has pytestmark = pytest.mark.eval
        # The number of sessions depends on whether we're in per-file or per-test mode
        # In per-file mode, should be 1 (just test_eval_marked.py)
        assert len(result.sessions_created) >= 1

        # Should NOT include symbolic-only tests
        symbolic_sessions = [
            s for s in result.sessions_created if "symbolic" in s.lower()
        ]
        assert (
            len(symbolic_sessions) == 0
        ), f"Should not run symbolic tests: {result.sessions_created}"

    def test_eval_only_with_per_test(self, runner, fixtures_dir):
        """--eval-only with -t should create sessions only for eval test functions."""
        result = runner.run(
            "--eval-only",
            "-t",
            fixtures_dir,
            wait_for_completion=True,
        )

        # Should find the eval tests (2 in test_eval_marked.py)
        assert len(result.sessions_created) >= 1


class TestSymbolicOnlyFlag:
    """Tests for --symbolic-only flag."""

    def test_symbolic_only_excludes_eval_tests(self, runner, fixtures_dir):
        """--symbolic-only should exclude tests marked with pytest.mark.eval."""
        result = runner.run(
            "--symbolic-only",
            fixtures_dir,
            wait_for_completion=True,
        )

        # Should NOT include eval-marked tests
        eval_sessions = [s for s in result.sessions_created if "eval_marked" in s]
        assert (
            len(eval_sessions) == 0
        ), f"Should not run eval tests: {result.sessions_created}"

    def test_symbolic_only_with_per_test(self, runner):
        """--symbolic-only with -t should work correctly."""
        result = runner.run(
            "--symbolic-only",
            "-t",
            runner.fixture_path("test_symbolic_only.py"),
            wait_for_completion=True,
        )

        # test_symbolic_only.py has 2 tests, none marked as eval
        assert len(result.sessions_created) == 2


class TestEvalSymbolicMutualExclusion:
    """Tests for --eval-only and --symbolic-only mutual exclusion."""

    def test_eval_and_symbolic_mutually_exclusive(self, runner):
        """--eval-only and --symbolic-only together should error."""
        result = runner.run(
            "--eval-only",
            "--symbolic-only",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code != 0
        assert "mutually exclusive" in result.stderr.lower()


class TestRepeatFlag:
    """Tests for --repeat flag."""

    def test_repeat_creates_multiple_sessions(self, runner):
        """--repeat N should create N sessions per test target."""
        result = runner.run(
            "--repeat",
            "3",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        # Should create 3 sessions for the single file
        assert (
            len(result.sessions_created) == 3
        ), f"Expected 3 sessions, got {len(result.sessions_created)}"

    def test_repeat_with_per_test(self, runner):
        """--repeat with -t should multiply sessions correctly."""
        result = runner.run(
            "--repeat",
            "2",
            "-t",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        # test_single_test.py has 1 test, repeated 2 times = 2 sessions
        assert (
            len(result.sessions_created) == 2
        ), f"Expected 2 sessions, got {len(result.sessions_created)}"

    def test_repeat_requires_number(self, runner):
        """--repeat without a number should error."""
        result = runner.run(
            "--repeat",
        )

        assert result.exit_code != 0

    def test_repeat_requires_positive(self, runner):
        """--repeat with 0 or negative should error."""
        result = runner.run(
            "--repeat",
            "0",
            runner.fixture_path("test_single_test.py"),
        )

        assert result.exit_code != 0


class TestTagsFlag:
    """Tests for --tags flag."""

    def test_tags_accepted(self, runner):
        """--tags should be accepted without error."""
        result = runner.run(
            "--tags",
            "test-tag",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert result.exit_code == 0

    def test_tags_multiple(self, runner):
        """--tags with comma-separated values should work."""
        result = runner.run(
            "--tags",
            "tag1,tag2,tag3",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert result.exit_code == 0

    def test_tags_requires_value(self, runner):
        """--tags without a value should error."""
        result = runner.run(
            "--tags",
        )

        assert result.exit_code != 0 or "Error" in result.stderr
