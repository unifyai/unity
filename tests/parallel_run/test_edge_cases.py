"""
Edge case and error handling tests for parallel_run.sh.

Tests unusual inputs, error conditions, and boundary cases:
- Empty directories
- Non-existent paths
- Invalid arguments
- Special characters in paths
- Very long session names
- Concurrent runs
"""

from __future__ import annotations


from tests.parallel_run.conftest import (
    REPO_ROOT,
)


class TestInvalidInputs:
    """Tests for handling invalid inputs."""

    def test_nonexistent_file(self, runner):
        """Non-existent file should produce warning."""
        result = runner.run(
            "tests/this_file_does_not_exist_ever.py",
        )

        assert "Warning" in result.stderr or "Skipping" in result.stderr
        assert (
            "non-existent" in result.stderr.lower()
            or "skipping" in result.stderr.lower()
        )

    def test_nonexistent_directory(self, runner):
        """Non-existent directory should produce warning."""
        result = runner.run(
            "tests/nonexistent_directory_xyz/",
        )

        assert (
            "Warning" in result.stderr
            or "No valid" in result.stderr
            or "Skipping" in result.stderr
        )

    def test_nonexistent_test_node(self, runner):
        """Non-existent test node should produce warning."""
        test_path = (
            runner.fixture_path("test_always_pass.py") + "::test_nonexistent_function"
        )
        result = runner.run(test_path)

        # Should either warn or the test will fail when pytest runs
        # The script should still create a session
        assert result.exit_code == 0 or "Warning" in result.stderr

    def test_mixed_valid_invalid_paths(self, runner):
        """Mix of valid and invalid paths should process valid ones."""
        result = runner.run(
            "nonexistent_file.py",
            runner.fixture_path("test_single_test.py"),
            "another_nonexistent.py",
            wait_for_completion=True,
        )

        # Should create session for the valid file
        assert len(result.sessions_created) >= 1

        # Should warn about invalid files
        assert "Warning" in result.stderr or "Skipping" in result.stderr


class TestArgumentErrors:
    """Tests for invalid argument handling."""

    def test_match_without_pattern(self, runner):
        """--match without pattern should error."""
        result = runner.run(
            "--match",
        )

        assert result.exit_code != 0
        assert "Error" in result.stderr

    def test_env_without_value(self, runner):
        """--env without KEY=VALUE should error."""
        result = runner.run(
            "--env",
        )

        assert result.exit_code != 0 or "Error" in result.stderr

    def test_env_without_equals(self, runner):
        """--env with KEY (no =VALUE) should error."""
        result = runner.run(
            "--env",
            "JUST_KEY",
            runner.fixture_path("test_single_test.py"),
        )

        assert result.exit_code != 0 or "Error" in result.stderr

    def test_repeat_without_number(self, runner):
        """--repeat without number should error."""
        result = runner.run(
            "--repeat",
        )

        assert result.exit_code != 0

    def test_repeat_with_zero(self, runner):
        """--repeat 0 should error."""
        result = runner.run(
            "--repeat",
            "0",
            runner.fixture_path("test_single_test.py"),
        )

        assert result.exit_code != 0

    def test_repeat_with_negative(self, runner):
        """--repeat with negative number should error."""
        result = runner.run(
            "--repeat",
            "-1",
            runner.fixture_path("test_single_test.py"),
        )

        assert result.exit_code != 0

    def test_repeat_with_non_number(self, runner):
        """--repeat with non-numeric value should error."""
        result = runner.run(
            "--repeat",
            "abc",
            runner.fixture_path("test_single_test.py"),
        )

        assert result.exit_code != 0

    def test_tags_without_value(self, runner):
        """--tags without value should error."""
        result = runner.run(
            "--tags",
        )

        assert result.exit_code != 0 or "Error" in result.stderr


class TestEmptyResults:
    """Tests for scenarios that result in no tests."""

    def test_no_matching_files(self, runner, fixtures_dir):
        """--match with no matches should report no tests."""
        result = runner.run(
            "--match",
            "*this_pattern_matches_nothing_xyz*",
            fixtures_dir,
        )

        assert "No tests" in result.stdout or len(result.sessions_created) == 0

    def test_eval_only_no_eval_tests(self, runner):
        """--eval-only with file that has no eval tests creates no sessions.

        In default per-test mode, the script pre-filters tests by marker.
        If a file has no matching tests, no sessions are created for it.
        """
        result = runner.run(
            "--eval-only",
            runner.fixture_path("test_symbolic_only.py"),
        )

        # test_symbolic_only.py has no eval marks, so no sessions created
        assert result.exit_code == 0
        assert len(result.sessions_created) == 0

    def test_symbolic_only_all_eval_tests(self, runner):
        """--symbolic-only with all-eval file creates no sessions.

        In default per-test mode, the script pre-filters tests by marker.
        If a file has no matching tests, no sessions are created for it.
        """
        result = runner.run(
            "--symbolic-only",
            runner.fixture_path("test_eval_marked.py"),
        )

        # test_eval_marked.py is all eval, so no symbolic tests to run
        assert result.exit_code == 0
        assert len(result.sessions_created) == 0


class TestPathFormats:
    """Tests for different path format handling."""

    def test_relative_path(self, runner):
        """Relative paths should work."""
        result = runner.run(
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        assert result.exit_code == 0
        assert len(result.sessions_created) == 1

    def test_absolute_path(self, runner):
        """Absolute paths should work."""
        abs_path = str(REPO_ROOT / runner.fixture_path("test_single_test.py"))
        result = runner.run(abs_path, wait_for_completion=True)

        assert result.exit_code == 0
        assert len(result.sessions_created) == 1

    def test_path_with_dot_prefix(self, runner):
        """Paths starting with ./ should work."""
        path = "./" + runner.fixture_path("test_single_test.py")
        result = runner.run(path, wait_for_completion=True)

        assert result.exit_code == 0
        assert len(result.sessions_created) == 1

    def test_trailing_slash_on_directory(self, runner, fixtures_dir):
        """Directory paths with trailing slash should work."""
        result = runner.run(
            fixtures_dir + "/",
            wait_for_completion=True,
        )

        assert result.exit_code == 0
        assert len(result.sessions_created) >= 1


class TestSpecialCharacters:
    """Tests for handling special characters."""

    def test_pattern_with_asterisks(self, runner, fixtures_dir):
        """--match with asterisks should work as glob."""
        result = runner.run(
            "--match",
            "*pass*",
            fixtures_dir,
            wait_for_completion=True,
        )

        assert len(result.sessions_created) >= 1

    def test_env_with_special_chars(self, runner):
        """--env with special characters in value should work."""
        result = runner.run(
            "--env",
            "MY_VAR=hello world",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        # Should at least not crash
        assert result.exit_code == 0

    def test_tags_with_hyphen(self, runner):
        """--tags with hyphens should work."""
        result = runner.run(
            "--tags",
            "my-test-tag",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        assert result.exit_code == 0


class TestOutputMessages:
    """Tests for expected output messages."""

    def test_shows_created_sessions(self, runner):
        """Output should list created sessions."""
        result = runner.run(
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        assert "Created" in result.stdout
        assert "tmux" in result.stdout.lower()

    def test_shows_usage_hints(self, runner):
        """Output should show usage hints."""
        result = runner.run(
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        # Should include hints about attaching/observing
        assert "attach" in result.stdout.lower() or "tmux" in result.stdout.lower()

    def test_shows_completion_message(self, runner):
        """Script should show completion message."""
        result = runner.run(
            runner.fixture_path("test_single_test.py"),
        )

        assert "completed" in result.stdout.lower() or "passed" in result.stdout.lower()

    def test_failure_shows_failure_info(self, runner):
        """Script with failures should show failure info."""
        result = runner.run(
            runner.fixture_path("test_always_fail.py"),
        )

        assert (
            "fail" in result.stdout.lower()
            or "Failure" in result.stdout
            or result.exit_code != 0
        )


class TestMultipleRuns:
    """Tests for running the script multiple times."""

    def test_second_run_creates_new_sessions(self, runner):
        """Second run with same file should create sessions with unique names."""
        # Start first run WITHOUT waiting (so session exists when second run starts)
        result1 = runner.run(
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=False,
        )

        # Start second run immediately (while first session still exists)
        # This should detect the collision and create a unique name
        result2 = runner.run(
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,  # Wait for both to complete
        )

        # Both should succeed in creating sessions
        assert result1.exit_code == 0
        assert result2.exit_code == 0

        # Sessions should have unique names (e.g., second gets -2 appended)
        all_sessions = result1.sessions_created + result2.sessions_created
        assert len(set(all_sessions)) == len(
            all_sessions,
        ), f"Session names should be unique: {all_sessions}"


class TestDefaultModeEdgeCases:
    """Edge cases for default per-test mode."""

    def test_single_test_file(self, runner):
        """Single-test file should create one session."""
        result = runner.run(
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 1

    def test_specific_test_node(self, runner):
        """Specific test node should create one session."""
        test_path = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        result = runner.run(
            test_path,
            wait_for_completion=True,
        )

        # Should be 1 session (specific test specified)
        assert len(result.sessions_created) == 1

    def test_multiple_specific_nodes(self, runner):
        """Multiple specific test nodes should work."""
        test1 = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        test2 = runner.fixture_path("test_always_pass.py") + "::test_pass_two"
        result = runner.run(
            test1,
            test2,
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 2
