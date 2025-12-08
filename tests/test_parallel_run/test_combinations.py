"""
Combination tests for parallel_run.sh.

Tests various flag combinations to ensure they work together correctly.
This is essentially a grid search across the flag space.

Flag combinations tested:
- --wait (default per-test mode)
- --wait + --serial
- --wait + --match
- --wait + --env
- --wait + --eval-only
- --wait + --symbolic-only
- --wait + --repeat
- --wait + --tags
- --serial + --match
- --serial + --eval-only
- --serial + --symbolic-only
- --match + --eval-only
- --match + --symbolic-only
- --eval-only + --repeat
- --symbolic-only + --repeat
- Triple combinations
- Quadruple combinations
"""

from __future__ import annotations


class TestWaitWithOtherFlags:
    """Tests combining --wait with other flags."""

    def test_wait_default_per_test(self, runner):
        """--wait should wait for all per-test sessions (default mode)."""
        result = runner.run(
            "--wait",
            runner.fixture_path("test_always_pass.py"),
        )

        # Should exit with 0 (all pass) and have waited for completion
        assert result.exit_code == 0, f"Failed with: {result.stderr}"

    def test_wait_default_per_test_failure(self, runner):
        """--wait should return non-zero if any test fails (default mode)."""
        result = runner.run(
            "--wait",
            runner.fixture_path("test_mixed_results.py"),
        )

        # test_mixed_results.py has one failing test
        assert result.exit_code != 0

    def test_wait_with_match(self, runner, fixtures_dir):
        """--wait with --match should wait for matched sessions."""
        result = runner.run(
            "--wait",
            "--match",
            "*single*",
            fixtures_dir,
        )

        assert result.exit_code == 0

    def test_wait_with_env(self, runner):
        """--wait with --env should pass env and wait."""
        result = runner.run(
            "--wait",
            "--env",
            "MY_VAR=my_value",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0

    def test_wait_with_eval_only(self, runner, fixtures_dir):
        """--wait with --eval-only should wait for eval tests only."""
        result = runner.run(
            "--wait",
            "--eval-only",
            fixtures_dir,
        )

        # Should succeed if eval tests pass
        assert result.exit_code == 0

    def test_wait_with_symbolic_only(self, runner):
        """--wait with --symbolic-only should wait for symbolic tests."""
        result = runner.run(
            "--wait",
            "--symbolic-only",
            runner.fixture_path("test_symbolic_only.py"),
        )

        assert result.exit_code == 0

    def test_wait_with_repeat(self, runner):
        """--wait with --repeat should wait for all repeated runs."""
        result = runner.run(
            "--wait",
            "--repeat",
            "2",
            runner.fixture_path("test_single_test.py"),
        )

        assert result.exit_code == 0

    def test_wait_with_tags(self, runner):
        """--wait with --tags should tag and wait."""
        result = runner.run(
            "--wait",
            "--tags",
            "my-tag",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0


class TestDefaultModeWithOtherFlags:
    """Tests combining default per-test mode with other flags."""

    def test_default_with_match(self, runner, fixtures_dir):
        """Default mode with --match should match files then split by test."""
        result = runner.run(
            "--match",
            "*always_pass*",
            fixtures_dir,
            wait_for_completion=True,
        )

        # test_always_pass.py has 3 tests
        assert (
            len(result.sessions_created) == 3
        ), f"Expected 3 sessions, got {len(result.sessions_created)}"

    def test_default_with_eval_only(self, runner, fixtures_dir):
        """Default mode with --eval-only should only run eval test functions."""
        result = runner.run(
            "--eval-only",
            fixtures_dir,
            wait_for_completion=True,
        )

        # Should only have eval test functions
        # test_eval_marked.py has 2 eval tests
        assert len(result.sessions_created) == 2

    def test_default_with_symbolic_only(self, runner, fixtures_dir):
        """Default mode with --symbolic-only should exclude eval tests."""
        result = runner.run(
            "--symbolic-only",
            runner.fixture_path("test_symbolic_only.py"),
            wait_for_completion=True,
        )

        # test_symbolic_only.py has 2 tests
        assert len(result.sessions_created) == 2

    def test_default_with_repeat(self, runner):
        """Default mode with --repeat should multiply correctly."""
        result = runner.run(
            "--repeat",
            "3",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        # 1 test * 3 repeats = 3 sessions
        assert (
            len(result.sessions_created) == 3
        ), f"Expected 3 sessions, got {len(result.sessions_created)}"

    def test_default_with_tags(self, runner):
        """Default mode with --tags should work."""
        result = runner.run(
            "--tags",
            "default-mode-tag",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 3

    def test_default_with_env(self, runner):
        """Default mode with --env should pass env to each test session."""
        result = runner.run(
            "--env",
            "TEST_ENV=value",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 3


class TestMatchWithOtherFlags:
    """Tests combining --match with other flags."""

    def test_match_with_eval_only(self, runner, fixtures_dir):
        """--match with --eval-only should apply both filters."""
        result = runner.run(
            "--match",
            "*eval*",
            "--eval-only",
            fixtures_dir,
            wait_for_completion=True,
        )

        # Should only find test_eval_marked.py (2 eval tests) with eval mark
        assert len(result.sessions_created) == 2

    def test_match_with_symbolic_only(self, runner, fixtures_dir):
        """--match with --symbolic-only should apply both filters."""
        result = runner.run(
            "--match",
            "*symbolic*",
            "--symbolic-only",
            fixtures_dir,
            wait_for_completion=True,
        )

        # Should find test_symbolic_only.py (2 tests, default per-test mode)
        assert len(result.sessions_created) == 2


class TestRepeatCombinations:
    """Tests combining --repeat with other flags."""

    def test_repeat_with_eval_only(self, runner, fixtures_dir):
        """--repeat with --eval-only should repeat only eval tests."""
        result = runner.run(
            "--repeat",
            "2",
            "--eval-only",
            fixtures_dir,
            wait_for_completion=True,
        )

        # test_eval_marked.py has 2 eval tests, repeated 2 times = 4 sessions
        assert len(result.sessions_created) == 4

    def test_repeat_with_symbolic_only(self, runner):
        """--repeat with --symbolic-only should repeat symbolic tests."""
        result = runner.run(
            "--repeat",
            "2",
            "--symbolic-only",
            runner.fixture_path("test_symbolic_only.py"),
            wait_for_completion=True,
        )

        # 2 tests * 2 repeats = 4 sessions (default per-test mode)
        assert len(result.sessions_created) == 4


class TestTripleCombinations:
    """Tests with three or more flags combined."""

    def test_wait_match(self, runner, fixtures_dir):
        """--wait + --match should all work together."""
        result = runner.run(
            "--wait",
            "--match",
            "*single*",
            fixtures_dir,
        )

        # test_single_test.py has 1 test
        assert result.exit_code == 0

    def test_wait_repeat(self, runner):
        """--wait + --repeat should all work together."""
        result = runner.run(
            "--wait",
            "--repeat",
            "2",
            runner.fixture_path("test_single_test.py"),
        )

        # 1 test * 2 repeats = 2 sessions, all should pass
        assert result.exit_code == 0

    def test_wait_eval_only(self, runner):
        """--wait + --eval-only should all work together."""
        # Use specific file instead of whole directory to avoid slow collection
        result = runner.run(
            "--wait",
            "--eval-only",
            runner.fixture_path("test_eval_marked.py"),
        )

        # Should wait for eval tests and return success
        assert result.exit_code == 0

    def test_wait_symbolic_only(self, runner):
        """--wait + --symbolic-only should all work together."""
        result = runner.run(
            "--wait",
            "--symbolic-only",
            runner.fixture_path("test_symbolic_only.py"),
        )

        assert result.exit_code == 0

    def test_wait_tags(self, runner):
        """--wait + --tags should all work together."""
        result = runner.run(
            "--wait",
            "--tags",
            "triple-combo",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0

    def test_wait_env(self, runner):
        """--wait + --env should all work together."""
        result = runner.run(
            "--wait",
            "--env",
            "COMBO_VAR=value",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0

    def test_match_eval_only(self, runner, fixtures_dir):
        """--match + --eval-only should all work together."""
        result = runner.run(
            "--match",
            "*eval*",
            "--eval-only",
            fixtures_dir,
            wait_for_completion=True,
        )

        # test_eval_marked.py has 2 tests
        assert len(result.sessions_created) == 2

    def test_repeat_env(self, runner):
        """--repeat + --env should all work together."""
        result = runner.run(
            "--repeat",
            "2",
            "--env",
            "REPEAT_VAR=value",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 2


class TestQuadrupleCombinations:
    """Tests with four or more flags combined."""

    def test_wait_match_repeat(self, runner, fixtures_dir):
        """--wait + --match + --repeat should all work together."""
        result = runner.run(
            "--wait",
            "--match",
            "*single*",
            "--repeat",
            "2",
            fixtures_dir,
        )

        # test_single_test.py has 1 test, repeated 2 times
        assert result.exit_code == 0

    def test_wait_eval_only_tags(self, runner):
        """--wait + --eval-only + --tags should all work together."""
        # Use specific file instead of whole directory to avoid slow collection
        result = runner.run(
            "--wait",
            "--eval-only",
            "--tags",
            "quad-combo",
            runner.fixture_path("test_eval_marked.py"),
        )

        assert result.exit_code == 0

    def test_wait_symbolic_only_env(self, runner):
        """--wait + --symbolic-only + --env should all work together."""
        result = runner.run(
            "--wait",
            "--symbolic-only",
            "--env",
            "QUAD_VAR=value",
            runner.fixture_path("test_symbolic_only.py"),
        )

        assert result.exit_code == 0

    def test_wait_repeat_tags(self, runner):
        """--wait + --repeat + --tags should all work together."""
        result = runner.run(
            "--wait",
            "--repeat",
            "2",
            "--tags",
            "repeat-tag",
            runner.fixture_path("test_single_test.py"),
        )

        assert result.exit_code == 0

    def test_wait_env_tags(self, runner):
        """--wait + --env + --tags should all work together."""
        result = runner.run(
            "--wait",
            "--env",
            "ENV_VAR=value",
            "--tags",
            "env-tag",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0


class TestFiveFlagCombinations:
    """Tests with five flags combined."""

    def test_wait_repeat_env_tags(self, runner):
        """--wait + --repeat + --env + --tags should all work together."""
        result = runner.run(
            "--wait",
            "--repeat",
            "2",
            "--env",
            "FULL_VAR=value",
            "--tags",
            "full-tag",
            runner.fixture_path("test_single_test.py"),
        )

        assert result.exit_code == 0

    def test_wait_match_repeat_tags(self, runner, fixtures_dir):
        """--wait + --match + --repeat + --tags should all work together."""
        result = runner.run(
            "--wait",
            "--match",
            "*single*",
            "--repeat",
            "2",
            "--tags",
            "match-repeat",
            fixtures_dir,
        )

        assert result.exit_code == 0


class TestInputTypeCombinations:
    """Tests combining different input types with flags."""

    def test_file_and_dir_with_wait(self, runner):
        """File + directory input with --wait should work."""
        result = runner.run(
            "--wait",
            runner.fixture_path("test_single_test.py"),
            runner.fixture_path("subdir"),
        )

        assert result.exit_code == 0

    def test_file_and_dir_default(self, runner):
        """File + directory input with default per-test mode should work."""
        result = runner.run(
            runner.fixture_path("test_single_test.py"),
            runner.fixture_path("subdir"),
            wait_for_completion=True,
        )

        # test_single_test.py has 1 test, subdir has 2 = 3 total
        assert len(result.sessions_created) == 3

    def test_specific_test_with_wait(self, runner):
        """Specific test node with --wait should work."""
        test_path = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        result = runner.run(
            "--wait",
            test_path,
        )

        assert result.exit_code == 0

    def test_specific_test_default(self, runner):
        """Specific test node with default mode should work."""
        test_path = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        result = runner.run(
            test_path,
            wait_for_completion=True,
        )

        # Still just 1 session for the specific test
        assert len(result.sessions_created) == 1

    def test_multiple_specific_tests_with_wait(self, runner):
        """Multiple specific tests with --wait should work."""
        test1 = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        test2 = runner.fixture_path("test_always_pass.py") + "::test_pass_two"
        result = runner.run(
            "--wait",
            test1,
            test2,
        )

        assert result.exit_code == 0

    def test_multiple_specific_tests_with_wait_blocks(self, runner):
        """Multiple explicit node IDs with --wait should block until complete.

        This tests the exact scenario of:
            parallel_run.sh --wait file.py::test_a file.py::test_b ...

        The --wait flag should cause the script to block until all tests complete.
        """
        import time

        test1 = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        test2 = runner.fixture_path("test_always_pass.py") + "::test_pass_two"
        test3 = runner.fixture_path("test_always_pass.py") + "::test_pass_three"
        test4 = runner.fixture_path("test_single_test.py") + "::test_single"

        start = time.time()
        result = runner.run(
            "--wait",
            test1,
            test2,
            test3,
            test4,
        )
        elapsed = time.time() - start

        # Should have blocked for at least some time
        assert (
            elapsed > 1.0
        ), f"--wait should block until tests complete, but returned in {elapsed:.2f}s"

        # All tests should pass
        assert result.exit_code == 0, f"All tests should pass: {result.stderr}"

    def test_mixed_file_and_specific_test_with_wait(self, runner):
        """File + specific test with --wait should work."""
        file_path = runner.fixture_path("test_single_test.py")
        specific_test = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        result = runner.run(
            "--wait",
            file_path,
            specific_test,
        )

        assert result.exit_code == 0


class TestMultipleEnvVars:
    """Tests with multiple --env flags."""

    def test_multiple_env_with_wait(self, runner):
        """Multiple --env with --wait should work."""
        result = runner.run(
            "--wait",
            "-e",
            "VAR1=value1",
            "-e",
            "VAR2=value2",
            "-e",
            "VAR3=value3",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0

    def test_multiple_env_default(self, runner):
        """Multiple --env with default per-test mode should work."""
        result = runner.run(
            "-e",
            "A=1",
            "-e",
            "B=2",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 1

    def test_multiple_env_wait(self, runner):
        """Multiple --env with --wait should work."""
        result = runner.run(
            "--wait",
            "-e",
            "X=1",
            "-e",
            "Y=2",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0
