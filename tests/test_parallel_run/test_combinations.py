"""
Combination tests for .parallel_run.sh.

Tests various flag combinations to ensure they work together correctly.
This is essentially a grid search across the flag space.

Flag combinations tested:
- --wait + --per-test
- --wait + --match
- --wait + --env
- --wait + --eval-only
- --wait + --symbolic-only
- --wait + --repeat
- --wait + --tags
- --per-test + --match
- --per-test + --eval-only
- --per-test + --symbolic-only
- --per-test + --repeat
- --per-test + --tags
- --per-test + --env
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

    def test_wait_with_per_test(self, runner):
        """--wait with -t should wait for all per-test sessions."""
        result = runner.run(
            "--wait",
            "-t",
            runner.fixture_path("test_always_pass.py"),
        )

        # Should exit with 0 (all pass) and have waited for completion
        assert result.exit_code == 0, f"Failed with: {result.stderr}"

    def test_wait_with_per_test_failure(self, runner):
        """--wait with -t should return non-zero if any test fails."""
        result = runner.run(
            "--wait",
            "-t",
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


class TestPerTestWithOtherFlags:
    """Tests combining --per-test with other flags."""

    def test_per_test_with_match(self, runner, fixtures_dir):
        """--per-test with --match should match files then split by test."""
        result = runner.run(
            "-t",
            "--match",
            "*always_pass*",
            fixtures_dir,
            wait_for_completion=True,
        )

        # test_always_pass.py has 3 tests
        assert (
            len(result.sessions_created) == 3
        ), f"Expected 3 sessions, got {len(result.sessions_created)}"

    def test_per_test_with_eval_only(self, runner, fixtures_dir):
        """--per-test with --eval-only should only run eval test functions."""
        result = runner.run(
            "-t",
            "--eval-only",
            fixtures_dir,
            wait_for_completion=True,
        )

        # Should only have eval test functions
        # test_eval_marked.py has 2 eval tests
        assert len(result.sessions_created) >= 2

    def test_per_test_with_symbolic_only(self, runner, fixtures_dir):
        """--per-test with --symbolic-only should exclude eval tests."""
        result = runner.run(
            "-t",
            "--symbolic-only",
            runner.fixture_path("test_symbolic_only.py"),
            wait_for_completion=True,
        )

        # test_symbolic_only.py has 2 tests
        assert len(result.sessions_created) == 2

    def test_per_test_with_repeat(self, runner):
        """--per-test with --repeat should multiply correctly."""
        result = runner.run(
            "-t",
            "--repeat",
            "3",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        # 1 test * 3 repeats = 3 sessions
        assert (
            len(result.sessions_created) == 3
        ), f"Expected 3 sessions, got {len(result.sessions_created)}"

    def test_per_test_with_tags(self, runner):
        """--per-test with --tags should work."""
        result = runner.run(
            "-t",
            "--tags",
            "per-test-tag",
            runner.fixture_path("test_always_pass.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 3

    def test_per_test_with_env(self, runner):
        """--per-test with --env should pass env to each test session."""
        result = runner.run(
            "-t",
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

        # Should only find test_eval_marked.py and it must have eval mark
        assert len(result.sessions_created) >= 1

    def test_match_with_symbolic_only(self, runner, fixtures_dir):
        """--match with --symbolic-only should apply both filters."""
        result = runner.run(
            "--match",
            "*symbolic*",
            "--symbolic-only",
            fixtures_dir,
            wait_for_completion=True,
        )

        # Should find test_symbolic_only.py
        assert len(result.sessions_created) == 1


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

        # Each eval test file repeated 2 times
        assert len(result.sessions_created) >= 2

    def test_repeat_with_symbolic_only(self, runner):
        """--repeat with --symbolic-only should repeat symbolic tests."""
        result = runner.run(
            "--repeat",
            "2",
            "--symbolic-only",
            runner.fixture_path("test_symbolic_only.py"),
            wait_for_completion=True,
        )

        # 1 file * 2 repeats = 2 sessions
        assert len(result.sessions_created) == 2


class TestTripleCombinations:
    """Tests with three or more flags combined."""

    def test_wait_per_test_match(self, runner, fixtures_dir):
        """--wait + -t + --match should all work together."""
        result = runner.run(
            "--wait",
            "-t",
            "--match",
            "*single*",
            fixtures_dir,
        )

        # test_single_test.py has 1 test
        assert result.exit_code == 0

    def test_wait_per_test_repeat(self, runner):
        """--wait + -t + --repeat should all work together."""
        result = runner.run(
            "--wait",
            "-t",
            "--repeat",
            "2",
            runner.fixture_path("test_single_test.py"),
        )

        # 1 test * 2 repeats = 2 sessions, all should pass
        assert result.exit_code == 0

    def test_wait_per_test_eval_only(self, runner, fixtures_dir):
        """--wait + -t + --eval-only should all work together."""
        result = runner.run(
            "--wait",
            "-t",
            "--eval-only",
            fixtures_dir,
        )

        # Should wait for eval tests and return success
        assert result.exit_code == 0

    def test_wait_per_test_symbolic_only(self, runner):
        """--wait + -t + --symbolic-only should all work together."""
        result = runner.run(
            "--wait",
            "-t",
            "--symbolic-only",
            runner.fixture_path("test_symbolic_only.py"),
        )

        assert result.exit_code == 0

    def test_wait_per_test_tags(self, runner):
        """--wait + -t + --tags should all work together."""
        result = runner.run(
            "--wait",
            "-t",
            "--tags",
            "triple-combo",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0

    def test_wait_per_test_env(self, runner):
        """--wait + -t + --env should all work together."""
        result = runner.run(
            "--wait",
            "-t",
            "--env",
            "COMBO_VAR=value",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0

    def test_per_test_match_eval_only(self, runner, fixtures_dir):
        """-t + --match + --eval-only should all work together."""
        result = runner.run(
            "-t",
            "--match",
            "*eval*",
            "--eval-only",
            fixtures_dir,
            wait_for_completion=True,
        )

        # test_eval_marked.py has 2 tests
        assert len(result.sessions_created) >= 2

    def test_per_test_repeat_env(self, runner):
        """-t + --repeat + --env should all work together."""
        result = runner.run(
            "-t",
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

    def test_wait_per_test_match_repeat(self, runner, fixtures_dir):
        """--wait + -t + --match + --repeat should all work together."""
        result = runner.run(
            "--wait",
            "-t",
            "--match",
            "*single*",
            "--repeat",
            "2",
            fixtures_dir,
        )

        # test_single_test.py has 1 test, repeated 2 times
        assert result.exit_code == 0

    def test_wait_per_test_eval_only_tags(self, runner, fixtures_dir):
        """--wait + -t + --eval-only + --tags should all work together."""
        result = runner.run(
            "--wait",
            "-t",
            "--eval-only",
            "--tags",
            "quad-combo",
            fixtures_dir,
        )

        assert result.exit_code == 0

    def test_wait_per_test_symbolic_only_env(self, runner):
        """--wait + -t + --symbolic-only + --env should all work together."""
        result = runner.run(
            "--wait",
            "-t",
            "--symbolic-only",
            "--env",
            "QUAD_VAR=value",
            runner.fixture_path("test_symbolic_only.py"),
        )

        assert result.exit_code == 0

    def test_wait_per_test_repeat_tags(self, runner):
        """--wait + -t + --repeat + --tags should all work together."""
        result = runner.run(
            "--wait",
            "-t",
            "--repeat",
            "2",
            "--tags",
            "repeat-tag",
            runner.fixture_path("test_single_test.py"),
        )

        assert result.exit_code == 0

    def test_wait_per_test_env_tags(self, runner):
        """--wait + -t + --env + --tags should all work together."""
        result = runner.run(
            "--wait",
            "-t",
            "--env",
            "ENV_VAR=value",
            "--tags",
            "env-tag",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0


class TestFiveFlagCombinations:
    """Tests with five flags combined."""

    def test_wait_per_test_repeat_env_tags(self, runner):
        """--wait + -t + --repeat + --env + --tags should all work together."""
        result = runner.run(
            "--wait",
            "-t",
            "--repeat",
            "2",
            "--env",
            "FULL_VAR=value",
            "--tags",
            "full-tag",
            runner.fixture_path("test_single_test.py"),
        )

        assert result.exit_code == 0

    def test_wait_per_test_match_repeat_tags(self, runner, fixtures_dir):
        """--wait + -t + --match + --repeat + --tags should all work together."""
        result = runner.run(
            "--wait",
            "-t",
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

    def test_file_and_dir_with_per_test(self, runner):
        """File + directory input with -t should work."""
        result = runner.run(
            "-t",
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

    def test_specific_test_with_per_test(self, runner):
        """Specific test node with -t should work (even though redundant)."""
        test_path = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        result = runner.run(
            "-t",
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

    def test_multiple_specific_tests_with_per_test_and_wait(self, runner):
        """Multiple explicit node IDs with -t and --wait should block and wait.

        This tests the exact scenario of:
            .parallel_run.sh -t --wait file.py::test_a file.py::test_b ...

        The --wait flag should cause the script to block until all tests complete,
        regardless of whether -t is specified (which is somewhat redundant with
        explicit node IDs, but should still work correctly).
        """
        import time

        test1 = runner.fixture_path("test_always_pass.py") + "::test_pass_one"
        test2 = runner.fixture_path("test_always_pass.py") + "::test_pass_two"
        test3 = runner.fixture_path("test_always_pass.py") + "::test_pass_three"
        test4 = runner.fixture_path("test_single_test.py") + "::test_single"

        start = time.time()
        result = runner.run(
            "-t",
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

    def test_multiple_env_with_per_test(self, runner):
        """Multiple --env with -t should work."""
        result = runner.run(
            "-t",
            "-e",
            "A=1",
            "-e",
            "B=2",
            runner.fixture_path("test_single_test.py"),
            wait_for_completion=True,
        )

        assert len(result.sessions_created) == 1

    def test_multiple_env_wait_per_test(self, runner):
        """Multiple --env with --wait and -t should work."""
        result = runner.run(
            "--wait",
            "-t",
            "-e",
            "X=1",
            "-e",
            "Y=2",
            runner.fixture_path("test_always_pass.py"),
        )

        assert result.exit_code == 0
