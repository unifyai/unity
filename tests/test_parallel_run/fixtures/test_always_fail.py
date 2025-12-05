"""
Fixture test file that always fails.
Used for testing parallel_run.sh behavior with failing tests.
"""


def test_fail_one():
    """First failing test."""
    assert False, "Intentional failure for testing"


def test_fail_two():
    """Second failing test."""
    assert 1 == 2, "Intentional failure for testing"
