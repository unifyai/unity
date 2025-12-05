"""
Fixture test file with mixed pass/fail results.
Used for testing parallel_run.sh behavior with partial failures.
"""


def test_mixed_pass():
    """This test passes."""
    assert True


def test_mixed_fail():
    """This test fails."""
    assert False, "Intentional failure for testing"


def test_mixed_pass_again():
    """Another passing test."""
    assert True
