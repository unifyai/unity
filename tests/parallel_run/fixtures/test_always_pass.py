"""
Fixture test file that always passes.
Used for testing parallel_run.sh behavior with passing tests.
"""


def test_pass_one():
    """First passing test."""
    assert True


def test_pass_two():
    """Second passing test."""
    assert 1 + 1 == 2


def test_pass_three():
    """Third passing test."""
    assert "hello".upper() == "HELLO"
