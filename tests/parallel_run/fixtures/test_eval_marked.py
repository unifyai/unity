"""
Fixture test file with eval-marked tests.
Used for testing --eval-only and --symbolic-only flags.
"""

import pytest

# Mark entire module as eval
pytestmark = pytest.mark.eval


def test_eval_one():
    """First eval test."""
    assert True


def test_eval_two():
    """Second eval test."""
    assert True
