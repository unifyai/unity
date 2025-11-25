import unity.common.prompt_helpers as prompt_helpers


def test_now_full_format():
    # Assistant timezone is treated as UTC in tests
    assert prompt_helpers.now() == "2025-06-13 12:00:00 UTC"


def test_now_time_only():
    assert prompt_helpers.now(time_only=True) == "12:00:00 UTC"
