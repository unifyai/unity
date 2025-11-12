import unity.common.prompt_helpers as prompt_helpers


def test_now_assistant_tz_full():
    # Assistant timezone is treated as UTC in tests
    assert prompt_helpers.now() == "2025-06-13 12:00:00 UTC"


def test_now_assistant_tz_time_only():
    assert prompt_helpers.now(time_only=True) == "12:00:00 UTC"
