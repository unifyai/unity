from datetime import datetime

import unity.common.prompt_helpers as prompt_helpers


def test_now_full_format():
    # Human-readable format with day, month, date, time, and timezone
    assert prompt_helpers.now() == "Friday, June 13, 2025 at 12:00 PM UTC"


def test_now_time_only():
    assert prompt_helpers.now(time_only=True) == "12:00 PM UTC"


def test_now_as_datetime():
    # When as_string=False, returns a datetime object
    result = prompt_helpers.now(as_string=False)
    assert isinstance(result, datetime)
    assert result.year == 2025
    assert result.month == 6
    assert result.day == 13
