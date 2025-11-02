import unity.common.prompt_helpers as prompt_helpers


def test_now_default_utc_full():
    assert prompt_helpers.now() == "2025-06-13 12:00:00 UTC"


def test_now_default_utc_time_only():
    assert prompt_helpers.now(time_only=True) == "12:00:00 UTC"


def test_now_pacific_full():
    assert (
        prompt_helpers.now(tz="America/Los_Angeles")
        == "2025-06-13 05:00:00 America/Los_Angeles"
    )


def test_now_pacific_time_only():
    assert (
        prompt_helpers.now(time_only=True, tz="America/Los_Angeles")
        == "05:00:00 America/Los_Angeles"
    )
