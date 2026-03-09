import functools
from datetime import datetime

import unity.common.prompt_helpers as prompt_helpers
from unity.common.tool_spec import ToolSpec


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


async def _sample_execute_code(
    thought: str,
    code: str | None = None,
    *,
    language: str = "python",
    _notification_up_q=None,
):
    """Execute arbitrary code in a specified language and state mode."""
    return None


def test_sig_dict_unwraps_toolspec_wrappers():
    spec = ToolSpec(fn=_sample_execute_code, display_label="Running code")

    @functools.wraps(spec.fn)
    async def wrapped_execute_code(*a, **kw):
        return await spec.fn(*a, **kw)

    wrapped_spec = ToolSpec(
        fn=wrapped_execute_code,
        display_label=spec.display_label,
    )

    sig = prompt_helpers.sig_dict({"execute_code": wrapped_spec})["execute_code"]
    assert sig.startswith("(thought: str")
    assert "language: str = 'python'" in sig
    assert "*a, **kw" not in sig
