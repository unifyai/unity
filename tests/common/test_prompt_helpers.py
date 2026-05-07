import functools
import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import unity.common.prompt_helpers as prompt_helpers
from unity.common.tool_spec import ToolSpec

pytestmark = pytest.mark.no_unify_context


def _real_prompt_helpers(monkeypatch):
    module = importlib.reload(prompt_helpers)
    monkeypatch.setattr(module, "log_startup_timing", lambda *args, **kwargs: None)
    module._assistant_timezone_cache = None
    return module


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


def test_assistant_timezone_lookup_caches_within_ttl(monkeypatch):
    module = _real_prompt_helpers(monkeypatch)
    calls = []

    def fake_get_logs(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(entries={"timezone": "Asia/Karachi"})]

    monkeypatch.setattr(module, "_contacts_context", lambda: "User/Assistant/Contacts")
    monkeypatch.setattr(module.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr("unify.get_logs", fake_get_logs)

    assert module.get_assistant_timezone() == "Asia/Karachi"
    assert module.get_assistant_timezone() == "Asia/Karachi"
    assert len(calls) == 1
    assert calls[0]["filter"] == "contact_id == 0"
    assert calls[0]["from_fields"] == ["timezone"]


def test_assistant_timezone_lookup_refreshes_after_ttl(monkeypatch):
    module = _real_prompt_helpers(monkeypatch)
    calls = []
    monotonic_now = {"value": 1000.0}

    def fake_get_logs(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(entries={"timezone": "Asia/Karachi"})]

    monkeypatch.setattr(module, "_contacts_context", lambda: "User/Assistant/Contacts")
    monkeypatch.setattr(module.time, "monotonic", lambda: monotonic_now["value"])
    monkeypatch.setattr("unify.get_logs", fake_get_logs)

    assert module.get_assistant_timezone() == "Asia/Karachi"
    monotonic_now["value"] += 299
    assert module.get_assistant_timezone() == "Asia/Karachi"
    monotonic_now["value"] += 2
    assert module.get_assistant_timezone() == "Asia/Karachi"
    assert len(calls) == 2


def test_now_recomputes_current_time_while_reusing_cached_timezone(monkeypatch):
    module = _real_prompt_helpers(monkeypatch)
    calls = []
    current_times = iter(
        [
            datetime(2026, 5, 7, 8, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 5, 7, 8, 0, 1, tzinfo=timezone.utc),
        ],
    )

    def fake_get_logs(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(entries={"timezone": "UTC"})]

    monkeypatch.setattr(module, "_contacts_context", lambda: "User/Assistant/Contacts")
    monkeypatch.setattr(module.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(module, "_utc_now", lambda: next(current_times))
    monkeypatch.setattr("unify.get_logs", fake_get_logs)

    first = module.now(as_string=False)
    second = module.now(as_string=False)

    assert first == datetime(2026, 5, 7, 8, 0, 0, tzinfo=timezone.utc)
    assert second == first + timedelta(seconds=1)
    assert len(calls) == 1


def test_now_falls_back_to_utc_when_timezone_lookup_fails(monkeypatch):
    module = _real_prompt_helpers(monkeypatch)
    calls = []

    def fake_get_logs(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(module, "_contacts_context", lambda: "User/Assistant/Contacts")
    monkeypatch.setattr(module.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(
        module,
        "_utc_now",
        lambda: datetime(2026, 5, 7, 8, 0, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("unify.get_logs", fake_get_logs)

    assert module.get_assistant_timezone() is None
    assert module.now() == "Thursday, May 07, 2026 at 08:00 AM UTC"
    assert len(calls) == 2


def test_failed_assistant_timezone_lookup_does_not_poison_cache(monkeypatch):
    module = _real_prompt_helpers(monkeypatch)
    calls = []

    def fake_get_logs(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("backend unavailable")
        return [SimpleNamespace(entries={"timezone": "Asia/Karachi"})]

    monkeypatch.setattr(module, "_contacts_context", lambda: "User/Assistant/Contacts")
    monkeypatch.setattr(module.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr("unify.get_logs", fake_get_logs)

    assert module.get_assistant_timezone() is None
    assert module.get_assistant_timezone() == "Asia/Karachi"
    assert len(calls) == 2


def test_missing_assistant_timezone_row_does_not_poison_cache(monkeypatch):
    module = _real_prompt_helpers(monkeypatch)
    calls = []

    def fake_get_logs(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return []
        return [SimpleNamespace(entries={"timezone": "Asia/Karachi"})]

    monkeypatch.setattr(module, "_contacts_context", lambda: "User/Assistant/Contacts")
    monkeypatch.setattr(module.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr("unify.get_logs", fake_get_logs)

    assert module.get_assistant_timezone() is None
    assert module.get_assistant_timezone() == "Asia/Karachi"
    assert len(calls) == 2


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
