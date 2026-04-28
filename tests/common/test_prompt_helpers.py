import functools
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import unity.common.prompt_helpers as prompt_helpers
from unity.common.context_registry import ContextRegistry
from unity.common.tool_spec import ToolSpec
from unity.session_details import SESSION_DETAILS


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


@pytest.fixture
def _self_timezone_session():
    """Pin SESSION_DETAILS + ContextRegistry to a per-body root with a
    resolved ``self_contact_id`` so the read goes through the production
    path. Each test toggles ``hive_id`` to flip between Hive-shared and
    per-body Contacts roots.
    """
    SESSION_DETAILS.reset()
    SESSION_DETAILS.populate(agent_id=7, user_id="u7", self_contact_id=5)
    previous_base = ContextRegistry._base_context
    ContextRegistry._base_context = "u7/7"
    try:
        yield
    finally:
        SESSION_DETAILS.reset()
        ContextRegistry._base_context = previous_base


@pytest.mark.parametrize(
    ("hive_id", "expected_context"),
    [
        (None, "u7/7/Contacts"),
        (42, "Hives/42/Contacts"),
    ],
    ids=["solo_body", "hive_member"],
)
def test_read_self_contact_timezone_routes_through_hive_aware_context(
    _self_timezone_session,
    hive_id,
    expected_context,
):
    """The self-timezone read must consult the same Contacts root every
    other Contacts reader and writer uses: a Hive member resolves to
    ``Hives/{hive_id}/Contacts`` and a solo body keeps the per-body
    root. Hardcoding the per-body literal here causes Hive members to
    render every prompt in UTC even when the assistant is configured
    for a local timezone.
    """
    SESSION_DETAILS.hive_id = hive_id

    captured: dict = {}

    def _fake_get_logs(**kwargs):
        captured.update(kwargs)
        return [SimpleNamespace(entries={"timezone": "America/New_York"})]

    with patch("unify.get_logs", side_effect=_fake_get_logs):
        result = prompt_helpers.read_self_contact_timezone()

    assert result == "America/New_York"
    assert captured["context"] == expected_context
    assert captured["filter"] == "contact_id == 5"


def test_read_self_contact_timezone_returns_none_when_self_contact_unresolved(
    _self_timezone_session,
):
    """Bootstrap may not yet have resolved the self-contact overlay.
    Returning ``None`` lets callers fall back to UTC instead of issuing
    a doomed lookup against ``contact_id == None``.
    """
    SESSION_DETAILS.assistant.contact_id = None

    with patch("unify.get_logs") as mock_get_logs:
        result = prompt_helpers.read_self_contact_timezone()

    assert result is None
    mock_get_logs.assert_not_called()


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
