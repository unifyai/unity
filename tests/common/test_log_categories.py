"""Tests for the log-category filtering system in hierarchical_logger.

These are pure-Python unit tests (no LLM, no async, no backend).  Their main
purpose is to catch silent breakage when someone adds a new event type to
``ICONS`` but forgets to assign it to a ``LOG_CATEGORIES`` entry.
"""

from __future__ import annotations

import logging

import pytest

from unity.common.hierarchical_logger import (
    ALL_LOG_CATEGORIES,
    DEFAULT_LOG_CATEGORIES,
    ICONS,
    LOG_CATEGORIES,
    configure_log_categories,
    get_enabled_categories,
    get_event_category,
    is_category_enabled,
    is_event_visible,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_categories():
    """Restore default categories after every test."""
    yield
    configure_log_categories(None)


# ── Coverage invariant ──────────────────────────────────────────────────────

# Event types that are intentionally uncategorized (severity indicators that
# should always be visible regardless of which categories are enabled).
INTENTIONALLY_UNCATEGORIZED = {"warning", "error"}


def test_every_icon_event_type_has_a_category():
    """Every event type in ICONS (except intentionally uncategorized ones)
    must appear in exactly one LOG_CATEGORIES entry."""
    all_categorized: set[str] = set()
    for events in LOG_CATEGORIES.values():
        all_categorized |= events

    uncovered = set(ICONS.keys()) - all_categorized - INTENTIONALLY_UNCATEGORIZED
    assert uncovered == set(), (
        f"Event types in ICONS but not assigned to any LOG_CATEGORY: {uncovered}. "
        "Add them to an appropriate category in LOG_CATEGORIES."
    )


def test_no_event_type_in_multiple_categories():
    """Each event type must belong to at most one category."""
    seen: dict[str, str] = {}
    for cat, events in LOG_CATEGORIES.items():
        for evt in events:
            assert (
                evt not in seen
            ), f"Event type {evt!r} appears in both {seen[evt]!r} and {cat!r}"
            seen[evt] = cat


def test_all_categorized_events_exist_in_icons():
    """Every event type referenced in LOG_CATEGORIES must exist in ICONS
    (except the special empty-set categories like bus/livekit)."""
    for cat, events in LOG_CATEGORIES.items():
        for evt in events:
            assert (
                evt in ICONS
            ), f"Event type {evt!r} in category {cat!r} is not in ICONS"


# ── Default categories ──────────────────────────────────────────────────────


def test_default_categories():
    assert get_enabled_categories() == DEFAULT_LOG_CATEGORIES


def test_default_brain_visible():
    assert is_event_visible("llm_thinking")
    assert is_event_visible("llm_response")
    assert is_event_visible("llm_log_file")


def test_default_speech_visible():
    assert is_event_visible("user_speech")
    assert is_event_visible("assistant_speech")
    assert is_event_visible("user_state")


def test_default_proactive_visible():
    assert is_event_visible("proactive_decision")
    assert is_event_visible("proactive_cancelled")


def test_default_comms_visible():
    assert is_event_visible("phone_call_received")
    assert is_event_visible("unify_meet_started")
    assert is_event_visible("sms_received")
    assert is_event_visible("email_received")


def test_default_guidance_visible():
    assert is_event_visible("guidance_received")
    assert is_event_visible("guidance_applied")


def test_default_actor_visible():
    assert is_event_visible("actor_request")
    assert is_event_visible("actor_result")
    assert is_event_visible("stop_requested")


def test_default_ipc_suppressed():
    assert not is_event_visible("ipc")
    assert not is_event_visible("ipc_inbound")
    assert not is_event_visible("ipc_outbound")


def test_default_infra_suppressed():
    assert not is_event_visible("managers_worker")
    assert not is_event_visible("liveview")


def test_default_lifecycle_suppressed():
    assert not is_event_visible("session_start")
    assert not is_event_visible("shutdown")


def test_default_bus_suppressed():
    assert not is_category_enabled("bus")


def test_default_livekit_suppressed():
    assert not is_category_enabled("livekit")


# ── Uncategorized events ────────────────────────────────────────────────────


def test_uncategorized_always_visible():
    """Event types not in any category must always pass through."""
    assert is_event_visible("warning")
    assert is_event_visible("error")
    assert is_event_visible("totally_unknown_future_event")


def test_uncategorized_visible_even_with_narrow_config():
    configure_log_categories({"speech"})
    assert is_event_visible("warning")
    assert is_event_visible("error")


# ── configure_log_categories ────────────────────────────────────────────────


def test_configure_explicit_set():
    configure_log_categories({"brain", "ipc"})
    assert is_event_visible("llm_thinking")
    assert is_event_visible("ipc")
    assert not is_event_visible("user_speech")
    assert not is_event_visible("managers_worker")


def test_configure_all():
    configure_log_categories(set(ALL_LOG_CATEGORIES))
    assert is_event_visible("ipc")
    assert is_event_visible("managers_worker")
    assert is_event_visible("session_start")
    assert is_category_enabled("bus")
    assert is_category_enabled("livekit")


def test_configure_none_resets_to_default():
    configure_log_categories({"ipc"})
    assert is_event_visible("ipc")
    configure_log_categories(None)
    assert not is_event_visible("ipc")
    assert get_enabled_categories() == DEFAULT_LOG_CATEGORIES


# ── get_event_category ──────────────────────────────────────────────────────


def test_get_event_category_known():
    assert get_event_category("llm_thinking") == "brain"
    assert get_event_category("ipc") == "ipc"
    assert get_event_category("managers_worker") == "infra"


def test_get_event_category_unknown():
    assert get_event_category("warning") is None
    assert get_event_category("nonexistent") is None


# ── Terminal filter (CategoryFilter) ────────────────────────────────────────


def test_category_filter_suppresses_bus():
    from unity.logger import _CategoryFilter

    f = _CategoryFilter()
    record = logging.LogRecord(
        "unity",
        logging.INFO,
        "",
        0,
        "Publishing bus event Foo",
        (),
        None,
    )
    assert not f.filter(record)


def test_category_filter_passes_brain_emoji():
    from unity.logger import _CategoryFilter

    f = _CategoryFilter()
    record = logging.LogRecord(
        "unity",
        logging.INFO,
        "",
        0,
        "🤖 LLM response",
        (),
        None,
    )
    assert f.filter(record)


def test_category_filter_suppresses_ipc_emoji():
    from unity.logger import _CategoryFilter

    f = _CategoryFilter()
    record = logging.LogRecord(
        "unity",
        logging.INFO,
        "",
        0,
        "🔌 IPC packet",
        (),
        None,
    )
    assert not f.filter(record)


def test_category_filter_always_passes_warnings():
    from unity.logger import _CategoryFilter

    f = _CategoryFilter()
    record = logging.LogRecord(
        "unity",
        logging.WARNING,
        "",
        0,
        "🔌 IPC warning",
        (),
        None,
    )
    assert f.filter(record)


def test_category_filter_skips_already_checked():
    from unity.logger import _CategoryFilter

    f = _CategoryFilter()
    record = logging.LogRecord(
        "unity",
        logging.INFO,
        "",
        0,
        "🔌 Looks like IPC but pre-checked",
        (),
        None,
    )
    record._category_checked = True  # type: ignore[attr-defined]
    assert f.filter(record)
