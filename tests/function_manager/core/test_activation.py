"""Symbolic: the memory-trace arithmetic behind function-store retrieval.

Each test pins one property of the design, not a magic number: the newborn
grace, rhythm-relative decay, the similarity cap, scope dropout, legacy-row
tolerance, and supersede inheritance. Settings are constructed explicitly so
a deliberate retuning edits the test alongside the setting — never silently.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from unify.function_manager.activation import (
    ActivationSettings,
    activation,
    in_scope,
    merged_usage,
    rank_score,
)

NOW = datetime(2026, 8, 22, 12, 0, 0, tzinfo=timezone.utc)
S = ActivationSettings()


def _calls(*days_ago: float) -> list[str]:
    return [(NOW - timedelta(days=d)).isoformat() for d in days_ago]


def test_newborn_grace_gives_fresh_functions_standing() -> None:
    """A just-created, never-called function surfaces (creation is a use)."""
    a = activation(
        now=NOW,
        created_at=NOW - timedelta(hours=1),
        call_count=0,
        settings=S,
    )
    assert a >= S.newborn_floor * 0.9
    assert in_scope(a, S)


def test_newborn_grace_is_not_immortality() -> None:
    """The same decay that retires veterans retires an unused newborn."""
    a = activation(
        now=NOW,
        created_at=NOW - timedelta(days=365),
        call_count=0,
        settings=S,
    )
    assert not in_scope(a, S)


def test_use_strengthens_standing() -> None:
    fresh_veteran = activation(
        now=NOW,
        created_at=NOW - timedelta(days=100),
        call_count=15,
        recent_calls=_calls(1, 8, 15, 22),
        settings=S,
    )
    fresh_newbie = activation(
        now=NOW,
        created_at=NOW - timedelta(days=1),
        call_count=0,
        settings=S,
    )
    assert fresh_veteran > fresh_newbie


def test_rhythm_relative_decay_spares_the_quarterly_function() -> None:
    """Dormancy is judged against a function's own cadence.

    Sixty days silent: fatal for a daily-rhythm skill, unremarkable for a
    quarterly one — the same wall-clock gap, opposite verdicts.
    """
    quarterly = activation(
        now=NOW,
        created_at=NOW - timedelta(days=400),
        call_count=4,
        recent_calls=_calls(60, 150, 240, 330),
        settings=S,
    )
    daily = activation(
        now=NOW,
        created_at=NOW - timedelta(days=400),
        call_count=4,
        recent_calls=_calls(60, 61, 62, 63),
        settings=S,
    )
    assert in_scope(quarterly, S)
    assert not in_scope(daily, S)
    assert quarterly > daily


def test_similarity_dominates_ranking() -> None:
    """The activation term is capped: a much better semantic match beats a
    much better-established mediocre one — standing is a tiebreaker."""
    strong_match_no_standing = rank_score(0.9, 0.0, S)
    weak_match_full_standing = rank_score(0.6, 1.0, S)
    assert strong_match_no_standing > weak_match_full_standing


def test_standing_breaks_ties() -> None:
    used = rank_score(0.8, 0.9, S)
    dusty = rank_score(0.8, 0.1, S)
    assert used > dusty


def test_legacy_rows_without_trace_surface_at_newborn_floor() -> None:
    """Pre-activation libraries keep working until usage data accrues."""
    a = activation(
        now=NOW,
        created_at=None,
        call_count=0,
        settings=S,
    )
    assert a == S.newborn_floor
    assert in_scope(a, S)


def test_supersede_inherits_standing() -> None:
    old = {
        "calls": 12,
        "last_called_at": (NOW - timedelta(days=2)).isoformat(),
        "recent_calls": _calls(2, 9, 16),
        "search_hits": 40,
    }
    merged = merged_usage(old, {"calls": 1, "recent_calls": _calls(0.5)}, S)
    assert merged["calls"] == 13
    assert merged["search_hits"] == 40
    assert len(merged["recent_calls"]) == 4
    # The replacement stands where the predecessor stood: recent last-call.
    a = activation(
        now=NOW,
        created_at=NOW - timedelta(days=300),
        call_count=merged["calls"],
        recent_calls=merged["recent_calls"],
        settings=S,
    )
    assert in_scope(a, S)


def test_recent_calls_bounded_by_setting() -> None:
    many = {"calls": 100, "recent_calls": _calls(*range(1, 60))}
    merged = merged_usage(many, None, S)
    assert len(merged["recent_calls"]) == S.recent_calls_kept
