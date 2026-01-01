"""
Tests for LLM cache statistics tracking.

These tests verify the cache statistics tracking functionality that records
cache hits/misses from LLM calls.
"""

from __future__ import annotations

from unity.common.llm_io_hooks import (
    get_cache_stats,
    reset_cache_stats,
    record_cache_status,
)


# --------------------------------------------------------------------------- #
#  Cache stats tests
# --------------------------------------------------------------------------- #


def test_get_cache_stats_initial():
    """Cache stats should track hits and misses."""
    # Reset to known state
    reset_cache_stats()

    stats = get_cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["hit_rate"] == 0.0


def test_record_cache_status_hit():
    """Recording a hit increments the hit counter."""
    reset_cache_stats()

    record_cache_status("hit")
    record_cache_status("hit")

    stats = get_cache_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 0
    assert stats["hit_rate"] == 100.0


def test_record_cache_status_miss():
    """Recording a miss increments the miss counter."""
    reset_cache_stats()

    record_cache_status("miss")
    record_cache_status("miss")
    record_cache_status("miss")

    stats = get_cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 3
    assert stats["hit_rate"] == 0.0


def test_record_cache_status_mixed():
    """Mixed hits and misses calculate correct hit rate."""
    reset_cache_stats()

    record_cache_status("hit")
    record_cache_status("miss")
    record_cache_status("hit")
    record_cache_status("miss")

    stats = get_cache_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 2
    assert stats["hit_rate"] == 50.0


def test_record_cache_status_unknown_ignored():
    """Unknown cache status is ignored."""
    reset_cache_stats()

    record_cache_status("hit")
    record_cache_status("unknown")
    record_cache_status("miss")

    stats = get_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_reset_cache_stats():
    """Resetting cache stats clears all counters."""
    record_cache_status("hit")
    record_cache_status("miss")

    reset_cache_stats()

    stats = get_cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
