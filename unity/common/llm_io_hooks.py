"""
LLM Cache Statistics Tracking
=============================

Provides global cache hit/miss statistics for LLM calls.

Statistics are captured using the ``unillm`` cache event context managers
(:pyfunc:`capture_cache_events` and :pyfunc:`acapture_cache_events`).

Usage:
    from unity.common.llm_io_hooks import get_cache_stats, reset_cache_stats

    # Get current statistics
    stats = get_cache_stats()
    print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")

    # Reset counters
    reset_cache_stats()
"""

from __future__ import annotations


# Global cache statistics
_CACHE_HITS = 0
_CACHE_MISSES = 0


def get_cache_stats() -> dict[str, int | float]:
    """Get cache hit/miss statistics for LLM calls.

    Returns a dict with:
        - hits: Number of cache hits
        - misses: Number of cache misses
        - hit_rate: Percentage of hits (0.0 if no calls)
    """
    total = _CACHE_HITS + _CACHE_MISSES
    hit_rate = (_CACHE_HITS / total * 100) if total > 0 else 0.0
    return {
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "hit_rate": hit_rate,
    }


def reset_cache_stats() -> None:
    """Reset the cache statistics counters to zero."""
    global _CACHE_HITS, _CACHE_MISSES
    _CACHE_HITS = 0
    _CACHE_MISSES = 0


def record_cache_status(cache_status: str) -> None:
    """Record a cache hit or miss in the global stats.

    This function is called automatically when using the context-aware
    cache tracking wrappers in the async tool loop.

    Args:
        cache_status: The cache status ("hit" or "miss"). Other values are ignored.
    """
    global _CACHE_HITS, _CACHE_MISSES
    if cache_status == "hit":
        _CACHE_HITS += 1
    elif cache_status == "miss":
        _CACHE_MISSES += 1
    # Ignore "unknown" or other statuses
