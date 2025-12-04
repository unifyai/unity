import os

import redis.asyncio as redis


redis_broker = None


def _get_redis_port() -> int:
    """Get Redis port from environment or default to 6379."""
    return int(os.environ.get("REDIS_PORT", "6379"))


def get_event_broker():
    global redis_broker

    if redis_broker is None:
        redis_broker = redis.Redis(
            host="localhost",
            port=_get_redis_port(),
            decode_responses=True,
        )
    return redis_broker


def create_event_broker():
    """
    Create and return a new Redis asyncio client instance. Use this when
    you need a separate client for a different asyncio event loop or thread.
    """
    return redis.Redis(
        host="localhost",
        port=_get_redis_port(),
        decode_responses=True,
    )
