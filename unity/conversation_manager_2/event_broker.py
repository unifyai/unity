import redis.asyncio as redis


redis_broker = None


def get_event_broker():
    global redis_broker

    if redis_broker is None:
        redis_broker = redis.Redis(host="localhost", port=6379, decode_responses=True)
    return redis_broker
