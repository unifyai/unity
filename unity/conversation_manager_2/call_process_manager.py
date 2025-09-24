import json

import redis.asyncio as redis


class CallProcessManager:
    def __init__(self, event_broker):
        self.event_broker: redis.Redis = event_broker

    async def wait_for_events(self):
        async with self.event_broker.pubsub() as pubsub:
            sub = await pubsub.subscribe("app:call_process")
            async for msg in sub.listen():
                if msg["type"] == "message":
                    msg = json.loads(msg["data"])
                    if msg["event_name"] == "start_call":
                        # start call
                        self.event_broker.publish(...)
