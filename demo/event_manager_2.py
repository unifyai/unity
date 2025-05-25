# The event manager job is to listen for events coming from GUI and call processes
# and send llm responses to both / ui updates
# the event manager will accumulate events and trigger an llm call when timeout happens or
# urgent event is sent, and cancel any running llm calls
import os
import asyncio
import json
from collections import defaultdict

import openai
from dotenv import load_dotenv

load_dotenv()

from comms_agent import CommsAgent
from comms_manager_2 import CommsManager

# globals


class EventManager:
    def __init__(self):
        self.servers = {}
        self.readers = {}
        self.writers: dict[str, asyncio.StreamWriter] = {}
        self.topic_to_subs = defaultdict(list)
        
        self.events_queue = asyncio.Queue()


    async def serve(self):
        self.servers["call"] = await asyncio.start_server(
            self.handle_call_client,
            "127.0.0.1",
            8090,
        )

        self.event_aggregator_task = asyncio.create_task(self.collect_events())
        async with self.servers["call"]:
            await self.servers["call"].serve_forever()
        
    
    async def collect_events(self):
        print("collecting...")
        while True:
            print(self.topic_to_subs)
            event = await self.events_queue.get()
            print(event)
            if event["topic"] == "call_process":
                print("recieved call event")
                # handle messages going to the call process
                # like gen
                self.writers["call"].write((json.dumps(event) + "\n").encode("utf-8"))
                await self.writers["call"].drain()
            else:
                for client in self.topic_to_subs[event["topic"]]:
                    client.handle_event(event)

    async def handle_call_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        self.readers["call"] = reader
        self.writers["call"] = writer

        print("Call connected")
        while True:
            try:
                raw = await reader.readline()
                if not raw:
                    break
                msg = json.loads(raw.decode())
                self.events_queue.put_nowait(msg)
            except Exception as e:
                print(str(e))
                print("CALL CLOSED")
                writer.close()
                await writer.wait_closed()

    def publish(self, event):
        self.events_queue.put_nowait(event)


def loop_exception_handler(loop, context):
    print("Error:", context.get("message"), context.get("exception"))

async def main():
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(loop_exception_handler)
    event_manager = EventManager()
    user_agent = CommsAgent("Yasser", "+17343611691", "+201092074828", "+12697784020", [], True)
    user_agent.set_event_manager(event_manager)
    user_agent.subscribe(["+201092074828", "+12697784020"])
    comms_manager = CommsManager(events_queue=event_manager.events_queue)
    event_manager_task = asyncio.create_task(event_manager.serve())
    comms_task = asyncio.create_task(comms_manager.start())
    user_manager_task = asyncio.create_task(user_agent.listen_for_events())
    await event_manager_task

if __name__ == "__main__":
    asyncio.run(main())
