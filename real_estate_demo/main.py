# The event manager job is to listen for events coming from GUI and call processes
# and send llm responses to both / ui updates
# the event manager will accumulate events and trigger an llm call when timeout happens or
# urgent event is sent, and cancel any running llm calls
import os
import asyncio
import json
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

from comms_agent import CommsAgent

# globals
user_agent = None


class EventManager:
    def __init__(self):
        self.servers = {}
        self.readers = {}
        self.writers: dict[str, asyncio.StreamWriter] = {}
        self.topic_to_subs = defaultdict(set)

        self.events_queue = asyncio.Queue()

        # Inactivity timeout management
        self.INACTIVITY_TIMEOUT = 360  # 6 minutes in seconds
        self.last_activity_time = asyncio.get_event_loop().time()
        self.is_shutting_down = False

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
            if self.is_shutting_down:
                break

            # print(self.topic_to_subs)
            event = await self.events_queue.get()
            print("EVENT MANAGER:", event)

            # Update activity time on any event
            self.last_activity_time = asyncio.get_event_loop().time()

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
            if self.is_shutting_down:
                break
            try:
                raw = await reader.readline()
                if not raw:
                    break
                msg = json.loads(raw.decode())
                # Update activity time on any message from call client
                self.last_activity_time = asyncio.get_event_loop().time()
                self.events_queue.put_nowait(msg)
            except Exception as e:
                print(str(e))
                print("CALL CLOSED")
                writer.close()
                await writer.wait_closed()
                break

    def publish(self, event):
        self.events_queue.put_nowait(event)


def loop_exception_handler(loop, context):
    print("Error:", context.get("message"), context.get("exception"))


async def main():
    global user_agent

    event_manager = EventManager()
    user_agent = CommsAgent(
        user_name=os.getenv("USER_NAME", ""),
        past_events=[],
        main_user_agent=True,
    )
    user_agent.set_event_manager(event_manager)
    user_agent.subscribe(
        [
            "user_call_step",
            "user_agent",
        ],
    )
    event_manager_task = asyncio.create_task(event_manager.serve())
    user_manager_task = asyncio.create_task(user_agent.listen_for_events())
    await event_manager_task


if __name__ == "__main__":
    asyncio.run(main())
