# The event manager job is to listen for events coming from GUI and call processes
# and send llm responses to both / ui updates
# the event manager will accumulate events and trigger an llm call when timeout happens or
# urgent event is sent, and cancel any running llm calls
import asyncio
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
import json
import os
import signal
import traceback

load_dotenv()
from unity.conversation_manager.comms_agent import CommsAgent
from unity.conversation_manager.comms_manager import CommsManager
from unity.constants import ASYNCIO_DEBUG


# globals
conv_context_length = 50
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
        # Start inactivity monitor
        self.inactivity_task = asyncio.create_task(self.check_inactivity())

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

            if event["topic"] == "ping":
                print("ping received - keeping event manager alive")
                continue
            elif event["topic"] == "call_process":
                print("recieved call event")
                # handle messages going to the call process
                # like gen
                try:
                    self.writers["call"].write(
                        (json.dumps(event) + "\n").encode("utf-8")
                    )
                    await self.writers["call"].drain()
                except Exception as e:
                    print(f"Error writing to call process: {e}")
                    traceback.print_exc()
            else:
                if event["topic"] == "startup":
                    self.topic_to_subs[event["event"]["payload"]["user_number"]] = (
                        self.topic_to_subs["tool_use"]
                    )
                    self.topic_to_subs[
                        event["event"]["payload"]["user_whatsapp_number"]
                    ] = self.topic_to_subs["tool_use"]
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
                traceback.print_exc()
                print(str(e))
                print("CALL CLOSED")
                writer.close()
                await writer.wait_closed()
                break

    def publish(self, event):
        # Update activity time when events are published
        self.last_activity_time = asyncio.get_event_loop().time()
        self.events_queue.put_nowait(event)

    async def check_inactivity(self):
        """Monitor for inactivity and shut down gracefully after timeout"""
        while True:
            if self.is_shutting_down:
                break

            await asyncio.sleep(30)  # Check every 30 seconds
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_activity_time > self.INACTIVITY_TIMEOUT:
                print(
                    f"Inactivity timeout reached ({self.INACTIVITY_TIMEOUT}s), "
                    "shutting down gracefully...",
                )
                await self.shutdown_gracefully()
                break

    async def shutdown_gracefully(self):
        """Gracefully shut down the event manager and all components"""
        print("Starting graceful shutdown...")
        self.is_shutting_down = True

        # Signal the global user agent to clean up
        global user_agent
        if user_agent:
            try:
                # Clean up main user agent call process
                user_agent.cleanup()
            except Exception as e:
                print(f"Error during user agent cleanup: {e}")

        # Close all connections
        for writer in self.writers.values():
            if writer and not writer.is_closing():
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception as e:
                    print(f"Error closing writer: {e}")

        # Close servers
        for server in self.servers.values():
            if server:
                try:
                    server.close()
                    await server.wait_closed()
                except Exception as e:
                    print(f"Error closing server: {e}")

        print("Graceful shutdown completed")

        # Exit the application
        os._exit(0)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        + " - [MAIN.PY] Received signal "
        + str(signum)
        + ", shutting down gracefully...",
    )

    # Clean up any running call processes
    global user_agent
    if user_agent:
        # Clean up main user agent call process
        user_agent.cleanup()


def loop_exception_handler(loop, context):
    print("Error:", context.get("message"), context.get("exception"))


async def main(
    start_local: bool = False,
    enabled_tools: list | str | None = "conductor",
    project_name: str = "Assistants",
):
    global user_agent

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    event_manager = EventManager()
    user_agent = CommsAgent(
        os.getenv("JOB_NAME", ""),
        os.getenv("USER_ID", ""),
        os.getenv("ASSISTANT_ID", ""),
        os.getenv("USER_NAME", ""),
        os.getenv("ASSISTANT_NAME", ""),
        os.getenv("ASSISTANT_AGE", ""),
        os.getenv("ASSISTANT_REGION", ""),
        os.getenv("ASSISTANT_ABOUT", ""),
        os.getenv("ASSISTANT_NUMBER", ""),
        os.getenv("ASSISTANT_EMAIL", ""),
        os.getenv("USER_NUMBER", ""),
        os.getenv("USER_WHATSAPP_NUMBER", ""),
        os.getenv("USER_EMAIL", ""),
        os.getenv("TTS_PROVIDER", "cartesia"),
        os.getenv("VOICE_ID", None),
        conv_context_length=conv_context_length,
        start_local=start_local,
        enabled_tools=enabled_tools,
        project_name=project_name,
    )
    user_agent.set_event_manager(event_manager)
    user_agent.subscribe(
        [
            os.getenv("USER_NUMBER", ""),
            os.getenv("USER_WHATSAPP_NUMBER", ""),
            "tool_use",
            "startup",
        ]
    )

    # Initialize Redis connection (waits for Redis to be ready)
    print("Initializing Redis connection...")
    await user_agent.initialize_redis()
    print("Redis connection initialized successfully")

    comms_manager = CommsManager(events_queue=event_manager.events_queue)
    event_manager_task = asyncio.create_task(event_manager.serve())
    asyncio.create_task(comms_manager.start())
    asyncio.create_task(user_agent.listen_for_events())
    await event_manager_task


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Mutually exclusive group for enabling or disabling tools
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--enabled-tools",
        dest="enabled_tools",
        type=lambda s: [t.strip() for t in s.split(",")],
        default=["conductor"],
        help="Comma-separated list of enabled tools with choices of conductor, contact, transcript, knowledge, scheduler. Default: conductor",
    )
    group.add_argument(
        "--no-tools",
        dest="enabled_tools",
        action="store_const",
        const=None,
        help="Disable all tool-based actions",
    )
    parser.add_argument(
        "--start-local",
        dest="start_local",
        action="store_true",
        default=False,
        help="Start local GUI instead of server",
    )
    parser.add_argument(
        "--project-name",
        dest="project_name",
        type=str,
        default="Assistants",
        help="Name of the project to use",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            start_local=args.start_local,
            enabled_tools=args.enabled_tools,
            project_name=args.project_name,
        ),
        debug=ASYNCIO_DEBUG,
    )
