# The event manager job is to listen for events coming from GUI and call processes
# and send llm responses to both / ui updates
# the event manager will accumulate events and trigger an llm call when timeout happens or
# urgent event is sent, and cancel any running llm calls
import asyncio
from collections import defaultdict
from dotenv import load_dotenv
import json
import logging
import os
import signal
from typing import Optional
import unify

load_dotenv()
LG = logging.getLogger("contact_manager_integration")

from unity.contact_manager.contact_manager import ContactManager
from comms_agent import CommsAgent
from comms_manager import CommsManager
from scenario_builder import ScenarioBuilder
from scenario_store import ScenarioStore

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
                    f"Inactivity timeout reached ({self.INACTIVITY_TIMEOUT}s), shutting down gracefully...",
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

                # Clean up all comm agents' call processes
                if hasattr(user_agent, "contact_num_to_comm_agent"):
                    for comm_agent in user_agent.contact_num_to_comm_agent.values():
                        comm_agent.cleanup()
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
    print(f"Received signal {signum}, shutting down gracefully...")

    # Clean up any running call processes
    global user_agent
    if user_agent:
        # Clean up main user agent call process
        user_agent.cleanup()

        # Clean up all comm agents' call processes
        if hasattr(user_agent, "contact_num_to_comm_agent"):
            for comm_agent in user_agent.contact_num_to_comm_agent.values():
                comm_agent.cleanup()


def loop_exception_handler(loop, context):
    print("Error:", context.get("message"), context.get("exception"))


async def _build_scenario(
    custom: Optional[str] = None,
) -> Optional[str]:
    """
    Populate the contact store **through the official tools** using
    :class:`ScenarioBuilder`.  Falls back to the fixed seed on any error.
    """
    cm = ContactManager()
    description = (
        custom.strip()
        if custom
        else (
            "Generate 10 realistic business contacts across EMEA, APAC and AMER. "
            "Each contact needs first_name, surname, email_address and phone_number. "
            "Also create custom columns with varying industries and locations."
        )
    )
    description += (
        "\nTry to get as much done as you can with each `update` and `ask` call. "
        "They can deal with complex multi-step requests just fine."
    )

    builder = ScenarioBuilder(
        description=description,
        tools={  # expose only the public surface
            "update": cm.update,
            "ask": cm.ask,  # allows the LLM to check for duplicates if it wishes
        },
    )

    try:
        await builder.create()
    except Exception as exc:
        raise (f"LLM seeding via ScenarioBuilder failed. {exc}")

    # The new flow doesn't produce a structured "theme"; preserve signature.
    return None


async def main():
    # prepare Unify context
    unify.activate("ContactManagerIntegration")
    unify.set_trace_context("Traces")
    ctxs = unify.get_contexts()
    if "Contacts" in ctxs:
        unify.delete_context("Contacts")
    unify.create_context("Contacts")
    if "Traces" in ctxs:
        unify.delete_context("Traces")
    unify.create_context("Traces")

    # logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LG.setLevel(logging.INFO)

    # manager & transcript vault
    store = ScenarioStore()

    # Obtain the transcript that seeds the scenario
    scenario_text: Optional[str] = None
    scenario_text = store.get(-1)
    LG.info(f"[seed] loaded transcript {scenario_text}")

    LG.info("[seed] building synthetic contacts – this can take 20-40 s…")
    await _build_scenario(scenario_text)
    LG.info("[seed] done.")

    store.save_named("default", scenario_text)
    LG.info(f"[seed] transcript saved as {scenario_text}.")

    global user_agent

    loop = asyncio.get_running_loop()
    # loop.set_exception_handler(loop_exception_handler)

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    event_manager = EventManager()
    user_agent = CommsAgent(
        os.getenv("USER_NAME", ""),
        os.getenv("ASSISTANT_NUMBER", ""),
        os.getenv("USER_NUMBER", ""),
        os.getenv("USER_PHONE_NUMBER", ""),
        None,
        [],
        True,
    )
    user_agent.set_event_manager(event_manager)
    user_agent.subscribe(
        [
            os.getenv("USER_NUMBER", ""),
            os.getenv("USER_PHONE_NUMBER", ""),
            "user_agent",
        ],
    )
    comms_manager = CommsManager(events_queue=event_manager.events_queue)
    event_manager_task = asyncio.create_task(event_manager.serve())
    comms_task = asyncio.create_task(comms_manager.start())
    user_manager_task = asyncio.create_task(user_agent.listen_for_events())
    await event_manager_task


if __name__ == "__main__":
    asyncio.run(main())
