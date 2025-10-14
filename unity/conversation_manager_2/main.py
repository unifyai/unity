from datetime import datetime
import signal
from dotenv import load_dotenv

load_dotenv()
import os
import asyncio

from unity.conversation_manager_2.conversation_manager import ConversationManager
from unity.conversation_manager_2.comms_manager import CommsManager
from unity.conversation_manager_2.managers_worker import ManagersWorker
from unity.conversation_manager_2.event_broker import (
    get_event_broker,
    create_event_broker,
)


stop = None
conversation_manager = None
managers_worker = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global conversation_manager, managers_worker, stop

    print(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        + " - [MAIN.PY] Received signal "
        + str(signum)
        + ", shutting down gracefully...",
    )
    if conversation_manager:
        print("Cleaning up conversation manager...")
        conversation_manager.cleanup()
        print("Cleanup finished")
    if managers_worker:
        managers_worker.stop()


async def main(local: bool = False, project_name: str = "Assistants"):
    global conversation_manager, managers_worker, stop

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Ensure Unify traced logging is disabled outside the main thread
    # (avoids ValueError: signal only works in main thread)
    os.environ.setdefault("UNIFY_TRACED", "false")

    stop = asyncio.Event()

    # passes events around, uses redis
    event_broker = get_event_broker()

    # Initialize ManagersWorker
    managers_worker = ManagersWorker()

    # Run ManagersWorker on a background thread via asyncio.to_thread
    def run_managers_worker():
        # Also enforce UNIFY_TRACED=false in the worker thread
        os.environ["UNIFY_TRACED"] = "false"
        # Create a fresh Redis client bound to the thread's event loop
        managers_worker._event_broker = create_event_broker()
        asyncio.run(managers_worker.wait_for_events())

    asyncio.create_task(asyncio.to_thread(run_managers_worker))

    # directly talks with the user
    conversation_manager = ConversationManager(
        event_broker,
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
        os.getenv("VOICE_PROVIDER", "cartesia"),
        os.getenv("VOICE_ID", None),
        project_name=project_name,
        stop=stop,
        user_turn_end_callback=None,
    )

    # listens for events coming from whatsapp, calls, and other media and passes it to the event_broker
    comms_manager = CommsManager(event_broker=event_broker)

    asyncio.create_task(conversation_manager.wait_for_events())
    asyncio.create_task(conversation_manager.check_inactivity())
    asyncio.create_task(comms_manager.start())

    print("Server is Running...")
    await stop.wait()

    print("Cleaning up conversation manager...")
    conversation_manager.cleanup()
    print("Cleanup finished")

    print("Shutting down managers worker...")
    managers_worker.stop()
    print("Shutdown finished")


if __name__ == "__main__":
    asyncio.run(main())
