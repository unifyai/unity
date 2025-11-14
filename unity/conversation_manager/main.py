from datetime import datetime
import signal
from dotenv import load_dotenv

load_dotenv()
import os
import asyncio

from unity.conversation_manager.comms_manager import CommsManager
from unity.conversation_manager.event_broker import (
    get_event_broker,
    create_event_broker,
)
from unity.conversation_manager.domains.utils import log_task_exc
from unity.conversation_manager.conversation_manager import ConversationManager
from unity.helpers import cleanup_dangling_call_processes


stop = None
conversation_manager = None


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
    # Set the stop event to trigger graceful shutdown in main()
    # This ensures cleanup happens only once, in the main async function
    if stop:
        stop.set()


async def main(use_realtime=False, project_name: str = "Assistants"):
    global conversation_manager, managers_worker, stop

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Ensure Unify traced logging is disabled outside the main thread
    # (avoids ValueError: signal only works in main thread)
    os.environ.setdefault("UNIFY_TRACED", "false")

    # Clean up any dangling call processes from previous runs
    # This prevents conflicts when multiple call processes can't run simultaneously
    print("Checking for dangling call processes from previous runs...")
    cleanup_dangling_call_processes()

    stop = asyncio.Event()

    # passes events around, uses redis
    event_broker = get_event_broker()

    # directly talks with the user
    conversation_manager = ConversationManager(
        event_broker,
        os.getenv("JOB_NAME", ""),
        os.getenv("USER_ID", ""),
        os.getenv("ASSISTANT_ID", ""),
        os.getenv("USER_NAME", ""),
        os.getenv("ASSISTANT_NAME", ""),
        os.getenv("ASSISTANT_AGE", ""),
        os.getenv("ASSISTANT_NATIONALITY", ""),
        os.getenv("ASSISTANT_ABOUT", ""),
        os.getenv("ASSISTANT_NUMBER", ""),
        os.getenv("ASSISTANT_EMAIL", ""),
        os.getenv("USER_NUMBER", ""),
        os.getenv("USER_WHATSAPP_NUMBER", ""),
        os.getenv("USER_EMAIL", ""),
        os.getenv("VOICE_PROVIDER", "cartesia"),
        os.getenv("VOICE_ID", None),
        os.getenv("VOICE_MODE", "tts"),
        project_name=project_name,
        stop=stop,
        user_turn_end_callback=None,
        # whether to use realtime settings or not
        realtime=use_realtime,
    )

    # listens for events coming from whatsapp, calls, and other media and passes it to the event_broker
    comms_manager = CommsManager(event_broker=event_broker)

    asyncio.create_task(conversation_manager.wait_for_events()).add_done_callback(
        log_task_exc
    )
    asyncio.create_task(conversation_manager.check_inactivity())
    if not os.getenv("TEST"):
        asyncio.create_task(comms_manager.start())

    print("Server is Running...")
    await stop.wait()

    print("Cleaning up conversation manager...")
    conversation_manager.cleanup()
    print("Cleanup finished")

    print("Shutdown finished")


if __name__ == "__main__":
    import sys

    use_realtime = "--realtime" in sys.argv
    asyncio.run(main(use_realtime=use_realtime))
