from datetime import datetime
import signal
from dotenv import load_dotenv

load_dotenv()
import os
import asyncio

from unity.session_details import SESSION_DETAILS
from unity.conversation_manager import debug_logger
from unity.conversation_manager.comms_manager import CommsManager
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.domains import comms_utils, managers_utils
from unity.conversation_manager.domains.event_handlers import EventHandler
from unity.conversation_manager.domains.utils import log_task_exc
from unity.conversation_manager.conversation_manager import ConversationManager
from unity.conversation_manager.events import SummarizeContext
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
    # Set the stop event to trigger graceful shutdown in main()
    # This ensures cleanup happens only once, in the main async function
    if stop:
        stop.set()


async def main(project_name: str = "Assistants"):
    global conversation_manager, managers_worker, stop

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Clean up any dangling call processes from previous runs
    # This prevents conflicts when multiple call processes can't run simultaneously
    print("Checking for dangling call processes from previous runs...")
    cleanup_dangling_call_processes()

    stop = asyncio.Event()

    # passes events around, uses redis
    event_broker = get_event_broker()

    # directly talks with the user
    # Use values from SESSION_DETAILS if already populated, otherwise defaults
    conversation_manager = ConversationManager(
        event_broker,
        os.getenv("JOB_NAME", ""),
        SESSION_DETAILS.user.id,
        SESSION_DETAILS.assistant.id,
        SESSION_DETAILS.user.name,
        SESSION_DETAILS.assistant.name,
        SESSION_DETAILS.assistant.age,
        SESSION_DETAILS.assistant.nationality,
        SESSION_DETAILS.assistant.about,
        SESSION_DETAILS.assistant.number,
        SESSION_DETAILS.assistant.email,
        SESSION_DETAILS.user.number,
        SESSION_DETAILS.user.whatsapp_number,
        SESSION_DETAILS.user.email,
        SESSION_DETAILS.voice.provider,
        SESSION_DETAILS.voice.id or None,
        SESSION_DETAILS.voice.mode,
        project_name=project_name,
        stop=stop,
        user_turn_end_callback=None,
    )

    # Monkeypatch functions/methods for testing
    if os.getenv("TEST"):

        def _sync_mock_success(*args, **kwargs):
            return {"success": True}

        async def _async_mock_success(*args, **kwargs):
            return {"success": True}

        comms_utils.send_sms_message_via_number = _async_mock_success
        comms_utils.send_unify_message = _async_mock_success
        comms_utils.send_email_via_address = _async_mock_success
        comms_utils.start_call = _async_mock_success
        conversation_manager.call_manager.start_call = _sync_mock_success
        conversation_manager.call_manager.start_unify_call = _sync_mock_success
        conversation_manager.schedule_proactive_speech = _async_mock_success
        debug_logger.log_job_startup = _sync_mock_success
        debug_logger.mark_job_done = _sync_mock_success
        managers_utils.log_message = _async_mock_success
        managers_utils.publish_bus_events = _async_mock_success
        EventHandler._registry[SummarizeContext] = _async_mock_success

    # listens for events coming from whatsapp, calls, and other media and passes it to the event_broker
    comms_manager = CommsManager(event_broker=event_broker)

    asyncio.create_task(conversation_manager.wait_for_events()).add_done_callback(
        log_task_exc,
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
    asyncio.run(main())
