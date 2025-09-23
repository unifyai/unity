from dotenv import load_dotenv

load_dotenv()
import os
import asyncio

from unity.conversation_manager_2.conversation_manager import ConversationManager
from unity.conversation_manager_2.comms_manager import CommsManager
from unity.conversation_manager_2.event_broker import get_event_broker


async def main(local: bool = False, project_name: str = "Assistants"):
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
    )

    # listens for events coming from whatsapp, calls, and other media and passes it to the event_broker
    comms_manager = CommsManager(event_broker=event_broker)

    asyncio.create_task(conversation_manager.wait_for_events())
    asyncio.create_task(comms_manager.start())

    print("Server is Running...")
    await stop.wait()


if __name__ == "__main__":
    asyncio.run(main())
