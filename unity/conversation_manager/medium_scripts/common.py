# unity/call_common.py

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Awaitable, Callable, Iterable

from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.events import (
    PhoneCallStarted,
    PhoneCallEnded,
    UnifyCallEnded,
    UnifyCallStarted,
)

# Shared event broker instance
event_broker = get_event_broker()

# Default inactivity timeout used by both agents
DEFAULT_INACTIVITY_TIMEOUT = 300  # 5 minutes


# -------- Call lifecycle helpers -------- #


async def publish_call_started(contact: dict, channel: str) -> None:
    event = (
        PhoneCallStarted(contact=contact)
        if channel == "phone"
        else UnifyCallStarted(contact=contact)
    )
    await event_broker.publish(f"app:comms:{channel}_call_started", event.to_json())


async def publish_call_ended(contact: dict, channel: str) -> None:
    event = (
        PhoneCallEnded(contact=contact)
        if channel == "phone"
        else UnifyCallEnded(contact=contact)
    )
    await event_broker.publish(f"app:comms:{channel}_call_ended", event.to_json())


def create_end_call(contact: dict, channel: str) -> Callable[[], Awaitable[None]]:
    """
    Returns an async function that:
      - publishes the call ended event
      - cancels all other asyncio tasks
    """

    async def end_call() -> None:
        print("Initiating graceful shutdown...")

        # Send end call event before cleaning tasks and closing connection
        await publish_call_ended(contact, channel)
        print("End call event sent")

        # Get all running tasks except current task
        tasks: Iterable[asyncio.Task] = [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task()
        ]

        if tasks:
            print(f"Cancelling {len(tasks)} running tasks...")
            # Cancel all tasks
            for task in tasks:
                task.cancel()

            # Wait for tasks to be cancelled gracefully
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                print("All tasks cancelled successfully")
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                print(f"Error during task cancellation: {e}")

        print("Graceful shutdown completed")

    return end_call


def setup_participant_disconnect_handler(room, end_call: Callable[[], Awaitable[None]]):
    """
    Registers a participant_disconnected handler that triggers end_call().
    """

    def on_participant_disconnected(*args, **kwargs):  # noqa: ANN001, ANN002
        asyncio.create_task(end_call())

    room.on("participant_disconnected", on_participant_disconnected)


def setup_inactivity_timeout(
    end_call: Callable[[], Awaitable[None]],
    timeout: float = DEFAULT_INACTIVITY_TIMEOUT,
) -> Callable[[], None]:
    """
    Starts an inactivity watchdog and returns a `touch()` function.

    Call the returned function whenever there is user/assistant activity
    that should reset the inactivity timer.
    """
    loop = asyncio.get_event_loop()
    state = {"last_activity": loop.time()}

    async def check_inactivity():
        while True:
            await asyncio.sleep(10)
            current_time = loop.time()
            if current_time - state["last_activity"] > timeout:
                print("Inactivity timeout reached, shutting down agent...")
                await end_call()
                break

    asyncio.create_task(check_inactivity())

    def touch() -> None:
        state["last_activity"] = loop.time()

    return touch


# -------- CLI / env helpers -------- #


def configure_from_cli(
    extra_env: list[tuple[str, bool]],
) -> str:
    """
    Shared CLI argument handling for both call scripts.

    extra_env: list of (ENV_NAME, is_json) describing additional arguments
               after OUTBOUND that should be stuffed into env vars.

    Layout (common to both scripts):
      argv[0] = script name
      argv[1] = "dev" | "connect" | "download-files"
      argv[2] = assistant_number
      argv[3] = VOICE_PROVIDER
      argv[4] = VOICE_ID
      argv[5] = OUTBOUND
      argv[6...] = extra_env[...]

    Returns the computed agent_name ("unity_<assistant_number>").
    """
    assistant_number = ""
    agent_name = ""
    room_name = ""
    print("sys.argv", sys.argv)

    # max index used = 6 + len(extra_env)
    required_len = 6 + len(extra_env)
    if len(sys.argv) > required_len:
        assistant_number = sys.argv[2]
        if " " in assistant_number:
            agent_name, room_name = assistant_number.split(":")
        else:
            agent_name = f"unity_{assistant_number}"
            room_name = agent_name
        os.environ["VOICE_PROVIDER"] = (
            sys.argv[3] if sys.argv[3] != "None" else "cartesia"
        )
        os.environ["VOICE_ID"] = sys.argv[4] if sys.argv[4] != "None" else ""
        os.environ["OUTBOUND"] = sys.argv[5]
        os.environ["CHANNEL"] = sys.argv[6]

        # extra env vars
        for idx, (env_name, is_json) in enumerate(extra_env, start=7):
            value = sys.argv[idx]
            os.environ[env_name] = value

            if is_json:
                try:
                    loaded = json.loads(value)
                except json.JSONDecodeError:
                    print(f"{env_name} payload is not valid JSON")
                    sys.exit(1)
                if not loaded:
                    print(f"{env_name} payload is invalid (empty)")
                    sys.exit(1)

        # keep only script name and the command ("dev" / "connect" / "download-files")
        sys.argv = sys.argv[:2]
    elif len(sys.argv) > 1 and sys.argv[1] != "download-files":
        print("Not enough arguments provided")
        sys.exit(1)

    return agent_name, room_name


def should_dispatch_agent() -> bool:
    """
    True when we should actually call dispatch_agent() for this process.
    """
    return len(sys.argv) > 1 and sys.argv[1] != "download-files"
