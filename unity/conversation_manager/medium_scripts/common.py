# unity/call_common.py

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Awaitable, Callable, Iterable, Optional

import unify

from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.events import (
    PhoneCallStarted,
    PhoneCallEnded,
    UnifyMeetEnded,
    UnifyMeetStarted,
)
from unity.session_details import SESSION_DETAILS

logger = logging.getLogger(__name__)

# Shared event broker instance
event_broker = get_event_broker()


# ---------------------------------------------------------------------------
# STS (Speech-to-Speech) Usage Tracking
# ---------------------------------------------------------------------------
#
# IMPORTANT: This is a ROUGH HEURISTIC for billing STS calls.
#
# OpenAI's Realtime API has a fundamentally different pricing model than text
# LLMs - it charges per audio minute rather than per token. However, Unify's
# usage tracking system is built around token-based billing.
#
# As a temporary workaround, we estimate token usage from call duration and
# log it against a text-based model (gpt-4@openai) that has similar-ish costs.
#
# Key assumptions (all approximate):
#   - Audio is processed at ~150 tokens/second (OpenAI's internal tokenization)
#   - We assume 50% of call duration is active speech (conservative estimate)
#   - We split usage 50/50 between input (user speech) and output (assistant)
#   - We use gpt-4@openai pricing which OVERESTIMATES actual Realtime API costs
#
# This approach intentionally OVERCHARGES users slightly to ensure we always
# cover our actual costs from OpenAI.
#
# TODO: Replace this heuristic with proper Realtime API usage tracking once
# Unify supports audio-minute billing or OpenAI Realtime pricing endpoints.
# ---------------------------------------------------------------------------

# Configuration for STS usage estimation
_STS_BILLING_ENDPOINT = "gpt-4@openai"  # Use expensive model to overestimate
_STS_TOKENS_PER_SECOND = 150  # Approximate audio tokenization rate
_STS_SPEECH_RATIO = 0.5  # Assume 50% of call is active speech
_STS_INPUT_OUTPUT_SPLIT = 0.5  # Assume roughly equal speaking time


def log_sts_usage(
    call_duration_seconds: float,
    contact: Optional[dict] = None,
    tags: Optional[list[str]] = None,
) -> None:
    """
    Log estimated usage for an STS (Speech-to-Speech) call.

    This is a ROUGH HEURISTIC - see module-level comments for details.
    The billing is intentionally set to OVERESTIMATE actual costs.

    Args:
        call_duration_seconds: Total duration of the call in seconds.
        contact: Optional contact dict for tagging/identification.
        tags: Optional additional tags for the query log.
    """
    if call_duration_seconds <= 0:
        logger.warning("Skipping STS usage logging: call duration <= 0")
        return

    # Estimate active speech time
    speech_seconds = call_duration_seconds * _STS_SPEECH_RATIO

    # Convert to estimated tokens
    total_tokens = int(speech_seconds * _STS_TOKENS_PER_SECOND)

    # Split between input (user) and output (assistant)
    input_tokens = int(total_tokens * _STS_INPUT_OUTPUT_SPLIT)
    output_tokens = total_tokens - input_tokens

    # Build query body for logging
    query_body = {
        "model": _STS_BILLING_ENDPOINT,
        "messages": [
            {"role": "system", "content": "[STS call - usage estimate]"},
            {"role": "user", "content": f"[Audio input: {speech_seconds:.1f}s]"},
        ],
    }

    # Build response body with usage stats
    response_body = {
        "model": _STS_BILLING_ENDPOINT,
        "object": "chat.completion",
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"[Audio output: {speech_seconds:.1f}s]",
                },
                "finish_reason": "stop",
            },
        ],
        # Include metadata for debugging/auditing
        "_sts_metadata": {
            "call_duration_seconds": call_duration_seconds,
            "estimated_speech_seconds": speech_seconds,
            "billing_heuristic_version": "v1",
            "note": "ROUGH ESTIMATE - see code comments for details",
        },
    }

    # Build tags
    all_tags = ["voice", "sts", "realtime", "usage-estimate"]
    if tags:
        all_tags.extend(tags)
    if contact:
        contact_id = contact.get("contact_id") or contact.get("id")
        if contact_id:
            all_tags.append(f"contact:{contact_id}")

    try:
        unify.log_query(
            endpoint=_STS_BILLING_ENDPOINT,
            query_body=query_body,
            response_body=response_body,
            tags=all_tags,
            consume_credits=True,
        )
        logger.info(
            f"Logged STS usage: {call_duration_seconds:.1f}s call → "
            f"{total_tokens} tokens ({input_tokens} in / {output_tokens} out)",
        )
    except Exception as e:
        # Don't let billing failures crash the call cleanup
        logger.error(f"Failed to log STS usage: {e}")


# Default inactivity timeout used by both agents
DEFAULT_INACTIVITY_TIMEOUT = 300  # 5 minutes


# -------- Call lifecycle helpers -------- #


async def publish_call_started(contact: dict, channel: str) -> None:
    event = (
        PhoneCallStarted(contact=contact)
        if channel == "phone"
        else UnifyMeetStarted(contact=contact)
    )
    await event_broker.publish(f"app:comms:{channel}_call_started", event.to_json())


async def publish_call_ended(contact: dict, channel: str) -> None:
    event = (
        PhoneCallEnded(contact=contact)
        if channel == "phone"
        else UnifyMeetEnded(contact=contact)
    )
    await event_broker.publish(f"app:comms:{channel}_call_ended", event.to_json())


def create_end_call(
    contact: dict,
    channel: str,
    pre_shutdown_callback: Optional[Callable[[], None]] = None,
) -> Callable[[], Awaitable[None]]:
    """
    Returns an async function that:
      - calls optional pre_shutdown_callback (e.g., for usage logging)
      - publishes the call ended event
      - cancels all other asyncio tasks

    Args:
        contact: Contact dictionary for the call.
        channel: Channel type ("phone" or other).
        pre_shutdown_callback: Optional sync callback to run before shutdown.
            Useful for logging call usage/metrics before tasks are cancelled.
    """

    async def end_call() -> None:
        print("Initiating graceful shutdown...")

        # Run pre-shutdown callback (e.g., usage logging) before cleanup
        if pre_shutdown_callback is not None:
            try:
                pre_shutdown_callback()
            except Exception as e:  # noqa: BLE001
                print(f"Error in pre-shutdown callback: {e}")

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
               after OUTBOUND that should be stuffed into SESSION_DETAILS.

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

        # Populate SESSION_DETAILS with voice config
        SESSION_DETAILS.voice.provider = (
            sys.argv[3] if sys.argv[3] != "None" else "cartesia"
        )
        SESSION_DETAILS.voice.id = sys.argv[4] if sys.argv[4] != "None" else ""
        SESSION_DETAILS.voice_call.outbound = sys.argv[5] == "True"
        SESSION_DETAILS.voice_call.channel = sys.argv[6]

        # Parse extra args (CONTACT, BOSS, ASSISTANT_BIO)
        for idx, (env_name, is_json) in enumerate(extra_env, start=7):
            value = sys.argv[idx]

            if is_json:
                try:
                    loaded = json.loads(value)
                except json.JSONDecodeError:
                    print(f"{env_name} payload is not valid JSON")
                    sys.exit(1)
                if not loaded:
                    print(f"{env_name} payload is invalid (empty)")
                    sys.exit(1)

            # Map known extra args to SESSION_DETAILS fields
            if env_name == "CONTACT":
                SESSION_DETAILS.voice_call.contact_json = value
            elif env_name == "BOSS":
                SESSION_DETAILS.voice_call.boss_json = value
            elif env_name == "ASSISTANT_BIO":
                SESSION_DETAILS.assistant.about = value

        # Export to env for subprocess inheritance
        SESSION_DETAILS.export_to_env()

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
