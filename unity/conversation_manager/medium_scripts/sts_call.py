from __future__ import annotations

import asyncio
import logging
import json
import time
from typing import AsyncIterable

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    FunctionTool,
    JobContext,
    llm,
    ModelSettings,
    RoomInputOptions,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from livekit import agents

load_dotenv()

# OpenAI Realtime API (voice-to-voice) requires livekit-plugins-openai;
# not available through unify due to different API architecture.
try:
    from livekit.plugins.openai import realtime as openai_realtime
except ImportError:
    openai_realtime = None

from unity.conversation_manager.utils import dispatch_livekit_agent
from unity.conversation_manager.events import *
from unity.conversation_manager.prompt_builders import build_voice_agent_prompt
from unity.session_details import SESSION_DETAILS

# Shared helpers
from unity.conversation_manager.medium_scripts.common import (
    event_broker,
    create_end_call,
    setup_inactivity_timeout,
    setup_participant_disconnect_handler,
    publish_call_started,
    configure_from_cli,
    should_dispatch_livekit_agent,
    log_sts_usage,
    start_event_broker_receive,
)

logger = logging.getLogger("gpt-realtime-agent")
logger.setLevel(logging.INFO)


class Assistant(Agent):
    def __init__(
        self,
        contact: dict,
        boss: dict,
        instructions: str = "",
        outbound: bool = False,
    ) -> None:
        self.contact = contact
        self.boss = boss
        self.call_received = not outbound

        super().__init__(instructions=instructions)

    def set_call_received(self) -> None:
        self.call_received = True

    async def on_user_turn_completed(self, turn_ctx, new_message):
        """
        Hook called when user finishes speaking.

        Note: User utterance publishing is handled by _on_chat_item_added
        to keep all transcript logging in one place alongside assistant utterances.
        """
        print(f"[on_user_turn_completed] {new_message.text_content}")

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        print("waiting for call to be received...")
        while not self.call_received:
            await asyncio.sleep(0.1)
        print("call received")

        print("running llm node...")
        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            yield chunk


async def entrypoint(ctx: JobContext) -> None:
    print("Connecting to room...")
    await ctx.connect()
    print("Connected to room")

    # Flag for call_answered that may arrive during initialization
    call_answered_flag = asyncio.Event()

    # Start receiving events from parent (callbacks registered later)
    await start_event_broker_receive()

    # Populate SESSION_DETAILS from environment (set by configure_from_cli)
    SESSION_DETAILS.populate_from_env()

    # Read config from SESSION_DETAILS
    voice_provider = SESSION_DETAILS.voice.provider
    voice_id = SESSION_DETAILS.voice.id
    outbound = SESSION_DETAILS.voice_call.outbound
    channel = SESSION_DETAILS.voice_call.channel
    assistant_bio = SESSION_DETAILS.assistant.about
    print("voice_provider", voice_provider)
    print("voice_id", voice_id)
    print("outbound", outbound)
    print("channel", channel)

    # Contact/boss payloads from SESSION_DETAILS
    contact = json.loads(SESSION_DETAILS.voice_call.contact_json or "{}")
    boss = json.loads(SESSION_DETAILS.voice_call.boss_json or "{}")

    # Configure the OpenAI Realtime model.
    # This requires the livekit-plugins-openai package with OpenAI's Realtime API.
    if openai_realtime is None:
        raise RuntimeError(
            "OpenAI Realtime API is required for voice_mode='sts' (speech-to-speech). "
            "This API is not available through unify. Install livekit-plugins-openai "
            "and configure OPENAI_API_KEY to use realtime voice mode.",
        )

    llm_model = openai_realtime.RealtimeModel(
        model="gpt-realtime",
        voice=voice_id,
        modalities=["audio"],
    )

    session = AgentSession(
        llm=llm_model,
        # If you prefer a separate TTS instead of Realtime audio,
        # set llm.modalities=["text"] above and provide a TTS here.
    )

    user_is_speaking = False
    if channel == "phone":
        user_utterance_event = InboundPhoneUtterance
        assistant_utterance_event = OutboundPhoneUtterance
    else:
        user_utterance_event = InboundUnifyMeetUtterance
        assistant_utterance_event = OutboundUnifyMeetUtterance

    # Track call start time for usage logging
    # See common.py for detailed comments on the STS billing heuristic
    call_start_time = time.time()

    def log_call_usage() -> None:
        """Log STS usage when call ends (pre-shutdown callback)."""
        call_duration = time.time() - call_start_time
        log_sts_usage(
            call_duration_seconds=call_duration,
            contact=contact,
            tags=[f"channel:{channel}"],
        )

    # Shared end_call + inactivity + participant disconnect handler
    # Pass usage logging callback to run before shutdown
    end_call = create_end_call(
        contact,
        channel,
        pre_shutdown_callback=log_call_usage,
    )
    touch_activity = setup_inactivity_timeout(end_call)
    setup_participant_disconnect_handler(ctx.room, end_call)

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: "UserStateChangedEvent"):
        nonlocal user_is_speaking
        user_is_speaking = ev.new_state == "speaking"
        touch_activity()

    @session.on("conversation_item_added")
    def _on_chat_item_added(ev: "UserInputTranscribedEvent"):
        """Publish both user and assistant utterances from a single location."""
        role = ev.item.role  # "user" | "assistant"
        text = ev.item.text_content or ""  # reliably the final text
        if role == "user":
            event = user_utterance_event(contact, content=text)
        else:
            event = assistant_utterance_event(contact, content=text)

        asyncio.create_task(
            event_broker.publish(f"app:comms:{channel}_utterance", event.to_json()),
        )
        print(role, text)
        touch_activity()

    rio = RoomInputOptions()

    # high-level behavior for the assistant.
    system = build_voice_agent_prompt(
        bio=assistant_bio,
        boss_first_name=boss["first_name"],
        boss_surname=boss["surname"],
        boss_email_address=boss["email_address"],
        boss_phone_number=boss["phone_number"],
        contact_first_name=contact["first_name"],
        contact_surname=contact["surname"],
        contact_phone_number=contact["phone_number"],
        contact_email=contact["email_address"],
        is_boss_user=contact["contact_id"] == 1,
    )
    print("PRINTING SYSTEM PROMPT")
    print(system)

    agent = Assistant(
        contact=contact,
        boss=boss,
        instructions=system,
        outbound=outbound,
    )

    # publish call started (shared helper)
    await publish_call_started(contact, channel)
    touch_activity()

    # Buffer for guidance that arrives before session is ready
    pending_guidance: list[str] = []
    session_ready = False

    def on_status(data: dict) -> None:
        """Handle status events (call_answered, stop)."""
        event_type = data.get("type", "")
        print(f"[Status] {event_type}")
        touch_activity()

        if event_type == "call_answered":
            call_answered_flag.set()
            agent.set_call_received()
        elif event_type == "stop":
            asyncio.create_task(end_call())

    def apply_guidance(content: str) -> None:
        """Apply guidance to chat context and optionally trigger reply."""
        chat_ctx = rt.chat_ctx
        chat_ctx.add_message(
            role="system",
            content=[f"[notification] {content}"],
        )

        async def update_and_reply():
            await rt.update_chat_ctx(chat_ctx)
            nonlocal user_is_speaking
            if not user_is_speaking and chat_ctx.items[-1].role != "assistant":
                session.generate_reply(allow_interruptions=True)

        asyncio.create_task(update_and_reply())

    def on_guidance(data: dict) -> None:
        """Handle guidance from conversation manager."""
        payload = data.get("payload") or data
        content = payload.get("content", "")
        print(
            (
                f"[Guidance] {content[:50]}..."
                if len(content) > 50
                else f"[Guidance] {content}"
            ),
        )
        touch_activity()

        if content:
            if not session_ready:
                pending_guidance.append(content)
            else:
                apply_guidance(content)

    event_broker.register_callback("app:call:status", on_status)
    event_broker.register_callback("app:call:call_guidance", on_guidance)

    # Handle call_answered that arrived during initialization
    if call_answered_flag.is_set():
        print("[Status] call_answered arrived during init - applying now")
        agent.set_call_received()

    logger.info("starting AgentSession")
    await session.start(room=ctx.room, agent=agent, room_input_options=rio)

    # Get realtime session (only available after session.start())
    rt = agent.realtime_llm_session

    await session.generate_reply(allow_interruptions=True)

    # Session is now ready - process buffered guidance and mark ready for future
    session_ready = True
    if pending_guidance:
        print(f"[Guidance] Processing {len(pending_guidance)} buffered message(s)")
        for content in pending_guidance:
            apply_guidance(content)
        pending_guidance.clear()


if __name__ == "__main__":
    # Shared CLI handling
    livekit_agent_name, room_name = configure_from_cli(
        extra_env=[
            ("CONTACT", True),
            ("BOSS", True),
            ("ASSISTANT_BIO", False),
        ],
    )

    # Dispatch LiveKit agent
    if should_dispatch_livekit_agent():
        print(f"Dispatching LiveKit agent {livekit_agent_name}")
        dispatch_livekit_agent(livekit_agent_name, room_name)
        print(f"LiveKit agent {livekit_agent_name} dispatched")

    # Run the agent using the standard CLI - this is the natural way to run LiveKit agents.
    # The process will be terminated via SIGTERM when cleanup_call_proc() is called.
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=livekit_agent_name,
            initialize_process_timeout=60,
        ),
    )
