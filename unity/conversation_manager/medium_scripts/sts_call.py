from __future__ import annotations

import asyncio
import logging
import json
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
    from livekit.plugins import openai as openai_plugin
    from livekit.plugins.openai import realtime as openai_realtime
except ImportError:
    openai_plugin = None
    openai_realtime = None

from unity.conversation_manager.utils import dispatch_livekit_agent
from unity.conversation_manager.events import *
from unity.conversation_manager.prompt_builders import build_voice_agent_prompt
from unity.session_details import SESSION_DETAILS

# Shared helpers
import unillm

from unity.conversation_manager.medium_scripts.common import (
    event_broker,
    create_end_call,
    setup_inactivity_timeout,
    setup_participant_disconnect_handler,
    publish_call_started,
    configure_from_cli,
    should_dispatch_livekit_agent,
    start_event_broker_receive,
    UserScreenCaptureManager,
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

    # User screen share capture (subscribes to LiveKit room tracks automatically)
    screen_capture = UserScreenCaptureManager(ctx.room)

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
        # OpenAI TTS with the same voice as the Realtime model, used exclusively
        # for session.say() when the guidance articulator provides pre-generated
        # speech. Normal conversation still uses the Realtime model's native audio.
        tts=openai_plugin.TTS(voice=voice_id) if voice_id else openai_plugin.TTS(),
    )

    user_is_speaking = False
    _queued_speech: list[str] = []
    if channel == "phone":
        user_utterance_event = InboundPhoneUtterance
        assistant_utterance_event = OutboundPhoneUtterance
    else:
        user_utterance_event = InboundUnifyMeetUtterance
        assistant_utterance_event = OutboundUnifyMeetUtterance

    # Shared end_call + inactivity + participant disconnect handler
    end_call = create_end_call(contact, channel)
    touch_activity = setup_inactivity_timeout(end_call)
    setup_participant_disconnect_handler(ctx.room, end_call)

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: "UserStateChangedEvent"):
        nonlocal user_is_speaking
        user_is_speaking = ev.new_state == "speaking"
        touch_activity()

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev):
        """Try queued speech only after the agent settles into a quiescent state.

        We intentionally do NOT trigger from user_state_changed because there is
        a gap between VAD silence detection and the turn detector confirming the
        turn. During that gap, agent_state is still "listening" and current_speech
        is None — firing then would race ahead of the fast brain's reply.

        Triggering here guarantees the full thinking → speaking → listening cycle
        has completed before queued guidance speech plays.
        """
        if ev.new_state in ("listening", "idle"):
            maybe_speak_queued()

    @session.on("conversation_item_added")
    def _on_chat_item_added(ev: "UserInputTranscribedEvent"):
        """Publish both user and assistant utterances from a single location."""
        role = ev.item.role  # "user" | "assistant"
        text = ev.item.text_content or ""  # reliably the final text
        if role == "user":
            event = user_utterance_event(contact, content=text)
            # Capture the user's screen if they are sharing it.
            b64 = screen_capture.capture_screenshot()
            if b64:
                from datetime import datetime, timezone

                asyncio.create_task(
                    event_broker.publish(
                        "app:comms:user_screen_screenshot",
                        json.dumps(
                            {
                                "b64": b64,
                                "utterance": text,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        ),
                    ),
                )
        else:
            event = assistant_utterance_event(contact, content=text)

        asyncio.create_task(
            event_broker.publish(f"app:comms:{channel}_utterance", event.to_json()),
        )
        print(role, text)
        touch_activity()

    rio = RoomInputOptions()

    # high-level behavior for the assistant.
    from unity.settings import SETTINGS

    assistant_name = SESSION_DETAILS.assistant.name
    system = build_voice_agent_prompt(
        bio=assistant_bio,
        assistant_name=assistant_name or None,
        boss_first_name=boss.get("first_name", ""),
        boss_surname=boss.get("surname", ""),
        boss_email_address=boss.get("email_address", ""),
        boss_phone_number=boss.get("phone_number", ""),
        boss_bio=boss.get("bio") or None,
        contact_first_name=contact.get("first_name", ""),
        contact_surname=contact.get("surname", ""),
        contact_phone_number=contact.get("phone_number", ""),
        contact_email=contact.get("email_address", ""),
        contact_bio=contact.get("bio") or None,
        is_boss_user=contact.get("contact_id") == 1,
        contact_rolling_summary=contact.get("rolling_summary", ""),
        demo_mode=SETTINGS.DEMO_MODE,
    ).flatten()
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
    pending_guidance: list[tuple[str, str, bool]] = []
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

    def apply_guidance(
        content: str,
        response_text: str = "",
        should_speak: bool = False,
    ) -> None:
        """Apply guidance to chat context and optionally queue pre-generated speech."""
        chat_ctx = rt.chat_ctx
        chat_ctx.add_message(
            role="system",
            content=[f"[notification] {content}"],
        )

        async def update_ctx():
            await rt.update_chat_ctx(chat_ctx)

        asyncio.create_task(update_ctx())

        if should_speak and response_text:
            _queued_speech.append(response_text)
            maybe_speak_queued()

    def maybe_speak_queued() -> None:
        """Speak the next queued response when user is silent and assistant is idle.

        Gates on agent_state to avoid racing with the fast brain's reply pipeline.
        After the user stops speaking, the agent transitions through thinking →
        speaking → listening. We only speak queued text once the agent has settled
        back to a quiescent state, guaranteeing the fast brain's reply comes first.
        """
        if not _queued_speech or user_is_speaking:
            return
        if session.agent_state not in ("listening", "idle"):
            return
        current = session.current_speech
        if current is not None and not current.done:
            return
        text = _queued_speech.pop(0)
        session.say(text, allow_interruptions=True, add_to_chat_ctx=True)

    def on_guidance(data: dict) -> None:
        """Handle guidance from conversation manager."""
        payload = data.get("payload") or data
        content = payload.get("content", "")
        response_text = payload.get("response_text", "")
        should_speak = payload.get("should_speak", False)
        print(
            (
                f"[Guidance] speak={should_speak} {content[:50]}..."
                if len(content) > 50
                else f"[Guidance] speak={should_speak} {content}"
            ),
        )
        touch_activity()

        if content:
            if not session_ready:
                pending_guidance.append((content, response_text, should_speak))
            else:
                apply_guidance(content, response_text, should_speak)

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

    # Log real usage per turn via unillm (replaces duration-based heuristic)
    @rt.on("metrics_collected")
    def _on_metrics(metrics):
        usage = {
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "total_tokens": metrics.total_tokens,
            "input_token_details": {
                "audio_tokens": metrics.input_token_details.audio_tokens,
                "text_tokens": metrics.input_token_details.text_tokens,
                "cached_tokens": metrics.input_token_details.cached_tokens,
            },
            "output_token_details": {
                "audio_tokens": metrics.output_token_details.audio_tokens,
                "text_tokens": metrics.output_token_details.text_tokens,
            },
        }
        transcript = [
            {"role": item.role, "content": item.text_content or ""}
            for item in rt.chat_ctx.items
            if item.text_content
        ]
        unillm.log_usage(
            metrics.model,
            usage,
            transcript=transcript,
            label=metrics.model,
        )

    # For outbound calls, wait for call_answered before speaking.
    # Unlike call.py, the Realtime API bypasses llm_node, so we must wait here.
    # Note: await on an already-set Event returns immediately.
    if outbound:
        print("Outbound call: waiting for call_answered before speaking...")
        await call_answered_flag.wait()
        print("Outbound call: call answered, proceeding to speak")

    # Mark session ready and process any buffered guidance BEFORE first utterance.
    # After this, the on_guidance callback will apply guidance immediately.
    session_ready = True
    if pending_guidance:
        print(f"[Guidance] Applying {len(pending_guidance)} buffered message(s)")
        for content, response_text, should_speak in pending_guidance:
            apply_guidance(content, response_text, should_speak)
        pending_guidance.clear()

    await session.generate_reply(allow_interruptions=True)


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
        dispatch_livekit_agent(
            livekit_agent_name,
            room_name,
            record=True,
            assistant_id=SESSION_DETAILS.assistant.id,
            user_id=SESSION_DETAILS.user.id,
        )
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
