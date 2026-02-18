import sys
import json
import asyncio

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    cartesia,
    deepgram,
    elevenlabs,
    silero,
)

from unity.conversation_manager.livekit_unify_adapter import UnifyLLM

if sys.platform == "darwin":
    from livekit.plugins import noise_cancellation

from livekit.plugins.turn_detector.english import EnglishModel
from livekit.agents import ChatContext, ChatMessage
from livekit.agents import ModelSettings, llm, FunctionTool

from typing import AsyncIterable

load_dotenv()

from unity.conversation_manager.events import *
from unity.conversation_manager.utils import dispatch_livekit_agent
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
    start_event_broker_receive,
    UserScreenCaptureManager,
)

# Globals initialized lazily or via prewarm to avoid duplicate heavy init
STT = None
VAD = None


def prewarm(_ctx=None):
    global STT, VAD
    try:
        print("Prewarm: initializing STT, VAD and turn detector...")
        STT = deepgram.STT(model="nova-3", language="en-GB")
        VAD = silero.VAD.load(min_speech_duration=0.15)
        print("Prewarm complete")
    except Exception as e:  # noqa: BLE001
        print(f"Prewarm failed: {e}")
        STT = None
        VAD = None


class Assistant(Agent):
    """
    TTS Fast Brain - handles real-time conversation independently.

    Uses a lightweight LLM (gpt-5-mini via UnifyLLM adapter) for fast
    conversational responses. Routes through unillm.AsyncUnify for local
    caching (CI) and usage tracking.
    Communicates with the Main CM Brain (slow brain) via Unix domain socket IPC.
    """

    def __init__(
        self,
        contact: dict,
        boss: dict,
        channel: str,
        instructions: str,
        outbound: bool = False,
    ) -> None:
        self.contact = contact
        self.boss = boss
        self.channel = channel
        self.utterance_event = (
            InboundPhoneUtterance if channel == "phone" else InboundUnifyMeetUtterance
        )
        self.assistant_utterance_event = (
            OutboundPhoneUtterance if channel == "phone" else OutboundUnifyMeetUtterance
        )
        self.call_received = not outbound

        super().__init__(instructions=instructions)

    def set_call_received(self):
        self.call_received = True

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
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
        """Wait for call connection then delegate to parent LLM."""
        print("waiting for call to be received...")
        while not self.call_received:
            await asyncio.sleep(0.1)
        print("call received")

        print("running llm node...")
        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            yield chunk


async def entrypoint(ctx: agents.JobContext):
    global STT, VAD

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

    # Fallback for whenever pre-loading fails
    if STT is None:
        STT = deepgram.STT(model="nova-3", language="en-GB")
        VAD = silero.VAD.load(min_speech_duration=0.15)

    from unity.settings import SETTINGS

    # Fast brain LLM - lightweight model for responsive conversation
    # Uses UnifyLLM adapter for local caching (CI) and usage tracking
    llm_model = UnifyLLM(
        model=SETTINGS.conversation.FAST_BRAIN_MODEL,
        reasoning_effort="low",
    )

    assistant_name = SESSION_DETAILS.assistant.name
    system_prompt = build_voice_agent_prompt(
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
    print(system_prompt)

    session = AgentSession(
        llm=llm_model,
        stt=STT,
        tts=(
            elevenlabs.TTS(
                voice_id=voice_id if voice_id != "" else elevenlabs.DEFAULT_VOICE_ID,
                model="eleven_multilingual_v2",
            )
            if voice_provider == "elevenlabs"
            else cartesia.TTS(
                voice=voice_id if voice_id != "" else cartesia.tts.TTSDefaultVoiceId,
            )
        ),
        vad=VAD,
        turn_detection=EnglishModel(),
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
    def _on_user_state_changed(ev):
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
    def _on_chat_item_added(ev):
        """Publish both user and assistant utterances from a single location."""
        role = ev.item.role  # "user" | "assistant"
        text = ev.item.text_content or ""
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

    assistant = Assistant(
        contact=contact,
        boss=boss,
        channel=channel,
        instructions=system_prompt,
        outbound=outbound,
    )

    rio = RoomInputOptions(
        noise_cancellation=(
            noise_cancellation.BVC() if sys.platform == "darwin" else None
        ),
    )

    # Publish call started (shared helper)
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
            assistant.set_call_received()
        elif event_type == "stop":
            asyncio.create_task(end_call())

    def apply_guidance(
        content: str,
        response_text: str = "",
        should_speak: bool = False,
    ) -> None:
        """Apply guidance to chat context and optionally queue pre-generated speech."""
        guidance_message = f"[notification] {content}"

        # generate_reply() reads from the agent chat context in TTS mode.
        assistant._chat_ctx.add_message(
            role="system",
            content=[guidance_message],
        )
        # Keep session history aligned for observability/debugging.
        session._chat_ctx.add_message(
            role="system",
            content=[guidance_message],
        )

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
        assistant.set_call_received()

    print("starting AgentSession")
    await session.start(room=ctx.room, agent=assistant, room_input_options=rio)

    # Mark session ready and process any buffered guidance BEFORE first utterance.
    # After this, the on_guidance callback will apply guidance immediately.
    # Note: For outbound calls, llm_node will wait for call_received (set by on_status).
    session_ready = True
    if pending_guidance:
        print(f"[Guidance] Applying {len(pending_guidance)} buffered message(s)")
        for content, response_text, should_speak in pending_guidance:
            apply_guidance(content, response_text, should_speak)
        pending_guidance.clear()

    await session.generate_reply(allow_interruptions=True)


if __name__ == "__main__":
    # Shared CLI handling (same as sts_call.py)
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
            prewarm_fnc=prewarm,
            initialize_process_timeout=60,
        ),
    )
