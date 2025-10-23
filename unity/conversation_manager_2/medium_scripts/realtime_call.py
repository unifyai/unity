"""
Quick-start LiveKit Agents example that uses OpenAI's gpt-realtime model.

Copy-paste into a file (e.g., livekit_gpt_realtime_agent.py), create a .env, and run:

  pip install "livekit-agents[openai]~=1.2" python-dotenv
  # Start a dev worker (hot-reload):
  python livekit_gpt_realtime_agent.py dev
  # Or connect the agent directly to a room for local testing:
  python livekit_gpt_realtime_agent.py connect --room test-room

Required environment variables (in .env or shell):
  LIVEKIT_URL=wss://<your-livekit-host>
  LIVEKIT_API_KEY=<your-key>
  LIVEKIT_API_SECRET=<your-secret>
  OPENAI_API_KEY=<your-openai-key>

In a browser/mobile client, join the same LiveKit room (e.g. "test-room").
The agent will join and converse using OpenAI Realtime with low-latency speech.
"""

from __future__ import annotations

import asyncio
import sys
import logging
import os
from dotenv import load_dotenv

from jinja2 import Template


from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    ConversationItemAddedEvent
)
from livekit import agents
from livekit.plugins.openai import realtime as openai_realtime
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    elevenlabs,
    # noise_cancellation,
    silero,
)
from livekit.plugins.google.beta import realtime as google_realtime
from livekit.agents.llm import ChatMessage
from livekit.plugins.turn_detector.english import EnglishModel
load_dotenv()

from unity.conversation_manager_2.utils import dispatch_agent
from unity.conversation_manager_2.event_broker import get_event_broker
from unity.conversation_manager_2.new_events import *

from pathlib import Path



event_broker = get_event_broker()

with open(Path(__file__).resolve().parent.parent / "prompts" / "realtime_phone_agent.md") as f:
    SYSTEM_PROMPT = f.read()

# Optional: tweak VAD/turn detection behavior using OpenAI's server VAD or semantic VAD
# from openai.types.beta.realtime.session import TurnDetection

logger = logging.getLogger("gpt-realtime-agent")
logger.setLevel(logging.INFO)

class Assistant(Agent):
    async def on_user_turn_completed(self, turn_ctx, new_message):
        print(turn_ctx)
        print(new_message)
    
    # async def transcription_node(self, text, model_settings): 
    #     async for delta in text:
    #         print(delta)
    #         yield delta.replace("😘", "")


async def entrypoint(ctx: JobContext) -> None:
    """Main job entry. Runs once per dispatched agent/room.

    The worker will call this function whenever it's assigned to a room.
    """
    logger.info("connecting to LiveKit room...")
    await ctx.connect()  # ensures ctx.room is usable

    # Configure the OpenAI Realtime model. The default model is 'gpt-realtime', so the
    # explicit model= parameter here is optional, but shown for clarity.
    llm = openai_realtime.RealtimeModel(
        model="gpt-realtime",
        # Pick a built-in OpenAI voice; 'alloy' is the default. Try 'marin', 'verse', etc.
        voice="alloy",
        # Example: run in speech-to-speech (audio) + text mode; set ["text"] to drive a separate TTS.
        modalities=["audio"],
        # Example (optional): customize server VAD / interrupt behavior
        # turn_detection=TurnDetection(
        #     type="server_vad",
        #     threshold=0.5,
        #     prefix_padding_ms=300,
        #     silence_duration_ms=500,
        #     create_response=True,
        #     interrupt_response=True,
        # ),
    )

    session = AgentSession(
        llm=llm,
        # If you prefer a separate TTS instead of Realtime audio, set llm.modalities=["text"] above
        # and provide a TTS here, e.g.: tts="cartesia/sonic-2"
        # tts="cartesia/sonic-2",
    )

    
    user_is_speaking = False

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: 'UserStateChangedEvent'):
        nonlocal user_is_speaking
        user_is_speaking = (ev.new_state == "speaking")
    
    @session.on("conversation_item_added")
    def _on_chat_item_added(ev: 'UserInputTranscribedEvent'):
        role = ev.item.role               # "user" | "assistant"
        text = ev.item.text_content or "" # reliably the final text
        if role == "user":
            event = PhoneUtterance(os.environ["CALL_FROM_NUMBER"], content=text)
        else:
            event = AssistantPhoneUtterance(os.environ["CALL_FROM_NUMBER"], content=text)
        asyncio.create_task(event_broker.publish("app:comms:phone_utterance", event.to_json()))
        print(role, text)

    voice_provider = os.environ.get("VOICE_PROVIDER", "cartesia")
    voice_id = os.environ.get("VOICE_ID", "")

    # Lightweight audio I/O options. You can add noise cancellation, custom VAD, etc.
    rio = RoomInputOptions(
        # noise_cancellation=noise_cancellation.BVC(),
    )

    # High-level behavior for the assistant.
    system = Template(SYSTEM_PROMPT).render(
            contact_id=1,
            first_name="Yasser",
            surname="Ahmed",
            phone_number="+12697784020",
            email_address="yasser@unify.ai",
        )
    agent = Assistant(
        instructions=system
    )

    await event_broker.publish(
        "app:comms:phone_call_started",
        PhoneCallStarted(contact=os.environ["CALL_FROM_NUMBER"]).to_json(),
    )

    async def wait_for_nudges():
        print("waiting")
        rt = agent.realtime_llm_session  # underlying OpenAI RealtimeSession
        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:call_notifs")
            while True:
                msg = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=None
                    )
                if msg is not None:
                    print("got notif", msg)
                    msg = json.loads(msg["data"])
                    chat_ctx = rt.chat_ctx
                    chat_ctx.add_message(role="user", content=[f"""[notification] {msg["content"]}"""])
                    await rt.update_chat_ctx(chat_ctx)
                    print(rt.chat_ctx.items)
                    nonlocal user_is_speaking
                    if not user_is_speaking and chat_ctx.items[-1].role != "assistant":
                        await session.generate_reply(allow_interruptions=True)
                await asyncio.sleep(0.1)


    logger.info("starting AgentSession")
    await session.start(room=ctx.room, agent=agent, room_input_options=rio)
    asyncio.create_task(wait_for_nudges())

    # Keep the job alive until the room ends or the worker shuts down.
    # The worker lifecycle will stop this when participants leave.
    while True:
        # If your app has a shutdown condition, check it here.
        await asyncio.sleep(2)


if __name__ == "__main__":
    # Extract phone numbers before passing to agents.cli
    from_number = ""
    assistant_number = ""
    to_number = ""
    voice_provider = "cartesia"
    voice_id = ""
    meet_id = ""
    outbound = "False"
    print("sys.argv", sys.argv)

    if len(sys.argv) > 7:
        # Remove phone numbers from sys.argv to prevent them from being passed to agents.cli
        from_number = sys.argv[2]
        assistant_number = sys.argv[3]
        voice_provider = sys.argv[4] if sys.argv[4] != "None" else "cartesia"
        voice_id = sys.argv[5]
        outbound = sys.argv[7]
        sys.argv = sys.argv[:2]  # Keep only script name and "dev" command

    # Store phone numbers in environment variables to be accessed by entrypoint
    os.environ["CALL_FROM_NUMBER"] = from_number
    os.environ["VOICE_PROVIDER"] = voice_provider
    if voice_id != "None":
        os.environ["VOICE_ID"] = voice_id
    # os.environ["CALL_TO_NUMBER"] = assistant_number
    os.environ["OUTBOUND"] = outbound

    agent_name = f"unity_{assistant_number}"

    # dispatch agent
    if sys.argv[1] == "dev":
        print("Dispatching agent")
        dispatch_agent(agent_name)
        print("Agent dispatched")

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=agent_name,
        ),
    )
