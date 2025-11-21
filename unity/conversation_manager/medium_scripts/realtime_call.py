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
    FunctionTool,
    JobContext,
    llm,
    ModelSettings,
    RoomInputOptions,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
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
from typing import AsyncIterable

load_dotenv()

from unity.conversation_manager.utils import dispatch_agent
from unity.conversation_manager.event_broker import get_event_broker
from unity.conversation_manager.new_events import *

from pathlib import Path


event_broker = get_event_broker()

contact_id = ""
contact_first_name = ""
contact_surname = ""
contact_is_boss_user = ""
contact_email = ""
boss_first_name = ""
boss_surname = ""
boss_phone_number = ""
boss_email = ""
is_boss_user = ""
assistant_bio = ""

with open(
    Path(__file__).resolve().parent.parent / "prompts" / "realtime_phone_agent.md",
) as f:
    SYSTEM_PROMPT = f.read()

# Optional: tweak VAD/turn detection behavior using OpenAI's server VAD or semantic VAD
# from openai.types.beta.realtime.session import TurnDetection

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

    async def on_user_turn_completed(self, turn_ctx, new_message):
        print(turn_ctx)
        print(new_message)

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

    # async def transcription_node(self, text, model_settings):
    #     async for delta in text:
    #         print(delta)
    #         yield delta.replace("😘", "")


async def entrypoint(ctx: JobContext) -> None:
    print("Connecting to room...")
    await ctx.connect()
    print("Connected to room")

    # read static config
    voice_provider = os.environ.get("VOICE_PROVIDER")
    voice_id = os.environ.get("VOICE_ID")
    outbound = os.environ.get("OUTBOUND") == "True"
    assistant_bio = os.environ.get("ASSISTANT_BIO")
    print("voice_provider", voice_provider)
    print("voice_id", voice_id)
    print("outbound", outbound)

    # contact/boss payloads passed as json env vars
    contact = json.loads(os.getenv("CONTACT", "{}"))
    boss = json.loads(os.getenv("BOSS", "{}"))

    # configure the OpenAI Realtime model. The default model is 'gpt-realtime', so the
    # explicit model= parameter here is optional, but shown for clarity.
    llm = openai_realtime.RealtimeModel(
        model=voice_provider,
        # pick a built-in OpenAI voice; 'alloy' is the default. Try 'marin', 'verse', etc.
        voice=voice_id,
        # example: run in speech-to-speech (audio) + text mode; set ["text"] to drive a separate TTS.
        modalities=["audio"],
        # example (optional): customize server VAD / interrupt behavior
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
        # ff you prefer a separate TTS instead of Realtime audio, set llm.modalities=["text"] above
        # and provide a TTS here, e.g.: tts="cartesia/sonic-2"
        # tts="cartesia/sonic-2",
    )

    user_is_speaking = False

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: "UserStateChangedEvent"):
        nonlocal user_is_speaking
        user_is_speaking = ev.new_state == "speaking"

    @session.on("conversation_item_added")
    def _on_chat_item_added(ev: "UserInputTranscribedEvent"):
        role = ev.item.role  # "user" | "assistant"
        text = ev.item.text_content or ""  # reliably the final text
        if role == "user":
            event = PhoneUtterance(contact, content=text)
        else:
            event = AssistantPhoneUtterance(contact, content=text)
        asyncio.create_task(
            event_broker.publish("app:comms:phone_utterance", event.to_json()),
        )
        print(role, text)

    async def end_call():
        print("Initiating graceful shutdown...")

        # Send end call event before cleaning tasks and closing connection
        await event_broker.publish(
            "app:comms:phone_call_ended",
            PhoneCallEnded(contact=contact).to_json(),
        )
        print("End call event sent")

        # Get all running tasks except current task
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

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
            except Exception as e:
                print(f"Error during task cancellation: {e}")

        print("Graceful shutdown completed")

    # add inactivity timeout
    INACTIVITY_TIMEOUT = 300  # 5 minutes in seconds
    last_activity_time = asyncio.get_event_loop().time()

    async def check_inactivity():
        nonlocal last_activity_time
        while True:
            await asyncio.sleep(10)  # check every 10 seconds
            current_time = asyncio.get_event_loop().time()
            if current_time - last_activity_time > INACTIVITY_TIMEOUT:
                print("Inactivity timeout reached, shutting down agent...")
                await end_call()
                break  # exit the loop after shutdown

    # start inactivity checker
    asyncio.create_task(check_inactivity())

    # create a wrapper for the room event handler since it expects a sync function
    def on_participant_disconnected(*args, **kwargs):
        asyncio.create_task(end_call())

    ctx.room.on("participant_disconnected", on_participant_disconnected)

    # lightweight audio I/O options. You can add noise cancellation, custom VAD, etc.
    rio = RoomInputOptions(
        # noise_cancellation=noise_cancellation.BVC(),
    )

    # high-level behavior for the assistant.
    system = Template(SYSTEM_PROMPT).render(
        bio=assistant_bio,
        boss_first_name=boss["first_name"],
        boss_surname=boss["surname"],
        boss_email_address=boss["email_address"],
        boss_phone_number=boss["phone_number"],
        contact_first_name=contact["first_name"],
        contact_surname=contact["surname"],
        contact_phone_number=contact["phone_number"],
        contact_email=contact["email_address"],
        is_boss_user=contact["is_boss"],
    )
    print("PRINTING SYSTEM PROMPT")
    print(system)
    agent = Assistant(
        contact=contact, boss=boss, instructions=system, outbound=outbound
    )

    await event_broker.publish(
        "app:comms:phone_call_started",
        PhoneCallStarted(contact=contact).to_json(),
    )

    async def wait_for_nudges():
        print("waiting")
        rt = agent.realtime_llm_session  # underlying OpenAI RealtimeSession
        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:call_notifs")
            while True:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=None,
                )
                if msg is not None:
                    print("got notif", msg)
                    msg = json.loads(msg["data"])
                    chat_ctx = rt.chat_ctx
                    chat_ctx.add_message(
                        role="user",
                        content=[f"""[notification] {msg["content"]}"""],
                    )
                    await rt.update_chat_ctx(chat_ctx)
                    print(rt.chat_ctx.items)
                    nonlocal user_is_speaking
                    if not user_is_speaking and chat_ctx.items[-1].role != "assistant":
                        await session.generate_reply(allow_interruptions=True)
                await asyncio.sleep(0.1)

    logger.info("starting AgentSession")
    await session.start(room=ctx.room, agent=agent, room_input_options=rio)
    asyncio.create_task(wait_for_nudges())


if __name__ == "__main__":
    assistant_number = ""
    print("sys.argv", sys.argv)

    if len(sys.argv) > 8:
        # get static config
        assistant_number = sys.argv[2]
        os.environ["VOICE_PROVIDER"] = (
            sys.argv[3] if sys.argv[3] != "None" else "cartesia"
        )
        os.environ["VOICE_ID"] = sys.argv[4] if sys.argv[4] != "None" else ""
        os.environ["OUTBOUND"] = sys.argv[5]

        # get contact/boss payloads
        os.environ["CONTACT"] = sys.argv[6]
        os.environ["BOSS"] = sys.argv[7]
        print(f"contact: {os.environ['CONTACT']}")
        print(f"boss: {os.environ['BOSS']}")
        if not json.loads(os.environ["CONTACT"]) or not json.loads(os.environ["BOSS"]):
            print("Contact or boss payload is invalid")
            sys.exit(1)

        # get assistant bio
        os.environ["ASSISTANT_BIO"] = sys.argv[8]

        sys.argv = sys.argv[:2]  # keep only script name and "dev" command
    else:
        print("Not enough arguments provided")
        sys.exit(1)

    agent_name = f"unity_{assistant_number}"

    # dispatch agent
    print(f"Dispatching agent {agent_name}")
    dispatch_agent(agent_name)
    print(f"Agent {agent_name} dispatched")

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=agent_name,
            initialize_process_timeout=60,
        ),
    )
