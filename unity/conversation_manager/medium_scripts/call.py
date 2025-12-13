import sys
import json
import os
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

if sys.platform == "darwin":
    from livekit.plugins import noise_cancellation

from livekit.plugins.turn_detector.english import EnglishModel
from livekit.agents import ChatContext, ChatMessage
from livekit.agents import ModelSettings, llm, FunctionTool

from typing import AsyncIterable

load_dotenv()

from unity.conversation_manager.events import *
from unity.conversation_manager.utils import dispatch_agent

# NEW: shared helpers
from unity.conversation_manager.medium_scripts.common import (
    event_broker,
    create_end_call,
    setup_inactivity_timeout,
    setup_participant_disconnect_handler,
    publish_call_started,
    configure_from_cli,
    should_dispatch_agent,
)

chunk_queue = asyncio.Queue()
current_running_response: asyncio.Task | None = None

# globals initialized lazily or via prewarm to avoid duplicate heavy init
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
        # ensure fallback path runs by resetting all globals
        STT = None
        VAD = None


class Assistant(Agent):
    def __init__(self, contact: dict, channel: str, outbound: bool = False) -> None:
        self.past_events = []
        self.new_events = []
        self.current_tasks_status = None
        self.contact = contact
        self.channel = channel
        self.utterance_event = (
            InboundPhoneUtterance if channel == "phone" else InboundUnifyMeetUtterance
        )
        self.call_received = not outbound

        super().__init__(instructions="")

    def set_call_received(self):
        self.call_received = True

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        print("sending user message...")
        await event_broker.publish(
            f"app:comms:{self.channel}_utterance",
            self.utterance_event(
                contact=self.contact,
                content=new_message.text_content,
            ).to_json(),
        )
        raise llm.StopResponse()

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
        while True:
            chunk = await chunk_queue.get()
            if chunk["type"] == "end_gen":
                break
            elif chunk["chunk"] is not None:
                yield chunk["chunk"]


async def entrypoint(ctx: agents.JobContext):
    global STT, VAD

    print("Connecting to room...")
    await ctx.connect()
    print("Connected to room")

    # read static config
    voice_provider = os.environ.get("VOICE_PROVIDER")
    voice_id = os.environ.get("VOICE_ID")
    outbound = os.environ.get("OUTBOUND") == "True"
    channel = os.environ.get("CHANNEL")
    print("voice_provider", voice_provider)
    print("voice_id", voice_id)
    print("outbound", outbound)
    print("channel", channel)

    # contact payloads passed as json env vars
    contact = json.loads(os.getenv("CONTACT", "{}"))

    # fallback for whenever pre-loading fails
    if STT is None:
        STT = deepgram.STT(model="nova-3", language="en-GB")
        VAD = silero.VAD.load(min_speech_duration=0.15)

    # LLM inference handled by Assistant.llm_node override via Redis/ConversationManager
    session = AgentSession(
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

    # NEW: shared end_call + inactivity + participant disconnect handler
    end_call = create_end_call(contact, channel)
    touch_activity = setup_inactivity_timeout(end_call)
    setup_participant_disconnect_handler(ctx.room, end_call)

    assistant = Assistant(contact=contact, channel=channel, outbound=outbound)

    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            noise_cancellation=(
                noise_cancellation.BVC() if sys.platform == "darwin" else None
            ),
        ),
    )

    # publish call started (shared helper)
    await publish_call_started(contact, channel)
    touch_activity()

    async def response_task():
        nonlocal session
        handle = await session.generate_reply()
        touch_activity()
        return handle.chat_items[-1].text_content, handle.interrupted

    def on_response_end(t: asyncio.Task):
        print("FIRED!!!")
        try:
            result = t.result()
            if result:
                print("RESULT", result)
                try:
                    utterance = result[0]
                except Exception:  # noqa: BLE001
                    utterance = ""

                # We could publish assistant utterances here if needed.
                # Update activity time on assistant response
                touch_activity()

                if result[1]:
                    asyncio.create_task(
                        event_broker.publish(
                            "app:comms:interrupt",
                            Interrupt(contact=contact).to_json(),
                        ),
                    )
        except asyncio.CancelledError:
            pass

    async def collect_events():
        global chunk_queue

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe(
                "app:call:response_gen",
                "app:unify_meet:response_gen",
                "app:call:status",
            )
            print("waiting for events...")
            while True:
                try:
                    msg = await pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=None,
                    )
                    if msg is None:
                        continue
                    print("done", msg)
                    msg = json.loads(msg["data"])
                    print("GOT", msg)

                    # Update activity time on any event
                    touch_activity()

                    # handle msg
                    if msg["type"] == "call_answered":
                        print("call received")
                        assistant.set_call_received()
                    elif msg["type"] == "start_gen":
                        chunk_queue = asyncio.Queue()
                        t = asyncio.create_task(response_task())
                        t.add_done_callback(on_response_end)
                    elif msg["type"] == "gen_chunk" or msg["type"] == "end_gen":
                        chunk_queue.put_nowait(msg)
                    elif msg["type"] == "stop":
                        print("STOPPING CALL")
                        await end_call()
                except Exception as e:  # noqa: BLE001
                    print(f"Error in collect_events: {e}")
                    break  # Exit the loop on error

    asyncio.create_task(collect_events())


if __name__ == "__main__":
    # Shared CLI handling
    agent_name, room_name = configure_from_cli(extra_env=[("CONTACT", True)])

    # dispatch agent
    if should_dispatch_agent():
        print(f"Dispatching agent {agent_name}")
        dispatch_agent(agent_name, room_name)
        print(f"Agent {agent_name} dispatched")

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=agent_name,
            prewarm_fnc=prewarm,
            initialize_process_timeout=60,
        ),
    )
