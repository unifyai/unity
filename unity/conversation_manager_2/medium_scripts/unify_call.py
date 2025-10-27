import sys
import json
import os


sys.path.append("..")
import asyncio

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    elevenlabs,
    silero,
)

if sys.platform == "darwin":
    from livekit.plugins import noise_cancellation
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.agents import ChatContext, ChatMessage

from livekit.agents import ModelSettings, llm, FunctionTool, Agent
from typing import AsyncIterable

from unity.conversation_manager_2.event_broker import get_event_broker


load_dotenv()

from unity.conversation_manager_2.new_events import *
from unity.conversation_manager_2.utils import dispatch_agent

event_broker = get_event_broker()
chunk_queue = asyncio.Queue()


# Pre-load STT, LLM and VAD so that we don't initialize inside entrypoint
try:
    print("[unify_call] Pre-loading STT, LLM and VAD...")
    STT = deepgram.STT(model="nova-3", language="en-GB")
    LLM = openai.LLM(model="gpt-4o")
    VAD = silero.VAD.load(min_speech_duration=0.15)
    print("[unify_call] Pre-loading complete")
except:
    print("[unify_call] Pre-loading failed")
    STT, LLM, VAD = None, None, None


class Assistant(Agent):
    def __init__(self, contact_id: int = 1) -> None:
        self.contact_id = contact_id
        super().__init__(instructions="", llm=LLM)

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        # Emit user utterance into Redis
        await event_broker.publish(
            "app:comms:unify_call_utterance",
            UnifyCallUtterance(
                contact=self.contact_id,
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
        while True:
            chunk = await chunk_queue.get()
            if chunk["type"] == "end_gen":
                break
            elif chunk["chunk"] is not None:
                yield chunk["chunk"]


async def entrypoint(ctx: agents.JobContext):
    global STT, LLM, VAD

    print("[unify_call] Connecting to room...")
    await ctx.connect()
    print("[unify_call] Connected to room")

    voice_provider = os.environ.get("VOICE_PROVIDER", "cartesia")
    voice_id = os.environ.get("VOICE_ID", "")
    # unify_call always addresses the boss contact (id=1)
    contact_id = 1

    # fallback for whenever pre-loading fails
    if STT is None:
        STT = deepgram.STT(model="nova-3", language="en-GB")
        LLM = openai.LLM(model="gpt-4o")
        VAD = silero.VAD.load(min_speech_duration=0.15)

    session = AgentSession(
        stt=STT,
        llm=LLM,
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

    async def end_call():
        print("[unify_call] Initiating graceful shutdown...")
        await event_broker.publish(
            "app:comms:unify_call_ended",
            UnifyCallEnded(contact=contact_id).to_json(),
        )

        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if tasks:
            for task in tasks:
                task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass
        print("[unify_call] Graceful shutdown completed")

    # Inactivity timeout
    INACTIVITY_TIMEOUT = 300
    last_activity_time = asyncio.get_event_loop().time()

    async def check_inactivity():
        nonlocal last_activity_time
        while True:
            await asyncio.sleep(10)
            current_time = asyncio.get_event_loop().time()
            if current_time - last_activity_time > INACTIVITY_TIMEOUT:
                print("[unify_call] Inactivity timeout reached, shutting down agent...")
                await end_call()
                break

    asyncio.create_task(check_inactivity())

    def on_participant_disconnected(*args, **kwargs):
        asyncio.create_task(end_call())

    ctx.room.on("participant_disconnected", on_participant_disconnected)

    assistant = Assistant(contact_id=contact_id)
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            noise_cancellation=(
                noise_cancellation.BVC() if sys.platform == "darwin" else None
            ),
        ),
    )

    # Worker has started and connected – publish UnifyCallStarted
    await event_broker.publish(
        "app:comms:unify_call_started",
        UnifyCallStarted(contact=contact_id).to_json(),
    )

    async def response_task():
        nonlocal session, last_activity_time
        handle = await session.generate_reply()
        last_activity_time = asyncio.get_event_loop().time()
        return handle.chat_items[-1].text_content, handle.interrupted

    def on_response_end(t: asyncio.Task):
        nonlocal last_activity_time
        try:
            result = t.result()
            if result:
                last_activity_time = asyncio.get_event_loop().time()
                if result[1]:
                    # interruption; we don't emit a special interrupt for unify_call
                    pass
        except asyncio.CancelledError:
            pass

    async def collect_events():
        nonlocal last_activity_time
        global chunk_queue

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:unify_call:response_gen")
            while True:
                try:
                    msg = await pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=None,
                    )
                    if msg is None:
                        continue
                    data = json.loads(msg["data"])
                    last_activity_time = asyncio.get_event_loop().time()
                    if data["type"] == "start_gen":
                        chunk_queue = asyncio.Queue()
                        t = asyncio.create_task(response_task())
                        t.add_done_callback(on_response_end)
                    elif data["type"] in ("gen_chunk", "end_gen"):
                        chunk_queue.put_nowait(data)
                except Exception as e:
                    print(f"[unify_call] Error in collect_events: {e}")
                    break

    asyncio.create_task(collect_events())


if __name__ == "__main__":
    # Allow running locally for dev worker
    voice_provider = "cartesia"
    voice_id = ""
    contact_id = 1
    agent_name = f"unity_unify_call_{contact_id}"

    # Parse optional args passed after the "dev" subcommand
    # Example invocation from run_script:
    #   unify_call.py dev <voice_provider> <voice_id> <agent_name>
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        if len(sys.argv) > 2:
            voice_provider = sys.argv[2]
        if len(sys.argv) > 3:
            voice_id = sys.argv[3]
        if len(sys.argv) > 4:
            agent_name = sys.argv[4]
        # Trim argv so livekit agents CLI doesn't see extra args
        sys.argv = sys.argv[:2]

    os.environ["UNIFY_CONTACT_ID"] = str(contact_id)
    os.environ["VOICE_PROVIDER"] = voice_provider
    if voice_id:
        os.environ["VOICE_ID"] = voice_id

    # dispatch agent
    print("[unify_call] Dispatching agent")
    dispatch_agent(agent_name)
    print("[unify_call] Agent dispatched")

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=agent_name,
        ),
    )
