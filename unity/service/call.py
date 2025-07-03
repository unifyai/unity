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
    # noise_cancellation,
    silero,
)

if sys.platform == "darwin":
    from livekit.plugins import noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import ChatContext, ChatMessage

from livekit.agents import ModelSettings, llm, FunctionTool, Agent
from typing import AsyncIterable
from pydantic_core import from_json

load_dotenv()

from unity.service.events import *
from unity.service.actions import AssistantOutput
from unity.service.utils import publish_event, close_connection, get_reader

events_queue = asyncio.Queue()
chunk_queue = asyncio.Queue()
current_running_response: asyncio.Task = None


async def process_structured_output(
    text: AsyncIterable[str],
) -> AsyncIterable[str]:
    last_response = ""
    acc_text = ""
    async for chunk in text:
        print("CHUNK FOR TTS", chunk)
        acc_text += chunk
        try:
            resp: AssistantOutput = from_json(
                acc_text,
                allow_partial="trailing-strings",
            )
        except ValueError:
            continue

        if not resp.get("phone_utterance"):
            continue

        new_delta = resp["phone_utterance"][len(last_response) :]
        if new_delta:
            print("delta", new_delta)
            yield new_delta
        last_response = resp["phone_utterance"]


class Assistant(Agent):
    def __init__(self, from_number: str = "", to_number: str = "") -> None:
        self.past_events = []
        self.new_events = []
        # self.client = client
        self.current_tasks_status = None
        self.from_number = from_number
        super().__init__(instructions="", llm=openai.LLM(model="gpt-4o"))

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        # events_queue.put_nowait(PhoneUtteranceEvent(role="User", content=new_message.text_content))
        # we will handle this through the events manager
        await publish_event(
            {
                "topic": self.from_number,
                "to": "pending",
                "event": PhoneUtteranceEvent(
                    role="User",
                    content=new_message.text_content,
                ).to_dict(),
            },
        )
        raise llm.StopResponse()

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        print("running llm node...")
        while True:
            chunk = await chunk_queue.get()
            if chunk["type"] == "end_gen":
                break
            elif chunk["chunk"] is not None:
                yield chunk["chunk"]

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable:
        return Agent.default.tts_node(
            self,
            process_structured_output(text),
            model_settings,
        )


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # Get phone numbers from environment variables
    from_number = os.environ.get("CALL_FROM_NUMBER", "")
    tts_provider = os.environ.get("TTS_PROVIDER", "cartesia")
    voice_id = os.environ.get("VOICE_ID", "")
    # to_number = os.environ.get("CALL_TO_NUMBER", "")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o"),
        tts=(
            elevenlabs.TTS(
                voice_id=voice_id if voice_id != "" else elevenlabs.DEFAULT_VOICE_ID,
                model="eleven_multilingual_v2",
            )
            if tts_provider == "elevenlabs"
            else cartesia.TTS(
                voice=voice_id if voice_id != "" else cartesia.tts.TTSDefaultVoiceId,
            )
        ),
        vad=silero.VAD.load(),
        # turn_detection=MultilingualModel(),
    )

    async def end_call():
        print("Initiating graceful shutdown...")

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
            except Exception as e:
                print(f"Error during task cancellation: {e}")

        # Close the connection gracefully
        try:
            # Send end call event before closing connection
            await publish_event(
                {
                    "topic": from_number,
                    "to": "past",
                    "event": PhoneCallEndedEvent().to_dict(),
                },
            )
            print("End call event sent")

            # Close the connection using utility function
            await close_connection()
            print("Connection closed gracefully")
        except Exception as e:
            print(f"Error during connection cleanup: {e}")

        print("Graceful shutdown completed")

    # Add inactivity timeout
    INACTIVITY_TIMEOUT = 300  # 5 minutes in seconds
    last_activity_time = asyncio.get_event_loop().time()

    async def check_inactivity():
        nonlocal last_activity_time
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            current_time = asyncio.get_event_loop().time()
            if current_time - last_activity_time > INACTIVITY_TIMEOUT:
                print("Inactivity timeout reached, shutting down agent...")
                await end_call()
                break  # Exit the loop after shutdown

    # Start inactivity checker
    asyncio.create_task(check_inactivity())

    # Create a wrapper for the room event handler since it expects a sync function
    def on_participant_disconnected(*args, **kwargs):
        asyncio.create_task(end_call())

    ctx.room.on("participant_disconnected", on_participant_disconnected)

    await session.start(
        room=ctx.room,
        agent=Assistant(from_number=from_number),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=(
                noise_cancellation.BVC() if sys.platform == "darwin" else None
            ),
        ),
    )

    # Initialize connection using utility function
    reader = await get_reader()
    await publish_event(
        {
            "topic": from_number,
            "to": "pending",
            "event": PhoneCallStartedEvent().to_dict(),
        },
    )

    async def response_task():
        nonlocal session, last_activity_time
        handle = await session.generate_reply()
        last_activity_time = asyncio.get_event_loop().time()  # Update activity time
        return handle.chat_message.text_content, handle.interrupted

    def on_response_end(t: asyncio.Task):
        nonlocal last_activity_time
        print("FIRED!!!")
        try:
            result = t.result()
            if result:
                print("RESULT", result)
                try:
                    assistant_res = from_json(
                        result[0],
                        allow_partial="trailing-strings",
                    )
                except:
                    assistant_res = {}
                if assistant_res.get("phone_utterance"):
                    # send assistant response as an event to be added in past events
                    asyncio.create_task(
                        publish_event(
                            {
                                "to": "past",
                                "topic": from_number,
                                "event": PhoneUtteranceEvent(
                                    role="Assistant",
                                    content=assistant_res.get("phone_utterance"),
                                ).to_dict(),
                            },
                        ),
                    )
                    # Update activity time on assistant response
                    last_activity_time = asyncio.get_event_loop().time()
                    # send interupt as an event to be added to pending events (?)
                    # this might confuse things a bit actually, maybe it should be sent to past events instead
                    # to prevent re-triggering events if nothing happens
                    # another way would be to signal the event manager that the user is talking now and prevent any
                    # agent response until the user finishes talking
                    if result[1]:
                        asyncio.create_task(
                            publish_event(
                                {
                                    "to": "past",
                                    "topic": from_number,
                                    "event": InterruptEvent().to_dict(),
                                },
                            ),
                        )
        except asyncio.CancelledError:
            pass

    async def collect_events():
        nonlocal last_activity_time, reader
        global chunk_queue

        while True:
            try:
                raw = await reader.readline()
                if not raw:
                    break
                msg = json.loads(raw.decode())
                print("GOT", msg)
                # Update activity time on any event
                last_activity_time = asyncio.get_event_loop().time()
                # handle msg
                if msg["type"] == "start_gen":
                    # nonlocal session
                    # await session.current_speech()
                    chunk_queue = asyncio.Queue()
                    t = asyncio.create_task(response_task())
                    t.add_done_callback(on_response_end)
                elif msg["type"] == "gen_chunk" or msg["type"] == "end_gen":
                    chunk_queue.put_nowait(msg)
            except Exception as e:
                print(f"Error in collect_events: {e}")
                # Connection will be handled by utils module
                break  # Exit the loop on error

    asyncio.create_task(collect_events())


if __name__ == "__main__":
    # Extract phone numbers before passing to agents.cli
    from_number = ""
    assistant_number = ""
    to_number = ""
    tts_provider = "cartesia"
    voice_id = ""
    outbound = ""
    if len(sys.argv) > 6:
        # Remove phone numbers from sys.argv to prevent them from being passed to agents.cli
        from_number = sys.argv[2]
        assistant_number = sys.argv[3]
        tts_provider = sys.argv[4] if sys.argv[4] != "None" else "cartesia"
        voice_id = sys.argv[5]
        outbound = sys.argv[6] if sys.argv[6] != "None" else ""
        sys.argv = sys.argv[:2]  # Keep only script name and "dev" command

    # Store phone numbers in environment variables to be accessed by entrypoint
    os.environ["CALL_FROM_NUMBER"] = from_number
    os.environ["TTS_PROVIDER"] = tts_provider
    if voice_id != "None":
        os.environ["VOICE_ID"] = voice_id
    # os.environ["CALL_TO_NUMBER"] = to_number

    agent_name = f"unity_{assistant_number}"
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=agent_name if not outbound else "",
        ),
    )
