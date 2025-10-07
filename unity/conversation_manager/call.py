import sys
import json
import os

sys.path.append("..")
import asyncio

from dotenv import load_dotenv

from livekit import agents, rtc, api
from livekit.agents import utils, tokenize, tts, stt
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.agents.log import logger
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    elevenlabs,
    # noise_cancellation,
    silero,
)

if not sys.platform.startswith("win"):
    from livekit.plugins import noise_cancellation
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.agents import ChatContext, ChatMessage

from livekit.agents import ModelSettings, llm, FunctionTool, Agent
from typing import AsyncIterable
import sounddevice as sd
import numpy as np

load_dotenv()

from unity.conversation_manager.events import *
from unity.conversation_manager.utils import (
    dispatch_agent,
    publish_event,
    close_connection,
    create_connection,
)

events_queue = asyncio.Queue()
chunk_queue = asyncio.Queue()
current_running_response: asyncio.Task = None
reader = None
writer = None


async def audio_from_meet_mic():
    def float32_to_int16(audio):
        return (audio * 32767).astype(np.int16)

    q = asyncio.Queue()

    def callback(indata, frames, time, status):
        if status:
            print("Input stream status:", status)
        # Push audio chunk to queue
        q.put_nowait(indata.copy())

    stream = sd.InputStream(
        channels=1,
        samplerate=16000,
        dtype="float32",
        blocksize=1024,
        callback=callback,
        # device=None  # Ensure system default is set to `meet_mic`
    )

    with stream:
        while True:
            data = await q.get()
            # Convert to AudioFrame
            frame = rtc.AudioFrame(
                data=float32_to_int16(data).tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=data.shape[0],
            )
            yield frame


async def process_structured_output(
    text: AsyncIterable[str],
) -> AsyncIterable[str]:
    async for chunk in text:
        yield chunk


class Assistant(Agent):
    def __init__(
        self,
        from_number: str = "",
        to_number: str = "",
        meet_id: str = None,
        outbound: bool = False,
    ) -> None:
        self.past_events = []
        self.new_events = []
        # self.client = client
        self.current_tasks_status = None
        self.from_number = from_number
        self._call_received = not outbound

        # meet conference
        self.meet_id = meet_id
        self.is_meet_call = meet_id is not None
        super().__init__(instructions="", llm=openai.LLM(model="gpt-4o"))

    def set_call_received(self):
        self._call_received = True

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
            writer=writer,
        )
        raise llm.StopResponse()

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        print("waiting for call to be received...")
        while not self._call_received:
            await asyncio.sleep(0.1)
        print("call received")
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
        activity = self._get_activity_or_raise()
        assert activity.tts is not None, "tts_node called but no TTS node is available"

        wrapped_tts = activity.tts

        if not activity.tts.capabilities.streaming:
            wrapped_tts = tts.StreamAdapter(
                tts=wrapped_tts,
                sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
            )

        async with wrapped_tts.stream() as stream:

            async def _forward_input():
                async for chunk in process_structured_output(text):
                    stream.push_text(chunk)

                stream.end_input()

            forward_task = asyncio.create_task(_forward_input())
            try:
                # real-time playback via OutputStream for meet calls
                out_stream = None
                async for ev in stream:
                    frame = ev.frame
                    if not self.is_meet_call:
                        yield frame
                        continue

                    # open output stream on first chunk
                    if out_stream is None:
                        out_stream = sd.OutputStream(
                            samplerate=frame.sample_rate,
                            channels=frame.num_channels,
                            dtype="int16",
                            latency="low",
                        )
                        out_stream.start()

                    # convert bytes to numpy array and write audio chunk
                    audio_arr = np.frombuffer(frame.data, dtype=np.int16)
                    if frame.num_channels > 1:
                        audio_arr = audio_arr.reshape(-1, frame.num_channels)
                    await asyncio.to_thread(out_stream.write, audio_arr)

                # close output stream when done
                if out_stream is not None:
                    out_stream.stop()
                    out_stream.close()
                    out_stream = None

            finally:
                await utils.aio.cancel_and_wait(forward_task)

    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings,
    ):
        activity = self._get_activity_or_raise()
        assert activity.stt is not None, "stt_node called but no STT node is available"

        wrapped_stt = activity.stt

        if not activity.stt.capabilities.streaming:
            if not activity.vad:
                raise RuntimeError(
                    f"The STT ({activity.stt.label}) does not support streaming, add a VAD to the AgentTask/VoiceAgent to enable streaming"  # noqa: E501
                    "Or manually wrap your STT in a stt.StreamAdapter",
                )

            wrapped_stt = stt.StreamAdapter(stt=wrapped_stt, vad=activity.vad)

        async with wrapped_stt.stream() as stream:

            @utils.log_exceptions(logger=logger)
            async def _forward_input():
                user_audio = audio_from_meet_mic() if self.is_meet_call else audio
                async for frame in user_audio:
                    stream.push_frame(frame)

            forward_task = asyncio.create_task(_forward_input())
            try:
                async for event in stream:
                    yield event
            finally:
                await utils.aio.cancel_and_wait(forward_task)


async def entrypoint(ctx: agents.JobContext):
    global reader, writer

    await ctx.connect()

    # Get phone numbers from environment variables
    from_number = os.environ.get("CALL_FROM_NUMBER", "")
    voice_provider = os.environ.get("VOICE_PROVIDER", "cartesia")
    voice_id = os.environ.get("VOICE_ID", "")
    # to_number = os.environ.get("CALL_TO_NUMBER", "")
    outbound = os.environ.get("OUTBOUND", "False") == "True"

    # meet conference
    meet_id = os.environ.get("MEET_ID", "")
    meet_token = None
    meet_user_room = None

    print("voice_provider", voice_provider)
    print("voice_id", voice_id)

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en-GB"),
        llm=openai.LLM(model="gpt-4o"),
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
        vad=silero.VAD.load(min_speech_duration=0.15),
        turn_detection=EnglishModel(),
    )

    async def end_call():
        print("Initiating graceful shutdown...")

        # Send end call event before cleaning tasks and closing connection
        await publish_event(
            {
                "topic": from_number,
                "to": "past",
                "event": PhoneCallEndedEvent().to_dict(),
            },
            writer=writer,
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
            except Exception as e:
                print(f"Error during task cancellation: {e}")

        if meet_user_room:
            await meet_user_room.disconnect()
            meet_user_room = None
            meet_token = None

        # Close the connection gracefully
        try:
            await close_connection(writer=writer)
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

    if meet_id:
        meet_token = (
            api.AccessToken(
                api_key=os.environ.get("LIVEKIT_API_KEY"),
                api_secret=os.environ.get("LIVEKIT_API_SECRET"),
            )
            .with_identity("meet-user")
            .with_grants(api.VideoGrants(room_join=True, room=meet_id))
            .with_room_config(
                api.RoomConfiguration(
                    agents=[api.RoomAgentDispatch(agent_name=meet_id)],
                ),
            )
            .to_jwt()
        )
        meet_user_room = rtc.Room()
        await meet_user_room.connect(
            url=os.environ.get("LIVEKIT_URL"),
            token=meet_token,
        )

    assistant = Assistant(
        from_number=from_number,
        meet_id=meet_id if meet_id else None,
        outbound=outbound,
    )
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=(
                noise_cancellation.BVC() if not sys.platform.startswith("win") else None
            ),
        ),
    )

    # Initialize connection using utility function
    reader, writer = await create_connection("call")
    await publish_event(
        {
            "topic": from_number,
            "to": "pending",
            "event": PhoneCallStartedEvent().to_dict(),
        },
        writer=writer,
    )

    async def response_task():
        nonlocal session, last_activity_time
        handle = await session.generate_reply()
        last_activity_time = asyncio.get_event_loop().time()  # Update activity time
        return handle.chat_items[-1].text_content, handle.interrupted

    def on_response_end(t: asyncio.Task):
        nonlocal last_activity_time
        print("FIRED!!!")
        try:
            result = t.result()
            if result:
                print("RESULT", result)
                try:
                    phone_utterance = result[0]
                except:
                    phone_utterance = ""
                if phone_utterance:
                    # send assistant response as an event to be added in past events
                    asyncio.create_task(
                        publish_event(
                            {
                                "to": "past",
                                "topic": from_number,
                                "event": PhoneUtteranceEvent(
                                    role="Assistant",
                                    content=phone_utterance,
                                ).to_dict(),
                            },
                            writer=writer,
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
                                writer=writer,
                            ),
                        )
        except asyncio.CancelledError:
            pass

    async def collect_events():
        nonlocal last_activity_time
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
                if msg["type"] == "call_received":
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
        meet_id = sys.argv[6] if sys.argv[6] != "None" else ""
        outbound = sys.argv[7]
        sys.argv = sys.argv[:2]  # Keep only script name and "dev" command

    # Store phone numbers in environment variables to be accessed by entrypoint
    os.environ["CALL_FROM_NUMBER"] = from_number
    os.environ["VOICE_PROVIDER"] = voice_provider
    os.environ["MEET_ID"] = meet_id
    if voice_id != "None":
        os.environ["VOICE_ID"] = voice_id
    # os.environ["CALL_TO_NUMBER"] = assistant_number
    os.environ["OUTBOUND"] = outbound

    agent_name = f"unity_{assistant_number}" if meet_id == "" else meet_id

    # dispatch agent
    if sys.argv[1] == "dev" and not meet_id:
        dispatch_agent(agent_name)

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=agent_name,
        ),
    )
