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

if sys.platform == "darwin":
    from livekit.plugins import noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import ChatContext, ChatMessage

from livekit.agents import ModelSettings, llm, FunctionTool, Agent
from typing import AsyncIterable, get_args

from unity.conversation_manager_2.event_broker import get_event_broker


load_dotenv()

from unity.conversation_manager_2.new_events import *
from unity.conversation_manager.utils import (
    dispatch_agent,
    publish_event,
)

event_broker = get_event_broker()
chunk_queue = asyncio.Queue()
current_running_response: asyncio.Task = None


class Assistant(Agent):
    def __init__(
        self,
        from_number: str = "",
        to_number: str = "",
        outbound: bool = False,
    ) -> None:
        self.past_events = []
        self.new_events = []
        # self.client = client
        self.current_tasks_status = None
        self.from_number = from_number
        self._call_received = not outbound

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
        print("sending user message...")
        await event_broker.publish("app:comms:phone_utterance", 
            PhoneUtterance(
                    contact=os.environ["CALL_FROM_NUMBER"],
                    content=new_message.text_content,
                ).to_json())
        raise llm.StopResponse()


    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        print("waiting for call to be received...")
        print("call received")
        print("running llm node...")
        # this should probably be done with a queue instead to avoid race conditions
        # async with event_broker.pubsub() as pubsub:
        #     await pubsub.subscribe("app:call:chunk")
        #     async for msg in pubsub.listen():
        #         ...


        while True:
            chunk = await chunk_queue.get()
            if chunk["type"] == "end_gen":
                break
            elif chunk["chunk"] is not None:
                yield chunk["chunk"]


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # Get phone numbers from environment variables
    from_number = os.environ.get("CALL_FROM_NUMBER", "")
    tts_provider = os.environ.get("TTS_PROVIDER", "cartesia")
    voice_id = os.environ.get("VOICE_ID", "")
    # to_number = os.environ.get("CALL_TO_NUMBER", "")
    outbound = os.environ.get("OUTBOUND", "False") == "True"

    print("tts_provider", tts_provider)
    print("voice_id", voice_id)

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
        turn_detection=MultilingualModel(),
    )

    async def end_call():
        print("Initiating graceful shutdown...")

        # Send end call event before cleaning tasks and closing connection
        await event_broker.publish("app:comms:phone_call_ended", PhoneCallEnded(contact=os.environ["CALL_FROM_NUMBER"]).to_json())
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

        # Close the connection gracefully
        # try:
        #     await close_connection(writer=writer)
        #     print("Connection closed gracefully")
        # except Exception as e:
        #     print(f"Error during connection cleanup: {e}")

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


    assistant = Assistant(
        from_number=from_number,
        # meet_id=meet_id if meet_id else None,
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
                noise_cancellation.BVC() if sys.platform == "darwin" else None
            ),
        ),
    )

    # Initialize connection using utility function
    # reader, writer = await create_connection("call")

    await event_broker.publish(
        "app:comms:phone_call_started",
        PhoneCallStarted(contact=os.environ["CALL_FROM_NUMBER"]).to_json()
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
                    phone_utterance = result[0]
                except:
                    phone_utterance = ""
                # if phone_utterance:
                #     # send assistant response as an event to be added in past events
                #     msg = {
                #                 "to": "past",
                #                 "topic": from_number,
                #                 "event": PhoneUtterance(
                #                     role="Assistant",
                #                     content=phone_utterance,
                #                 ).to_dict(),
                #             }
                #     asyncio.create_task(
                #         event_broker.publish("app:comms:phone_utterance",
                #             json.dumps({
                #                 "to": "past",
                #                 "topic": from_number,
                #                 "event": PhoneUtterance(
                #                     role="Assistant",
                #                     content=phone_utterance,
                #                 ).to_dict(),
                #             }),
                #         ),
                #     )
                    # Update activity time on assistant response
                    last_activity_time = asyncio.get_event_loop().time()
                    # send interupt as an event to be added to pending events (?)
                    # this might confuse things a bit actually, maybe it should be sent to past events instead
                    # to prevent re-triggering events if nothing happens
                    # another way would be to signal the event manager that the user is talking now and prevent any
                    # agent response until the user finishes talking
                    if result[1]:
                        asyncio.create_task(
                            event_broker.publish(
                                "app:comms:interrupt",
                                Interrupt(contact=os.environ["CALL_FROM_NUMBER"]).to_json()
                            ),
                        )
        except asyncio.CancelledError:
            pass

    async def collect_events():
        nonlocal last_activity_time
        global chunk_queue

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:response_gen")
            print("waiting for events...")
            while True:
                try:
                    msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=None)
                    if msg is None:
                        continue
                    print("done", msg)
                    msg = json.loads(msg["data"])
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
    tts_provider = "cartesia"
    voice_id = ""
    meet_id = ""
    outbound = "False"
    print("sys.argv", sys.argv)

    if len(sys.argv) > 7:
        # Remove phone numbers from sys.argv to prevent them from being passed to agents.cli
        from_number = sys.argv[2]
        assistant_number = sys.argv[3]
        tts_provider = sys.argv[4] if sys.argv[4] != "None" else "cartesia"
        voice_id = sys.argv[5]
        outbound = sys.argv[7]
        sys.argv = sys.argv[:2]  # Keep only script name and "dev" command

    # Store phone numbers in environment variables to be accessed by entrypoint
    os.environ["CALL_FROM_NUMBER"] = from_number
    os.environ["TTS_PROVIDER"] = tts_provider
    if voice_id != "None":
        os.environ["VOICE_ID"] = voice_id
    # os.environ["CALL_TO_NUMBER"] = assistant_number
    os.environ["OUTBOUND"] = outbound

    agent_name = f"unity_{assistant_number}"

    # dispatch agent
    if sys.argv[1] == "dev":
        dispatch_agent(agent_name)

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=agent_name,
        ),
    )
