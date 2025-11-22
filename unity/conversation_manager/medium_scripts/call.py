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
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.agents import ChatContext, ChatMessage

from livekit.agents import ModelSettings, llm, FunctionTool, Agent
from typing import AsyncIterable

from unity.conversation_manager.event_broker import get_event_broker


load_dotenv()

from unity.conversation_manager.new_events import *
from unity.conversation_manager.utils import dispatch_agent

event_broker = get_event_broker()
chunk_queue = asyncio.Queue()
current_running_response: asyncio.Task = None

# globals initialized lazily or via prewarm to avoid duplicate heavy init
STT = None
LLM = None
VAD = None


def prewarm(_ctx=None):
    global STT, LLM, VAD
    try:
        print("Prewarm: initializing STT, LLM, VAD and turn detector...")
        STT = deepgram.STT(model="nova-3", language="en-GB")
        LLM = openai.LLM(model="gpt-4o")
        VAD = silero.VAD.load(min_speech_duration=0.15)
        print("Prewarm complete")
    except Exception as e:
        print(f"Prewarm failed: {e}")
        # ensure fallback path runs by resetting all globals
        STT = None
        LLM = None
        VAD = None


class Assistant(Agent):
    def __init__(self, contact: dict, outbound: bool = False) -> None:
        self.past_events = []
        self.new_events = []
        self.current_tasks_status = None
        self.contact = contact
        self.call_received = not outbound

        super().__init__(instructions="", llm=LLM)

    def set_call_received(self):
        self.call_received = True

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        print("sending user message...")
        await event_broker.publish(
            "app:comms:phone_utterance",
            PhoneUtterance(
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
    global STT, LLM, VAD

    print("Connecting to room...")
    await ctx.connect()
    print("Connected to room")

    # read static config
    voice_provider = os.environ.get("VOICE_PROVIDER")
    voice_id = os.environ.get("VOICE_ID")
    outbound = os.environ.get("OUTBOUND") == "True"
    print("voice_provider", voice_provider)
    print("voice_id", voice_id)
    print("outbound", outbound)

    # contact payloads passed as json env vars
    contact = json.loads(os.getenv("CONTACT", "{}"))

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
        print("Initiating graceful shutdown...")

        # send end call event before cleaning tasks and closing connection
        await event_broker.publish(
            "app:comms:phone_call_ended",
            PhoneCallEnded(contact=contact).to_json(),
        )
        print("End call event sent")

        # get all running tasks except current task
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

        if tasks:
            print(f"Cancelling {len(tasks)} running tasks...")
            # cancel all tasks
            for task in tasks:
                task.cancel()

            # wait for tasks to be cancelled gracefully
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

    assistant = Assistant(contact=contact, outbound=outbound)
    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            noise_cancellation=(
                noise_cancellation.BVC() if sys.platform == "darwin" else None
            ),
        ),
    )

    await event_broker.publish(
        "app:comms:phone_call_started",
        PhoneCallStarted(contact=contact).to_json(),
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
                                Interrupt(
                                    contact=os.environ["CALL_FROM_NUMBER"],
                                ).to_json(),
                            ),
                        )
        except asyncio.CancelledError:
            pass

    async def collect_events():
        nonlocal last_activity_time
        global chunk_queue

        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:response_gen", "app:call:status")
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
                    last_activity_time = asyncio.get_event_loop().time()
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
                except Exception as e:
                    print(f"Error in collect_events: {e}")
                    # Connection will be handled by utils module
                    break  # Exit the loop on error

    asyncio.create_task(collect_events())


if __name__ == "__main__":
    assistant_number = ""
    print("sys.argv", sys.argv)

    if len(sys.argv) > 7:
        # get static config
        assistant_number = sys.argv[2]
        os.environ["VOICE_PROVIDER"] = (
            sys.argv[3] if sys.argv[3] != "None" else "cartesia"
        )
        os.environ["VOICE_ID"] = sys.argv[4] if sys.argv[4] != "None" else ""
        os.environ["OUTBOUND"] = sys.argv[5]

        # get contact payloads
        os.environ["CONTACT"] = sys.argv[6]
        print(f"contact: {os.environ['CONTACT']}")
        if not json.loads(os.environ["CONTACT"]):
            print("Contact payload is invalid")
            sys.exit(1)

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
            prewarm_fnc=prewarm,
            initialize_process_timeout=60,
        ),
    )
