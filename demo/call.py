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

from events import *
from actions_2 import AssistantOutput

events_queue = asyncio.Queue()
chunk_queue = asyncio.Queue()
current_running_response: asyncio.Task = None
READER: asyncio.StreamReader | None = None
WRITER: asyncio.StreamWriter | None = None


async def publish_event(ev: dict):
    global WRITER
    ev = json.dumps(ev) + "\n"
    WRITER.write(ev.encode())
    await WRITER.drain()


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
        self.new_events = [
            PhoneCallStartedEvent()
        ]
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
    # to_number = os.environ.get("CALL_TO_NUMBER", "")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )




    def end_call():
        asyncio.create_task(
            publish_event({"topic": from_number, "to": "past", "event": PhoneCallEndedEvent().to_dict()})
        )


    ctx.room.on("participant_disconnected", end_call)


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

    global READER, WRITER
    READER, WRITER = await asyncio.open_connection("127.0.0.1", 8090)
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
        nonlocal last_activity_time
        global chunk_queue
        while True:
            try:
                raw = await READER.readline()
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
            except:
                WRITER.close()
                await WRITER.wait_closed()

    asyncio.create_task(collect_events())


if __name__ == "__main__":
    # Extract phone numbers before passing to agents.cli
    from_number = ""
    to_number = ""
    if len(sys.argv) > 2:
        # Remove phone numbers from sys.argv to prevent them from being passed to agents.cli
        from_number = sys.argv[2]
        # to_number = sys.argv[3]
        sys.argv = sys.argv[:2]  # Keep only script name and "dev" command

    # Store phone numbers in environment variables to be accessed by entrypoint
    os.environ["CALL_FROM_NUMBER"] = from_number
    # os.environ["CALL_TO_NUMBER"] = to_number

    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
