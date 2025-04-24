import os
import queue
import asyncio
from typing import AsyncIterable
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

from user_facing.sys_msgs import USER_FACING
from intermediaries.task_request_listener import TaskRequestListener

FIRST_NAME = os.environ["FIRST_NAME"]

import unify

unify.activate("Unity")

task_q: queue.Queue[list[str]] = queue.Queue()
async_task_q: asyncio.Queue[str] = asyncio.Queue()

listener = TaskRequestListener(task_q)
listener.start()


# ─────────────────────────────────────────────────────────────────────
async def _demo_interjector(async_q: "asyncio.Queue[str]") -> None:
    """
    Wait 20 s after startup, then push a dummy message that should
    interrupt whatever the agent is saying.  Runs only once.
    """
    await asyncio.sleep(20)
    await async_q.put("Okay, done! What next?")


# ─────────────────────────────────────────────────────────────────────


async def _bridge_blocking_to_async(
    blocking_q: "queue.Queue[list[str]]",
    async_q: "asyncio.Queue[str]",
) -> None:
    """
    Runs forever in the event-loop, pulling from the blocking queue
    (in a thread-safe way) and forwarding into the asyncio queue.
    """
    loop = asyncio.get_running_loop()
    while True:
        # .get() is blocking; run it in the default executor
        msgs = await loop.run_in_executor(None, blocking_q.get)
        if not msgs:
            continue
        # only take the *latest* text in that list
        await async_q.put(msgs[-1])


async def _speech_dispatcher(
    session: AgentSession,
    async_q: "asyncio.Queue[str]",
) -> None:
    """
    Waits for text, interrupts any current speech, and speaks the new text.
    """
    while True:
        next_text = await async_q.get()
        # 1) stop whatever is playing
        await session.interrupt()  # Docs: “Interrupt current speech”
        # 2) speak the fresh text
        await session.say(
            next_text,
            allow_interruptions=True,  # let the user break in again if needed
        )


class VoiceAssistant(Agent):

    def __init__(self) -> None:
        super().__init__(instructions=USER_FACING.replace("{first_name}", FIRST_NAME))

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        unify.log(
            sender=FIRST_NAME,
            receiver="Unity",
            medium="phone call",
            msg=new_message.text_content,
        )
        task_q.put(
            [msg.content[0] for msg in self.chat_ctx.items[1:]]
            + [new_message.text_content],
        )

    async def transcription_node(
        self,
        text: AsyncIterable[str],
        model_settings,
    ) -> AsyncIterable[str]:
        # This method receives the LLM output as an async stream of text.
        collected_chunks = []
        async for chunk in text:
            collected_chunks.append(chunk)
            # Yield the chunk onward so TTS (and any client transcript) receives it without delay
            yield chunk
        unify.log(
            sender="Unity",
            receiver=FIRST_NAME,
            medium="phone call",
            msg="".join(collected_chunks),
        )


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=VoiceAssistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # ── background tasks ──────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    # start the bridge that listens to the thread-based queue
    asyncio.create_task(_bridge_blocking_to_async(task_q, async_task_q))
    # start the dispatcher that speaks anything it receives
    asyncio.create_task(_speech_dispatcher(session, async_task_q))

    # one-off demo publisher
    asyncio.create_task(_demo_interjector(async_task_q))

    await session.generate_reply(
        instructions=f"Greet {FIRST_NAME} by name, and ask how it's going.",
    )


if __name__ == "__main__":
    # if len(sys.argv) == 1:          # no sub-command provided
    #     sys.argv.append("console")  # pretend the user typed “… console”
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
