import os
import sys
import asyncio
from typing import AsyncIterable
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()
FIRST_NAME = os.environ["FIRST_NAME"]

from user_facing.sys_msgs import PHONE_AGENT
from busses.bus_manager import BusManager


import unify

unify.activate("Unity")

bus_manager = BusManager()
bus_manager.start()


# ─────────────────────────────────────────────────────────────────────


async def _bridge_blocking_to_async() -> None:
    """
    Runs forever in the event-loop, pulling from the blocking queue
    (in a thread-safe way) and forwarding into the asyncio queue.
    """
    loop = asyncio.get_running_loop()
    while True:
        # .get() is blocking; run it in the default executor
        msgs = await loop.run_in_executor(None, bus_manager._transcript_q.get)
        if not msgs:
            continue
        # only take the *latest* text in that list
        await bus_manager.task_completion_q.put(msgs[-1])


async def _speech_dispatcher(
    session: AgentSession,
) -> None:
    """
    Waits for text, interrupts any current speech, and speaks the new text.
    """
    while True:
        next_text = await bus_manager.task_completion_q.get()
        # 1) stop whatever is playing
        await session.interrupt()  # Docs: “Interrupt current speech”
        # 2) speak the fresh text
        await session.say(
            next_text,
            allow_interruptions=True,  # let the user break in again if needed
        )


class VoiceAssistant(Agent):

    def __init__(self) -> None:
        super().__init__(instructions=PHONE_AGENT.replace("{first_name}", FIRST_NAME))

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        unify.log(
            context="Exchanges",
            sender=FIRST_NAME,
            receiver="Unity",
            medium="phone call",
            msg=new_message.text_content,
        )
        bus_manager.transcript_q.put(
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
            context="Exchanges",
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
    asyncio.create_task(_bridge_blocking_to_async())
    # start the dispatcher that speaks anything it receives
    asyncio.create_task(_speech_dispatcher(session))

    await session.generate_reply(
        instructions=f"Greet {FIRST_NAME} by name, and ask how it's going.",
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:  # no sub-command provided
        sys.argv.append("console")  # pretend the user typed “… console”
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
