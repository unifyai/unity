import os
from typing import AsyncIterable
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from sys_msgs import COMMUNICATOR

FIRST_NAME = os.environ["FIRST_NAME"]

import unify

unify.activate("Unity")


load_dotenv()


class Assistant(Agent):

    def __init__(self) -> None:
        super().__init__(instructions=COMMUNICATOR.replace("{first_name}", FIRST_NAME))

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        unify.log(
            sender=FIRST_NAME,
            receiver="Unity",
            medium="phone call",
            msg=new_message.text_content,
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
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions=f"Greet {FIRST_NAME} by name, and ask how it's going.",
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
