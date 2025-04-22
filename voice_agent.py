import os
from typing import AsyncIterable

import unify
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

unify.activate("Stream LiveKit")
unify.initialize_async_logger()

load_dotenv()


class Assistant(Agent):

    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        unify.log(name=os.environ["FIRST_NAME"], msg=new_message.text_content)

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
        # Once the LLM output stream is done, combine into full reply and log it
        unify.log(name="Unity", msg="".join(collected_chunks))


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
        instructions="Greet the user and offer your assistance.",
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
