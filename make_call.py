import os
import sys
import asyncio
import random
import logging
from datetime import datetime, timezone
from typing import AsyncIterable
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Hack to prevent terminal writies
from livekit.agents.voice import chat_cli


def _silent_print_audio_mode(self, *args, **kwargs):
    sys.stdout.flush()


chat_cli.ChatCLI._print_audio_mode = _silent_print_audio_mode
# End hack

logging.disable(logging.CRITICAL)

load_dotenv()
FIRST_NAME = os.environ["FIRST_NAME"]

from user_facing.sys_msgs import PHONE_AGENT
from busses.bus_manager import BusManager


import unify

unify.activate("Unity")

bus_manager = BusManager()


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
            random.choice(
                ("And... Done.", "Done.", "Finished.", "That's now done."),
            )
            + random.choice(
                (
                    "Anything else?",
                    "What next?",
                    "Anything else I can do?",
                    "Need anything else?",
                    "",
                ),
            ),
            allow_interruptions=True,  # let the user break in again if needed
        )


class VoiceAssistant(Agent):

    def __init__(self) -> None:
        super().__init__(instructions=PHONE_AGENT)

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
        print(f"\n🎙️ Transcribed user speach [{t}]\n")
        unify.log(
            context="Exchanges",
            sender=FIRST_NAME,
            receiver="Unity",
            medium="phone call",
            msg=new_message.text_content,
        )
        msgs = [{msg.role: msg.content[0]} for msg in self.chat_ctx.items[1:]] + [
            {"user": new_message.text_content},
        ]
        bus_manager.transcript_q.put(msgs)

    async def transcription_node(
        self,
        text: AsyncIterable[str],
        model_settings,
    ) -> AsyncIterable[str]:
        t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
        print(f"\n🔈 Playing assistant audio [{t}]\n")
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
    voice_loop = asyncio.get_running_loop()
    bus_manager.set_coms_asyncio_loop(voice_loop)
    bus_manager.start()

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
    # start the dispatcher that speaks anything it receives
    asyncio.create_task(_speech_dispatcher(session))

    await session.generate_reply(
        instructions=f"Greet {FIRST_NAME} by name, and ask how it's going.",
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:  # no sub-command provided
        sys.argv.append("console")  # pretend the user typed “… console”
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
