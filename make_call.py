from dotenv import load_dotenv

load_dotenv()
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
import argparse
import asyncio
import json
import random
import redis
import pathlib
import logging
import logging.config
from datetime import datetime, timezone
from typing import AsyncIterable
from livekit import agents
from livekit.agents.voice import chat_cli
from livekit.agents import Agent, AgentSession, RoomInputOptions, function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from communication.sys_msgs import NEW_AGENT, PHONE_AGENT
from busses.bus_manager import BusManager
from constants import SESSION_ID

import unify
from state import State

# ── bootstrap (runs before other imports) ──────────────────────────────
_boot = argparse.ArgumentParser(add_help=False)
_boot.add_argument(
    "--log",
    metavar="FILE",
    help="Write Unify logs to FILE instead of stderr",
)
_opts, _rest = _boot.parse_known_args()
# scrub the option so nobody else complains if they re-parse argv later
sys.argv[:] = [sys.argv[0], *_rest]


def _setup_logging(dest: str | None) -> None:
    handlers = {}
    active = []

    if dest:  #  --log given
        path = pathlib.Path(dest).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.FileHandler",
            "filename": str(path),
            "encoding": "utf-8",
            "mode": "a",
            "formatter": "default",
        }
        active = ["file"]
    else:  # default: stderr
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "stream": sys.stderr,
            "formatter": "default",
        }
        active = ["console"]

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": True,
            "formatters": {
                "default": {
                    "format": "%(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": handlers,
            "root": {"level": "CRITICAL", "handlers": []},
            "loggers": {
                "unity": {
                    "level": "INFO",
                    "handlers": active,
                    "propagate": False,
                },
            },
        },
    )


_setup_logging(_opts.log)
from constants import LOGGER

# ── end bootstrap ─────────────────────────────────────────────────────


# Hack to prevent terminal writies
def _silent_print_audio_mode(self, *args, **kwargs):
    sys.stdout.flush()


chat_cli.ChatCLI._print_audio_mode = _silent_print_audio_mode
# End hack

FIRST_NAME = os.environ["FIRST_NAME"]

unify.activate("Unity")


bus_manager = BusManager(with_browser_use=bool(os.environ.get("OFF_THE_SHELF", False)))


async def _speech_dispatcher(
    session: AgentSession,
) -> None:
    """
    Waits for text, interrupts any current speech, and speaks the new text.
    """
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    pubsub = redis_client.pubsub()
    pubsub.subscribe("task_completion")
    for task_completion in pubsub.listen():
        if task_completion["type"] != "message":
            continue
        # 1) stop whatever is playing
        await session.interrupt()  # Docs: "Interrupt current speech"
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
        super().__init__(instructions=NEW_AGENT)  # PHONE_AGENT)
        self._redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self._latest_dialogue_window: list[dict[str, str]] = []
        self._state = State()

    # --------------------------- TOOLS -----------------------------------
    @function_tool()
    async def is_task_running(self) -> bool:
        """Return *True* if a browser task is currently underway."""
        return self._state.task_running

    @function_tool()
    async def is_task_paused(self) -> bool:
        """Return *True* if a browser task is currently paused."""
        return self._state.task_paused

    @function_tool()
    async def create_task(self) -> str:
        """Send the latest user-assistant exchange to the browser helper.

        Should be called *after* the assistant has clarified the desired
        action and is ready to launch the task.
        """
        if self._state.task_running:
            return "I'm already working on something for you. Ask me anything else meanwhile!"
        cmd = json.dumps(
            {"action": "create_task", "payload": self._latest_dialogue_window}
        )
        self._redis_client.publish("transcript", cmd)
        return "Alright, let me get on with that. I'll let you know how it goes!"

    @function_tool()
    async def pause_task(self) -> str:
        """Pause the current task."""
        if not self._state.task_running:
            return "No task is currently running."
        self._redis_client.publish("command", json.dumps({"action": "pause_task"}))
        return "Task paused. You can resume it later."

    @function_tool()
    async def cancel_task(self) -> str:
        """Cancel the current task."""
        if not (self._state.task_running or self._state.task_paused):
            return "No task is currently running."
        self._redis_client.publish("command", json.dumps({"action": "cancel_task"}))
        return "Task cancelled. Let me know if there's anything else."

    @function_tool()
    async def resume_task(self) -> str:
        """Resume the current task."""
        if not self._state.task_paused:
            return "No task is currently paused."
        self._redis_client.publish("command", json.dumps({"action": "resume_task"}))
        return "Task resumed."

    @function_tool()
    async def get_last_task_result(self) -> str:
        """Fetch the final result once a task has completed."""
        if self._state.task_running:
            return "Still working on it – I'll have an update soon."
        result = self._state.last_task_result
        if not result:
            return "There isn't a completed task yet."
        return result

    @function_tool()
    async def get_last_step_results(self) -> list[str]:
        """Fetch the steps of the current running task (oldest first)."""
        if not (self._state.task_running or self._state.task_paused):
            return ["No task is currently running."]
        return self._state.last_step_results

    # -------------------- RUNTIME HOOKS (LiveKit) ------------------------
    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
        LOGGER.info(f"\n🎙️ Transcribed user speech [⏱️ {t}]\n{new_message.text_content}")
        unify.log(
            context="Transcripts",
            session_id=SESSION_ID,
            sender=FIRST_NAME,
            receiver="Unity",
            medium="phone call",
            msg=new_message.text_content,
        )
        self._latest_dialogue_window = [
            {msg.role: msg.content[0]}
            for msg in self.chat_ctx.items[1:]
            if msg.type not in ["function_call", "function_call_output"]
        ] + [{"user": new_message.text_content}]

    async def transcription_node(
        self,
        text: AsyncIterable[str],
        model_settings,
    ) -> AsyncIterable[str]:
        t = datetime.now(timezone.utc).time().isoformat(timespec="milliseconds")
        LOGGER.info(f"\n🔈 Playing assistant audio [⏱️ {t}]\n")
        # This method receives the LLM output as an async stream of text.
        collected_chunks = []
        async for chunk in text:
            collected_chunks.append(chunk)
            # Yield the chunk onward so TTS (and any client transcript) receives it without delay
            yield chunk
        unify.log(
            context="Transcripts",
            session_id=SESSION_ID,
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
        llm=openai.LLM(model="o4-mini"),
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
    # asyncio.create_task(_speech_dispatcher(session))

    await session.generate_reply(
        instructions=f"Greet {FIRST_NAME} by name, and ask how it's going.",
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:  # no sub-command provided
        sys.argv.append("console")  # pretend the user typed "… console"
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
