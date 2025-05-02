import json
from dotenv import load_dotenv
import os
import sys
import asyncio
from typing import Optional
import threading

load_dotenv()

from browser_use import Agent as BrowserAgent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from langchain_openai import ChatOpenAI

from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from communication.sys_msgs import NEW_AGENT  # single prompt is enough here

notify_task_completed = None


# ---------------------------------------------------------------------------
#  1) THE PRIMARY VOICE ASSISTANT
# ---------------------------------------------------------------------------
class VoiceAssistant(Agent):
    """Converses with the user *and* orchestrates browser tasks.

    – Maintains an internal flag so it knows when a task is running.
    – Exposes 8 tools so the LLM can explicitly start tasks and query progress.
    – Receives task-completion notifications via `notify_task_completed()`.
    """

    # --------------------------- INIT ------------------------------------
    def __init__(self) -> None:
        super().__init__(instructions=NEW_AGENT)

        # Create a separate event-loop dedicated to browser tasks
        self._browser_loop = asyncio.new_event_loop()

        # Spin up a daemon thread that runs this loop forever
        def _run_loop(loop: asyncio.AbstractEventLoop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._browser_thread = threading.Thread(
            target=_run_loop, args=(self._browser_loop,), daemon=True
        )
        self._browser_thread.start()
        self._task_running: bool = False
        self._task_paused: bool = False
        self._latest_dialogue_window: list[dict[str, str]] = []
        self._last_task_result: Optional[str] = None
        self._last_step_results: list[str] = []
        self._browser = Browser(config=BrowserConfig(disable_security=True))
        self._browser_context = BrowserContext(browser=self._browser)
        self._browser_agent = BrowserAgent(
            task="You're a web assistant. Wait for user instructions.",
            llm=ChatOpenAI(model="gpt-4.1"),
            browser=self._browser,
            browser_context=self._browser_context,
        )
        # call the action and track the result

    async def set_last_task_result(self, result: BrowserAgent):
        """Set the result of the previous task (async to satisfy BrowserAgent callback)."""
        last_action = result.state.history.last_action()
        self._last_step_results.append(json.dumps({} if last_action is None else last_action))

    async def browser_run(self):
        """Run the browser agent to fulfill the task represented by the current conversation."""
        result = await self._browser_agent.run(on_step_end=self.set_last_task_result)
        result = json.loads(result.model_dump_json())
        history_list = []
        for history in result["history"]:
            history.pop("state")
            history.pop("metadata")
            history_list.append(history)
        result = json.dumps({"result": history_list}, indent=4)
        if notify_task_completed is not None:
            notify_task_completed(self, result)

    # --------------------------- TOOLS -----------------------------------
    @function_tool()
    async def is_task_running(self) -> bool:
        """Return *True* if a browser task is currently underway."""
        return self._task_running

    @function_tool()
    async def is_task_paused(self) -> bool:
        """Return *True* if a browser task is currently paused."""
        return self._task_paused

    @function_tool()
    async def create_task(self) -> str:
        """Send the latest user-assistant exchange to the browser helper.

        Should be called *after* the assistant has clarified the desired
        action and is ready to launch the task.
        """
        if self._task_running:
            return "I'm already working on something for you. Ask me anything else meanwhile!"

        self._browser_agent.add_new_task(
            "\n" + json.dumps(self._latest_dialogue_window, indent=4) + "\n"
        )
        if not self._task_paused:
            # reset the state of the browser agent
            self._task_running = True
            self._last_task_result = None
            self._last_step_results = []
            self._browser_agent.state.history.history = []

            # submit the coroutine to the dedicated browser loop safely
            fut = asyncio.run_coroutine_threadsafe(
                self.browser_run(),
                self._browser_loop,
            )

            # Log any exception that might occur in the background task
            def _log_bg_exc(fut):
                try:
                    fut.result()
                except Exception as e:
                    print("[BrowserLoop] task raised an exception:", e)

            fut.add_done_callback(_log_bg_exc)
        return "Alright, let me get on with that. I'll let you know how it goes!"

    @function_tool()
    async def pause_task(self) -> str:
        """Pause the current task."""
        if not self._task_running:
            return "No task is currently running."
        self._browser_agent.pause()
        self._task_running = False
        self._task_paused = True
        return "Task paused. You can resume it later."

    @function_tool()
    async def cancel_task(self) -> str:
        """Cancel the current task."""
        if not self._task_running:
            return "No task is currently running."
        self._browser_agent.stop()
        self._task_running = False

    @function_tool()
    async def resume_task(self) -> str:
        """Resume the current task."""
        if not self._task_running:
            return "No task is currently running."
        self._browser_agent.resume()
        self._task_running = True
        self._task_paused = False
        return "Task resumed. I'll let you know how it goes!"

    @function_tool()
    async def get_last_task_result(self) -> str:
        """Fetch the final result once a task has completed."""
        if self._task_running:
            return "Still working on it – I'll have an update soon."
        if self._last_task_result is None:
            return "There isn't a completed task yet."
        return self._last_task_result

    @function_tool()
    async def get_last_step_results(self) -> list[str]:
        """Fetch the step of the current running task."""
        if not self._task_running:
            return "No task is currently running."
        return self._last_step_results

    # -------------------- RUNTIME HOOKS (LiveKit) ------------------------
    async def on_user_turn_completed(self, turn_ctx, new_message):
        # Build a compact dialogue window to hand over to the browser helper
        self._latest_dialogue_window = [
            {msg.role: msg.content[0]}
            for msg in self.chat_ctx.items[1:]
            if msg.type not in ["function_call", "function_call_output"]
        ] + [{"user": new_message.text_content}]
        print(self._latest_dialogue_window)


# ---------------------------------------------------------------------------
#  2) Function for task completion
# ---------------------------------------------------------------------------
def notify_task_completed_wrapped(
    session: AgentSession, loop: asyncio.AbstractEventLoop
):
    """Return a thread-safe callback to be called from the browser thread."""

    async def _send_completion_reply(result: str):
        # coroutine executed inside the LiveKit loop
        await session.generate_reply(
            instructions=(
                'By the way, your requested browser task has completed. Here are the details:\n\n'
                f'{result}'
            )
        )

    def notify_task_completed(assistant: VoiceAssistant, result: str) -> None:
        # executed from the browser thread / loop
        assistant._task_running = False
        assistant._last_task_result = result

        # Define a function that will run in the LiveKit (main) loop
        def _on_main_thread():
            # interrupt any ongoing TTS/inference safely inside the right loop
            session.interrupt()
            # schedule the async reply coroutine
            asyncio.create_task(_send_completion_reply(result))

        # marshal to the main event-loop thread-safely
        loop.call_soon_threadsafe(_on_main_thread)

    return notify_task_completed


# ---------------------------------------------------------------------------
#  3) LIVEKIT ENTRYPOINT
# ---------------------------------------------------------------------------
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # Create the LiveKit voice session ------------------------------------
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4.1"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    # Capture the main LiveKit loop so we can safely schedule from the browser thread
    main_loop = asyncio.get_running_loop()

    assistant = VoiceAssistant()

    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Initial greeting ----------------------------------------------------
    await session.generate_reply(
        instructions=f"Hi {os.environ.get('FIRST_NAME')}! What can I do for you today?",
    )


# ---------------------------------------------------------------------------
#  4) SCRIPT LAUNCHER (mirror `make_call.py` behaviour)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("console")
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
