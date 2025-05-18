"""
text_chat_app.py  ──────────────────────────────────────────────────────────

Pure‑text, asynchronous console chat that **re‑uses the same BusManager
queues** (`transcript_q` and `task_completion_q`) employed by the LiveKit
voice demo.  Swap the `call_llm()` coroutine with your real LLM invocation.

Features
========
• Either side can type multiple messages in a row ("double‑messaging").
• "Unity is typing…" spinner while the agent is generating a reply.
• Every time a user/assistant message is added, the full transcript list is
  pushed into `bus_manager.transcript_q`.
• A background task listens to `bus_manager.task_completion_q` and prints a
  confirmation line when a task has completed.
• No external deps – just Python ≥3.9 and your own `busses.bus_manager`.

Run with:  python text_chat_app.py      (Ctrl‑C or Ctrl‑D to quit)
"""

from __future__ import annotations

import asyncio
import random
import sys
from datetime import datetime, timezone
from typing import List, Dict

from unity.constants import LOGGER
from unity.busses.bus_manager import BusManager  # ← your existing helper

# ─────────── configuration ──────────────────────────────────────────────
USER_NAME = "You"
AGENT_NAME = "Unity"

# ─────────── global state & queues ─────────────────────────────────────
user_to_agent_q: asyncio.Queue[str] = asyncio.Queue()
agent_to_user_q: asyncio.Queue[str] = asyncio.Queue()
chat_history: List[Dict[str, str]] = []  # shared conversational memory

# BusManager instance (thread‑safe queues to rest of the system) ----------
bus_manager = BusManager()


# ─────────── placeholder LLM call ───────────────────────────────────────
async def call_llm(transcript: List[Dict[str, str]]) -> str:
    """Simulate an LLM call; replace with your own HTTP request."""
    await asyncio.sleep(random.uniform(0.8, 2.0))  # pretend to think
    last_user_msg = transcript[-1]["user"]
    return f"I heard you say: '{last_user_msg}'.  (LLM stub response)"


# ─────────── workers ────────────────────────────────────────────────────
async def user_input_worker() -> None:
    """Read lines from stdin in a thread; enqueue for the agent."""
    while True:
        try:
            line = await asyncio.to_thread(input, f"{USER_NAME}: ")
        except EOFError:
            break  # Ctrl‑D / pipe closed
        if not line.strip():
            continue  # skip blanks
        timestamp = datetime.now(timezone.utc).time().isoformat(timespec="seconds")
        LOGGER.info(f"[📤 {timestamp}] Sent")

        # Update transcript & broadcast through BusManager
        chat_history.append({"user": line})
        bus_manager.transcript_q.put(chat_history.copy())

        await user_to_agent_q.put(line)


async def agent_worker() -> None:
    """Consume user messages, generate replies, broadcast back."""
    while True:
        user_msg = await user_to_agent_q.get()

        # ─ show typing spinner ─
        typing_done = asyncio.Event()
        spinner_task = asyncio.create_task(_typing_spinner(typing_done))

        reply = await call_llm(chat_history)
        typing_done.set()
        await spinner_task

        chat_history.append({"assistant": reply})
        bus_manager.transcript_q.put(chat_history.copy())

        await agent_to_user_q.put(reply)


async def display_agent_worker() -> None:
    """Display assistant replies as they arrive."""
    while True:
        reply = await agent_to_user_q.get()
        timestamp = datetime.now(timezone.utc).time().isoformat(timespec="seconds")
        LOGGER.info(f"{AGENT_NAME} [📥 {timestamp}]: {reply}")


async def task_completion_listener() -> None:
    """Listen for completion messages and announce them."""
    while True:
        msg = await bus_manager.task_completion_q.get()
        timestamp = datetime.now(timezone.utc).time().isoformat(timespec="seconds")
        LOGGER.info(f"\n✅ Task completed [⏱ {timestamp}]: {msg}\n")


async def _typing_spinner(done_evt: asyncio.Event) -> None:
    frames = ["   ", ".  ", ".. ", "..."]
    i = 0
    while not done_evt.is_set():
        frame = frames[i % len(frames)]
        LOGGER.info(f"\r{AGENT_NAME} is typing{frame}", end="", flush=True)
        await asyncio.sleep(0.4)
        i += 1
    # clear line
    LOGGER.info("\r" + " " * (len(AGENT_NAME) + 15) + "\r", end="", flush=True)


# ─────────── orchestrator ───────────────────────────────────────────────
async def main() -> None:
    # Hook BusManager into this event‑loop before starting its threads
    loop = asyncio.get_running_loop()
    bus_manager.set_coms_asyncio_loop(loop)
    bus_manager.start()

    await asyncio.gather(
        user_input_worker(),
        agent_worker(),
        display_agent_worker(),
        task_completion_listener(),
    )


# ─────────── entry ‑ point ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("\nBye!")
        sys.exit(0)
