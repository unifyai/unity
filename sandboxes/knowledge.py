"""knowledge_sandbox.py  (voice mode, Deepgram SDK v4, sync)
================================================================
Interactive sandbox for **KnowledgeManager** with optional voice input.

Features
--------
* Fixed richly‑seeded scenario covering multiple tables and attributes
  (people, products, purchases, geometry, pets, misc facts).
* `--scenario llm` flag – generate 120‑180 factual sentences via LLM and
  ingest them automatically.
* Shared audio/STT/TTS helpers imported from `utils.py`.
* Minimal dispatcher routes utterances to `KnowledgeManager.store` or
  `.retrieve` using a lightweight LLM intent/cleanup step.
* Supports interruptions during LLM processing via AsyncToolLoopHandle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import select
import os
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unify

from unity.constants import LOGGER as _LG  # type: ignore
from unity.knowledge_manager.knowledge_manager import KnowledgeManager  # type: ignore
from unity.common.llm_helpers import AsyncToolLoopHandle  # type: ignore
from sandboxes.utils import (
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    run_in_loop,
    get_custom_scenario,
)  # type: ignore


# ---------------------------------------------------------------------------
# Scenario seeding
# ---------------------------------------------------------------------------


def _seed_fixed(km: KnowledgeManager) -> None:
    """Populate KnowledgeManager with a deterministic, multi‑table world."""
    seed_statements: List[str] = [
        # People & attributes
        "Adrian was born in 1994.",
        "Bob is 35 years old.",
        "Bob's favourite colour is green and his height is 180 centimetres.",
        "Carol owns a dog named Fido.",
        "Carol also owns a cat named Luna.",
        "Daniel's employee ID is E‑421 and he works in the London office.",
        # Products & purchases
        "The Apple iPhone 15 costs 999 US dollars.",
        "Daniel bought an iPhone 15 on 3 May 2025 using his credit card.",
        "A Logitech MX Master 4 mouse costs 129 US dollars.",
        "Sara ordered two MX Master 4 mice on 1 May 2025.",
        # Geometry
        "Point P has coordinates x = 3 and y = 4.",
        "Point Q has coordinates x = 1 and y = 10.",
        "Point R has coordinates x = ‑5 and y = 7.",
        # Random knowledge
        "The capital of Spain is Madrid.",
        "Mount Everest is 8,848 metres tall.",
        "The chemical symbol for gold is Au.",
    ]

    for stmt in seed_statements:
        km.store(stmt)


def _seed_llm(
    km: KnowledgeManager,
    custom_scenario: Optional[str] = None,
) -> Optional[str]:
    """Use an LLM to generate a large set of factual sentences."""
    if custom_scenario:
        prompt = (
            f"User-provided scenario:\n{custom_scenario}\n\n"
            "Generate 120‑180 short factual sentences based on this scenario for ingestion by a knowledge base. "
            "Avoid any personally identifying sensitive data. "
            'Return as JSON {"statements": [...], "theme": <string>} and nothing else.'
        )
    else:
        prompt = (
            "Generate 120‑180 short factual sentences suitable for ingestion by a knowledge base. "
            "Cover diverse domains: personal bios, product pricing, purchases, geography, science facts, pets, coordinates, sports scores etc. "
            "Avoid any personally identifying sensitive data. "
            'Return as JSON {"statements": [...], "theme": <string>} and nothing else.'
        )
    client = unify.Unify(
        "o4-mini@openai",
        cache=eval(os.environ.get("UNIFY_CACHE")),
        traced=eval(os.environ.get("UNIFY_TRACED")),
    )
    client.set_system_message(prompt)
    raw = client.generate("Produce knowledge scenario").strip()

    try:
        payload = json.loads(raw)
    except Exception:
        _LG.warning("LLM scenario failed – falling back to fixed seed.")
        _seed_fixed(km)
        return None

    for stmt in payload.get("statements", []):
        km.store(stmt)

    return payload.get("theme")


# ---------------------------------------------------------------------------
# Dispatcher – decide between store vs retrieve
# ---------------------------------------------------------------------------


class _IntentResp(BaseModel):
    action: str = Field(..., description="either 'store' or 'retrieve'")
    cleaned_text: str


_INTENT_PROMPT = (
    "You interface with a knowledge base. Incoming text may be: (a) a fact to be stored, or (b) a question to be answered via retrieval. "
    "If it ends with a question mark, it's *probably* a retrieval, but clarifying words like 'remember', 'note that', 'please store', etc. force storage. "
    "Respond with JSON {action:'store'|'retrieve', cleaned_text:'<sanitised>'}. For storage, cleaned_text should be the fact statement; for retrieval, it should be the user question."
)


async def _dispatch(
    km: KnowledgeManager,
    raw: str,
    *,
    show_steps: bool,
) -> Tuple[str, AsyncToolLoopHandle, List | None]:
    raw = raw.strip()

    # Quick rule: voice input often lacks punctuation – fall back to heuristic + LLM judge if ambiguous
    heuristic_store = bool(
        re.match(r"^(remember|note|store|add)\b", raw, re.I),
    ) and not raw.endswith("?")

    if heuristic_store:
        handle = km.store(raw, return_reasoning_steps=show_steps)
        return "store", handle, None

    llm = unify.Unify("gpt-4o@openai", response_format=_IntentResp)
    intent_json = llm.set_system_message(_INTENT_PROMPT).generate(raw)
    intent = _IntentResp.model_validate_json(intent_json)

    if intent.action == "store":
        handle = km.store(intent.cleaned_text, return_reasoning_steps=show_steps)
        return "store", handle, None

    # Retrieval path
    handle = km.retrieve(intent.cleaned_text, return_reasoning_steps=show_steps)
    return "retrieve", handle, None


# ---------------------------------------------------------------------------
# Input polling helpers
# ---------------------------------------------------------------------------


def _poll_for_input(timeout: float = 0.1) -> Optional[str]:
    """Non-blocking check for user input from stdin."""
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None

    line = sys.stdin.readline().strip()
    return line if line else None


async def _handle_interruptions(
    handle: AsyncToolLoopHandle,
    answer_task: asyncio.Task,
    *,
    voice_mode: bool = False,
) -> str:
    """
    Poll for user interruptions while waiting for the answer task to complete.
    Returns the kind of operation and the final result.
    """
    result = ""

    try:
        # Loop until the answer task is done
        while not answer_task.done():
            # Check for user input
            if voice_mode:
                # In voice mode, we just check if the user pressed Enter
                user_input = _poll_for_input(0.1)
                if user_input is not None:
                    print("⚠️ Interruption detected. Recording new input...")
                    audio_bytes = await asyncio.to_thread(_record_until_enter)
                    user_text = await asyncio.to_thread(
                        _transcribe_deepgram,
                        audio_bytes,
                    )
                    if user_text:
                        print(f"▶️  New input: {user_text}")
                        if user_text.lower() in {"stop", "cancel"}:
                            print("🛑 Stopping current operation...")
                            handle.stop()
                        else:
                            print("⚡ Interjecting new information...")
                            run_in_loop(handle.interject(user_text))
            else:
                # In text mode, we check for any input
                user_input = _poll_for_input(0.1)
                if user_input is not None:
                    if user_input.lower() in {"stop", "cancel"}:
                        print("🛑 Stopping current operation...")
                        handle.stop()
                    else:
                        print(f"⚡ Interjecting: {user_input}")
                        run_in_loop(handle.interject(user_input))

            # Small sleep to prevent CPU spinning
            await asyncio.sleep(0.1)

        # Get the result from the completed task
        result = answer_task.result()

        # If we have a tuple (for return_reasoning_steps=True), extract just the answer
        if isinstance(result, tuple) and len(result) >= 1:
            result = result[0]

        return result
    except asyncio.CancelledError:
        return "Operation was cancelled."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _main_async(args, scenario_text: Optional[str] = None) -> None:
    # Logging
    if not args.silent:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        _LG.setLevel(logging.INFO)
        if not args.debug:
            for noisy in ("unify", "unify.utils", "unify.logging", "requests", "httpx"):
                logging.getLogger(noisy).setLevel(logging.WARNING)

    # Unify project context
    unify.activate("KnowledgeSandbox")
    fresh = "Knowledge" not in unify.get_contexts() or args.new
    unify.set_context("Knowledge", overwrite=fresh)

    # Manager
    km = KnowledgeManager()

    if fresh:
        if scenario_text is not None:
            theme = _seed_llm(km, custom_scenario=scenario_text)
            if theme:
                _LG.info(f"[Seed] LLM scenario theme: {theme}")
        elif args.scenario == "llm":
            theme = _seed_llm(km)
            if theme:
                _LG.info(f"[Seed] LLM scenario theme: {theme}")
        else:
            _seed_fixed(km)

    print("KnowledgeManager sandbox – speak or type. 'quit' to exit.")
    print("Press Enter during processing to interject or type 'stop' to cancel.\n")

    # Interaction loop
    if args.voice:
        while True:
            audio_bytes = _record_until_enter()
            user_text = _transcribe_deepgram(audio_bytes).strip()
            if not user_text:
                continue
            print(f"▶️  {user_text}")
            if user_text.lower() in {"quit", "exit"}:
                break

            _speak("Working on this now…")

            # Get the handle and create a task for the result
            kind, handle, steps = await _dispatch(
                km,
                user_text,
                show_steps=not args.silent,
            )
            answer_task = asyncio.create_task(handle.result())

            # Handle interruptions while waiting for the result
            result = await _handle_interruptions(
                handle,
                answer_task,
                voice_mode=True,
            )

            print(f"[{kind}] => {result}\n")
            if kind == "retrieve":
                _speak(result)
    else:
        try:
            while True:
                line = input("> ").strip()
                if line.lower() in {"quit", "exit"}:
                    break
                if not line:
                    continue

                # Get the handle and create a task for the result
                kind, handle, steps = await _dispatch(
                    km,
                    line,
                    show_steps=not args.silent,
                )
                answer_task = asyncio.create_task(handle.result())

                # Handle interruptions while waiting for the result
                result = await _handle_interruptions(handle, answer_task)

                print(f"[{kind}] => {result}\n")
        except (EOFError, KeyboardInterrupt):
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KnowledgeManager sandbox with shared voice mode",
    )
    parser.add_argument(
        "--voice",
        "-v",
        action="store_true",
        help="enable voice capture/playback",
    )
    parser.add_argument(
        "--scenario",
        choices=["fixed", "llm"],
        default="fixed",
        help="scenario type (overridden by --custom-scenario flags)",
    )
    parser.add_argument("--new", "-n", action="store_true", help="wipe & reseed data")
    parser.add_argument(
        "--silent",
        "-s",
        action="store_true",
        help="suppress tool logs",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="verbose HTTP/LLM logs",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--custom-scenario", type=str, help="Provide a custom scenario")
    group.add_argument(
        "--custom-scenario-voice",
        action="store_true",
        help="Describe custom scenario via voice",
    )

    args = parser.parse_args()

    scenario_text = get_custom_scenario(args, silent=args.silent)

    # Run the async main function
    asyncio.run(_main_async(args, scenario_text))


if __name__ == "__main__":
    main()
