"""
======================================================================
Interactive sandbox for **TranscriptManager**.

Features
--------
• Fixed or LLM-generated seed data via :class:`ScenarioBuilder`.
• Voice or plain-text input (shared helpers).
• Automatic dispatch to `ask` *or* `summarize` depending on intent.
• Mid-conversation interruption (pause / interject / cancel) for `ask` calls.
• Scenario builder tool-loop exposes **private** ``_log_messages`` alongside
  the public `ask` and `summarize` methods so that the LLM can inject raw
  transcripts directly.
"""

from __future__ import annotations

# ─────────────────────────────── stdlib / vendored ──────────────────────────
import os
import argparse
import asyncio
import logging
import re
import select
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union

import unify

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from scenario_builder import ScenarioBuilder

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

# ────────────────────────────────  unity imports  ───────────────────────────
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.common.llm_helpers import SteerableToolHandle
from sandboxes.utils import (  # shared helpers reused in other sandboxes
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    run_in_loop,
    get_custom_scenario,
)

LG = logging.getLogger("transcript_sandbox")

# ═════════════════════════════════ seed helpers ═════════════════════════════


async def _build_scenario(custom: Optional[str] = None) -> Optional[str]:
    """Populate the transcript store **via the official tools** using
    :class:`ScenarioBuilder`.

    The tool-loop exposes:
    • ``_log_messages`` – for inserting raw transcript messages.
    • ``ask``             – so the LLM can sanity-check its inserts.
    • ``summarize``       – optional summarisation during seeding.
    """

    tm = TranscriptManager()

    description = (
        custom.strip()
        if custom
        else (
            "Generate 15 realistic message exchanges across email, Slack and "
            "WhatsApp between 5 colleagues over the last two weeks. Vary the "
            "topics (project updates, meeting scheduling, casual banter). "
            "Provide rich, time-ordered message content so that questions "
            "about context, participants and timing are interesting."
        )
    )
    description += (
        "\nYou have three tools:\n"
        "• `_log_messages` — write raw messages (you *must* supply all schema "
        "  fields).\n"
        "• `ask` — query what you've created.\n"
        "• `summarize` — store concise summaries of exchanges.\n"
        "Work in batches: insert several related messages at once, then call "
        "`ask` to confirm, and finally `summarize` important threads."
    )

    builder = ScenarioBuilder(
        description=description,
        tools={
            "_log_messages": tm._log_messages,  # IMPORTANT: private helper
            "ask": tm.ask,
            "summarize": tm.summarize,
        },
    )

    try:
        await builder.create()
    except Exception as exc:
        raise RuntimeError(f"LLM seeding via ScenarioBuilder failed. {exc}")

    return None  # kept for signature parity with other sandboxes


# ═════════════════════════════ intent dispatcher ════════════════════════════


class _Intent(BaseModel):
    """Lightweight schema used when delegating intent detection to an LLM."""

    action: str = Field(..., pattern=r"^(ask|summarize)$")
    cleaned_text: str


_INTENT_SYS_MSG = (
    "Decide whether the user input is a question about the transcripts "
    "(`ask`) or a summarisation request (`summarize`). Summarisation inputs "
    "may mention one or more *exchange IDs* (integers). Return JSON "
    "{'action':'ask'|'summarize','cleaned_text':<fixed_input>}."
)


_EXCHANGE_ID_RE = re.compile(r"\b\d+\b")


def _extract_exchange_ids(text: str) -> List[int]:
    """Return all integer tokens found in *text* as a list of `int`."""

    return [int(m.group()) for m in _EXCHANGE_ID_RE.finditer(text)]


async def _dispatch_with_context(
    tm: TranscriptManager,
    raw: str,
    *,
    show_steps: bool,
    parent_chat_context: List[Dict[str, str]],
) -> Tuple[str, Union[SteerableToolHandle, str]]:
    """Route *raw* to either `ask` or `summarize`, forwarding chat context."""

    lowered = raw.lower()

    # ───── heuristic fast-paths ───────────────────────────────────────────
    if lowered.startswith(("summarize ", "summarise ", "summary ")):
        ids = _extract_exchange_ids(raw)
        if not ids:
            raise ValueError(
                "Please mention at least one exchange/thread ID to summarise.",
            )
        # Guidance is whatever remains after stripping the first word & IDs
        guidance = " ".join(w for w in raw.split()[1:] if not w.isdigit()).strip()
        summary = await tm.summarize(exchange_ids=ids, guidance=guidance or None)
        return "summarize", summary

    # ───── otherwise maybe ask an LLM judge (rare) ───────────────────────
    judge = unify.Unify("gpt-4o@openai", response_format=_Intent)
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )

    if intent.action == "summarize":
        ids = _extract_exchange_ids(intent.cleaned_text)
        if not ids:
            raise ValueError(
                "The summarisation request didn't include any exchange IDs.",
            )
        guidance = " ".join(
            w for w in intent.cleaned_text.split() if not w.isdigit()
        ).strip()
        summary = await tm.summarize(exchange_ids=ids, guidance=guidance or None)
        return "summarize", summary

    # default: ask
    handle = await tm.ask(
        raw,
        parent_chat_context=parent_chat_context,
        _return_reasoning_steps=show_steps,
    )
    return "ask", handle


# ═════════════════════════════ interruption helper ══════════════════════════


def _input_now(timeout: float = 0.1) -> Optional[str]:
    """Non-blocking stdin check (POSIX & Windows)."""
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.readline().strip() if r else None


async def _await_with_interrupt(handle: SteerableToolHandle) -> str:
    """Wait on *handle*, permitting live user interjection."""

    while not handle.done():
        txt = _input_now(0.1)
        if txt:
            if txt.lower() in {"stop", "cancel"}:
                handle.stop()
                break
            run_in_loop(handle.interject(txt))
        await asyncio.sleep(0.05)

    return await handle.result()


# ══════════════════════════════════  CLI  ═══════════════════════════════════


async def _main_async() -> None:
    parser = argparse.ArgumentParser(description="TranscriptManager sandbox")
    parser.add_argument(
        "--voice",
        "-v",
        action="store_true",
        help="enable voice capture + TTS",
    )
    parser.add_mutually_exclusive_group()
    parser.add_argument("--reuse", "-r", action="store_true", help="re-use old data")
    parser.add_argument("--debug", "-d", action="store_true", help="verbose tool logs")
    parser.add_argument("--traced", "-t", action="store_true", help="include tracing")
    args = parser.parse_args()

    # tracing flag → env var consumed by `unify`
    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    # prepare Unify context
    unify.activate("TranscriptSandbox")
    unify.set_trace_context("Traces")
    if not args.reuse:
        ctxs = unify.get_contexts()
        # Remove everything under the sandbox context for a clean run
        for ctx in list(ctxs):
            if ctx.startswith("TranscriptSandbox") or ctx.startswith("Traces"):
                unify.delete_context(ctx)
        unify.create_context("Traces")

    # logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LG.setLevel(logging.INFO)

    # manager (optionally traced)
    tm: TranscriptManager = TranscriptManager()
    if args.traced:
        tm = unify.traced(tm)

    # seed synthetic transcripts unless --reuse
    if not args.reuse:
        # custom scenario
        scenario_text = get_custom_scenario(args)
        if scenario_text:
            LG.info("[voice] transcript: “%s”", scenario_text)
            LG.info("[seed] building synthetic transcript store – can take 30-60 s…")
            if args.voice:
                _speak("Sure thing, building your custom scenario now.")
            theme = await _build_scenario(scenario_text)
            LG.info("[seed] done.")
            if args.voice:
                _speak(
                    "All done, your custom scenario is built and ready to go.",
                )
            if theme:
                LG.info(f"[seed] theme: {theme}")
        else:
            raise Exception("No text provided for building the custom scenario")

    print("TranscriptManager sandbox – type or speak. 'quit' to exit.\n")

    _speak(
        "Press enter to ask questions or request a summary from the transcripts.",
    )

    # running memory of the dialogue (passed back into tm.ask for context)
    chat_history: List[Dict[str, str]] = []

    # interaction loop
    while True:
        try:
            # ─────────────────── capture input (voice / text) ──────────────────
            if args.voice:
                audio = _record_until_enter()
                raw = _transcribe_deepgram(audio).strip()
                if not raw:
                    continue
                print(f"▶️  {raw}")
            else:
                raw = input("> ").strip()

            if raw.lower() in {"quit", "exit"}:
                break
            if not raw:
                continue

            # ───────────── remember the user's utterance before dispatch ──────
            _kind, result = await _dispatch_with_context(
                tm,
                raw,
                show_steps=args.debug,
                parent_chat_context=list(chat_history),
            )
            chat_history.append({"role": "user", "content": raw})
            if args.voice:
                _speak("Let me take a look, give me a moment")

            # ───────────── process result (handle or immediate string) ─────────
            if isinstance(result, SteerableToolHandle):
                answer = await _await_with_interrupt(result)
                if isinstance(answer, tuple):  # reasoning steps requested
                    answer, _steps = answer
            else:  # already a string (summarize)
                answer = result

            if args.voice:
                _speak("Okay, that's all done")
            print(f"[{_kind}] → {answer}\n")

            # ───────────── remember assistant's reply for follow-up ────────────
            chat_history.append({"role": "assistant", "content": answer})
            if args.voice:
                _speak(f"{answer} Anything else I can help with?")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break
        except Exception as exc:
            LG.error("[error] %s", exc)


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
