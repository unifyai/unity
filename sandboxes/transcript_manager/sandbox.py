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
import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
from datetime import datetime

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
    get_custom_scenario,
    await_with_interrupt as _await_with_interrupt,
    build_cli_parser,
    activate_project,
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

    def log_messages(messages: list[dict]) -> str:
        for msg in messages:
            tm.log_messages(msg)
        tm.join_published()
        return "messages logged successfully"

    builder = ScenarioBuilder(
        description=description,
        tools={
            "log_messages": log_messages,
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
        summary = await tm.summarize(from_exchanges=ids, guidance=guidance or None)
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
        summary = await tm.summarize(from_exchanges=ids, guidance=guidance or None)
        return "summarize", summary

    # default: ask
    handle = await tm.ask(
        raw,
        parent_chat_context=parent_chat_context,
        _return_reasoning_steps=show_steps,
    )
    return "ask", handle


# ══════════════════════════════════  CLI  ═══════════════════════════════════


async def _main_async() -> None:
    parser = build_cli_parser("TranscriptManager sandbox")
    args = parser.parse_args()

    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    activate_project(args.project_name, args.overwrite)
    unify.set_trace_context("Traces")
    if args.overwrite:
        ctxs = unify.get_contexts()
        for tbl in (
            "Transcripts",
            "Contacts",
            "Traces",
        ):
            if tbl in ctxs:
                unify.delete_context(tbl)
        unify.create_context("Traces")

    # ─────────────────── project version handling ────────────────────
    if args.project_version != -1:
        commits = unify.get_project_commits(args.project_name)
        if commits:
            try:
                target = commits[args.project_version]
                unify.rollback_project(args.project_name, target["commit_hash"])
                LG.info("[version] Rolled back to commit %s", target["commit_hash"])
            except IndexError:
                LG.warning(
                    "[version] project_version index %s out of range, ignoring",
                    args.project_version,
                )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LG.setLevel(logging.INFO)

    tm: TranscriptManager = TranscriptManager()
    if args.traced:
        tm = unify.traced(tm)

    scenario_text: Optional[str] = get_custom_scenario(args)
    LG.info("[seed] building synthetic transcript store – can take 30-60 s…")
    if args.voice:
        _speak("Sure thing, building your custom scenario now.")
    await _build_scenario(scenario_text)
    LG.info("[seed] done.")
    if args.voice:
        _speak("All done, your custom scenario is built and ready to go.")

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

            # ─────────────── save project snapshot ────────────────
            if raw.lower() in {"save_project", "sp"}:
                commit_hash = unify.commit_project(
                    args.project_name,
                    commit_message=f"Sandbox save {datetime.utcnow().isoformat()}",
                ).get("commit_hash")
                print(f"💾 Project saved at commit {commit_hash}")
                if args.voice:
                    _speak("Project saved")
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
