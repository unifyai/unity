"""knowledge_sandbox.py  (optional voice mode, Deepgram SDK v4, sync)
====================================================================
Interactive sandbox for **KnowledgeManager**.

It supports:
• Fixed or LLM‑generated seed data.
• Voice or plain‑text input (same helpers as the other sandboxes).
• Automatic dispatch to `retrieve`, `store` *or* `refactor` depending on intent.
• Mid‑conversation interruption (pause / interject / cancel).
"""

from __future__ import annotations

# ─────────────────────────────── stdlib / vendored ──────────────────────────
import os
import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

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
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.common.llm_helpers import SteerableToolHandle
from sandboxes.utils import (  # shared helpers reused in other sandboxes
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    run_in_loop,
    get_custom_scenario,
    await_with_interrupt as _await_with_interrupt,
)
from sandboxes.scenario_store import ScenarioStore

LG = logging.getLogger("knowledge_sandbox")

# ═════════════════════════════════ seed helpers ═════════════════════════════


async def _build_scenario(custom: Optional[str] = None) -> Optional[str]:
    """
    Populate the knowledge store **via the official tools** (`store`/`retrieve`)
    using :class:`ScenarioBuilder`.  Falls back to the fixed seed on error.
    """
    km = KnowledgeManager()

    description = (
        custom.strip()
        if custom
        else (
            "Generate 20 diverse facts about electric-vehicle manufacturers. "
            "Cover launch years, battery capacities, warranty terms and sales "
            "figures in different regions.  Include numbers, dates and named "
            "entities so the schema has to evolve."
        )
    )
    description += (
        "\nTry to batch actions – each `store` can add multiple rows/columns "
        "and `retrieve` can verify to avoid duplication."
    )

    builder = ScenarioBuilder(
        description=description,
        tools={
            "store": km.update,
            "retrieve": km.ask,
        },
    )

    try:
        await builder.create()
    except Exception as exc:
        raise RuntimeError(f"LLM seeding via ScenarioBuilder failed. {exc}")

    return None


# ═════════════════════════════ intent dispatcher ════════════════════════════


class _Intent(BaseModel):
    action: str = Field(..., pattern="^(retrieve|store|refactor)$")
    cleaned_text: str


_INTENT_SYS_MSG = (
    "Decide whether the user input is a *query* about existing knowledge "
    "(`retrieve`), a *mutation* that adds/updates knowledge (`store`), "
    "or a schema-level restructuring (`refactor`). "
    "Return JSON "
    "{'action':'retrieve'|'store'|'refactor','cleaned_text':<fixed_input>}."
)


async def _dispatch_with_context(
    km: KnowledgeManager,
    raw: str,
    *,
    show_steps: bool,
    parent_chat_context: List[Dict[str, str]],
) -> Tuple[str, SteerableToolHandle]:
    """
    Figure out whether to call `store`, `retrieve` or `refactor`, forwarding
    *parent_chat_context* to the KnowledgeManager methods.
    """

    lowered = raw.lower()

    # ───── quick heuristics (fast-path) ───────────────────────────────
    if lowered.startswith(
        (
            "add ",
            "create ",
            "update ",
            "change ",
            "delete ",
            "store ",
            "remember ",
            "note ",
        ),
    ):
        handle = await km.update(
            raw,
            parent_chat_context=parent_chat_context,
            _return_reasoning_steps=show_steps,
        )
        return "store", handle

    if lowered.startswith(
        (
            "refactor ",
            "restructure ",
            "normalize ",
            "normalise ",
            "schema ",
        ),
    ):
        handle = await km.refactor(
            raw,
            parent_chat_context=parent_chat_context,
            _return_reasoning_steps=show_steps,
        )
        return "refactor", handle

    # ───── everything else – ask an LLM judge ────────────────────────
    judge = unify.Unify("gpt-4o@openai", response_format=_Intent)
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )

    fn = (
        km.update
        if intent.action == "store"
        else km.refactor if intent.action == "refactor" else km.ask
    )
    handle = await fn(
        raw,
        parent_chat_context=parent_chat_context,
        _return_reasoning_steps=show_steps,
    )
    return intent.action, handle


# ══════════════════════════════════  CLI  ═══════════════════════════════════


async def _main_async() -> None:
    parser = argparse.ArgumentParser(description="KnowledgeManager sandbox")
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
    parser.add_argument(
        "--load_custom",
        "-L",
        metavar="NAME|-N",
        help="Load a stored transcript by name or negative history index",
    )
    parser.add_argument(
        "--save_custom",
        "-S",
        metavar="NAME",
        help="Save the transcript used to seed this run under NAME",
    )
    args = parser.parse_args()

    # tracing flag
    if args.traced:
        os.environ["UNIFY_TRACED"] = "true"
    else:
        os.environ["UNIFY_TRACED"] = "false"

    # prepare Unify context
    unify.activate("KnowledgeSandbox")
    unify.set_trace_context("Traces")
    if not args.reuse:
        [
            unify.delete_context(table)
            for table in unify.get_contexts(prefix="Knowledge").keys()
        ]
        if "Traces" in unify.get_contexts():
            unify.delete_context("Traces")
        unify.create_context("Traces")

    # logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LG.setLevel(logging.INFO)

    # manager & transcript vault
    km = KnowledgeManager()
    if args.traced:
        km = unify.traced(km)
    store = ScenarioStore()

    # Seeding with vault support
    if not args.reuse:
        scenario_text: Optional[str] = None

        if args.load_custom:
            try:
                key = int(args.load_custom)
            except ValueError:
                key = args.load_custom
            scenario_text = store.get(key)
            LG.info(f"[seed] loaded transcript {key} → {scenario_text}")
            if args.voice:
                _speak("Loading your saved scenario, give me a second.")

        if scenario_text is None:
            scenario_text = get_custom_scenario(args)
            if not scenario_text:
                raise Exception("No text provided for building the custom scenario")
            store.add_to_history(scenario_text)
            LG.info(f"[voice] transcript: {scenario_text}")

        LG.info("[seed] building synthetic knowledge base – this can take 20-40 s…")
        if args.voice:
            _speak("Sure thing, building your custom scenario now.")
        await _build_scenario(scenario_text)
        LG.info("[seed] done.")
        if args.voice:
            _speak("All done, your custom scenario is built and ready to go.")

        if args.save_custom:
            store.save_named(args.save_custom, scenario_text)
            LG.info(f"[seed] transcript saved as {args.save_custom}.")

    print("KnowledgeManager sandbox – type or speak. 'quit' to exit.\n")

    _speak(
        "Press enter to record a question or request an update for the knowledge base.",
    )

    # running memory of the dialogue
    chat_history: List[Dict[str, str]] = []

    # interaction loop
    while True:
        try:
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

            # ──────────────── remember the user's utterance ────────────────
            _kind, _handle = await _dispatch_with_context(
                km,
                raw,
                show_steps=args.debug,
                parent_chat_context=list(chat_history),
            )
            chat_history.append({"role": "user", "content": raw})
            if args.voice:
                _speak("Let me take a look, give me a moment")

            answer = await _await_with_interrupt(_handle)
            if args.voice:
                _speak("Okay that's all done")
            if isinstance(answer, tuple):  # reasoning steps requested
                answer, _steps = answer
            print(f"[{_kind}] → {answer}\n")

            # ──────────────── remember the assistant's reply ───────────────
            chat_history.append({"role": "assistant", "content": answer})
            if args.voice:
                _speak(f"{answer} Anything else I can help with?")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
