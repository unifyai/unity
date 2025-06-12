"""contact_sandbox.py  (optional voice mode, Deepgram SDK v4, sync)
===================================================================
Interactive sandbox for **ContactManager**.

It supports:
• Fixed or LLM‑generated seed data.
• Voice or plain‑text input (same helpers as the other sandboxes).
• Automatic dispatch to `ask` *or* `update` depending on intent.
• Mid‑conversation interruption (pause / interject / cancel).

Run:
    poetry run python -m sandboxes.contact       # text mode
    poetry run python -m sandboxes.contact --voice   # with STT/TTS
"""

from __future__ import annotations

# ─────────────────────────────── stdlib / vendored ──────────────────────────
import os
import argparse
import asyncio
import logging
import select
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import unify

unify.set_trace_context("Traces")
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from scenario_builder import ScenarioBuilder

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

# ────────────────────────────────  unity imports  ───────────────────────────
from unity.contact_manager.contact_manager import ContactManager
from unity.common.llm_helpers import SteerableToolHandle
from sandboxes.utils import (  # shared helpers reused in other sandboxes
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    run_in_loop,
    get_custom_scenario,
)

LG = logging.getLogger("contact_sandbox")

# ═════════════════════════════════ seed helpers ═════════════════════════════


async def _build_scenario(
    custom: Optional[str] = None,
) -> Optional[str]:
    """
    Populate the contact store **through the official tools** using
    :class:`ScenarioBuilder`.  Falls back to the fixed seed on any error.
    """
    cm = ContactManager(batched=True)
    description = (
        custom.strip()
        if custom
        else (
            "Generate 10 realistic business contacts across EMEA, APAC and AMER. "
            "Each contact needs first_name, surname, email_address and phone_number. "
            "Also create custom columns with varying industries and locations."
        )
    )
    description += (
        "\nTry to get as much done as you can with each `update` and `ask` call. "
        "They can deal with complex multi-step requests just fine."
    )

    builder = ScenarioBuilder(
        description=description,
        tools={  # expose only the public surface
            "update": cm.update,
            "ask": cm.ask,  # allows the LLM to check for duplicates if it wishes
        },
    )

    try:
        await builder.create()
    except Exception as exc:
        raise (f"LLM seeding via ScenarioBuilder failed. {exc}")

    # The new flow doesn't produce a structured "theme"; preserve signature.
    return None


# ═════════════════════════════ intent dispatcher ════════════════════════════


class _Intent(BaseModel):
    action: str = Field(..., pattern="^(ask|update)$")
    cleaned_text: str


_INTENT_SYS_MSG = (
    "Decide whether the user input is a *query* about existing contacts "
    "or a *mutation* (create / update).  "
    "Return JSON {'action':'ask'|'update','cleaned_text':<fixed_input>}."
)


async def _dispatch_with_context(
    cm: ContactManager,
    raw: str,
    *,
    show_steps: bool,
    parent_chat_context: List[Dict[str, str]],
) -> Tuple[str, SteerableToolHandle]:
    """
    Same as :pyfunc:`_dispatch` but forwards *parent_chat_context* to the CM
    methods.  This indirection keeps the diff minimal.
    """

    # quick heuristic – verbs that virtually always imply an update
    if raw.lower().startswith(("add ", "create ", "update ", "change ", "delete ")):
        handle = await cm.update(
            raw,
            parent_chat_context=parent_chat_context,
            _return_reasoning_steps=show_steps,
        )
        return "update", handle

    # ask an LLM if less obvious
    judge = unify.Unify("gpt-4o@openai", response_format=_Intent)
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )
    fn = cm.update if intent.action == "update" else cm.ask
    handle = await fn(
        intent.cleaned_text,
        parent_chat_context=parent_chat_context,
        _return_reasoning_steps=show_steps,
    )
    return intent.action, handle


# ═════════════════════════════ interruption helper ══════════════════════════


def _input_now(timeout: float = 0.1) -> Optional[str]:
    """Non‑blocking stdin check (POSIX & Windows)."""
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.readline().strip() if r else None


async def _await_with_interrupt(
    handle: SteerableToolHandle,
) -> str:  # returns final answer
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
    parser = argparse.ArgumentParser(description="ContactManager sandbox")
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

    # tracing flag
    if args.traced:
        os.environ["UNIFY_TRACED"] = "true"
    else:
        os.environ["UNIFY_TRACED"] = "false"

    # prepare Unify context
    unify.activate("ContactSandbox")
    if not args.reuse:
        ctxs = unify.get_contexts()
        if "Contacts" in ctxs:
            unify.delete_context("Contacts")
        unify.create_context("Contacts")
        if "Traces" in ctxs:
            unify.delete_context("Traces")
        unify.create_context("Traces")

    # logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LG.setLevel(logging.INFO)

    # custom scenario
    scenario_text = get_custom_scenario(args)

    # manager
    cm = ContactManager()
    if args.traced:
        cm = unify.traced(cm)

    # seed
    if not args.reuse:
        if scenario_text:
            LG.info("[voice] transcript: “%s”", scenario_text)
            LG.info("[seed] building synthetic contacts – this can take 20-40 s…")
            if args.voice:
                _speak("Sure thing, building your custom scenario now.")
            theme = await _build_scenario(scenario_text)
            LG.info("[seed] done.")
            if args.voice:
                _speak(
                    "All done, your custom scenario is built and ready to go. Press enter to record a question or request an update for the contact list.",
                )
            if theme:
                LG.info(f"[seed] theme: {theme}")
        else:
            raise Exception("No text provided for building the custom scenario")

    print("ContactManager sandbox – type or speak. 'quit' to exit.\n")

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
            chat_history.append({"role": "user", "content": raw})
            _kind, _handle = await _dispatch_with_context(
                cm,
                raw,
                show_steps=args.debug,
                parent_chat_context=chat_history,
            )
            if args.voice:
                _speak("Sure, working on this now")

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
