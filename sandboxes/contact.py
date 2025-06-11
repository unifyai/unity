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
import argparse
import asyncio
import logging
import select
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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
from unity.events.event_bus import EventBus
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

_FIXED_CONTACTS: List[dict] = [
    dict(
        first_name="Alice",
        surname="Smith",
        email_address="alice.smith@example.com",
        phone_number="+14155550001",
        whatsapp_number="+14155550001",
        description="Key account at Acme Corp.",
    ),
    dict(
        first_name="Bob",
        surname="Jones",
        email_address="bob.jones@example.com",
        phone_number="+447700900010",
        whatsapp_number="+447700900010",
        description="London sales lead.",
    ),
    dict(
        first_name="Carlos",
        surname="Diaz",
        email_address="carlos.diaz@example.com",
        phone_number="+34911222333",
        whatsapp_number="+34911222333",
        description="Madrid‑based logistics contact.",
    ),
]


def _seed_fixed(cm: ContactManager) -> None:
    """Populate three predictable contacts."""
    for c in _FIXED_CONTACTS:
        cm._create_contact(**c)


async def _build_scenario(
    cm: ContactManager,
    custom: Optional[str] = None,
) -> Optional[str]:
    """
    Populate the contact store **through the official tools** using
    :class:`ScenarioBuilder`.  Falls back to the fixed seed on any error.
    """
    description = (
        custom.strip()
        if custom
        else (
            "Generate 30-50 realistic business contacts across EMEA, APAC and AMER. "
            "Each contact needs first_name, surname, email_address and phone_number. "
            "Vary industries and locations."
        )
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
        LG.warning(
            "LLM seeding via ScenarioBuilder failed – falling back to fixed seed. (%s)",
            exc,
        )
        _seed_fixed(cm)

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


async def _dispatch(
    cm: ContactManager,
    raw: str,
    *,
    show_steps: bool,
) -> Tuple[str, SteerableToolHandle]:
    raw = raw.strip()

    # quick heuristic – verbs that virtually always imply an update
    if raw.lower().startswith(("add ", "create ", "update ", "change ", "delete ")):
        handle = await cm.update(raw, _return_reasoning_steps=show_steps)
        return "update", handle

    # ask an LLM if less obvious
    judge = unify.Unify("gpt-4o@openai", response_format=_Intent)
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )
    fn = cm.update if intent.action == "update" else cm.ask
    handle = await fn(intent.cleaned_text, _return_reasoning_steps=show_steps)
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--custom-scenario",
        action="store_true",
        help="whether to create a custom scenario",
    )
    parser.add_argument("--reuse", "-r", action="store_true", help="re-use old data")
    parser.add_argument("--debug", "-d", action="store_true", help="verbose tool logs")
    args = parser.parse_args()

    scenario_text = get_custom_scenario(args)

    # logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LG.setLevel(logging.INFO)

    # prepare Unify context
    unify.activate("ContactSandbox")
    fresh = "Contacts" not in unify.get_contexts() or (not args.reuse)
    unify.set_context("Contacts", overwrite=fresh)

    # manager
    bus = EventBus()
    cm = ContactManager(event_bus=bus)

    # seed
    if fresh:
        if scenario_text:
            LG.info("[voice] transcript: “%s”", scenario_text)
            LG.info("[seed] building synthetic contacts – this can take 20-40 s…")
            theme = await _build_scenario(cm, scenario_text)
            LG.info("[seed] done.")
            if theme:
                LG.info(f"[seed] theme: {theme}")
        else:
            _seed_fixed(cm)

    print("ContactManager sandbox – type or speak. 'quit' to exit.\n")

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

            _kind, _handle = await _dispatch(cm, raw, show_steps=args.debug)
            if args.voice:
                _speak("Working on this")

            answer = await _await_with_interrupt(_handle)
            if isinstance(answer, tuple):  # reasoning steps requested
                answer, _steps = answer
            print(f"[{_kind}] → {answer}\n")
            if args.voice and _kind == "ask":
                _speak(answer)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
