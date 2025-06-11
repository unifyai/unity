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
import json
import logging
import os
import select
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import unify
from dotenv import load_dotenv
from pydantic import BaseModel, Field

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


def _seed_llm(cm: ContactManager, custom: Optional[str] = None) -> Optional[str]:
    """Let an LLM invent a richer contact list."""
    if custom:
        prompt = (
            f"User scenario:\n{custom}\n\n"
            "Generate 30‑50 realistic business contacts as JSON "
            "array under key 'contacts'; optional 'theme' field."
        )
    else:
        prompt = (
            "Generate 30‑50 realistic business contacts across EMEA/APAC/AMER. "
            "Return JSON {'contacts':[...],'theme':<str>}. "
            "Each contact needs first_name, surname, email_address and phone_number."
        )

    client = unify.Unify(
        "o4-mini@openai",
        cache=json.loads(os.getenv("UNIFY_CACHE", "true")),
        traced=json.loads(os.getenv("UNIFY_TRACED", "true")),
    )
    client.set_system_message(prompt)
    try:
        payload = json.loads(client.generate("produce contacts"))
    except Exception:
        LG.warning("LLM seeding failed – falling back to fixed seed.")
        _seed_fixed(cm)
        return None

    for c in payload.get("contacts", []):
        # prune unknown keys to avoid validation errors
        allowed = {
            k: v
            for k, v in c.items()
            if k
            in {
                "first_name",
                "surname",
                "email_address",
                "phone_number",
                "whatsapp_number",
                "description",
            }
        }
        cm._create_contact(**allowed)

    return payload.get("theme")


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
    if os.name == "nt":  # Windows
        import msvcrt, time as _t

        start = _t.time()
        buf = []
        while _t.time() - start < timeout:
            if msvcrt.kbhit():
                ch = msvcrt.getwche()
                if ch == "\r":
                    return "".join(buf)
                buf.append(ch)
            _t.sleep(0.01)
        return None

    # POSIX
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
    parser.add_argument(
        "--scenario",
        choices=["fixed", "llm"],
        default="fixed",
        help="initial data set (LLM or deterministic)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--custom-scenario", type=str)
    group.add_argument("--custom-scenario-voice", action="store_true")
    parser.add_argument("--new", "-n", action="store_true", help="wipe existing data")
    parser.add_argument("--silent", "-s", action="store_true", help="suppress logs")
    parser.add_argument("--debug", "-d", action="store_true", help="verbose tool logs")
    args = parser.parse_args()

    scenario_text = get_custom_scenario(args, silent=args.silent)

    # logging
    if not args.silent:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        LG.setLevel(logging.INFO)

    # prepare Unify context
    unify.activate("ContactSandbox", overwrite=True)
    fresh = "Contacts" not in unify.get_contexts() or args.new
    unify.set_context("Contacts", overwrite=fresh)

    # manager
    bus = EventBus()
    cm = ContactManager(event_bus=bus)

    # seed
    if fresh:
        if scenario_text:
            theme = _seed_llm(cm, scenario_text)
            if theme:
                LG.info(f"[seed] theme: {theme}")
        elif args.scenario == "llm":
            theme = _seed_llm(cm)
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
                _speak("Working on this…")

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
