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
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unify

from ..unity.constants import LOGGER as _LG
from ..knowledge_manager.knowledge_manager import KnowledgeManager
from .utils import (
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
)

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


def _seed_llm(km: KnowledgeManager) -> Optional[str]:
    """Use an LLM to generate a large set of factual sentences."""
    prompt = (
        "Generate 120‑180 short factual sentences suitable for ingestion by a knowledge base. "
        "Cover diverse domains: personal bios, product pricing, purchases, geography, science facts, pets, coordinates, sports scores etc. "
        "Avoid any personally identifying sensitive data. "
        'Return as JSON {"statements": [...], "theme": <string>} and nothing else.'
    )
    client = unify.Unify("o4-mini@openai", cache=True)
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


def _dispatch(
    km: KnowledgeManager,
    raw: str,
    *,
    show_steps: bool,
) -> Tuple[str, str, List | None]:
    raw = raw.strip()

    # Quick rule: voice input often lacks punctuation – fall back to heuristic + LLM judge if ambiguous
    heuristic_store = bool(
        re.match(r"^(remember|note|store|add)\b", raw, re.I),
    ) and not raw.endswith("?")

    if heuristic_store:
        km.store(raw)
        return "store", "Got it – I've stored that.", None

    llm = unify.Unify("gpt-4o@openai", response_format=_IntentResp)
    intent_json = llm.set_system_message(_INTENT_PROMPT).generate(raw)
    intent = _IntentResp.model_validate_json(intent_json)

    if intent.action == "store":
        km.store(intent.cleaned_text)
        return "store", "Noted.", None

    # Retrieval path
    answer, steps = km.retrieve(intent.cleaned_text, return_reasoning_steps=show_steps)
    return "retrieve", answer, steps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
    parser.add_argument("--scenario", choices=["fixed", "llm"], default="fixed")
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
    args = parser.parse_args()

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
    km.start()

    if fresh:
        if args.scenario == "llm":
            theme = _seed_llm(km)
            if theme:
                _LG.info(f"[Seed] LLM scenario theme: {theme}")
        else:
            _seed_fixed(km)

    print("KnowledgeManager sandbox – speak or type. 'quit' to exit.\n")

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
            kind, result, _ = _dispatch(km, user_text, show_steps=not args.silent)
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
                kind, result, _ = _dispatch(km, line, show_steps=not args.silent)
                print(f"[{kind}] => {result}\n")
        except (EOFError, KeyboardInterrupt):
            print()


if __name__ == "__main__":
    main()
