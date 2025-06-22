# memory_manager/prompt_builders.py
from __future__ import annotations

import json
import inspect
from datetime import datetime, timezone
from typing import Callable, Dict


# ── utils ───────────────────────────────────────────────────────────────
def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    return {n: str(inspect.signature(fn)) for n, fn in tools.items()}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ── three tiny builders (one per public method) ─────────────────────────
def build_contact_update_prompt(tools: Dict[str, Callable]) -> str:
    return "\n".join(
        [
            "You are the **offline MemoryManager** tasked with *creating or amending*",
            "contact records — names, phone numbers, emails, etc. — after reading",
            "a 50-message transcript chunk.",
            "",
            "Work **only** via the tools below.  First figure out what changed,",
            "then call the appropriate update tool(s).  Finally return a short",
            "human-readable summary of what you did.",
            "",
            "Tools (name → argspec):",
            json.dumps(_sig_dict(tools), indent=4),
            "",
            "Current UTC time: " + _now(),
        ],
    )


def build_bio_prompt(tools: Dict[str, Callable]) -> str:
    return "\n".join(
        [
            "You are the **MemoryManager** updating the *bio* column for ONE contact.",
            "Input is the last 50 messages plus the *existing* bio (if any).",
            "",
            "1️⃣ Decide whether the bio should change.",
            "2️⃣ If yes, call the specialised `set_bio` tool.",
            "3️⃣ Return only the *new* bio text (or the unchanged one).",
            "",
            "Tools (name → argspec):",
            json.dumps(_sig_dict(tools), indent=4),
            "",
            "Current UTC time: " + _now(),
        ],
    )


def build_rolling_prompt(tools: Dict[str, Callable]) -> str:
    return "\n".join(
        [
            "You are the **MemoryManager** refreshing the 50-message *rolling summary*",
            "for ONE contact.  Start from the previous rolling summary (if supplied).",
            "",
            "Produce a concise, up-to-date summary **<= 120 words** capturing:",
            "• main conversation theme(s)",
            "• immediate goals / outstanding tasks",
            "• tone or sentiment shifts if relevant",
            "",
            "You may call the specialised `set_rolling_summary` tool exactly once.",
            "Finally, return the summary text you stored.",
            "",
            "Tools (name → argspec):",
            json.dumps(_sig_dict(tools), indent=4),
            "",
            "Current UTC time: " + _now(),
        ],
    )
