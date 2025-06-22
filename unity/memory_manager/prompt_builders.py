from __future__ import annotations

import inspect
import json
import textwrap
from datetime import datetime, timezone
from typing import Callable, Dict

# Re‑use schemas so the model sees the full structure

# ––– helpers –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    return {n: str(inspect.signature(fn)) for n, fn in tools.items()}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ––– builders –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def build_update_contacts_prompt(tools: Dict[str, Callable]) -> str:
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    transcript_ask = next((n for n in tools if "transcript" in n), "transcript_ask")
    contact_ask = next((n for n in tools if "contact_ask" in n), "contact_ask")
    contact_update = next((n for n in tools if "contact_update" in n), "contact_update")

    examples = textwrap.dedent(
        f"""
        Examples
        --------
        • Determine whether *new* contacts need creating based on the transcript:
          1️⃣ Ask contacts table → `{contact_ask}(text="Is there a contact named Zoe Zhang?")`
          2️⃣ If not found, create → `{contact_update}(text="Create a contact Zoe Zhang …")`

        • Normalise phone numbers before inserting (strip spaces & dashes).
        • Ensure you *never* overwrite a unique field (email / phone) with a value used by another contact.
        """,
    ).strip()

    return "\n".join(
        [
            "You are an assistant charged with **keeping the Contacts table in sync** with newly ingested transcripts.",
            "Given *only* the raw transcript text, decide which contacts should be **created** or **updated**.",
            "Work *exclusively* through the high‑level tools provided – do NOT call low‑level CRUD helpers directly.",
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            examples,
            "",
            f"Current UTC time: {_now()}.",
        ],
    )


def build_update_bio_prompt(tools: Dict[str, Callable]) -> str:
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    set_bio = next((n for n in tools if "set_bio" in n), "set_bio")
    return "\n".join(
        [
            "You maintain a short **biography** for a single contact.",
            "Use the latest transcript excerpt to *improve or expand* the existing bio, keeping it concise (< 120 words).",
            "Only update when you have genuinely new, relevant information.",
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            f"When appropriate call `{set_bio}(contact_id=<id>, bio=<new_bio>)`.",
            "Do **not** modify any other columns.",
            "",
            f"Current UTC time: {_now()}.",
        ],
    )


def build_update_rolling_summary_prompt(tools: Dict[str, Callable]) -> str:
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    set_sum = next(
        (n for n in tools if "set_rolling_summary" in n),
        "set_rolling_summary",
    )
    return "\n".join(
        [
            "Update the **rolling_summary**: a ~500‑character moving window capturing goals, next steps and the gist of the latest ≈50 messages.",
            "Focus on *purpose* of communication, current objectives, and any commitments.",
            "Never exceed 500 characters.",
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            f"Call `{set_sum}(contact_id=<id>, rolling_summary=<new_summary>)` when you have a revised summary.",
            "Do **not** touch any other fields.",
            "",
            f"Current UTC time: {_now()}.",
        ],
    )
