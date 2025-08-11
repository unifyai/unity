# tests/test_recursive_ask_real.py
"""
End-to-end verification that a pending *outer* tool-loop can answer an
`ask()` inspection that propagates **two tiers down**:

Conductor.update           (tier-0  – outer loop, pending)
└── ContactManager.update    (tier-1  – pending tool call)
    └── *nested ask* helper  (tier-2)

The test sequence:

1.  Create a fresh “Daniel Smith” contact so that the system has exactly
    *one* “Daniel”.
2.  Launch `Conductor.request("change Daniel Smith's first name to Dan")`
    **without** awaiting its `.result()` – this returns a *running*
    `SteerableToolHandle` (`h_update`).
3.  Immediately call `h_update.ask("How many Daniel's …?")`.
    ·   The inspection must propagate to the still-pending
        **ContactManager.update** handle.
    ·   The nested loop should answer `1` (the rename isn’t committed yet).
4.  Await `h_update.result()` so the rename completes, then issue a *final*
    read-only `Conductor.ask` to confirm the count is now `0`.

No monkey-patches are required; the assertions rely only on the returned
natural-language answers containing an integer.
"""

from __future__ import annotations

import re

import pytest

from unity.contact_manager.contact_manager import ContactManager
from unity.conductor.conductor import Conductor
from tests.helpers import _handle_project


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
_NUMBER_RX = re.compile(r"\b(\d+)\b")


def _extract_int(text: str) -> int:
    """
    Pull the *first* integer out of *text* (raises if none found).
    """
    m = _NUMBER_RX.search(text)
    assert m is not None, f"Expected at least one integer in answer but got: {text!r}"
    return int(m.group(1))


# --------------------------------------------------------------------------- #
# Main integration test                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_two_tier_ask_propagation():

    # 1️⃣  Ensure we have exactly **one** “Daniel Smith” contact up-front
    cm = ContactManager()
    create = await cm.update(
        "Create a contact called Daniel Smith with email daniel.smith@example.com.",
    )
    await create.result()

    # 2️⃣  Kick off the *outer* mutation via the real Conductor
    cond = Conductor()

    h_update = await cond.request(
        "Change Daniel Smith's first name to Dan.",
        _log_tool_steps=False,
    )

    # 3️⃣  While the rename is *still pending*, inspect via .ask()
    h_nested = await h_update.ask(
        "How many Daniel's do we have in our contacts?",
    )
    pre_answer = await h_nested.result()
    assert (
        _extract_int(pre_answer) == 1
    ), "Before rename completes, one contact should still be called Daniel."

    # 4️⃣  Let the outer update finish, then verify the rename took effect
    await h_update.result()

    h_verify = await cond.ask("How many Daniel's do we have in our contacts?")
    post_answer = await h_verify.result()
    assert (
        _extract_int(post_answer) == 0
    ), "After rename, no contact should be called Daniel."
