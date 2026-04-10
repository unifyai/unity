from __future__ import annotations

import re
import pytest

from unity.secret_manager.simulated import (
    SimulatedSecretManager,
)

from tests.helpers import (
    _handle_project,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Doc-string inheritance
# ─────────────────────────────────────────────────────────────────────────────
def test_docstrings_match_base():
    """Public methods in SimulatedSecretManager should copy BaseSecretManager doc-strings."""
    from unity.secret_manager.base import BaseSecretManager
    from unity.secret_manager.simulated import SimulatedSecretManager

    assert (
        BaseSecretManager.ask.__doc__.strip()
        in SimulatedSecretManager.ask.__doc__.strip()
    )
    assert (
        BaseSecretManager.update.__doc__.strip()
        in SimulatedSecretManager.update.__doc__.strip()
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Basic start-and-ask
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_start_and_ask():
    sm = SimulatedSecretManager("Demo Secret Manager for unit-tests.")
    h = await sm.ask("List all secret keys.")
    answer = await h.result()
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Stateful memory – serial asks
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_stateful_memory_serial_asks():
    sm = SimulatedSecretManager()

    h1 = await sm.ask(
        "Please propose a safe placeholder, output only the placeholder name like ${token_name}.",
    )
    placeholder = (await h1.result()).strip()
    assert placeholder, "Placeholder should not be empty"
    # Extract a single ${key} token from the first answer
    m = re.search(r"\$\{[^}]+\}", placeholder)
    assert m, "Response should contain a ${name} placeholder token"
    token = m.group(0).lower()

    h2 = await sm.ask("What placeholder did you just propose?")
    answer2 = (await h2.result()).lower()
    assert token in answer2, "LLM should recall the placeholder token it generated"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Update then ask – state carries through
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_stateful_update_then_ask():
    sm = SimulatedSecretManager()

    upd = await sm.update(
        "Create a secret named api_key with a value (do not reveal it).",
    )
    await upd.result()

    hq = await sm.ask("Confirm that ${api_key} exists and is stored.")
    ans = (await hq.result()).lower()
    assert (
        "${api_key}" in ans
    ), "Secret created via update should be referenced by placeholder"


# 10.  Clear – reset and remain usable
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_clear():
    """
    SimulatedSecretManager.clear should reset the manager and remain usable afterwards.
    """
    sm = SimulatedSecretManager()
    # Do an update/ask to create some prior state in the stateful LLM
    h_upd = await sm.update("Create a temporary secret named temp_token.")
    await h_upd.result()

    # Clear should not raise and should be quick (no LLM roundtrip requirement)
    sm.clear()

    # Post-clear, an ask should still work
    h_q = await sm.ask("List any secret placeholders referenced in the system.")
    answer = await h_q.result()
    assert isinstance(answer, str) and answer.strip()


# 11.  Placeholder conversion helpers
@pytest.mark.asyncio
@_handle_project
async def test_from_and_to_placeholder_roundtrip():
    """
    Verify the public placeholder conversion helpers round-trip as expected.
    """
    sm = SimulatedSecretManager()
    original = "Use ${api_key} for requests and ${db_password} for DB."

    # Convert to opaque value tokens
    to_values = await sm.from_placeholder(original)
    assert "<value:api_key>" in to_values
    assert "<value:db_password>" in to_values

    # Convert back to placeholders
    back_to_placeholders = await sm.to_placeholder(to_values)
    assert "${api_key}" in back_to_placeholders
    assert "${db_password}" in back_to_placeholders
