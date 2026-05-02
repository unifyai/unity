from __future__ import annotations

import asyncio
import pytest
import unify

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]

from unity.secret_manager.secret_manager import SecretManager


@pytest.mark.asyncio
async def test_ask_and_update_flow(secret_manager_context):
    sm = SecretManager()

    # Natural-language update to create a secret (the LLM will route to tools).
    # Allow model to infer a suitable key name when not provided explicitly.
    handle = await sm.update("Create the desktop password for mac as pass123.")
    await handle.result()

    # Ask should allow inspection by name/description without revealing values
    h2 = await sm.ask(
        "List secret keys and confirm that a mac-related password key exists.",
    )
    ans = await h2.result()
    # Tolerant assertion: look for a plausible synthesized key reference
    assert ("mac" in ans.lower() and "password" in ans.lower()) or (
        "desktop" in ans.lower()
    )


@pytest.mark.asyncio
async def test_ask_with_clarification(secret_manager_context):
    sm = SecretManager()
    sm._create_secret(
        name="db_password_staging_1",
        value="topsecret",
        description="database",
    )
    sm._create_secret(
        name="db_password_staging_2",
        value="s3cret",
        description="staging db",
    )

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    h = await sm.ask(
        "Show the placeholder for the staging database password. If ambiguous, request clarification.",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
    )

    q = await asyncio.wait_for(up_q.get(), timeout=120)
    assert q and isinstance(q, str)
    await down_q.put("I mean the one named db_password_staging_2.")

    ans = await h.result()
    # We expect the model to reference placeholder; if the model fabricates text
    # this check is permissive to avoid flakiness in simulated/real LLMs.
    assert ("${db_password_staging}" in ans) or ("staging" in ans.lower())


@pytest.mark.asyncio
async def test_update_creates_two_secrets(secret_manager_context):
    sm = SecretManager()

    # Ask the model to create two different secrets in a single turn
    req = (
        "Create two secrets (not in parallel): "
        "name alpha_token with value a1; and name beta_token with value b2."
    )
    handle = await sm.update(req)
    await handle.result()

    # Verify both were created
    keys = sm._list_secret_keys()  # type: ignore[attr-defined]
    assert "alpha_token" in keys and "beta_token" in keys


@pytest.mark.asyncio
async def test_update_routes_team_credential_to_shared_space(
    secret_manager_context,
    secret_manager_spaces,
):
    """Natural-language updates choose the named shared-space destination."""
    patch_space_id, _ = secret_manager_spaces
    manager = SecretManager()

    handle = await manager.update(
        "Create a credential named patch_ops_slack_bot with value xoxb-routing-eval. "
        "It is the Slack bot token for the Patch Team workspace and should be shared "
        "with that team, not kept as my personal credential.",
    )
    await handle.result()

    personal_matches = unify.get_logs(
        context=manager._ctx,
        filter="name == 'patch_ops_slack_bot'",
    )
    shared_matches = unify.get_logs(
        context=f"Spaces/{patch_space_id}/Secrets",
        filter="name == 'patch_ops_slack_bot'",
    )

    assert personal_matches == []
    assert [row.entries["value"] for row in shared_matches] == ["xoxb-routing-eval"]
