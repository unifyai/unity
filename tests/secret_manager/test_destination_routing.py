from __future__ import annotations

import pytest
import unify

from unity.common.tool_outcome import ToolError
from unity.function_manager.function_manager import VenvPool
from unity.secret_manager.secret_manager import SecretManager


def _rows(context: str) -> list[dict]:
    return [log.entries for log in unify.get_logs(context=context)]


def test_secret_writes_route_to_destination_and_reads_merge_roots(
    secret_manager_context,
    secret_manager_spaces,
):
    """Credential writes land in one vault while read tools see every reachable vault."""
    first_space, second_space = secret_manager_spaces
    manager = SecretManager()

    manager._create_secret(
        name="shared_api",
        value="personal-value",
        description="Private API key",
    )
    manager._create_secret(
        name="shared_api",
        value="space-one-value",
        description="Patch team service account",
        destination=f"space:{first_space}",
    )
    manager._create_secret(
        name="family_calendar",
        value="space-two-value",
        description="Family calendar shared key",
        destination=f"space:{second_space}",
    )

    personal_rows = _rows(manager._ctx)
    first_space_rows = _rows(f"Spaces/{first_space}/Secrets")
    second_space_rows = _rows(f"Spaces/{second_space}/Secrets")

    assert [row["value"] for row in personal_rows] == ["personal-value"]
    assert [row["value"] for row in first_space_rows] == ["space-one-value"]
    assert [row["value"] for row in second_space_rows] == ["space-two-value"]

    merged_rows = manager._filter_secrets()
    assert [row.name for row in merged_rows].count("shared_api") == 2
    assert {row.name for row in merged_rows} == {"shared_api", "family_calendar"}
    assert {(row.name, row.destination) for row in merged_rows} == {
        ("shared_api", "personal"),
        ("shared_api", f"space:{first_space}"),
        ("family_calendar", f"space:{second_space}"),
    }
    assert set(manager._list_secret_keys()) == {"shared_api", "family_calendar"}

    shared_only = manager._filter_secrets(
        filter=f"name == 'shared_api' and destination == 'space:{first_space}'",
    )
    assert [(row.name, row.destination) for row in shared_only] == [
        ("shared_api", f"space:{first_space}"),
    ]


def test_create_secret_invalid_destination_returns_tool_error(
    secret_manager_context,
    secret_manager_spaces,
):
    """Invalid shared-space destinations return a structured tool error."""
    manager = SecretManager()

    outcome = manager._create_secret(
        name="bad_destination",
        value="nope",
        destination="space:987699",
    )

    assert isinstance(outcome, dict)
    assert outcome["error_kind"] == "invalid_destination"
    assert outcome["details"]["destination"] == "space:987699"


def test_mutating_secret_invalid_destination_returns_tool_error(
    secret_manager_context,
    secret_manager_spaces,
):
    """Update and delete operations reject inaccessible shared-space destinations."""
    manager = SecretManager()

    update_outcome = manager._update_secret(
        name="missing",
        value="ignored",
        destination="space:987699",
    )
    delete_outcome = manager._delete_secret(
        name="missing",
        destination="space:987699",
    )

    assert update_outcome["error_kind"] == "invalid_destination"
    assert delete_outcome["error_kind"] == "invalid_destination"
    assert update_outcome["details"]["destination"] == "space:987699"
    assert delete_outcome["details"]["destination"] == "space:987699"


def test_get_credential_reads_exact_destination_root(
    secret_manager_context,
    secret_manager_spaces,
):
    """Credential use reads one vault and never falls back across scopes."""
    space_id, _ = secret_manager_spaces
    manager = SecretManager()

    manager._create_secret(name="sendgrid", value="personal-sendgrid")
    manager._create_secret(
        name="sendgrid",
        value="team-sendgrid",
        destination=f"space:{space_id}",
    )
    manager._create_secret(
        name="space_only",
        value="team-only",
        destination=f"space:{space_id}",
    )

    assert manager.get_credential("sendgrid") == "personal-sendgrid"
    assert (
        manager.get_credential("sendgrid", destination=f"space:{space_id}")
        == "team-sendgrid"
    )

    with pytest.raises(KeyError):
        manager.get_credential("space_only")


@pytest.mark.asyncio
async def test_placeholder_resolution_inherits_task_destination(
    monkeypatch,
    secret_manager_context,
    secret_manager_spaces,
):
    """Shared task execution resolves placeholders from the task's destination vault."""
    space_id, _ = secret_manager_spaces
    manager = SecretManager()

    manager._create_secret(name="mail_key", value="personal-mail")
    manager._create_secret(
        name="mail_key",
        value="team-mail",
        destination=f"space:{space_id}",
    )

    monkeypatch.setenv("TASK_DESTINATION", f"space:{space_id}")

    assert await manager.from_placeholder("token=${mail_key}") == "token=team-mail"


def test_credential_writes_invalidate_pooled_subprocesses(
    monkeypatch,
    secret_manager_context,
    secret_manager_spaces,
):
    """Every successful credential mutation invalidates stateful execution pools."""
    calls: list[str] = []

    def record_invalidation(cls) -> int:
        calls.append("invalidate")
        return 0

    monkeypatch.setattr(
        VenvPool,
        "invalidate_all_pools",
        classmethod(record_invalidation),
    )

    space_id, _ = secret_manager_spaces
    manager = SecretManager()

    error: ToolError = manager._create_secret(
        name="bad_destination",
        value="nope",
        destination="space:987699",
    )  # type: ignore[assignment]
    assert error["error_kind"] == "invalid_destination"
    assert calls == []

    manager._create_secret(name="rotating", value="v1", destination=f"space:{space_id}")
    manager._update_secret(name="rotating", value="v2", destination=f"space:{space_id}")
    manager._delete_secret(name="rotating", destination=f"space:{space_id}")

    assert calls == ["invalidate", "invalidate", "invalidate"]


def test_personal_env_sync_failure_does_not_skip_invalidation(
    monkeypatch,
    secret_manager_context,
):
    """A backend credential write still invalidates processes if .env sync fails."""
    calls: list[str] = []

    def record_invalidation(cls) -> int:
        calls.append("invalidate")
        return 0

    monkeypatch.setattr(
        VenvPool,
        "invalidate_all_pools",
        classmethod(record_invalidation),
    )

    manager = SecretManager()
    monkeypatch.setattr(
        manager,
        "_env_set",
        lambda name, value: (_ for _ in ()).throw(OSError("disk unavailable")),
    )

    result = manager._create_secret(name="env_failure", value="stored")

    assert result["outcome"] == "secret created"
    assert calls == ["invalidate"]
    assert manager.get_credential("env_failure") == "stored"
