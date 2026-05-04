from __future__ import annotations

import uuid

import unify

from tests.destination_routing_helpers import (
    manager_routing_context as manager_routing_context,  # noqa: F401
)
from unity.function_manager.function_manager import FunctionManager


def test_function_writes_route_to_destination_and_reads_merge_roots(
    manager_routing_context,
):
    """Composed functions write to one root while list tools read all reachable roots."""

    _, space_id = manager_routing_context
    manager = FunctionManager(include_primitives=False)

    manager.add_functions(
        implementations="def personal_helper():\n    return 'personal'",
    )
    manager.add_functions(
        implementations="def team_helper():\n    return 'shared'",
        destination=f"space:{space_id}",
    )
    shared_venv_id = manager.add_venv(
        venv="[project]\nname = 'team-tools'\nversion = '0.1.0'\ndependencies = []",
        destination=f"space:{space_id}",
    )

    personal_rows = unify.get_logs(context=manager._compositional_ctx)
    shared_rows = unify.get_logs(
        context=f"Spaces/{space_id}/Functions/Compositional",
    )

    assert {row.entries["name"] for row in personal_rows} == {
        "personal_helper",
    }
    assert {row.entries["name"] for row in shared_rows} == {
        "team_helper",
    }
    assert set(manager.list_functions()) == {
        "personal_helper",
        "team_helper",
    }
    assert "team-tools" in manager.get_venv(venv_id=shared_venv_id)["venv"]

    outcome = manager.add_functions(
        implementations="def invisible_helper():\n    return 'nope'",
        destination="space:404404",
    )
    assert outcome["error_kind"] == "invalid_destination"


def test_function_reads_keep_personal_implementation_when_names_overlap(
    manager_routing_context,
):
    """Personal functions win when a shared function has the same name."""

    _, space_id = manager_routing_context
    manager = FunctionManager(include_primitives=False)

    manager.add_functions(
        implementations="def duplicate_helper():\n    return 'personal'",
    )
    manager.add_functions(
        implementations="def duplicate_helper():\n    return 'shared'",
        destination=f"space:{space_id}",
    )

    duplicate = manager.list_functions(include_implementations=True)["duplicate_helper"]
    assert "return 'personal'" in duplicate["implementation"]


def test_sync_custom_venvs_routes_each_destination_independently(
    manager_routing_context,
):
    """Custom venv syncs keep personal and shared roots independent."""

    _, space_id = manager_routing_context
    manager = FunctionManager(include_primitives=False)
    venv_name = f"custom-venv-{uuid.uuid4().hex[:8]}"
    source_venvs = {
        venv_name: {
            "name": venv_name,
            "venv": (
                "[project]\n"
                f"name = '{venv_name}'\n"
                "version = '0.1.0'\n"
                "dependencies = []"
            ),
            "custom_hash": uuid.uuid4().hex[:16],
        },
    }

    shared_ids = manager.sync_custom_venvs(
        source_venvs=source_venvs,
        destination=f"space:{space_id}",
    )
    personal_ids = manager.sync_custom_venvs(source_venvs=source_venvs)

    personal_rows = unify.get_logs(
        context=manager._venvs_ctx,
        filter=f"name == '{venv_name}'",
    )
    shared_rows = unify.get_logs(
        context=f"Spaces/{space_id}/Functions/VirtualEnvs",
        filter=f"name == '{venv_name}'",
    )

    assert venv_name in shared_ids
    assert venv_name in personal_ids
    assert [row.entries["custom_hash"] for row in personal_rows] == [
        source_venvs[venv_name]["custom_hash"],
    ]
    assert [row.entries["custom_hash"] for row in shared_rows] == [
        source_venvs[venv_name]["custom_hash"],
    ]

    source_venvs[venv_name] = {
        **source_venvs[venv_name],
        "venv": source_venvs[venv_name]["venv"] + "\n# refreshed",
        "custom_hash": uuid.uuid4().hex[:16],
    }
    manager.sync_custom_venvs(
        source_venvs=source_venvs,
        destination=f"space:{space_id}",
    )
    refreshed_shared_rows = unify.get_logs(
        context=f"Spaces/{space_id}/Functions/VirtualEnvs",
        filter=f"name == '{venv_name}'",
    )
    assert [row.entries["custom_hash"] for row in refreshed_shared_rows] == [
        source_venvs[venv_name]["custom_hash"],
    ]


def test_sync_custom_functions_routes_each_destination_independently(
    manager_routing_context,
):
    """Custom function syncs keep personal and shared roots independent."""

    _, space_id = manager_routing_context
    manager = FunctionManager(include_primitives=False)
    function_name = f"custom_sync_helper_{uuid.uuid4().hex[:8]}"
    source_functions = {
        function_name: {
            "name": function_name,
            "implementation": f"def {function_name}():\n    return 'synced'",
            "argspec": "() -> str",
            "docstring": "Return a deterministic sync marker.",
            "custom_hash": uuid.uuid4().hex[:16],
            "embedding_text": f"{function_name} deterministic sync marker",
            "is_primitive": False,
            "verify": False,
        },
    }

    assert (
        manager.sync_custom_functions(
            source_functions=source_functions,
            destination=f"space:{space_id}",
        )
        is True
    )
    assert manager.sync_custom_functions(source_functions=source_functions) is True

    personal_rows = unify.get_logs(
        context=manager._compositional_ctx,
        filter=f"name == '{function_name}'",
    )
    shared_rows = unify.get_logs(
        context=f"Spaces/{space_id}/Functions/Compositional",
        filter=f"name == '{function_name}'",
    )

    assert [row.entries["custom_hash"] for row in personal_rows] == [
        source_functions[function_name]["custom_hash"],
    ]
    assert [row.entries["custom_hash"] for row in shared_rows] == [
        source_functions[function_name]["custom_hash"],
    ]

    source_functions[function_name] = {
        **source_functions[function_name],
        "implementation": f"def {function_name}():\n    return 'refreshed'",
        "custom_hash": uuid.uuid4().hex[:16],
    }
    assert (
        manager.sync_custom_functions(
            source_functions=source_functions,
            destination=f"space:{space_id}",
        )
        is True
    )
    refreshed_shared_rows = unify.get_logs(
        context=f"Spaces/{space_id}/Functions/Compositional",
        filter=f"name == '{function_name}'",
    )
    assert [row.entries["custom_hash"] for row in refreshed_shared_rows] == [
        source_functions[function_name]["custom_hash"],
    ]

    outcome = manager.sync_custom(
        source_functions=source_functions,
        source_venvs={},
        destination="space:404404",
    )
    assert outcome["error_kind"] == "invalid_destination"
