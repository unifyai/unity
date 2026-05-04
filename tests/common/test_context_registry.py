from unittest.mock import patch

import pytest
from unify.logs import CONTEXT_READ, CONTEXT_WRITE

from unity.common.context_registry import ContextRegistry, TableContext
from unity.common.tool_outcome import ToolErrorException
from unity.session_details import SESSION_DETAILS


class RegistryExampleManager:
    class Config:
        required_contexts = [
            TableContext(
                name="Tasks",
                description="Scheduled work items.",
            ),
            TableContext(
                name="Contacts",
                description="People and organizations the assistant knows.",
            ),
            TableContext(name="Secrets", description="Credentials."),
            TableContext(name="FileRecords", description="File indexes."),
            TableContext(name="Files", description="File content."),
            TableContext(name="Data", description="Datasets."),
            TableContext(name="BlackList", description="Blocked contacts."),
        ]


@pytest.fixture(autouse=True)
def reset_context_registry():
    ContextRegistry.clear()
    SESSION_DETAILS.reset()
    CONTEXT_READ.set("user123/42")
    CONTEXT_WRITE.set("user123/42")
    yield
    ContextRegistry.clear()
    SESSION_DETAILS.reset()


def test_write_root_resolves_personal_and_space_destinations():
    SESSION_DETAILS.space_ids = [3, 7]

    with patch("unity.common.context_registry._create_context_with_retry"):
        assert (
            ContextRegistry.write_root(
                RegistryExampleManager,
                "Tasks",
                destination=None,
            )
            == "user123/42"
        )
        assert (
            ContextRegistry.write_root(
                RegistryExampleManager,
                "Tasks",
                destination="personal",
            )
            == "user123/42"
        )
        assert (
            ContextRegistry.write_root(
                RegistryExampleManager,
                "Contacts",
                destination="space:7",
            )
            == "Spaces/7"
        )


def test_write_root_resolves_all_manager_destination_tables_to_shared_spaces():
    class DestinationAwareManager:
        class Config:
            required_contexts = [
                TableContext(
                    name="Knowledge",
                    description="Structured assistant memory.",
                ),
                TableContext(
                    name="Guidance",
                    description="Assistant behavior guidance.",
                ),
                TableContext(
                    name="Functions/Compositional",
                    description="Assistant-authored functions.",
                ),
                TableContext(
                    name="Functions/VirtualEnvs",
                    description="Custom function environments.",
                ),
                TableContext(
                    name="Functions/Primitives",
                    description="Runtime-provided primitive functions.",
                ),
                TableContext(
                    name="Functions/Meta",
                    description="Function synchronization metadata.",
                ),
            ]

    SESSION_DETAILS.space_ids = [37]

    with patch("unity.common.context_registry._create_context_with_retry"):
        for table_name in (
            "Knowledge",
            "Guidance",
            "Functions/Compositional",
            "Functions/VirtualEnvs",
            "Functions/Primitives",
            "Functions/Meta",
        ):
            assert (
                ContextRegistry.write_root(
                    DestinationAwareManager,
                    table_name,
                    destination="space:37",
                )
                == "Spaces/37"
            )


def test_invalid_destination_raises_structured_error():
    SESSION_DETAILS.space_ids = [3, 7]

    with pytest.raises(ToolErrorException) as exc_info:
        ContextRegistry.write_root(
            RegistryExampleManager,
            "Tasks",
            destination="space:999",
        )

    assert exc_info.value.payload["error_kind"] == "invalid_destination"
    assert exc_info.value.payload["details"]["destination"] == "space:999"
    assert exc_info.value.payload["details"]["space_ids"] == [3, 7]


def test_read_roots_returns_personal_then_sorted_spaces():
    SESSION_DETAILS.space_ids = [7, 3]

    with patch("unity.common.context_registry._create_context_with_retry"):
        task_roots = ContextRegistry.read_roots(RegistryExampleManager, "Tasks")
        contact_roots = ContextRegistry.read_roots(RegistryExampleManager, "Contacts")

    assert task_roots == ["user123/42", "Spaces/3", "Spaces/7"]
    assert contact_roots == ["user123/42", "Spaces/3", "Spaces/7"]


def test_files_data_and_blacklist_are_shared_scoped():
    SESSION_DETAILS.space_ids = [7]

    with patch("unity.common.context_registry._create_context_with_retry"):
        assert ContextRegistry.read_roots(RegistryExampleManager, "Secrets") == [
            "user123/42",
            "Spaces/7",
        ]
        assert ContextRegistry.read_roots(RegistryExampleManager, "FileRecords") == [
            "user123/42",
            "Spaces/7",
        ]
        assert ContextRegistry.read_roots(RegistryExampleManager, "Files") == [
            "user123/42",
            "Spaces/7",
        ]
        assert ContextRegistry.read_roots(RegistryExampleManager, "Data") == [
            "user123/42",
            "Spaces/7",
        ]
        assert ContextRegistry.read_roots(RegistryExampleManager, "BlackList") == [
            "user123/42",
            "Spaces/7",
        ]


def test_lazy_provisioning_is_cached_per_root():
    SESSION_DETAILS.space_ids = [7]

    with patch(
        "unity.common.context_registry._create_context_with_retry",
    ) as create_context:
        ContextRegistry.write_root(
            RegistryExampleManager,
            "Tasks",
            destination="space:7",
        )
        ContextRegistry.write_root(
            RegistryExampleManager,
            "Tasks",
            destination="space:7",
        )

    create_context.assert_called_once()
    assert create_context.call_args.args[0] == "Spaces/7/Tasks"
    assert (
        ContextRegistry._registry[("RegistryExampleManager", "Tasks", "Spaces/7")]
        == "Spaces/7/Tasks"
    )
