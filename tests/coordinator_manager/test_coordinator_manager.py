from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from unify.logs import CONTEXT_READ, CONTEXT_WRITE
from unify.utils.http import RequestError

from unity.common.context_registry import ContextRegistry
from unity.coordinator_manager.coordinator_manager import (
    COORDINATOR_CHECKLIST_CONTEXT,
    COORDINATOR_STATE_CONTEXT,
    CoordinatorOnboardingManager,
)
from unity.manager_registry import ManagerRegistry
from unity.session_details import SESSION_DETAILS


@pytest.fixture(autouse=True)
def reset_coordinator_manager_state():
    ContextRegistry.clear()
    ManagerRegistry.clear()
    SESSION_DETAILS.reset()
    CONTEXT_READ.set("user123/42")
    CONTEXT_WRITE.set("user123/42")
    yield
    ContextRegistry.clear()
    ManagerRegistry.clear()
    SESSION_DETAILS.reset()


class TestCoordinatorOnboardingManager:
    def test_contexts_are_private_to_assistant_scope(self):
        SESSION_DETAILS.space_ids = [3, 7]

        with (
            patch("unity.common.context_registry._create_context_with_retry"),
            patch(
                "unity.common.context_registry.create_fields",
            ),
        ):
            manager = CoordinatorOnboardingManager()
            state_context = manager._get_state_context()
            checklist_context = manager._get_checklist_context()
        state_root = state_context.removesuffix("/Coordinator/State")
        checklist_root = checklist_context.removesuffix(
            "/Coordinator/Checklist",
        )
        assert state_root == checklist_root
        assert state_context.endswith("/Coordinator/State")
        assert checklist_context.endswith("/Coordinator/Checklist")
        assert ContextRegistry.read_roots(
            CoordinatorOnboardingManager,
            COORDINATOR_STATE_CONTEXT,
        ) == [state_root]
        assert ContextRegistry.read_roots(
            CoordinatorOnboardingManager,
            COORDINATOR_CHECKLIST_CONTEXT,
        ) == [state_root]

    def test_direct_instantiation_registers_singleton_instance(self):
        with patch("unity.common.context_registry._create_context_with_retry"):
            manager = CoordinatorOnboardingManager()

        assert ManagerRegistry.get_instance(CoordinatorOnboardingManager) is manager

    def test_checklist_context_has_stable_auto_counted_item_id(self):
        checklist_context = next(
            context
            for context in CoordinatorOnboardingManager.Config.required_contexts
            if context.name == COORDINATOR_CHECKLIST_CONTEXT
        )

        assert checklist_context.unique_keys == {"item_id": "int"}
        assert checklist_context.auto_counting == {"item_id": None}

    @pytest.mark.parametrize("initial_status", ["pending", "done", "skipped"])
    def test_add_checklist_item_accepts_initial_status(self, initial_status):
        with (
            patch("unity.common.context_registry._create_context_with_retry"),
            patch(
                "unity.coordinator_manager.coordinator_manager.unity_log",
                return_value=SimpleNamespace(entries={"item_id": 17}),
            ) as write_log,
            patch(
                "unity.coordinator_manager.coordinator_manager.publish_coordinator_activity",
            ) as publish_activity,
        ):
            manager = CoordinatorOnboardingManager()
            with patch.object(
                manager,
                "_get_checklist_context",
                return_value="assistants/42/Coordinator/Checklist",
            ):
                result = manager.add_checklist_item(
                    title="Backfilled checklist phase",
                    initial_status=initial_status,
                )

        assert result == {
            "outcome": "checklist item added",
            "details": {"item_id": 17},
        }
        assert write_log.call_args.kwargs["status"] == initial_status
        expected_phase = "progress" if initial_status == "pending" else "completed"
        assert publish_activity.call_args.kwargs["phase"] == expected_phase

    def test_add_checklist_item_rejects_invalid_initial_status(self):
        with (
            patch("unity.common.context_registry._create_context_with_retry"),
            patch(
                "unity.coordinator_manager.coordinator_manager.unity_log",
            ) as write_log,
            patch(
                "unity.coordinator_manager.coordinator_manager.publish_coordinator_activity",
            ) as publish_activity,
        ):
            manager = CoordinatorOnboardingManager()
            result = manager.add_checklist_item(
                title="Invalid status checklist phase",
                initial_status="blocked",
            )

        assert result["error_kind"] == "invalid_argument"
        assert result["details"] == {"status": "blocked"}
        write_log.assert_not_called()
        publish_activity.assert_not_called()

    def test_personal_coordinator_has_empty_authorized_humans(self):
        with patch("unity.common.context_registry._create_context_with_retry"):
            manager = CoordinatorOnboardingManager()

        with patch(
            "unity.coordinator_manager.coordinator_manager.unify.list_org_members",
        ) as list_members:
            assert manager.get_org_members() == []

        list_members.assert_not_called()

    def test_org_members_are_cached(self):
        SESSION_DETAILS.org_id = 7
        SESSION_DETAILS.unify_key = "owner-key"

        with patch("unity.common.context_registry._create_context_with_retry"):
            manager = CoordinatorOnboardingManager()

        with patch(
            "unity.coordinator_manager.coordinator_manager.unify.list_org_members",
            return_value=[{"email": "dana@acme.com"}],
        ) as list_members:
            assert manager.get_org_members() == [{"email": "dana@acme.com"}]
            assert manager.get_org_members() == [{"email": "dana@acme.com"}]

        list_members.assert_called_once_with(7, api_key="owner-key")

    def test_org_member_failures_do_not_poison_cache(self):
        SESSION_DETAILS.org_id = 7
        SESSION_DETAILS.unify_key = "owner-key"

        with patch("unity.common.context_registry._create_context_with_retry"):
            manager = CoordinatorOnboardingManager()

        response = requests.Response()
        response.status_code = 500
        response._content = b"temporary"
        with patch(
            "unity.coordinator_manager.coordinator_manager.unify.list_org_members",
            side_effect=[
                RequestError("https://api.unify.ai", "GET", response),
                [{"email": "dana@acme.com"}],
            ],
        ) as list_members:
            assert manager.get_org_members() == []
            assert manager.get_org_members() == [{"email": "dana@acme.com"}]

        assert list_members.call_count == 2

    def test_org_coordinator_name_uses_org_wide_assistant_listing(self):
        SESSION_DETAILS.org_id = 7
        SESSION_DETAILS.unify_key = "owner-key"

        with patch("unity.common.context_registry._create_context_with_retry"):
            manager = CoordinatorOnboardingManager()

        with patch(
            "unity.coordinator_manager.coordinator_manager.unify.list_assistants",
            return_value=[
                {
                    "first_name": "Patch",
                    "surname": "One",
                    "is_coordinator": False,
                },
                {
                    "first_name": "Avery",
                    "surname": "Coordinator",
                    "is_coordinator": True,
                },
            ],
        ) as list_assistants:
            assert manager.get_org_coordinator_name() == "Avery Coordinator"
            assert manager.get_org_coordinator_name() == "Avery Coordinator"

        list_assistants.assert_called_once_with(
            list_all_org=True,
            api_key="owner-key",
        )

    def test_org_coordinator_name_returns_none_when_no_coordinator_exists(self):
        SESSION_DETAILS.org_id = 7
        SESSION_DETAILS.unify_key = "owner-key"

        with patch("unity.common.context_registry._create_context_with_retry"):
            manager = CoordinatorOnboardingManager()

        with patch(
            "unity.coordinator_manager.coordinator_manager.unify.list_assistants",
            return_value=[{"first_name": "Patch", "surname": "One"}],
        ):
            assert manager.get_org_coordinator_name() is None

    def test_org_coordinator_name_failures_do_not_poison_cache(self):
        SESSION_DETAILS.org_id = 7
        SESSION_DETAILS.unify_key = "owner-key"

        with patch("unity.common.context_registry._create_context_with_retry"):
            manager = CoordinatorOnboardingManager()

        response = requests.Response()
        response.status_code = 500
        response._content = b"temporary"
        with patch(
            "unity.coordinator_manager.coordinator_manager.unify.list_assistants",
            side_effect=[
                RequestError("https://api.unify.ai", "GET", response),
                [
                    {
                        "first_name": "Avery",
                        "surname": "Coordinator",
                        "is_coordinator": True,
                    },
                ],
            ],
        ) as list_assistants:
            assert manager.get_org_coordinator_name() is None
            assert manager.get_org_coordinator_name() == "Avery Coordinator"

        assert list_assistants.call_count == 2

    def test_org_coordinator_name_cache_follows_session_identity(self):
        SESSION_DETAILS.org_id = 7
        SESSION_DETAILS.unify_key = "owner-key"

        with patch("unity.common.context_registry._create_context_with_retry"):
            manager = CoordinatorOnboardingManager()

        with patch(
            "unity.coordinator_manager.coordinator_manager.unify.list_assistants",
            side_effect=[
                [
                    {
                        "first_name": "Avery",
                        "surname": "Coordinator",
                        "is_coordinator": True,
                    },
                ],
                [
                    {
                        "first_name": "Blair",
                        "surname": "Coordinator",
                        "is_coordinator": True,
                    },
                ],
            ],
        ) as list_assistants:
            assert manager.get_org_coordinator_name() == "Avery Coordinator"
            SESSION_DETAILS.org_id = 8
            SESSION_DETAILS.unify_key = "admin-key"
            assert manager.get_org_coordinator_name() == "Blair Coordinator"

        assert list_assistants.call_count == 2

    def test_coordinator_name_lookup_does_not_create_onboarding_contexts(self):
        SESSION_DETAILS.org_id = 7
        SESSION_DETAILS.unify_key = "owner-key"

        with patch(
            "unity.common.context_registry._create_context_with_retry",
        ) as create_context:
            manager = CoordinatorOnboardingManager()
            with patch(
                "unity.coordinator_manager.coordinator_manager.unify.list_assistants",
                return_value=[
                    {
                        "first_name": "Avery",
                        "surname": "Coordinator",
                        "is_coordinator": True,
                    },
                ],
            ):
                assert manager.get_org_coordinator_name() == "Avery Coordinator"

        create_context.assert_not_called()

    def test_org_member_cache_follows_session_identity(self):
        SESSION_DETAILS.org_id = 7
        SESSION_DETAILS.unify_key = "owner-key"

        with patch("unity.common.context_registry._create_context_with_retry"):
            manager = CoordinatorOnboardingManager()

        with patch(
            "unity.coordinator_manager.coordinator_manager.unify.list_org_members",
            side_effect=[
                [{"email": "owner@acme.com"}],
                [{"email": "admin@beta.com"}],
            ],
        ) as list_members:
            assert manager.get_org_members() == [{"email": "owner@acme.com"}]
            SESSION_DETAILS.org_id = 8
            SESSION_DETAILS.unify_key = "admin-key"
            assert manager.get_org_members() == [{"email": "admin@beta.com"}]

        assert list_members.call_count == 2
