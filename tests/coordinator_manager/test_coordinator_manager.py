from __future__ import annotations

from unittest.mock import patch

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


def setup_function():
    ContextRegistry.clear()
    ManagerRegistry.clear()
    SESSION_DETAILS.reset()
    CONTEXT_READ.set("user123/42")
    CONTEXT_WRITE.set("user123/42")


def teardown_function():
    ContextRegistry.clear()
    ManagerRegistry.clear()
    SESSION_DETAILS.reset()


class TestCoordinatorOnboardingManager:
    def test_contexts_are_private_to_assistant_scope(self):
        SESSION_DETAILS.space_ids = [3, 7]

        with patch("unity.common.context_registry._create_context_with_retry"):
            manager = CoordinatorOnboardingManager()

        state_root = manager._state_context.removesuffix("/Coordinator/State")
        checklist_root = manager._checklist_context.removesuffix(
            "/Coordinator/Checklist",
        )
        assert state_root == checklist_root
        assert manager._state_context.endswith("/Coordinator/State")
        assert manager._checklist_context.endswith("/Coordinator/Checklist")
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
