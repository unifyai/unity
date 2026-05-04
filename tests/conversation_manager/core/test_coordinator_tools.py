from __future__ import annotations

import requests

from unify.utils.http import RequestError

from unity.conversation_manager.domains.coordinator_tools import CoordinatorTools
from unity.session_details import SESSION_DETAILS


class TestCoordinatorTools:
    def setup_method(self):
        SESSION_DETAILS.reset()
        SESSION_DETAILS.unify_key = "owner-key"
        SESSION_DETAILS.org_id = 7

    def teardown_method(self):
        SESSION_DETAILS.reset()

    def test_as_tools_exposes_exact_lifecycle_surface(self):
        tools = CoordinatorTools(cm=object()).as_tools()

        assert set(tools) == {
            "create_assistant",
            "delete_assistant",
            "update_assistant_config",
            "list_assistants",
            "list_org_members",
        }

    def test_list_assistants_uses_owner_key_and_current_sdk_shape(self, monkeypatch):
        calls = []

        def fake_list_assistants(**kwargs):
            calls.append(kwargs)
            return [{"agent_id": 42, "first_name": "Ops"}]

        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.list_assistants",
            fake_list_assistants,
        )

        result = CoordinatorTools(cm=object()).list_assistants()

        assert result == [{"agent_id": 42, "first_name": "Ops"}]
        assert calls == [
            {
                "phone": None,
                "email": None,
                "agent_id": None,
                "list_all_org": True,
                "api_key": "owner-key",
            },
        ]

    def test_delete_requires_reachable_assistant(self, monkeypatch):
        delete_calls = []

        list_calls = []

        def fake_list_assistants(**kwargs):
            list_calls.append(kwargs)
            return [{"agent_id": 42}]

        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.list_assistants",
            fake_list_assistants,
        )
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.delete_assistant",
            lambda *args, **kwargs: delete_calls.append((args, kwargs)),
        )

        missing = CoordinatorTools(cm=object()).delete_assistant(agent_id=99)

        assert missing["error_kind"] == "not_found"
        assert list_calls[0]["agent_id"] == 99
        assert delete_calls == []

    def test_filtered_lists_do_not_poison_reachability_cache(self, monkeypatch):
        list_calls = []

        def fake_list_assistants(**kwargs):
            list_calls.append(kwargs)
            if kwargs["phone"] is not None:
                return [{"agent_id": 42}]
            if kwargs["agent_id"] == 99:
                return [{"agent_id": 99}]
            return []

        delete_calls = []
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.list_assistants",
            fake_list_assistants,
        )
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.delete_assistant",
            lambda *args, **kwargs: delete_calls.append((args, kwargs)) or {},
        )

        tools = CoordinatorTools(cm=object())
        assert tools.list_assistants(phone="+15555551234") == [{"agent_id": 42}]

        result = tools.delete_assistant(agent_id=99)

        assert result == {}
        assert list_calls[1]["agent_id"] == 99
        assert delete_calls == [((99,), {"api_key": "owner-key"})]

    def test_request_errors_return_tool_error(self, monkeypatch):
        def failing_create_assistant(**_):
            response = requests.Response()
            response.status_code = 403
            response._content = b"forbidden"
            raise RequestError("https://api.unify.ai", "POST", response)

        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.create_assistant",
            failing_create_assistant,
        )

        result = CoordinatorTools(cm=object()).create_assistant(first_name="Ops")

        assert result["error_kind"] == "permission_denied"
        assert result["details"]["status_code"] == 403

    def test_personal_coordinator_returns_empty_org_members(self, monkeypatch):
        SESSION_DETAILS.org_id = None
        called = False

        def fake_list_org_members(*_, **__):
            nonlocal called
            called = True
            return []

        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.list_org_members",
            fake_list_org_members,
        )

        result = CoordinatorTools(cm=object()).list_org_members()

        assert result == []
        assert called is False
