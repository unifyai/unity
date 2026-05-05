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

    def test_as_tools_exposes_exact_workspace_surface(self):
        tools = CoordinatorTools(cm=object()).as_tools()

        assert set(tools) == {
            "create_assistant",
            "delete_assistant",
            "update_assistant_config",
            "list_assistants",
            "list_org_members",
            "create_space",
            "delete_space",
            "update_space",
            "add_space_member",
            "remove_space_member",
            "list_spaces",
            "list_space_members",
            "list_spaces_for_assistant",
            "invite_assistant_to_space",
            "cancel_space_invitation",
            "list_pending_invitations",
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

    def test_create_space_uses_owner_key_and_current_workspace_scope(
        self,
        monkeypatch,
    ):
        calls = []

        def fake_create_space(**kwargs):
            calls.append(kwargs)
            return {"space_id": 11, "name": "Ops"}

        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.create_space",
            fake_create_space,
        )

        result = CoordinatorTools(cm=object()).create_space(
            name="Ops",
            organization_id=999,
            owner_user_id="other-user",
        )

        assert result == {"space_id": 11, "name": "Ops"}
        assert calls == [
            {
                "name": "Ops",
                "organization_id": 7,
                "api_key": "owner-key",
            },
        ]

    def test_list_spaces_cache_authorizes_follow_up_space_writes(self, monkeypatch):
        list_calls = []
        delete_calls = []

        def fake_list_spaces(**kwargs):
            list_calls.append(kwargs)
            return [{"space_id": 11, "name": "Ops"}]

        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.list_spaces",
            fake_list_spaces,
        )
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.delete_space",
            lambda *args, **kwargs: delete_calls.append((args, kwargs)) or {},
        )

        tools = CoordinatorTools(cm=object())
        assert tools.list_spaces(
            organization_id=999,
            owner_user_id="other-user",
        ) == [{"space_id": 11, "name": "Ops"}]

        result = tools.delete_space(space_id=11)

        assert result == {}
        assert list_calls == [{"organization_id": 7, "api_key": "owner-key"}]
        assert delete_calls == [((11,), {"api_key": "owner-key"})]

    def test_space_member_writes_require_reachable_space_and_assistant(
        self,
        monkeypatch,
    ):
        member_calls = []

        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.list_spaces",
            lambda **_: [{"space_id": 11}],
        )
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.list_assistants",
            lambda **_: [{"agent_id": 42}],
        )
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.add_space_member",
            lambda *args, **kwargs: member_calls.append((args, kwargs))
            or {"membership_status": "active"},
        )

        result = CoordinatorTools(cm=object()).add_space_member(
            space_id=11,
            assistant_id=42,
        )

        assert result == {"membership_status": "active"}
        assert member_calls == [((11, 42), {"api_key": "owner-key"})]

    def test_remaining_space_wrappers_forward_after_reachability(
        self,
        monkeypatch,
    ):
        calls = []

        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.list_spaces",
            lambda **_: [{"space_id": 11, "name": "Ops"}],
        )
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.list_assistants",
            lambda **_: [{"agent_id": 42}],
        )
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.update_space",
            lambda *args, **kwargs: calls.append(("update", args, kwargs))
            or {"space_id": 11, "name": "Ops Team"},
        )
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.remove_space_member",
            lambda *args, **kwargs: calls.append(("remove", args, kwargs)) or {},
        )
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.list_space_members",
            lambda *args, **kwargs: calls.append(("members", args, kwargs))
            or [{"assistant_id": 42}],
        )
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.list_spaces_for_assistant",
            lambda *args, **kwargs: calls.append(("assistant_spaces", args, kwargs))
            or [{"space_id": 11}],
        )
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.invite_assistant_to_space",
            lambda *args, **kwargs: calls.append(("invite", args, kwargs))
            or {"invite_id": 13},
        )
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.list_pending_invitations",
            lambda **kwargs: calls.append(("pending", (), kwargs))
            or [{"invite_id": 13}],
        )

        tools = CoordinatorTools(cm=object())

        assert tools.update_space(
            space_id=11,
            patch={"name": "Ops Team"},
        ) == {"space_id": 11, "name": "Ops Team"}
        assert tools.list_space_members(space_id=11) == [{"assistant_id": 42}]
        assert tools.list_spaces_for_assistant(assistant_id=42) == [{"space_id": 11}]
        assert tools.invite_assistant_to_space(
            space_id=11,
            assistant_id=42,
        ) == {"invite_id": 13}
        assert tools.remove_space_member(space_id=11, assistant_id=42) == {}
        assert tools.list_pending_invitations() == [{"invite_id": 13}]
        assert calls == [
            ("update", (11, {"name": "Ops Team"}), {"api_key": "owner-key"}),
            ("members", (11,), {"api_key": "owner-key"}),
            ("assistant_spaces", (42,), {"api_key": "owner-key"}),
            ("invite", (11, 42), {"api_key": "owner-key"}),
            ("remove", (11, 42), {"api_key": "owner-key"}),
            ("pending", (), {"api_key": "owner-key"}),
        ]

    def test_fabricated_space_id_returns_tool_error_without_sdk_write(
        self,
        monkeypatch,
    ):
        member_calls = []

        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.list_spaces",
            lambda **_: [{"space_id": 11}],
        )
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.add_space_member",
            lambda *args, **kwargs: member_calls.append((args, kwargs)),
        )

        result = CoordinatorTools(cm=object()).add_space_member(
            space_id=99,
            assistant_id=42,
        )

        assert result["error_kind"] == "not_found"
        assert result["details"] == {"space_id": 99}
        assert member_calls == []

    def test_space_request_errors_return_tool_error(self, monkeypatch):
        def failing_delete_space(*_, **__):
            response = requests.Response()
            response.status_code = 409
            response._content = b"conflict"
            raise RequestError("https://api.unify.ai", "DELETE", response)

        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.list_spaces",
            lambda **_: [{"space_id": 11}],
        )
        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.delete_space",
            failing_delete_space,
        )

        result = CoordinatorTools(cm=object()).delete_space(space_id=11)

        assert result["error_kind"] == "conflict"
        assert result["details"]["status_code"] == 409

    def test_cancel_space_invitation_uses_owner_key(self, monkeypatch):
        calls = []

        monkeypatch.setattr(
            "unity.conversation_manager.domains.coordinator_tools.unify.cancel_space_invitation",
            lambda *args, **kwargs: calls.append((args, kwargs)) or {},
        )

        result = CoordinatorTools(cm=object()).cancel_space_invitation(invite_id=13)

        assert result == {}
        assert calls == [((13,), {"api_key": "owner-key"})]
