"""
Tests verifying that Conductor respects global settings for manager configuration.

These tests verify:
1. Optional managers are disabled by default
2. Foundational managers are always present regardless of settings
3. Explicit manager arguments override settings
4. DISABLED sentinel explicitly disables managers
5. Tools reflect enabled/disabled managers
6. SimulatedConductor bypasses settings and has all managers
"""

from unittest.mock import patch

from tests.helpers import _handle_project


# ---------------------------------------------------------------------------
#  Test: Default settings disable optional managers
# ---------------------------------------------------------------------------


@_handle_project
def test_default_settings_disable_optional_managers():
    """Verify that optional managers are disabled by default."""
    from unity.settings import SETTINGS

    # Optional managers should be disabled by default
    assert SETTINGS.knowledge.ENABLED is False
    assert SETTINGS.guidance.ENABLED is False
    assert SETTINGS.secret.ENABLED is False
    assert SETTINGS.web.ENABLED is False
    assert SETTINGS.file.ENABLED is False


@_handle_project
def test_default_settings_use_real_implementations():
    """Verify that all implementations default to 'real' (or 'hierarchical' for Actor)."""
    from unity.settings import SETTINGS

    # Foundational managers default to real implementations
    assert SETTINGS.actor.IMPL == "hierarchical"
    assert SETTINGS.contact.IMPL == "real"
    assert SETTINGS.transcript.IMPL == "real"
    assert SETTINGS.task.IMPL == "real"
    assert SETTINGS.conversation.IMPL == "real"

    # Optional managers also default to real (when enabled)
    assert SETTINGS.knowledge.IMPL == "real"
    assert SETTINGS.guidance.IMPL == "real"
    assert SETTINGS.secret.IMPL == "real"
    assert SETTINGS.web.IMPL == "real"
    assert SETTINGS.file.IMPL == "real"


# ---------------------------------------------------------------------------
#  Test: Conductor respects disabled optional managers
# ---------------------------------------------------------------------------


@_handle_project
def test_conductor_optional_managers_disabled_by_default():
    """Verify that Conductor has None for optional managers when using defaults."""
    from unity.conductor import Conductor
    from unity.conversation_manager.simulated import SimulatedConversationManagerHandle

    # Create Conductor with explicit conversation_manager (required for real impl)
    conv = SimulatedConversationManagerHandle(assistant_id="test", contact_id="1")
    conductor = Conductor(conversation_manager=conv)

    # Foundational managers should be present
    assert conductor._contact_manager is not None
    assert conductor._transcript_manager is not None
    assert conductor._task_scheduler is not None
    assert conductor._actor is not None
    assert conductor._cm_handle is not None

    # Optional managers should be None (disabled by default)
    assert conductor._knowledge_manager is None
    assert conductor._guidance_manager is None
    assert conductor._secret_manager is None
    assert conductor._web_searcher is None
    assert conductor._file_manager is None


# ---------------------------------------------------------------------------
#  Test: SimulatedConductor bypasses settings (always has all managers)
# ---------------------------------------------------------------------------


@_handle_project
def test_simulated_conductor_bypasses_settings():
    """Verify that SimulatedConductor has all managers regardless of settings."""
    from unity.conductor.simulated import SimulatedConductor

    conductor = SimulatedConductor()

    # All managers should be present (simulated versions)
    assert conductor._contact_manager is not None
    assert conductor._transcript_manager is not None
    assert conductor._task_scheduler is not None
    assert conductor._actor is not None
    assert conductor._cm_handle is not None

    # Optional managers should also be present (SimulatedConductor bypasses settings)
    assert conductor._knowledge_manager is not None
    assert conductor._guidance_manager is not None
    assert conductor._secret_manager is not None
    assert conductor._web_searcher is not None
    assert conductor._file_manager is not None


# ---------------------------------------------------------------------------
#  Test: Explicit manager arguments override settings
# ---------------------------------------------------------------------------


@_handle_project
def test_explicit_manager_overrides_disabled_setting():
    """Verify that explicitly passed managers take precedence over disabled settings."""
    from unity.conductor import Conductor
    from unity.conversation_manager.simulated import SimulatedConversationManagerHandle
    from unity.knowledge_manager.simulated import SimulatedKnowledgeManager

    # Settings have knowledge disabled, but we pass an explicit instance
    explicit_km = SimulatedKnowledgeManager(description="explicit override")
    conv = SimulatedConversationManagerHandle(assistant_id="test", contact_id="1")

    conductor = Conductor(
        conversation_manager=conv,
        knowledge_manager=explicit_km,
    )

    # The explicit instance should be used
    assert conductor._knowledge_manager is explicit_km


@_handle_project
def test_explicit_actor_overrides_settings():
    """Verify that explicitly passed Actor takes precedence over settings."""
    from unity.conductor import Conductor
    from unity.conversation_manager.simulated import SimulatedConversationManagerHandle
    from unity.actor.simulated import SimulatedActor

    explicit_actor = SimulatedActor()
    conv = SimulatedConversationManagerHandle(assistant_id="test", contact_id="1")

    conductor = Conductor(
        conversation_manager=conv,
        actor=explicit_actor,
    )

    assert conductor._actor is explicit_actor


@_handle_project
def test_explicit_contact_manager_overrides_settings():
    """Verify that explicitly passed ContactManager takes precedence over settings."""
    from unity.conductor import Conductor
    from unity.conversation_manager.simulated import SimulatedConversationManagerHandle
    from unity.contact_manager.simulated import SimulatedContactManager

    explicit_cm = SimulatedContactManager()
    conv = SimulatedConversationManagerHandle(assistant_id="test", contact_id="1")

    conductor = Conductor(
        conversation_manager=conv,
        contact_manager=explicit_cm,
    )

    assert conductor._contact_manager is explicit_cm


# ---------------------------------------------------------------------------
#  Test: DISABLED sentinel explicitly disables managers
# ---------------------------------------------------------------------------


@_handle_project
def test_disabled_sentinel_disables_optional_manager():
    """Verify that DISABLED sentinel explicitly disables an optional manager."""
    from unity.conductor import Conductor
    from unity.conversation_manager.simulated import SimulatedConversationManagerHandle
    from unity.common.sentinels import DISABLED

    conv = SimulatedConversationManagerHandle(assistant_id="test", contact_id="1")

    # Pass DISABLED for knowledge (already disabled by default, but explicit)
    conductor = Conductor(
        conversation_manager=conv,
        knowledge_manager=DISABLED,
    )

    assert conductor._knowledge_manager is None


@_handle_project
def test_disabled_sentinel_overrides_explicit_enable_via_patch():
    """Verify that DISABLED sentinel takes precedence even when settings would enable."""
    from unity.conductor import Conductor
    from unity.conversation_manager.simulated import SimulatedConversationManagerHandle
    from unity.common.sentinels import DISABLED

    conv = SimulatedConversationManagerHandle(assistant_id="test", contact_id="1")

    # Even with patched settings that would enable knowledge, DISABLED wins
    with patch("unity.conductor.conductor.SETTINGS") as mock_settings:
        mock_settings.knowledge.ENABLED = True
        mock_settings.knowledge.IMPL = "simulated"
        mock_settings.guidance.ENABLED = False
        mock_settings.secret.ENABLED = False
        mock_settings.web.ENABLED = False
        mock_settings.file.ENABLED = False
        mock_settings.actor.IMPL = "hierarchical"
        mock_settings.contact.IMPL = "real"
        mock_settings.transcript.IMPL = "real"
        mock_settings.task.IMPL = "real"

        conductor = Conductor(
            conversation_manager=conv,
            knowledge_manager=DISABLED,
        )

        assert conductor._knowledge_manager is None


# ---------------------------------------------------------------------------
#  Test: Tools reflect enabled/disabled managers
# ---------------------------------------------------------------------------


@_handle_project
def test_disabled_managers_have_no_tools():
    """Verify that disabled managers don't expose tools on ask/request surfaces."""
    from unity.conductor import Conductor
    from unity.conversation_manager.simulated import SimulatedConversationManagerHandle

    conv = SimulatedConversationManagerHandle(assistant_id="test", contact_id="1")
    conductor = Conductor(conversation_manager=conv)

    ask_tools = list(conductor.get_tools("ask").keys())
    request_tools = list(conductor.get_tools("request").keys())

    # Foundational manager tools should be present
    assert any("ContactManager" in t for t in ask_tools)
    assert any("TranscriptManager" in t for t in ask_tools)
    assert any("TaskScheduler" in t for t in ask_tools)

    # Disabled optional manager tools should NOT be present
    assert not any("KnowledgeManager" in t for t in ask_tools)
    assert not any("GuidanceManager" in t for t in ask_tools)
    assert not any("SecretManager" in t for t in ask_tools)
    assert not any("WebSearcher" in t for t in ask_tools)
    assert not any("GlobalFileManager" in t for t in ask_tools)

    # Same for request surface
    assert not any("KnowledgeManager" in t for t in request_tools)
    assert not any("GuidanceManager" in t for t in request_tools)


@_handle_project
def test_enabled_managers_have_tools():
    """Verify that explicitly enabled managers expose their tools."""
    from unity.conductor import Conductor
    from unity.conversation_manager.simulated import SimulatedConversationManagerHandle
    from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
    from unity.guidance_manager.simulated import SimulatedGuidanceManager

    conv = SimulatedConversationManagerHandle(assistant_id="test", contact_id="1")
    conductor = Conductor(
        conversation_manager=conv,
        knowledge_manager=SimulatedKnowledgeManager(description="test"),
        guidance_manager=SimulatedGuidanceManager(description="test"),
    )

    ask_tools = list(conductor.get_tools("ask").keys())
    request_tools = list(conductor.get_tools("request").keys())

    # Knowledge and Guidance tools should now be present
    assert any("KnowledgeManager" in t for t in ask_tools)
    assert any("GuidanceManager" in t for t in ask_tools)
    assert any("KnowledgeManager" in t for t in request_tools)
    assert any("GuidanceManager" in t for t in request_tools)


# ---------------------------------------------------------------------------
#  Test: Manager registry lookup works correctly
# ---------------------------------------------------------------------------


@_handle_project
def test_manager_registry_has_all_implementations():
    """Verify that the manager registry contains all expected implementations."""
    from unity.manager_registry import ManagerRegistry

    # Actor implementations
    from unity.actor.hierarchical_actor import HierarchicalActor
    from unity.actor.single_function_actor import SingleFunctionActor
    from unity.actor.code_act_actor import CodeActActor
    from unity.actor.simulated import SimulatedActor

    assert ManagerRegistry.get_class("actor", "hierarchical") is HierarchicalActor
    assert ManagerRegistry.get_class("actor", "single_function") is SingleFunctionActor
    assert ManagerRegistry.get_class("actor", "code_act") is CodeActActor
    assert ManagerRegistry.get_class("actor", "simulated") is SimulatedActor

    # Contact implementations
    from unity.contact_manager.contact_manager import ContactManager
    from unity.contact_manager.simulated import SimulatedContactManager

    assert ManagerRegistry.get_class("contacts", "real") is ContactManager
    assert ManagerRegistry.get_class("contacts", "simulated") is SimulatedContactManager

    # Knowledge implementations
    from unity.knowledge_manager.knowledge_manager import KnowledgeManager
    from unity.knowledge_manager.simulated import SimulatedKnowledgeManager

    assert ManagerRegistry.get_class("knowledge", "real") is KnowledgeManager
    assert (
        ManagerRegistry.get_class("knowledge", "simulated") is SimulatedKnowledgeManager
    )


@_handle_project
def test_manager_registry_raises_for_unknown():
    """Verify that registry raises for unknown implementations."""
    import pytest
    from unity.manager_registry import ManagerRegistry

    with pytest.raises(ValueError, match="Unknown implementation"):
        ManagerRegistry.get_class("actor", "nonexistent")

    with pytest.raises(ValueError, match="Unknown implementation"):
        ManagerRegistry.get_class("nonexistent_manager", "real")
