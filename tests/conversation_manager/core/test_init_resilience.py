"""
tests/conversation_manager/core/test_init_resilience.py
=======================================================

Symbolic tests verifying that failures in non-essential initialization steps
degrade gracefully instead of preventing the pod from becoming operational.

Each test mocks a single degradable subsystem to raise during init, then
asserts that ``cm.initialized`` still becomes ``True`` — proving the pod
would survive and serve requests (with reduced capability) rather than
becoming a zombie.
"""

import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock

from tests.helpers import scenario_file_lock

# ---------------------------------------------------------------------------
# Fixture: lightweight CM factory for resilience tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def resilience_cm():
    """Create a ConversationManager, run init_conv_manager, and tear down.

    Yields the CM *before* init so individual tests can apply mocks around
    the ``init_conv_manager`` call.
    """
    from unity.conversation_manager.event_broker import reset_event_broker
    from unity.conversation_manager import start_async, stop_async

    reset_event_broker()

    cm = await start_async(
        project_name="TestInitResilience",
        enable_comms_manager=False,
        apply_test_mocks=True,
    )

    yield cm

    await stop_async()
    reset_event_broker()


async def _init(cm, lock_name="init_resilience"):
    """Helper: run manager init with a SimulatedActor under file lock."""
    from unity.actor.simulated import SimulatedActor
    from unity.conversation_manager.domains import managers_utils

    actor = SimulatedActor(
        steps=None,
        duration=None,
        log_mode="log",
        emit_notifications=False,
    )
    with scenario_file_lock(lock_name):
        await managers_utils.init_conv_manager(cm, actor=actor)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDegradableStepResilience:
    """Failures in optional init steps must not prevent initialization."""

    @pytest.mark.asyncio
    async def test_memory_manager_init_failure(self, resilience_cm):
        cm = resilience_cm
        with patch(
            "unity.conversation_manager.domains.managers_utils.ManagerRegistry.get_memory_manager",
            side_effect=ConnectionError("Orchestra unreachable"),
        ):
            # Force the feature flag on so the guarded path is exercised
            from unity.settings import SETTINGS

            original = SETTINGS.memory.ENABLED
            SETTINGS.memory.ENABLED = True
            try:
                await _init(cm, "resilience_memory")
            finally:
                SETTINGS.memory.ENABLED = original

        assert cm.initialized is True
        assert cm.memory_manager is None

    @pytest.mark.asyncio
    async def test_file_manager_init_failure(self, resilience_cm):
        cm = resilience_cm
        with patch(
            "unity.conversation_manager.domains.managers_utils.ManagerRegistry.get_file_manager",
            side_effect=ConnectionError("Orchestra unreachable"),
        ):
            await _init(cm, "resilience_file")

        assert cm.initialized is True

    @pytest.mark.asyncio
    async def test_custom_function_sync_failure(self, resilience_cm):
        cm = resilience_cm
        with patch(
            "unity.function_manager.custom_functions.collect_functions_from_directories",
            side_effect=RuntimeError("filesystem error"),
        ):
            await _init(cm, "resilience_funcsyns")

        assert cm.initialized is True

    @pytest.mark.asyncio
    async def test_primitive_sync_failure(self, resilience_cm):
        cm = resilience_cm
        with patch(
            "unity.conversation_manager.domains.managers_utils.ManagerRegistry.get_function_manager",
        ) as mock_fm:
            mock_instance = MagicMock()
            mock_instance.sync_primitives.side_effect = RuntimeError("sync error")
            mock_instance.warm_embeddings.side_effect = RuntimeError("warm error")
            mock_fm.return_value = mock_instance
            await _init(cm, "resilience_prim")

        assert cm.initialized is True

    @pytest.mark.asyncio
    async def test_embedding_warmup_failure(self, resilience_cm):
        cm = resilience_cm
        with patch(
            "unity.conversation_manager.domains.managers_utils.ManagerRegistry.warm_all_embeddings",
            side_effect=ConnectionError("Orchestra unreachable"),
        ):
            await _init(cm, "resilience_embed")

        assert cm.initialized is True


class TestContextRegistryResilience:
    """Individual context creation failures must not crash setup()."""

    def test_partial_context_creation_failure_does_not_raise(self):
        """ContextRegistry.setup() tolerates individual context creation errors."""
        from unity.common.context_registry import ContextRegistry

        original = ContextRegistry._create_context_wrapper

        call_count = 0

        @classmethod
        def _flaky(cls, manager_name, entry):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("transient network failure")
            return original.__func__(cls, manager_name, entry)

        ContextRegistry._setup_complete = False
        try:
            with patch.object(ContextRegistry, "_create_context_wrapper", _flaky):
                ContextRegistry.setup()
        finally:
            ContextRegistry._setup_complete = False

        assert call_count > 1, "Mock was not exercised"


class TestUserDetailsFetchResilience:
    """Orchestra user-info fetch failure should fall back to SESSION_DETAILS."""

    @pytest.mark.asyncio
    async def test_user_details_fetch_failure_uses_fallback(self, resilience_cm):
        cm = resilience_cm
        with patch(
            "unity.contact_manager.system_contacts.unify.get_user_basic_info",
            side_effect=ConnectionError("Orchestra unreachable"),
        ):
            await _init(cm, "resilience_userinfo")

        assert cm.initialized is True
        contact_info = cm.contact_manager.get_contact_info(1)
        user_contact = contact_info.get(1, {})
        assert user_contact is not None


class TestLivekitWorkerResilience:
    """LiveKit worker failure must not prevent manager initialization."""

    @pytest.mark.asyncio
    async def test_livekit_failure_does_not_block_init(self, resilience_cm):
        cm = resilience_cm

        original_start = cm.call_manager.start_persistent_worker
        cm.call_manager.start_persistent_worker = MagicMock(
            side_effect=RuntimeError("worker script not found"),
        )
        try:
            await _init(cm, "resilience_livekit")
        finally:
            cm.call_manager.start_persistent_worker = original_start

        assert cm.initialized is True
