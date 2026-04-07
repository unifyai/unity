"""Tests for the persistent LiveKit agent worker architecture.

Covers:
- Worker startup and process management
- Job dispatch with metadata (replacing subprocess spawning)
- Entrypoint config parsing from job metadata
- IPC socket initialisation from metadata
- Worker watchdog / restart
- Fallback to legacy subprocess when worker is not running
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unity.conversation_manager.domains.call_manager import (
    CallConfig,
    LivekitCallManager,
    make_room_name,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return CallConfig(
        assistant_id="42",
        user_id="user_abc",
        assistant_bio="Test assistant bio",
        assistant_number="+15551234567",
        voice_provider="elevenlabs",
        voice_id="test_voice_id",
        job_name="unity-test-pod-abc123",
    )


@pytest.fixture
def call_manager(config):
    return LivekitCallManager(config)


@pytest.fixture
def sample_contact():
    return {
        "contact_id": 2,
        "first_name": "Alice",
        "surname": "Smith",
        "phone_number": "+15552222222",
    }


@pytest.fixture
def boss_contact():
    return {
        "contact_id": 1,
        "first_name": "Boss",
        "surname": "User",
        "phone_number": "+15551111111",
    }


# ---------------------------------------------------------------------------
# Worker agent name
# ---------------------------------------------------------------------------


class TestWorkerAgentName:
    def test_agent_name_uses_job_name(self, call_manager):
        assert call_manager.worker_agent_name == "unity_unity-test-pod-abc123"

    def test_agent_name_updates_with_job_name(self, call_manager):
        call_manager.job_name = "unity-other-pod-xyz"
        assert call_manager.worker_agent_name == "unity_unity-other-pod-xyz"


# ---------------------------------------------------------------------------
# Persistent worker startup
# ---------------------------------------------------------------------------


class TestPersistentWorkerStartup:
    @pytest.mark.asyncio
    async def test_start_persistent_worker_spawns_process(self, call_manager):
        with patch(
            "unity.conversation_manager.domains.call_manager.run_script",
        ) as mock_run:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_run.return_value = mock_proc

            call_manager.start_persistent_worker()

            mock_run.assert_called_once()
            args = mock_run.call_args[0]
            assert "worker.py" in str(args[0])
            assert "dev" in args
            assert "unity_unity-test-pod-abc123" in args
            assert call_manager._worker_proc is mock_proc

    @pytest.mark.asyncio
    async def test_start_persistent_worker_idempotent(self, call_manager):
        with patch(
            "unity.conversation_manager.domains.call_manager.run_script",
        ) as mock_run:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_run.return_value = mock_proc

            call_manager.start_persistent_worker()
            call_manager.start_persistent_worker()

            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_persistent_worker_restarts_if_dead(self, call_manager):
        with patch(
            "unity.conversation_manager.domains.call_manager.run_script",
        ) as mock_run:
            dead_proc = MagicMock()
            dead_proc.poll.return_value = 1
            call_manager._worker_proc = dead_proc

            new_proc = MagicMock()
            new_proc.poll.return_value = None
            mock_run.return_value = new_proc

            call_manager.start_persistent_worker()

            mock_run.assert_called_once()
            assert call_manager._worker_proc is new_proc


# ---------------------------------------------------------------------------
# Job dispatch
# ---------------------------------------------------------------------------


class TestJobDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_job_creates_agent_dispatch(
        self,
        call_manager,
        sample_contact,
        boss_contact,
    ):
        mock_dispatch = MagicMock()
        mock_dispatch.id = "dispatch_123"

        mock_lk = AsyncMock()
        mock_lk.agent_dispatch.create_dispatch = AsyncMock(return_value=mock_dispatch)
        mock_lk.aclose = AsyncMock()

        with (
            patch(
                "unity.conversation_manager.domains.call_manager.LiveKitAPI",
                return_value=mock_lk,
            ),
            patch.object(
                call_manager,
                "_ensure_socket_server",
                new_callable=AsyncMock,
                return_value="/tmp/test.sock",
            ),
        ):
            await call_manager._dispatch_job(
                "unity_42_phone",
                "phone",
                sample_contact,
                boss_contact,
                False,
            )

        mock_lk.agent_dispatch.create_dispatch.assert_called_once()
        req = mock_lk.agent_dispatch.create_dispatch.call_args[0][0]
        assert req.agent_name == "unity_unity-test-pod-abc123"
        assert req.room == "unity_42_phone"

        meta = json.loads(req.metadata)
        assert meta["voice_provider"] == "elevenlabs"
        assert meta["voice_id"] == "test_voice_id"
        assert meta["channel"] == "phone"
        assert meta["outbound"] is False
        assert meta["contact"] == sample_contact
        assert meta["boss"] == boss_contact
        assert meta["ipc_socket_path"] == "/tmp/test.sock"
        assert call_manager._active_job is True

    @pytest.mark.asyncio
    async def test_start_call_dispatches_when_worker_running(
        self,
        call_manager,
        sample_contact,
        boss_contact,
    ):
        mock_worker = MagicMock()
        mock_worker.poll.return_value = None
        call_manager._worker_proc = mock_worker

        with patch.object(
            call_manager,
            "_dispatch_job",
            new_callable=AsyncMock,
        ) as mock_dispatch:
            await call_manager.start_call(sample_contact, boss_contact)

            mock_dispatch.assert_called_once_with(
                make_room_name("42", "phone"),
                "phone_call",
                sample_contact,
                boss_contact,
                False,
            )

    @pytest.mark.asyncio
    async def test_start_unify_meet_dispatches_when_worker_running(
        self,
        call_manager,
        sample_contact,
        boss_contact,
    ):
        mock_worker = MagicMock()
        mock_worker.poll.return_value = None
        call_manager._worker_proc = mock_worker

        with patch.object(
            call_manager,
            "_dispatch_job",
            new_callable=AsyncMock,
        ) as mock_dispatch:
            await call_manager.start_unify_meet(
                sample_contact,
                boss_contact,
                room_name="unity_42_meet",
            )

            mock_dispatch.assert_called_once_with(
                "unity_42_meet",
                "unify_meet",
                sample_contact,
                boss_contact,
                False,
            )


# ---------------------------------------------------------------------------
# Fallback to legacy subprocess
# ---------------------------------------------------------------------------


class TestLegacyFallback:
    @pytest.mark.asyncio
    async def test_start_call_falls_back_without_worker(
        self,
        call_manager,
        sample_contact,
        boss_contact,
    ):
        assert call_manager._worker_proc is None

        with patch(
            "unity.conversation_manager.domains.call_manager.run_script",
        ) as mock_run:
            mock_proc = MagicMock()
            mock_run.return_value = mock_proc

            await call_manager.start_call(sample_contact, boss_contact)

            mock_run.assert_called_once()
            assert call_manager._call_proc is mock_proc

    @pytest.mark.asyncio
    async def test_start_call_falls_back_when_worker_dead(
        self,
        call_manager,
        sample_contact,
        boss_contact,
    ):
        dead_worker = MagicMock()
        dead_worker.poll.return_value = 1
        call_manager._worker_proc = dead_worker

        with patch(
            "unity.conversation_manager.domains.call_manager.run_script",
        ) as mock_run:
            mock_proc = MagicMock()
            mock_run.return_value = mock_proc

            await call_manager.start_call(sample_contact, boss_contact)

            mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# has_active_call guard
# ---------------------------------------------------------------------------


class TestActiveCallGuard:
    @pytest.mark.asyncio
    async def test_rejects_second_call_during_dispatch(
        self,
        call_manager,
        sample_contact,
        boss_contact,
    ):
        call_manager._active_job = True

        with patch.object(
            call_manager,
            "_dispatch_job",
            new_callable=AsyncMock,
        ) as mock_dispatch:
            await call_manager.start_call(sample_contact, boss_contact)
            mock_dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_second_call_during_subprocess(
        self,
        call_manager,
        sample_contact,
        boss_contact,
    ):
        call_manager._call_proc = MagicMock()

        with patch(
            "unity.conversation_manager.domains.call_manager.run_script",
        ) as mock_run:
            await call_manager.start_call(sample_contact, boss_contact)
            mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_resets_active_job_flag(self, call_manager):
        call_manager._active_job = True
        await call_manager.cleanup_call_proc()
        assert call_manager._active_job is False

    @pytest.mark.asyncio
    async def test_cleanup_does_not_terminate_worker(self, call_manager):
        mock_worker = MagicMock()
        mock_worker.poll.return_value = None
        call_manager._worker_proc = mock_worker
        call_manager._active_job = True

        await call_manager.cleanup_call_proc()

        assert call_manager._worker_proc is mock_worker
        assert call_manager._active_job is False


# ---------------------------------------------------------------------------
# Entrypoint metadata parsing
# ---------------------------------------------------------------------------


class TestEntrypointMetadata:
    def test_load_config_from_metadata_valid(self):
        from unity.conversation_manager.medium_scripts.call import (
            _load_config_from_metadata,
        )

        ctx = MagicMock()
        ctx.job.metadata = json.dumps(
            {
                "voice_provider": "elevenlabs",
                "voice_id": "abc123",
                "channel": "phone",
                "outbound": False,
                "contact": {"first_name": "Alice"},
                "boss": {"first_name": "Boss"},
                "assistant_bio": "A helpful assistant",
                "ipc_socket_path": "/tmp/sock.sock",
            },
        )
        result = _load_config_from_metadata(ctx)
        assert result is not None
        assert result["voice_provider"] == "elevenlabs"
        assert result["contact"]["first_name"] == "Alice"
        assert result["ipc_socket_path"] == "/tmp/sock.sock"

    def test_load_config_from_metadata_empty(self):
        from unity.conversation_manager.medium_scripts.call import (
            _load_config_from_metadata,
        )

        ctx = MagicMock()
        ctx.job.metadata = ""
        assert _load_config_from_metadata(ctx) is None

    def test_load_config_from_metadata_invalid_json(self):
        from unity.conversation_manager.medium_scripts.call import (
            _load_config_from_metadata,
        )

        ctx = MagicMock()
        ctx.job.metadata = "not json"
        assert _load_config_from_metadata(ctx) is None


# ---------------------------------------------------------------------------
# IPC socket initialisation from metadata
# ---------------------------------------------------------------------------


class TestIPCSocketInit:
    def test_init_socket_for_job_sets_env_and_singleton(self):
        from unity.conversation_manager.domains.ipc_socket import (
            CM_EVENT_SOCKET_ENV,
            init_socket_for_job,
        )

        client = init_socket_for_job("/tmp/test_job.sock")
        assert client is not None
        assert os.environ[CM_EVENT_SOCKET_ENV] == "/tmp/test_job.sock"
        assert client._socket_path == "/tmp/test_job.sock"

        del os.environ[CM_EVENT_SOCKET_ENV]
