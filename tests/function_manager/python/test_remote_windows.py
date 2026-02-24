"""
Tests for remote Windows execution in FunctionManager.

Tests the complete remote execution flow including:
- Routing logic (when to execute remotely)
- FileSync integration (sync before/after execution)
- Venv preparation (uv sync)
- Script execution with arguments
- Result capture and error handling
"""

from __future__ import annotations

import base64
import json
import pytest
from typing import Any, Dict, List

from unity.function_manager.function_manager import FunctionManager
from unity.common.context_registry import ContextRegistry
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Sample Functions
# ────────────────────────────────────────────────────────────────────────────

SIMPLE_WINDOWS_FUNC = """
def process_data(input_path: str) -> dict:
    \"\"\"Process data from input path.\"\"\"
    return {"processed": True, "path": input_path}
""".strip()

MINIMAL_VENV_CONTENT = """
[project]
name = "test-venv"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def function_manager_factory():
    """Factory fixture that creates FunctionManager instances."""
    managers = []

    def _create():
        ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
        ContextRegistry.forget(FunctionManager, "Functions/Compositional")
        ContextRegistry.forget(FunctionManager, "Functions/Primitives")
        ContextRegistry.forget(FunctionManager, "Functions/Meta")
        fm = FunctionManager()
        managers.append(fm)
        return fm

    yield _create

    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


@pytest.fixture
def mock_session_details_windows(monkeypatch):
    """Configure SESSION_DETAILS for Windows VM execution."""
    from unity.session_details import SESSION_DETAILS

    monkeypatch.setattr(SESSION_DETAILS.assistant, "desktop_mode", "windows")
    monkeypatch.setattr(SESSION_DETAILS.assistant, "id", "test-assistant")
    monkeypatch.setattr(
        SESSION_DETAILS.assistant,
        "desktop_url",
        "https://test-vm.unify.ai",
    )
    monkeypatch.setattr(SESSION_DETAILS, "unify_key", "test-api-key")

    yield SESSION_DETAILS


@pytest.fixture
def mock_session_details_ubuntu(monkeypatch):
    """Configure SESSION_DETAILS for Ubuntu VM execution (managed VM, not Windows)."""
    from unity.session_details import SESSION_DETAILS

    monkeypatch.setattr(SESSION_DETAILS.assistant, "desktop_mode", "ubuntu")
    monkeypatch.setattr(SESSION_DETAILS.assistant, "id", "test-assistant-ubuntu")
    monkeypatch.setattr(
        SESSION_DETAILS.assistant,
        "desktop_url",
        "https://test-ubuntu-vm.unify.ai",
    )

    yield SESSION_DETAILS


class MockResponse:
    """Mock aiohttp response."""

    def __init__(self, payload: Any, status: int = 200):
        self._payload = payload
        self.status = status
        self.ok = status < 400

    async def json(self):
        return self._payload

    async def text(self):
        return (
            json.dumps(self._payload)
            if isinstance(self._payload, dict)
            else str(self._payload)
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockClientSession:
    """Mock aiohttp.ClientSession that records requests."""

    def __init__(self):
        self.requests: List[Dict[str, Any]] = []
        self._responses: Dict[str, MockResponse] = {}

    def set_response(self, url_pattern: str, response: MockResponse):
        """Set a response for a URL pattern."""
        self._responses[url_pattern] = response

    def _find_response(self, url: str, method: str, kwargs: Dict) -> MockResponse:
        """Find matching response or return default."""
        # Check for exact match first
        for pattern, response in self._responses.items():
            if pattern in url:
                return response

        # Default responses based on endpoint
        if "/api/files" in url:
            action = kwargs.get("json", {}).get("action", "save")
            if action == "save":
                return MockResponse({"status": "saved", "files": ["test.txt"]})
            elif action == "read":
                return MockResponse(
                    {
                        "content": base64.b64encode(b"test content").decode(),
                        "encoding": "base64",
                        "filename": "test.txt",
                    },
                )
            elif action == "list":
                return MockResponse({"files": [], "path": "."})
            elif action == "delete":
                return MockResponse({"status": "deleted", "files": []})
        elif "/api/exec" in url:
            return MockResponse(
                {
                    "exitCode": 0,
                    "stdout": "__EXECUTION_COMPLETE__",
                    "stderr": "",
                    "duration": 100,
                },
            )

        return MockResponse({})

    def post(self, url: str, **kwargs):
        """Record POST request and return mock response (sync - returns context manager)."""
        self.requests.append({"method": "POST", "url": url, **kwargs})
        return self._find_response(url, "POST", kwargs)

    def get(self, url: str, **kwargs):
        """Record GET request and return mock response (sync - returns context manager)."""
        self.requests.append({"method": "GET", "url": url, **kwargs})
        return self._find_response(url, "GET", kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_aiohttp_session(monkeypatch):
    """Mock aiohttp.ClientSession for HTTP request testing."""
    mock_session = MockClientSession()

    def patched_client_session(*args, **kwargs):
        return mock_session

    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", patched_client_session)

    yield mock_session


@pytest.fixture
def skip_vm_wait(monkeypatch):
    """Skip the VM readiness wait by returning desktop_url directly."""
    from unity.function_manager.function_manager import FunctionManager

    async def mock_wait_for_vm_ready(self):
        from unity.session_details import SESSION_DETAILS

        return SESSION_DETAILS.assistant.desktop_url

    monkeypatch.setattr(
        FunctionManager,
        "_wait_for_remote_windows_vm_ready",
        mock_wait_for_vm_ready,
    )


# ────────────────────────────────────────────────────────────────────────────
# 1. Routing Logic Tests
# ────────────────────────────────────────────────────────────────────────────


class TestRemoteWindowsRoutingLogic:
    """Tests for _should_execute_python_function_on_remote_windows."""

    @_handle_project
    def test_returns_false_when_windows_os_required_is_false(
        self,
        function_manager_factory,
        mock_session_details_windows,
    ):
        """Standard functions don't route to Windows."""
        fm = function_manager_factory()
        func_data = {"windows_os_required": False}
        assert fm._should_execute_python_function_on_remote_windows(func_data) is False

    @_handle_project
    def test_returns_false_when_windows_os_required_missing(
        self,
        function_manager_factory,
        mock_session_details_windows,
    ):
        """Missing windows_os_required defaults to False."""
        fm = function_manager_factory()
        func_data = {}
        assert fm._should_execute_python_function_on_remote_windows(func_data) is False

    @_handle_project
    def test_returns_false_when_desktop_mode_is_ubuntu(
        self,
        function_manager_factory,
        mock_session_details_ubuntu,
    ):
        """Ubuntu VM mode stays local - windows_os_required functions execute on the Ubuntu VM."""
        fm = function_manager_factory()
        func_data = {"windows_os_required": True}
        assert fm._should_execute_python_function_on_remote_windows(func_data) is False

    @_handle_project
    def test_returns_true_when_all_conditions_met(
        self,
        function_manager_factory,
        mock_session_details_windows,
    ):
        """All conditions met → remote execution."""
        fm = function_manager_factory()
        func_data = {"windows_os_required": True}
        assert fm._should_execute_python_function_on_remote_windows(func_data) is True


# ────────────────────────────────────────────────────────────────────────────
# 2. Wait Time Calculation Tests
# ────────────────────────────────────────────────────────────────────────────


class TestCalculateWaitTime:
    """Tests for _calculate_wait_time_from_vm_ready_at."""

    @_handle_project
    def test_returns_default_when_vm_ready_at_is_none(self, function_manager_factory):
        """None timestamp returns default wait time."""
        fm = function_manager_factory()
        result = fm._calculate_wait_time_from_vm_ready_at(None)
        assert result == 10

    @_handle_project
    def test_returns_default_on_invalid_timestamp(self, function_manager_factory):
        """Invalid timestamp returns default wait time."""
        fm = function_manager_factory()
        result = fm._calculate_wait_time_from_vm_ready_at("not-a-timestamp")
        assert result == 10

    @_handle_project
    def test_returns_minimum_five_for_past_timestamp(self, function_manager_factory):
        """Past timestamps return minimum 5 seconds."""
        fm = function_manager_factory()
        # A timestamp far in the past
        result = fm._calculate_wait_time_from_vm_ready_at("2020-01-01T00:00:00Z")
        assert result == 5

    @_handle_project
    def test_calculates_future_delta(self, function_manager_factory):
        """Future timestamp calculates correct delta."""
        from datetime import datetime, timezone, timedelta

        fm = function_manager_factory()
        # Timestamp 30 seconds in the future
        future = datetime.now(timezone.utc) + timedelta(seconds=30)
        future_str = future.isoformat()

        result = fm._calculate_wait_time_from_vm_ready_at(future_str)
        # Should be approximately 30 (allow some tolerance for execution time)
        assert 25 <= result <= 35


# ────────────────────────────────────────────────────────────────────────────
# 3. Venv Preparation Tests
# ────────────────────────────────────────────────────────────────────────────


class TestPrepareVenvOnRemoteWindows:
    """Tests for _prepare_venv_on_remote_windows."""

    @_handle_project
    @pytest.mark.asyncio
    async def test_raises_on_venv_not_found(
        self,
        function_manager_factory,
        mock_session_details_windows,
    ):
        """ValueError when venv_id doesn't exist."""
        fm = function_manager_factory()

        with pytest.raises(ValueError, match="not found"):
            await fm._prepare_venv_on_remote_windows(
                desktop_url="https://test-vm.unify.ai",
                venv_id=99999,  # Non-existent
            )

    @_handle_project
    @pytest.mark.asyncio
    async def test_writes_pyproject_and_runs_sync(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
    ):
        """Venv preparation writes pyproject.toml and runs uv sync."""
        fm = function_manager_factory()

        # Create a venv first
        venv_id = fm.add_venv(venv=MINIMAL_VENV_CONTENT)

        try:
            result = await fm._prepare_venv_on_remote_windows(
                desktop_url="https://test-vm.unify.ai",
                venv_id=venv_id,
            )

            # Should return python path with new Local\venvs path
            assert "Local\\venvs\\venv_" in result
            assert ".venv\\Scripts\\python.exe" in result

            # Should have made requests for:
            # 1. Write pyproject.toml (/api/files)
            # 2. pip install uv (/api/exec)
            # 3. uv sync (/api/exec)
            files_requests = [
                r for r in mock_aiohttp_session.requests if "/api/files" in r["url"]
            ]
            exec_requests = [
                r for r in mock_aiohttp_session.requests if "/api/exec" in r["url"]
            ]

            assert len(files_requests) >= 1  # At least pyproject.toml write
            assert len(exec_requests) >= 2  # pip install uv + uv sync
        finally:
            # Cleanup
            try:
                fm.delete_venv(venv_id=venv_id)
            except Exception:
                pass


# ────────────────────────────────────────────────────────────────────────────
# 4. Full Execution Flow Tests
# ────────────────────────────────────────────────────────────────────────────


class TestExecutePythonFunctionOnRemoteWindows:
    """End-to-end tests for _execute_python_function_on_remote_windows."""

    @_handle_project
    @pytest.mark.asyncio
    async def test_executes_simple_function(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
        skip_vm_wait,
    ):
        """Basic execution with result capture."""
        fm = function_manager_factory()

        # Set up response for result file read
        mock_aiohttp_session.set_response(
            "read",
            MockResponse(
                {
                    "content": base64.b64encode(
                        json.dumps({"result": 42, "error": None}).encode(),
                    ).decode(),
                    "encoding": "base64",
                },
            ),
        )

        func_data = {
            "name": "test_func",
            "windows_os_required": True,
        }

        result = await fm._execute_python_function_on_remote_windows(
            func_data=func_data,
            implementation=SIMPLE_WINDOWS_FUNC,
            call_kwargs={"input_path": "/test/path"},
        )

        # Should have made requests
        assert len(mock_aiohttp_session.requests) > 0

        # Check that script was written
        files_requests = [
            r
            for r in mock_aiohttp_session.requests
            if "/api/files" in r["url"] and r.get("json", {}).get("action") == "save"
        ]
        assert len(files_requests) >= 1

        # Check that exec was called
        exec_requests = [
            r for r in mock_aiohttp_session.requests if "/api/exec" in r["url"]
        ]
        assert len(exec_requests) >= 1

    @_handle_project
    @pytest.mark.asyncio
    async def test_uses_powershell_shell_mode(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
        skip_vm_wait,
    ):
        """Execution uses PowerShell shell mode."""
        fm = function_manager_factory()

        func_data = {
            "name": "test_func",
            "windows_os_required": True,
        }

        await fm._execute_python_function_on_remote_windows(
            func_data=func_data,
            implementation=SIMPLE_WINDOWS_FUNC,
            call_kwargs={"input_path": "/test/path"},
        )

        # Find exec request and check shell_mode
        exec_requests = [
            r for r in mock_aiohttp_session.requests if "/api/exec" in r["url"]
        ]
        assert len(exec_requests) >= 1

        # Check that shell_mode is powershell
        exec_json = exec_requests[-1].get("json", {})
        assert exec_json.get("shell_mode") == "powershell"


# ────────────────────────────────────────────────────────────────────────────
# 5. Integration Tests (execute_function routing)
# ────────────────────────────────────────────────────────────────────────────


class TestExecuteFunctionRemoteRouting:
    """Tests that execute_function correctly routes to remote execution."""

    @_handle_project
    @pytest.mark.asyncio
    async def test_routes_to_remote_when_conditions_met(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
        skip_vm_wait,
        monkeypatch,
    ):
        """execute_function dispatches to remote execution when conditions met."""
        fm = function_manager_factory()

        # Add a function
        fm.add_functions(
            implementations=SIMPLE_WINDOWS_FUNC,
            language="python",
        )

        # Mock the routing check to return True (simulating windows_os_required=True)
        monkeypatch.setattr(
            fm,
            "_should_execute_python_function_on_remote_windows",
            lambda func_data: True,
        )

        # Execute - should route to remote
        result = await fm.execute_function(
            function_name="process_data",
            call_kwargs={"input_path": "/test/path"},
        )

        # Should have made HTTP requests to the mock agent service
        assert len(mock_aiohttp_session.requests) > 0

    @_handle_project
    @pytest.mark.asyncio
    async def test_stays_local_when_windows_os_required_false(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
    ):
        """execute_function stays local when windows_os_required=False."""
        fm = function_manager_factory()

        # Add a function without windows_os_required
        fm.add_functions(
            implementations=SIMPLE_WINDOWS_FUNC,
            language="python",
        )

        # Execute - should NOT route to remote
        result = await fm.execute_function(
            function_name="process_data",
            call_kwargs={"input_path": "/test/path"},
        )

        # Should NOT have made HTTP requests to agent service
        # (local execution doesn't use aiohttp)
        agent_requests = [
            r for r in mock_aiohttp_session.requests if "test-vm.unify.ai" in r["url"]
        ]
        assert len(agent_requests) == 0

        # Should have a result (local execution worked)
        assert "result" in result or "error" in result


# ────────────────────────────────────────────────────────────────────────────
# 6. FileSync Integration Tests
# ────────────────────────────────────────────────────────────────────────────


class TestSyncIntegration:
    """Tests for FileSync integration with remote execution."""

    @_handle_project
    @pytest.mark.asyncio
    async def test_sync_called_before_and_after_execution(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
        skip_vm_wait,
        monkeypatch,
    ):
        """Verify sync_to_remote and sync_from_remote are called."""
        fm = function_manager_factory()
        sync_calls = []

        async def mock_sync_to_remote(self):
            sync_calls.append("sync_to_remote")
            return True

        async def mock_sync_from_remote(self):
            sync_calls.append("sync_from_remote")
            return True

        monkeypatch.setattr(
            FunctionManager,
            "_sync_to_remote",
            mock_sync_to_remote,
        )
        monkeypatch.setattr(
            FunctionManager,
            "_sync_from_remote",
            mock_sync_from_remote,
        )

        func_data = {"name": "test_func", "windows_os_required": True}

        await fm._execute_python_function_on_remote_windows(
            func_data=func_data,
            implementation=SIMPLE_WINDOWS_FUNC,
            call_kwargs={"input_path": "/Unity/data/file.txt"},
        )

        # Verify sync order
        assert sync_calls == ["sync_to_remote", "sync_from_remote"]

    @_handle_project
    @pytest.mark.asyncio
    async def test_execution_continues_without_sync_manager(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
        skip_vm_wait,
        monkeypatch,
    ):
        """Execution proceeds if no SyncManager available."""
        fm = function_manager_factory()

        # Ensure no sync manager
        monkeypatch.setattr(fm, "_get_sync_manager", lambda: None)

        func_data = {"name": "test_func", "windows_os_required": True}

        # Should not raise
        await fm._execute_python_function_on_remote_windows(
            func_data=func_data,
            implementation=SIMPLE_WINDOWS_FUNC,
            call_kwargs={"input_path": "/Unity/data/file.txt"},
        )

        # HTTP requests still made (script write + exec + result read)
        assert len(mock_aiohttp_session.requests) > 0

    @_handle_project
    def test_get_sync_manager_returns_none_without_file_manager(
        self,
        function_manager_factory,
    ):
        """_get_sync_manager returns None when no FileManager."""
        fm = function_manager_factory()
        fm._fm = None
        assert fm._get_sync_manager() is None
