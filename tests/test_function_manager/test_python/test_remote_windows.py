"""
Tests for remote Windows execution in FunctionManager.

Tests the complete remote execution flow including:
- Path conversion (Unix → Windows)
- Routing logic (when to execute remotely)
- Data upload/download (multipart)
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

ASYNC_WINDOWS_FUNC = """
async def async_process(x: int) -> int:
    \"\"\"Async function for testing.\"\"\"
    import asyncio
    await asyncio.sleep(0.01)
    return x * 2
""".strip()

WINDOWS_WITH_DATA = """
def extract_data(input_dir: str, output_dir: str) -> dict:
    \"\"\"Extract data with input/output directories.\"\"\"
    import os
    files = os.listdir(input_dir)
    return {"files_found": len(files), "output": output_dir}
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
    monkeypatch.setattr(SESSION_DETAILS.assistant, "is_user_desktop", False)
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
    """Configure SESSION_DETAILS for Ubuntu (non-Windows) mode."""
    from unity.session_details import SESSION_DETAILS

    monkeypatch.setattr(SESSION_DETAILS.assistant, "desktop_mode", "ubuntu")
    monkeypatch.setattr(SESSION_DETAILS.assistant, "is_user_desktop", False)

    yield SESSION_DETAILS


@pytest.fixture
def temp_data_files(tmp_path):
    """Create temporary test files for upload/download tests."""
    # Single file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    test_file = data_dir / "input.xlsx"
    test_file.write_bytes(b"test excel content")

    # Nested directory
    nested = tmp_path / "nested"
    subdir = nested / "subdir"
    subdir.mkdir(parents=True)
    (nested / "root_file.txt").write_text("root content")
    (subdir / "nested_file.txt").write_text("nested content")

    # Empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    yield {
        "file": test_file,
        "directory": nested,
        "empty_dir": empty_dir,
        "root": tmp_path,
    }


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

    async def post(self, url: str, **kwargs):
        """Record POST request and return mock response."""
        self.requests.append({"method": "POST", "url": url, **kwargs})
        return self._find_response(url, "POST", kwargs)

    async def get(self, url: str, **kwargs):
        """Record GET request and return mock response."""
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


# ────────────────────────────────────────────────────────────────────────────
# 1. Path Conversion Tests
# ────────────────────────────────────────────────────────────────────────────


class TestLocalToRemotePath:
    """Tests for _local_to_remote_path path conversion."""

    @_handle_project
    def test_converts_absolute_unix_path(self, function_manager_factory):
        """Convert absolute Unix path to Windows equivalent."""
        fm = function_manager_factory()
        result = fm._local_to_remote_path("/Users/julia/data/file.xlsx")
        assert result == "C:\\Unity\\Users\\julia\\data\\file.xlsx"

    @_handle_project
    def test_handles_nested_directories(self, function_manager_factory):
        """Deep paths are preserved correctly."""
        fm = function_manager_factory()
        result = fm._local_to_remote_path("/a/b/c/d/e/file.txt")
        assert result == "C:\\Unity\\a\\b\\c\\d\\e\\file.txt"

    @_handle_project
    def test_handles_single_component_path(self, function_manager_factory):
        """Single-level path conversion."""
        fm = function_manager_factory()
        result = fm._local_to_remote_path("/data")
        assert result == "C:\\Unity\\data"

    @_handle_project
    def test_handles_path_with_spaces(self, function_manager_factory):
        """Paths with spaces are preserved."""
        fm = function_manager_factory()
        result = fm._local_to_remote_path("/Users/My User/Documents/file name.xlsx")
        assert result == "C:\\Unity\\Users\\My User\\Documents\\file name.xlsx"

    @_handle_project
    def test_handles_tmp_path(self, function_manager_factory, tmp_path):
        """Real tmp_path conversion works correctly."""
        fm = function_manager_factory()
        local_path = str(tmp_path / "test" / "file.txt")
        result = fm._local_to_remote_path(local_path)

        # Should start with C:\Unity\ and contain the path components
        assert result.startswith("C:\\Unity\\")
        assert "test" in result
        assert "file.txt" in result


# ────────────────────────────────────────────────────────────────────────────
# 2. Routing Logic Tests
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
        """Ubuntu desktop mode stays local."""
        fm = function_manager_factory()
        func_data = {"windows_os_required": True}
        assert fm._should_execute_python_function_on_remote_windows(func_data) is False

    @_handle_project
    def test_returns_false_when_is_user_desktop_is_true(
        self,
        function_manager_factory,
        monkeypatch,
    ):
        """User's own desktop stays local (no managed VM)."""
        from unity.session_details import SESSION_DETAILS

        monkeypatch.setattr(SESSION_DETAILS.assistant, "desktop_mode", "windows")
        monkeypatch.setattr(SESSION_DETAILS.assistant, "is_user_desktop", True)

        fm = function_manager_factory()
        func_data = {"windows_os_required": True}
        assert fm._should_execute_python_function_on_remote_windows(func_data) is False

    @_handle_project
    def test_returns_true_when_all_conditions_met(
        self,
        function_manager_factory,
        mock_session_details_windows,
    ):
        """All three conditions met → remote execution."""
        fm = function_manager_factory()
        func_data = {"windows_os_required": True}
        assert fm._should_execute_python_function_on_remote_windows(func_data) is True


# ────────────────────────────────────────────────────────────────────────────
# 3. Wait Time Calculation Tests
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
# 4. Data Upload Tests
# ────────────────────────────────────────────────────────────────────────────


class TestUploadDataToRemote:
    """Tests for _upload_data_to_remote."""

    @_handle_project
    @pytest.mark.asyncio
    async def test_raises_on_nonexistent_path(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
    ):
        """FileNotFoundError for missing paths."""
        fm = function_manager_factory()

        with pytest.raises(FileNotFoundError, match="does not exist"):
            await fm._upload_data_to_remote(
                session=mock_aiohttp_session,
                desktop_url="https://test-vm.unify.ai",
                headers={"Authorization": "Bearer test"},
                local_path="/nonexistent/path/file.txt",
            )

    @_handle_project
    @pytest.mark.asyncio
    async def test_returns_correct_remote_path_for_file(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
        temp_data_files,
    ):
        """Return value is Windows path equivalent for file."""
        fm = function_manager_factory()

        result = await fm._upload_data_to_remote(
            session=mock_aiohttp_session,
            desktop_url="https://test-vm.unify.ai",
            headers={"Authorization": "Bearer test"},
            local_path=str(temp_data_files["file"]),
        )

        # Should be Windows path under C:\Unity
        assert result.startswith("C:\\Unity\\")
        assert "input.xlsx" in result

    @_handle_project
    @pytest.mark.asyncio
    async def test_returns_correct_remote_path_for_directory(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
        temp_data_files,
    ):
        """Return value is Windows path equivalent for directory."""
        fm = function_manager_factory()

        result = await fm._upload_data_to_remote(
            session=mock_aiohttp_session,
            desktop_url="https://test-vm.unify.ai",
            headers={"Authorization": "Bearer test"},
            local_path=str(temp_data_files["directory"]),
        )

        # Should be Windows path under C:\Unity
        assert result.startswith("C:\\Unity\\")
        assert "nested" in result

    @_handle_project
    @pytest.mark.asyncio
    async def test_handles_empty_directory(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
        temp_data_files,
    ):
        """Empty directory returns remote path without HTTP call."""
        fm = function_manager_factory()
        initial_request_count = len(mock_aiohttp_session.requests)

        result = await fm._upload_data_to_remote(
            session=mock_aiohttp_session,
            desktop_url="https://test-vm.unify.ai",
            headers={"Authorization": "Bearer test"},
            local_path=str(temp_data_files["empty_dir"]),
        )

        # Should return remote path
        assert result.startswith("C:\\Unity\\")
        # No HTTP requests should have been made (no files to upload)
        assert len(mock_aiohttp_session.requests) == initial_request_count

    @_handle_project
    @pytest.mark.asyncio
    async def test_uploads_via_multipart(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
        temp_data_files,
    ):
        """Files are uploaded via multipart form data."""
        fm = function_manager_factory()

        await fm._upload_data_to_remote(
            session=mock_aiohttp_session,
            desktop_url="https://test-vm.unify.ai",
            headers={"Authorization": "Bearer test"},
            local_path=str(temp_data_files["file"]),
        )

        # Should have made a POST request to /api/files
        assert len(mock_aiohttp_session.requests) >= 1
        request = mock_aiohttp_session.requests[-1]
        assert request["method"] == "POST"
        assert "/api/files" in request["url"]
        # Should have form data (multipart)
        assert "data" in request


# ────────────────────────────────────────────────────────────────────────────
# 5. Data Download Tests
# ────────────────────────────────────────────────────────────────────────────


class TestDownloadDataFromRemote:
    """Tests for _download_data_from_remote."""

    @_handle_project
    @pytest.mark.asyncio
    async def test_handles_empty_listing(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
        tmp_path,
    ):
        """No error when remote directory is empty."""
        fm = function_manager_factory()

        # Set up response for empty directory listing
        mock_aiohttp_session.set_response(
            "/api/files",
            MockResponse({"files": [], "path": "test"}),
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Should not raise
        await fm._download_data_from_remote(
            session=mock_aiohttp_session,
            desktop_url="https://test-vm.unify.ai",
            headers={"Authorization": "Bearer test"},
            local_path=str(output_dir),
        )

    @_handle_project
    @pytest.mark.asyncio
    async def test_handles_list_failure(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
        tmp_path,
    ):
        """No error when listing fails (e.g., path doesn't exist on remote)."""
        fm = function_manager_factory()

        # Set up failed response for directory listing
        mock_aiohttp_session.set_response(
            "/api/files",
            MockResponse({"error": "not found"}, status=404),
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Should not raise, just log and return
        await fm._download_data_from_remote(
            session=mock_aiohttp_session,
            desktop_url="https://test-vm.unify.ai",
            headers={"Authorization": "Bearer test"},
            local_path=str(output_dir),
        )


# ────────────────────────────────────────────────────────────────────────────
# 6. Venv Preparation Tests
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

            # Should return python path
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
# 7. Full Execution Flow Tests
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

    @_handle_project
    @pytest.mark.asyncio
    async def test_uses_user_session(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
    ):
        """Execution uses user_session=True for COM automation."""
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

        # Find exec request and check user_session
        exec_requests = [
            r for r in mock_aiohttp_session.requests if "/api/exec" in r["url"]
        ]
        assert len(exec_requests) >= 1

        # Check that user_session is True
        exec_json = exec_requests[-1].get("json", {})
        assert exec_json.get("user_session") is True

    @_handle_project
    @pytest.mark.asyncio
    async def test_handles_data_required(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
        temp_data_files,
    ):
        """data_required paths are uploaded before execution."""
        fm = function_manager_factory()

        func_data = {
            "name": "test_func",
            "windows_os_required": True,
            "data_required": ["input_dir"],
        }

        result = await fm._execute_python_function_on_remote_windows(
            func_data=func_data,
            implementation=WINDOWS_WITH_DATA,
            call_kwargs={
                "input_dir": str(temp_data_files["directory"]),
                "output_dir": "/tmp/output",
            },
        )

        # Should have made upload request before exec
        requests = mock_aiohttp_session.requests

        # Find the upload request (multipart to /api/files)
        upload_requests = [
            r for r in requests if "/api/files" in r["url"] and "data" in r
        ]

        # Should have uploaded the directory
        assert len(upload_requests) >= 1

    @_handle_project
    @pytest.mark.asyncio
    async def test_handles_data_output(
        self,
        function_manager_factory,
        mock_session_details_windows,
        mock_aiohttp_session,
        temp_data_files,
        tmp_path,
    ):
        """data_output paths are downloaded after execution."""
        fm = function_manager_factory()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        func_data = {
            "name": "test_func",
            "windows_os_required": True,
            "data_output": ["output_dir"],
        }

        await fm._execute_python_function_on_remote_windows(
            func_data=func_data,
            implementation=WINDOWS_WITH_DATA,
            call_kwargs={
                "input_dir": "/remote/input",
                "output_dir": str(output_dir),
            },
        )

        # Should have made download requests (list + read for each file)
        list_requests = [
            r
            for r in mock_aiohttp_session.requests
            if "/api/files" in r["url"] and r.get("json", {}).get("action") == "list"
        ]

        # At least one list request for the output directory
        assert len(list_requests) >= 1


# ────────────────────────────────────────────────────────────────────────────
# 8. Integration Tests (execute_function routing)
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
    ):
        """execute_function dispatches to remote execution when conditions met."""
        fm = function_manager_factory()

        # Add a function with windows_os_required=True
        fm.add_functions(
            implementations=SIMPLE_WINDOWS_FUNC,
            language="python",
        )

        # Update the function to have windows_os_required=True
        # (add_functions doesn't support this directly, so we update it)
        funcs = fm.filter_functions(filter="name == 'process_data'")
        if funcs:
            func = funcs[0]
            fm._compositional_ctx.update(
                int(func["function_id"]),
                {"windows_os_required": True},
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
