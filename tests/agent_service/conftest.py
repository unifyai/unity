"""Fixtures for agent-service integration tests.

Starts the real agent-service (Node.js / ts-node) as a subprocess and
provides its base URL to tests.  Skips automatically when prerequisites
are missing (no Node.js, no ``node_modules``, magnitude-core not built).
"""

import os

os.environ["SKIP_UNITY_TEST_INIT"] = "1"

import shutil
import socket
import subprocess
import time

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_AGENT_SERVICE_DIR = os.path.join(_REPO_ROOT, "agent-service")
_MAGNITUDE_CORE_DIST = os.path.join(
    _REPO_ROOT,
    "magnitude",
    "packages",
    "magnitude-core",
    "dist",
    "index.cjs",
)
_API_KEY = os.environ.get("UNIFY_KEY", "")
_REAL_HOME = os.path.expanduser("~")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _check_prerequisites():
    """Return a skip reason string, or *None* if everything looks good."""
    if shutil.which("node") is None:
        return "Node.js is not installed"
    if not os.path.isdir(os.path.join(_AGENT_SERVICE_DIR, "node_modules")):
        return "agent-service/node_modules missing (run: npm install)"
    if not os.path.isfile(_MAGNITUDE_CORE_DIST):
        return "magnitude-core not built (run: npm run build in magnitude/packages/magnitude-core)"
    return None


@pytest.fixture(scope="module")
def agent_service_url():
    """Start the real agent-service and yield ``http://localhost:<port>``."""
    reason = _check_prerequisites()
    if reason:
        pytest.skip(reason)
    if not _API_KEY:
        pytest.skip("UNIFY_KEY not set")

    port = _find_free_port()
    env = {
        **os.environ,
        "PORT": str(port),
        "HOME": _REAL_HOME,
    }

    proc = subprocess.Popen(
        ["npx", "ts-node", "src/index.ts"],
        cwd=_AGENT_SERVICE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    url = f"http://localhost:{port}"
    deadline = time.monotonic() + 30
    ready = False

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read().decode(errors="replace")
            pytest.fail(
                f"agent-service exited with code {proc.returncode}:\n{output}",
            )
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                ready = True
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)

    if not ready:
        proc.kill()
        output = proc.stdout.read().decode(errors="replace")
        pytest.fail(f"agent-service did not become ready within 30s:\n{output}")

    yield url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def auth_headers():
    """Authorization headers for agent-service requests."""
    return {"authorization": f"Bearer {_API_KEY}"}
