"""Fixtures for URL mapping end-to-end tests.

Starts the example demo site on a random port and the real agent-service
(Node.js) as a subprocess. Both are module-scoped so they persist across
all tests in this directory.
"""

import os
import shutil
import socket
import subprocess
import threading
import time
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEMO_SITES_DIR = _REPO_ROOT / "agent-service" / "demo-sites" / "example"
_AGENT_SERVICE_DIR = _REPO_ROOT / "agent-service"
_MAGNITUDE_CORE_DIST = (
    _REPO_ROOT / "magnitude" / "packages" / "magnitude-core" / "dist" / "index.cjs"
)
_API_KEY = os.environ.get("UNIFY_KEY", "")
_REAL_HOME = os.path.expanduser("~")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _check_prerequisites():
    if shutil.which("node") is None:
        return "Node.js is not installed"
    if not (_AGENT_SERVICE_DIR / "node_modules").is_dir():
        return "agent-service/node_modules missing (run: npm install)"
    if not _MAGNITUDE_CORE_DIST.is_file():
        return "magnitude-core not built (run: npm run build in magnitude/packages/magnitude-core)"
    return None


@pytest.fixture(scope="module")
def demo_site_url():
    """Serve the example demo site on a random port."""
    port = _find_free_port()
    handler = partial(SimpleHTTPRequestHandler, directory=str(_DEMO_SITES_DIR))
    server = HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)

    yield url
    server.shutdown()


@pytest.fixture(scope="module")
def agent_service_url():
    """Start the real agent-service and yield its base URL."""
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
        cwd=str(_AGENT_SERVICE_DIR),
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
