"""
Lazy startup for demo site processes.

When an assistant's ActorConfig contains url_mappings that point to
localhost URLs, this module ensures the corresponding demo site processes
are running before the browser session tries to reach them.

Called from managers_utils.py at actor initialization time.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEMO_SITES_DIR = Path(__file__).resolve().parents[2] / "demo-sites"

_RUNNING: dict[int, subprocess.Popen] = {}


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def _start_site(site_dir: Path, port: int) -> subprocess.Popen | None:
    if (site_dir / "app.py").exists():
        logger.info("Starting FastAPI demo site: %s on port %d", site_dir.name, port)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
            ],
            cwd=str(site_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc

    if (site_dir / "index.html").exists():
        logger.info("Starting static demo site: %s on port %d", site_dir.name, port)
        proc = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port)],
            cwd=str(site_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc

    logger.warning("Demo site directory %s has no app.py or index.html", site_dir)
    return None


def _wait_for_port(port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_port_open(port):
            return True
        time.sleep(0.2)
    return False


def ensure_demo_sites_running(url_mappings: dict[str, str]) -> None:
    """Start demo site processes for any localhost replacement URLs.

    Scans the url_mappings values for localhost URLs, extracts the port,
    and if nothing is listening on that port, attempts to find and start
    the corresponding demo site from the demo-sites/ directory.
    """
    if not _DEMO_SITES_DIR.is_dir():
        logger.debug("No demo-sites directory found at %s", _DEMO_SITES_DIR)
        return

    site_dirs = {d.name: d for d in _DEMO_SITES_DIR.iterdir() if d.is_dir()}

    for original_url, replacement_url in url_mappings.items():
        parsed = urlparse(replacement_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port

        if not port:
            continue
        if host not in ("localhost", "127.0.0.1", "0.0.0.0"):
            continue

        if _is_port_open(port, "127.0.0.1"):
            logger.debug("Port %d already open, skipping", port)
            continue

        if port in _RUNNING:
            proc = _RUNNING[port]
            if proc.poll() is None:
                logger.debug(
                    "Process for port %d still running (pid %d)",
                    port,
                    proc.pid,
                )
                continue
            logger.warning("Process for port %d has exited, restarting", port)

        matched_dir = None
        for name, d in site_dirs.items():
            if name == "example" and port == 4001:
                matched_dir = d
                break
            if name == "vantage-portal" and port == 4002:
                matched_dir = d
                break

        if not matched_dir:
            for name, d in site_dirs.items():
                if (d / "app.py").exists() or (d / "index.html").exists():
                    if not any(p == port for p in _RUNNING):
                        matched_dir = d
                        break

        if not matched_dir:
            logger.warning(
                "No demo site found for port %d (mapping %s -> %s)",
                port,
                original_url,
                replacement_url,
            )
            continue

        proc = _start_site(matched_dir, port)
        if proc is None:
            continue

        _RUNNING[port] = proc

        if _wait_for_port(port):
            logger.info(
                "Demo site %s ready on port %d (pid %d)",
                matched_dir.name,
                port,
                proc.pid,
            )
        else:
            stderr_output = ""
            if proc.stderr:
                try:
                    stderr_output = proc.stderr.read1(4096).decode(errors="replace")
                except Exception:
                    pass
            if proc.poll() is not None:
                stderr_output += f" (exit code: {proc.returncode})"
            logger.error(
                "Demo site %s failed to start on port %d within timeout. stderr: %s",
                matched_dir.name,
                port,
                stderr_output or "(no output)",
            )
