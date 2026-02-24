"""
ConversationManager sandbox support for the Docker-based virtual desktop.

When `agent_mode == "desktop"`, the sandbox needs a full Linux virtual desktop
(TigerVNC + Fluxbox + noVNC) plus the Magnitude agent-service, all bundled
inside a Docker container built from ``desktop/Dockerfile``.

This module mirrors the pattern in ``agent_service_bootstrap.py``:
- Structured result types with progress callbacks
- Two-tier bootstrap: direct start (image already built) and full auto-bootstrap
- Best-effort cleanup on sandbox exit

Design notes
------------
- These helpers are intentionally conservative and UI-agnostic.
- They return structured results; callers decide how to display progress and errors.
- Container spawning is best-effort and should never be the only supported path.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

ProgressCallback = Callable[[str], None]

DESKTOP_IMAGE_TAG = "unity-desktop"
DESKTOP_CONTAINER_NAME = "unity-desktop-sandbox"


def _desktop_novnc_url() -> str:
    """Return the noVNC viewer URL with the VNC password embedded."""
    key = os.environ.get("UNIFY_KEY", "")
    if key:
        # Query params are parsed with URLSearchParams in custom.html; '+' must be
        # percent-encoded as %2B (otherwise it is decoded as a space).
        encoded_key = quote(key, safe="")
        return f"http://localhost:6080/custom.html?password={encoded_key}"
    return "http://localhost:6080/custom.html"


@dataclass(frozen=True)
class DesktopBootstrapResult:
    ok: bool
    summary: str
    container_id: Optional[str] = None


def _docker_available() -> bool:
    """Return True if the Docker CLI is installed and the daemon is reachable."""
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=True,
        )
        return True
    except Exception:
        return False


def _desktop_image_exists(tag: str = DESKTOP_IMAGE_TAG) -> bool:
    """Return True if the desktop Docker image is already built."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _check_magnitude_packages(repo_root: Path) -> Optional[str]:
    """
    Return an error message if the magnitude packages required by agent-service
    are missing or unbuilt. Returns None if everything is present.
    """
    magnitude_dir = repo_root / "magnitude"
    if not magnitude_dir.exists():
        return (
            "Missing `magnitude/` directory. This is Unity's Magnitude fork, required "
            "because agent-service depends on local magnitude-core and magnitude-extract.\n\n"
            "Clone it into the repo root:\n"
            "  git clone <magnitude-repo-url> magnitude"
        )
    for pkg in ("magnitude-core", "magnitude-extract"):
        pkg_dir = magnitude_dir / "packages" / pkg
        if not pkg_dir.exists():
            return f"Missing magnitude package directory: {pkg_dir}"
        dist_dir = pkg_dir / "dist"
        if not dist_dir.exists() or not any(dist_dir.iterdir()):
            return (
                f"Magnitude package '{pkg}' has not been built (missing dist/).\n\n"
                "Build it with:\n"
                f"  cd magnitude/packages/{pkg} && npm ci && npm run build"
            )
    return None


def _build_desktop_image(
    *,
    repo_root: Path,
    tag: str = DESKTOP_IMAGE_TAG,
    progress: Optional[ProgressCallback] = None,
) -> bool:
    """Build the desktop Docker image. Returns True on success."""
    progress = progress or (lambda _m: None)
    dockerfile = repo_root / "desktop" / "Dockerfile"
    if not dockerfile.exists():
        progress(f"[desktop] Missing Dockerfile at {dockerfile}")
        return False

    # The Dockerfile COPYs magnitude packages into the image; verify they exist.
    mag_err = _check_magnitude_packages(repo_root)
    if mag_err:
        progress(f"[desktop] {mag_err}")
        return False

    progress(
        f"[desktop] Building Docker image '{tag}' (this may take a few minutes on first run)...",
    )
    try:
        subprocess.run(
            ["docker", "build", "-t", tag, "-f", str(dockerfile), "."],
            cwd=str(repo_root),
            check=True,
            timeout=600,
        )
        progress(f"[desktop] Image '{tag}' built successfully")
        return True
    except subprocess.CalledProcessError as exc:
        progress(
            f"[desktop] Docker build failed (exit {exc.returncode})",
        )
        return False
    except subprocess.TimeoutExpired:
        progress("[desktop] Docker build timed out (10 min limit)")
        return False


def _find_running_container(name: str = DESKTOP_CONTAINER_NAME) -> Optional[str]:
    """Return the container ID if a container with the given name is running."""
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.Id}}",
                name,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        container_id = (result.stdout or "").strip()
        if not container_id:
            return None
        # Verify it's actually running.
        state_result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Running}}",
                name,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if (state_result.stdout or "").strip().lower() == "true":
            return container_id
        return None
    except Exception:
        return None


def stop_desktop_container(
    name: str = DESKTOP_CONTAINER_NAME,
    *,
    progress: Optional[ProgressCallback] = None,
) -> None:
    """Best-effort stop and remove the desktop container."""
    progress = progress or (lambda _m: None)
    try:
        subprocess.run(
            ["docker", "stop", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        progress(f"[desktop] Stopped container '{name}'")
    except Exception:
        pass
    try:
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        pass


def _validate_agent_service(
    *,
    agent_server_url: str,
    unify_key: str,
    timeout_s: float = 2.5,
) -> bool:
    """Return True iff the agent-service inside the container responds with HTTP 200."""
    try:
        import httpx
    except Exception:
        httpx = None  # type: ignore

    url = str(agent_server_url).rstrip("/") + "/sessions"
    headers = {"authorization": f"Bearer {unify_key}"}
    try:
        if httpx is None:
            from urllib.request import Request, urlopen

            req = Request(url, headers=headers)
            with urlopen(req, timeout=float(timeout_s)) as resp:  # nosec B310
                return int(getattr(resp, "status", 0) or 0) == 200
        resp = httpx.get(url, headers=headers, timeout=float(timeout_s))
        return int(resp.status_code) == 200
    except Exception:
        return False


def _wait_for_container_ready(
    *,
    container_name: str,
    agent_server_url: str,
    unify_key: str,
    timeout_s: float = 60.0,
    poll_interval_s: float = 1.0,
) -> DesktopBootstrapResult:
    """Poll the containerized agent-service until it responds or the container dies."""
    start_t = time.time()
    deadline = start_t + float(timeout_s)
    while time.time() < deadline:
        # Check if the container is still running.
        if _find_running_container(container_name) is None:
            return DesktopBootstrapResult(
                ok=False,
                summary="Desktop container exited during startup",
                container_id=None,
            )
        if _validate_agent_service(
            agent_server_url=agent_server_url,
            unify_key=unify_key,
        ):
            container_id = _find_running_container(container_name)
            return DesktopBootstrapResult(
                ok=True,
                summary=f"Desktop container ready — view at {_desktop_novnc_url()}",
                container_id=container_id,
            )
        time.sleep(float(poll_interval_s))
    return DesktopBootstrapResult(
        ok=False,
        summary=f"Desktop container did not become ready within {timeout_s:.0f}s",
        container_id=_find_running_container(container_name),
    )


def _remap_localhost_env_overrides(env_file: Path) -> list[str]:
    """
    Parse the .env file and return ``-e KEY=VALUE`` args for any env vars whose
    values reference ``localhost``, rewritten to use ``host.docker.internal``.

    Inside Docker on macOS/Windows, ``localhost`` resolves to the container's own
    loopback — not the host machine. Docker's ``host.docker.internal`` hostname
    provides the correct route to host-side services (e.g. Orchestra).
    """
    overrides: list[str] = []
    try:
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            # Strip surrounding quotes that Python-style .env files may use.
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            if "localhost" in value:
                remapped = value.replace("localhost", "host.docker.internal")
                overrides.extend(["-e", f"{key}={remapped}"])
    except Exception:
        pass
    return overrides


def _start_container(
    *,
    repo_root: Path,
    tag: str = DESKTOP_IMAGE_TAG,
    name: str = DESKTOP_CONTAINER_NAME,
    progress: Optional[ProgressCallback] = None,
) -> Optional[str]:
    """Run the desktop container in detached mode. Returns container ID or None."""
    progress = progress or (lambda _m: None)

    env_file = repo_root / ".env"
    if not env_file.exists():
        progress(
            "[desktop] Warning: no .env file found; container may lack required keys",
        )

    cmd = [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        name,
        "-p",
        "6080:6080",
        "-p",
        "5900:5900",
        "-p",
        "3000:3000",
        # Allow the container to reach host-side services (e.g. Orchestra).
        "--add-host=host.docker.internal:host-gateway",
    ]
    if env_file.exists():
        cmd.extend(["--env-file", str(env_file)])
        # Override any env vars that reference localhost so they route to the host.
        cmd.extend(_remap_localhost_env_overrides(env_file))
    cmd.append(tag)

    progress(f"[desktop] Starting container '{name}' from image '{tag}'...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(repo_root),
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            progress(f"[desktop] docker run failed: {stderr}")
            return None
        container_id = (result.stdout or "").strip()
        progress(f"[desktop] Container started (ID: {container_id[:12]})")
        return container_id
    except Exception as exc:
        progress(f"[desktop] Failed to start container: {exc}")
        return None


def try_start_desktop_direct(
    *,
    repo_root: Path,
    agent_server_url: str,
    progress: Optional[ProgressCallback] = None,
    timeout_s: float = 60.0,
) -> DesktopBootstrapResult:
    """
    Start the desktop container assuming the Docker image is already built.

    If a container with the expected name is already running and healthy,
    reuse it. Otherwise, start a fresh one.
    """
    progress = progress or (lambda _m: None)
    unify_key = os.environ.get("UNIFY_KEY") or ""
    if not unify_key:
        return DesktopBootstrapResult(
            ok=False,
            summary="UNIFY_KEY is not set (required for agent-service auth)",
        )

    # Already running and healthy?
    existing = _find_running_container(DESKTOP_CONTAINER_NAME)
    if existing and _validate_agent_service(
        agent_server_url=agent_server_url,
        unify_key=unify_key,
    ):
        progress(
            f"[desktop] Reusing existing desktop container — view at {_desktop_novnc_url()}",
        )
        return DesktopBootstrapResult(
            ok=True,
            summary=f"Desktop container already running — view at {_desktop_novnc_url()}",
            container_id=existing,
        )

    # Stop any stale container with the same name.
    if existing:
        progress("[desktop] Stopping stale container...")
        stop_desktop_container(DESKTOP_CONTAINER_NAME, progress=progress)

    if not _docker_available():
        return DesktopBootstrapResult(
            ok=False,
            summary=(
                "Docker is not available. Desktop mode requires Docker to run the "
                "virtual desktop environment. Install Docker or use --agent-mode web"
            ),
        )

    if not _desktop_image_exists():
        return DesktopBootstrapResult(
            ok=False,
            summary=(
                f"Docker image '{DESKTOP_IMAGE_TAG}' not found. "
                "Run with --agent-service-bootstrap auto to build it automatically, "
                f"or build manually: docker build -t {DESKTOP_IMAGE_TAG} -f desktop/Dockerfile ."
            ),
        )

    container_id = _start_container(
        repo_root=repo_root,
        progress=progress,
    )
    if not container_id:
        return DesktopBootstrapResult(
            ok=False,
            summary="Failed to start desktop container",
        )

    return _wait_for_container_ready(
        container_name=DESKTOP_CONTAINER_NAME,
        agent_server_url=agent_server_url,
        unify_key=unify_key,
        timeout_s=timeout_s,
    )


def try_auto_bootstrap_desktop(
    *,
    repo_root: Path,
    agent_server_url: str,
    progress: Optional[ProgressCallback] = None,
    timeout_s: float = 90.0,
) -> DesktopBootstrapResult:
    """
    Full bootstrap: check Docker, build image if needed, start container.

    This is the "auto" tier that handles everything from scratch.
    """
    progress = progress or (lambda _m: None)
    unify_key = os.environ.get("UNIFY_KEY") or ""
    if not unify_key:
        return DesktopBootstrapResult(
            ok=False,
            summary="UNIFY_KEY is not set (required for agent-service auth)",
        )

    # Already running and healthy?
    existing = _find_running_container(DESKTOP_CONTAINER_NAME)
    if existing and _validate_agent_service(
        agent_server_url=agent_server_url,
        unify_key=unify_key,
    ):
        progress(
            f"[desktop] Reusing existing desktop container — view at {_desktop_novnc_url()}",
        )
        return DesktopBootstrapResult(
            ok=True,
            summary=f"Desktop container already running — view at {_desktop_novnc_url()}",
            container_id=existing,
        )

    # Stop any stale container.
    if existing:
        progress("[desktop] Stopping stale container...")
        stop_desktop_container(DESKTOP_CONTAINER_NAME, progress=progress)

    if not _docker_available():
        return DesktopBootstrapResult(
            ok=False,
            summary=(
                "Docker is not available (either not installed or daemon not running). "
                "Desktop mode requires Docker to run the virtual desktop environment.\n\n"
                "Options:\n"
                "  1. Install and start Docker, then re-run the sandbox\n"
                "  2. Use --agent-mode web for browser-only mode (no Docker required)"
            ),
        )

    # Build the image if it doesn't exist.
    if not _desktop_image_exists():
        if not _build_desktop_image(repo_root=repo_root, progress=progress):
            return DesktopBootstrapResult(
                ok=False,
                summary="Failed to build desktop Docker image",
            )

    container_id = _start_container(
        repo_root=repo_root,
        progress=progress,
    )
    if not container_id:
        return DesktopBootstrapResult(
            ok=False,
            summary="Failed to start desktop container",
        )

    return _wait_for_container_ready(
        container_name=DESKTOP_CONTAINER_NAME,
        agent_server_url=agent_server_url,
        unify_key=unify_key,
        timeout_s=timeout_s,
    )


def diagnose_desktop_setup(
    *,
    repo_root: Path,
    agent_server_url: str,
) -> str:
    """Return user-facing help text for desktop mode setup issues."""
    lines: list[str] = []
    lines.append(
        "Desktop mode requires a Docker container running the virtual desktop environment.",
    )
    lines.append("")

    if not _docker_available():
        lines.append("Issue: Docker is not available.")
        lines.append("  - Install Docker: https://docs.docker.com/get-docker/")
        lines.append("  - Ensure the Docker daemon is running")
        lines.append("")
    else:
        lines.append("Docker: OK")
        lines.append("")

    mag_err = _check_magnitude_packages(repo_root)
    if mag_err:
        lines.append(f"Issue: {mag_err}")
        lines.append("")
    else:
        lines.append("Magnitude packages: OK")
        lines.append("")

    if not _desktop_image_exists():
        lines.append(f"Issue: Docker image '{DESKTOP_IMAGE_TAG}' not found.")
        lines.append("  Build it with:")
        lines.append(f"    docker build -t {DESKTOP_IMAGE_TAG} -f desktop/Dockerfile .")
        lines.append("")
    else:
        lines.append(f"Image '{DESKTOP_IMAGE_TAG}': OK")
        lines.append("")

    running = _find_running_container(DESKTOP_CONTAINER_NAME)
    if running:
        lines.append(f"Container '{DESKTOP_CONTAINER_NAME}': running ({running[:12]})")
    else:
        lines.append(f"Container '{DESKTOP_CONTAINER_NAME}': not running")
        lines.append("  Start it with:")
        lines.append(
            f"    docker run --rm -d --name {DESKTOP_CONTAINER_NAME} "
            f"-p 6080:6080 -p 5900:5900 -p 3000:3000 --env-file .env {DESKTOP_IMAGE_TAG}",
        )
    lines.append("")

    env_file = repo_root / ".env"
    if not env_file.exists():
        lines.append("Warning: no .env file found in repo root.")
        lines.append(
            "  The container needs ANTHROPIC_API_KEY, ORCHESTRA_URL, and UNIFY_KEY.",
        )
        lines.append("")

    lines.append(
        "Alternative: use --agent-mode web for browser-only mode (no Docker required)",
    )
    lines.append("")
    lines.append("Docs:")
    lines.append("  - desktop/README.md")

    return "\n".join(lines)
