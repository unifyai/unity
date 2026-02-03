"""
ConversationManager sandbox support for Magnitude `agent-service`.

Mode 3 of the ConversationManager sandbox uses a real computer backend (Magnitude),
which requires the Node.js `agent-service` to be running and reachable.

This module is *sandbox-only* and provides two capabilities:
- **Diagnosis**: return a user-facing explanation + next steps when `agent-service`
  cannot be reached or authenticated.
- **Bootstrap**: optionally start `agent-service` (direct), and if needed install/build
  dependencies and start it (full bootstrap).

Design notes
------------
- These helpers are intentionally conservative and UI-agnostic.
- They return structured results; callers decide how to display progress and errors.
- Subprocess spawning is best-effort and should never be the only supported path.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class AgentServiceDiagnosis:
    """User-facing diagnosis for `agent-service` availability."""

    ok: bool
    summary: str
    help_text: str


@dataclass(frozen=True)
class AgentServiceBootstrapResult:
    ok: bool
    summary: str
    process: Optional[subprocess.Popen[str]] = None


def _which(cmd: str) -> bool:
    from shutil import which

    return which(cmd) is not None


def _parse_port(agent_server_url: str) -> str:
    try:
        u = urlparse(str(agent_server_url))
        if u.port:
            return str(int(u.port))
    except Exception:
        pass
    return "3000"


def _terminate_process(proc: subprocess.Popen[str], *, timeout_s: float = 2.0) -> None:
    """
    Best-effort shutdown for a subprocess.

    We prefer a graceful terminate, then a hard kill if needed. Errors are swallowed
    intentionally since this is a developer sandbox UX helper.
    """
    try:
        if proc.poll() is not None:
            return
    except Exception:
        return
    # When started with `start_new_session=True`, `agent-service` is the leader
    # of a new process group. Terminating only the parent (e.g. `npm exec`) can
    # leave the child `node` process running, so prefer killing the whole group.
    try:
        import os as _os
        import signal as _signal

        pid = int(getattr(proc, "pid", 0) or 0)
        if pid > 0:
            try:
                pgid = _os.getpgid(pid)
            except Exception:
                pgid = None
            if pgid:
                try:
                    _os.killpg(pgid, _signal.SIGTERM)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        proc.terminate()
    except Exception:
        return
    try:
        proc.wait(timeout=float(timeout_s))
    except Exception:
        try:
            import os as _os
            import signal as _signal

            pid = int(getattr(proc, "pid", 0) or 0)
            if pid > 0:
                try:
                    pgid = _os.getpgid(pid)
                except Exception:
                    pgid = None
                if pgid:
                    try:
                        _os.killpg(pgid, _signal.SIGKILL)
                        return
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass


def _agent_service_log_path(*, repo_root: Path, port: int) -> Path:
    """
    Return a stable log file path for sandbox-started agent-service instances.

    We put this under `logs/` (gitignored) to avoid polluting the repo root and to
    keep logs discoverable for debugging.
    """
    return Path(repo_root) / "logs" / "agent_service" / f"agent-service_{port}.log"


def get_agent_service_log_path(*, repo_root: Path, agent_server_url: str) -> Path:
    """
    Public helper: return the sandbox log file path for `agent-service` at the given URL.

    This mirrors the path used when the sandbox starts agent-service itself.
    """
    port_s = _parse_port(agent_server_url)
    try:
        port = int(port_s)
    except Exception:
        port = 3000
    return _agent_service_log_path(repo_root=repo_root, port=port)


def _open_agent_service_log(*, repo_root: Path, port: int):
    path = _agent_service_log_path(repo_root=repo_root, port=port)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        fh = path.open("a", encoding="utf-8")
    except Exception:
        # Fallback: discard logs rather than breaking startup.
        fh = open(os.devnull, "w")  # noqa: P201
    return fh, path


@dataclass(frozen=True)
class PortReleaseResult:
    """Outcome of a best-effort attempt to free a listening port."""

    released: bool
    summary: str
    pid: Optional[int] = None


def _extract_repo_path_markers(repo_root: Path) -> list[str]:
    """
    Return stable substrings used to identify processes belonging to this repo.

    We keep these conservative: only match clearly repo-scoped paths.
    """
    rr = str(repo_root)
    return [
        str(Path(rr) / "agent-service"),
        str(Path(rr) / "agent-service" / "src" / "index.ts"),
    ]


def _looks_like_repo_agent_service_process(
    *,
    repo_root: Path,
    cmdline: str,
    cwd: str,
) -> bool:
    markers = _extract_repo_path_markers(repo_root)
    if any(m in cmdline for m in markers):
        return True
    if any(m in cwd for m in markers):
        return True
    # Also handle `cwd=.../agent-service` with cmdline like `ts-node src/index.ts`
    if str(Path(repo_root) / "agent-service") in cwd and "src/index.ts" in cmdline:
        return True
    return False


def _find_listening_pid(port: int) -> Optional[int]:
    """
    Return the PID of a process listening on `port` (best-effort).

    Prefers psutil for portability; falls back to lsof if psutil is unavailable.
    """
    try:
        import psutil  # type: ignore

        # Prefer inet listeners (tcp/udp) and filter for LISTEN.
        for c in psutil.net_connections(kind="inet"):
            try:
                if not c.laddr:
                    continue
                if int(getattr(c.laddr, "port", -1)) != int(port):
                    continue
                # psutil uses 'LISTEN' for TCP listeners; UDP may not have status.
                if getattr(c, "status", None) not in (None, "LISTEN"):
                    continue
                pid = getattr(c, "pid", None)
                if pid:
                    return int(pid)
            except Exception:
                continue
    except Exception:
        pass

    # Fallback: lsof -t prints PIDs only.
    try:
        out = subprocess.check_output(
            ["lsof", f"-iTCP:{int(port)}", "-sTCP:LISTEN", "-n", "-P", "-t"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in (out or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                return int(line)
            except Exception:
                continue
    except Exception:
        pass
    return None


def free_agent_service_port(
    *,
    repo_root: Path,
    agent_server_url: str,
    progress: Optional[ProgressCallback] = None,
) -> PortReleaseResult:
    """
    Best-effort: if the configured port is in use by *this repo's* agent-service,
    terminate that process so a fresh instance can start.

    Safety: we do **not** kill unrelated processes. If the port is in use by something
    else, we return a helpful message instead.
    """
    progress = progress or (lambda _m: None)
    port_s = _parse_port(agent_server_url)
    try:
        port = int(port_s)
    except Exception:
        return PortReleaseResult(
            released=False,
            summary=f"Invalid port in URL: {agent_server_url}",
        )

    pid = _find_listening_pid(port)
    if not pid:
        return PortReleaseResult(
            released=True,
            summary=f"Port {port} is free",
            pid=None,
        )

    # Resolve process info and only terminate if it looks like repo agent-service.
    try:
        import psutil  # type: ignore

        p = psutil.Process(pid)
        cmdline = " ".join(p.cmdline() or [])
        cwd = ""
        try:
            cwd = str(p.cwd() or "")
        except Exception:
            cwd = ""
        name = ""
        try:
            name = str(p.name() or "")
        except Exception:
            name = ""

        if not _looks_like_repo_agent_service_process(
            repo_root=repo_root,
            cmdline=cmdline,
            cwd=cwd,
        ):
            return PortReleaseResult(
                released=False,
                summary=(
                    f"Port {port} is already in use by PID {pid} ({name or 'unknown'}). "
                    "Not stopping it automatically. Use --agent-server-url to choose a different port "
                    "or stop the process manually."
                ),
                pid=pid,
            )

        progress(f"[agent-service] Port {port} is in use by PID {pid}; stopping it...")
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.wait(timeout=2.0)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

        # Verify the port is now free.
        if _find_listening_pid(port) is None:
            return PortReleaseResult(
                released=True,
                summary=f"Freed port {port} (stopped PID {pid})",
                pid=pid,
            )
        return PortReleaseResult(
            released=False,
            summary=f"Failed to free port {port} (PID {pid} still listening)",
            pid=pid,
        )
    except Exception:
        return PortReleaseResult(
            released=False,
            summary=(
                f"Port {port} is in use (PID {pid}), but the sandbox could not inspect/terminate it."
            ),
            pid=pid,
        )


def _validate_agent_service(
    *,
    agent_server_url: str,
    unify_key: str,
    timeout_s: float = 2.5,
) -> bool:
    """Return True iff `/sessions` responds with HTTP 200 using the provided token."""
    try:
        import httpx  # type: ignore
    except Exception:
        httpx = None  # type: ignore

    url = str(agent_server_url).rstrip("/") + "/sessions"
    headers = {"authorization": f"Bearer {unify_key}"}
    try:
        if httpx is None:
            from urllib.request import Request
            from urllib.request import urlopen

            req = Request(url, headers=headers)
            with urlopen(req, timeout=float(timeout_s)) as resp:  # nosec B310
                return int(getattr(resp, "status", 0) or 0) == 200
        resp = httpx.get(url, headers=headers, timeout=float(timeout_s))
        return int(resp.status_code) == 200
    except Exception:
        return False


def _probe_agent_service_noauth_status(
    *,
    agent_server_url: str,
    timeout_s: float = 1.5,
) -> Optional[int]:
    """
    Probe reachability without Unify auth.

    agent-service applies an auth middleware globally; `/sessions` without a token
    should return 401 quickly. This helps distinguish:
    - connection errors (service down/unreachable) vs
    - service running but auth failing (401 when token is supplied).
    """
    try:
        import httpx  # type: ignore
    except Exception:
        httpx = None  # type: ignore

    url = str(agent_server_url).rstrip("/") + "/sessions"
    try:
        if httpx is None:
            from urllib.error import HTTPError
            from urllib.request import Request
            from urllib.request import urlopen

            req = Request(url)
            try:
                with urlopen(req, timeout=float(timeout_s)) as resp:  # nosec B310
                    return int(getattr(resp, "status", 0) or 0)
            except HTTPError as e:  # 401 etc
                return int(getattr(e, "code", 0) or 0)
        resp = httpx.get(url, timeout=float(timeout_s))
        return int(resp.status_code)
    except Exception:
        return None


def _require_node_tooling() -> Optional[str]:
    for cmd in ("node", "npm", "npx"):
        if not _which(cmd):
            return cmd
    return None


def _wait_for_ready_or_explain_auth(
    *,
    proc: subprocess.Popen[str],
    agent_server_url: str,
    unify_key: str,
    timeout_s: float,
    poll_interval_s: float,
    early_auth_hint_after_s: float,
) -> AgentServiceBootstrapResult:
    start_t = time.time()
    deadline = start_t + float(timeout_s)
    while time.time() < deadline:
        if proc.poll() is not None:
            return AgentServiceBootstrapResult(
                ok=False,
                summary="agent-service process exited during startup",
                process=proc,
            )
        if _validate_agent_service(
            agent_server_url=agent_server_url,
            unify_key=unify_key,
        ):
            return AgentServiceBootstrapResult(
                ok=True,
                summary="agent-service started",
                process=proc,
            )
        if (time.time() - start_t) > float(early_auth_hint_after_s):
            st = _probe_agent_service_noauth_status(agent_server_url=agent_server_url)
            if st == 401:
                return AgentServiceBootstrapResult(
                    ok=False,
                    summary=(
                        "agent-service is running but authentication failed "
                        "(check UNIFY_KEY + ORCHESTRA_URL reachability)"
                    ),
                    process=proc,
                )
        time.sleep(float(poll_interval_s))
    return AgentServiceBootstrapResult(
        ok=False,
        summary=f"agent-service did not become ready within {timeout_s:.0f}s",
        process=proc,
    )


def try_start_agent_service_direct(
    *,
    repo_root: Path,
    agent_server_url: str,
    progress: Optional[ProgressCallback] = None,
    timeout_s: float = 20.0,
) -> AgentServiceBootstrapResult:
    """
    Start `agent-service` without modifying dependencies.

    This "direct start" path is intended for developers who already have Node deps
    installed and simply forgot to launch the service.
    """
    progress = progress or (lambda _m: None)
    unify_key = os.environ.get("UNIFY_KEY") or ""
    if not unify_key:
        return AgentServiceBootstrapResult(
            ok=False,
            summary="UNIFY_KEY is not set (required for agent-service auth)",
            process=None,
        )

    if _validate_agent_service(agent_server_url=agent_server_url, unify_key=unify_key):
        return AgentServiceBootstrapResult(
            ok=True,
            summary="agent-service already running",
            process=None,
        )

    missing = _require_node_tooling()
    if missing:
        return AgentServiceBootstrapResult(
            ok=False,
            summary=f"Missing `{missing}` (install Node.js / npm first)",
            process=None,
        )

    # If the port is already bound, attempt to free it (only if it's our agent-service).
    # This prevents confusing EADDRINUSE errors when the user retries Mode 3.
    port_status = _probe_agent_service_noauth_status(agent_server_url=agent_server_url)
    if port_status in {400, 401, 404}:
        r = free_agent_service_port(
            repo_root=repo_root,
            agent_server_url=agent_server_url,
            progress=progress,
        )
        if not r.released:
            return AgentServiceBootstrapResult(
                ok=False,
                summary=r.summary,
                process=None,
            )

    agent_dir = repo_root / "agent-service"
    if not agent_dir.exists():
        return AgentServiceBootstrapResult(
            ok=False,
            summary=f"Missing `agent-service/` at {agent_dir}",
            process=None,
        )

    port = _parse_port(agent_server_url)
    env = os.environ.copy()
    env.setdefault("PORT", port)

    port_int = int(env.get("PORT") or port)
    log_fh, log_path = _open_agent_service_log(repo_root=repo_root, port=port_int)
    progress(
        f"[agent-service] Starting on port {env.get('PORT')} (direct). "
        f"Logs: {log_path}",
    )
    proc = subprocess.Popen(
        ["npx", "ts-node", "src/index.ts"],
        cwd=str(agent_dir),
        env=env,
        stdout=log_fh,
        stderr=log_fh,
        text=True,
        start_new_session=True,
    )
    try:
        log_fh.close()
    except Exception:
        pass
    return _wait_for_ready_or_explain_auth(
        proc=proc,
        agent_server_url=agent_server_url,
        unify_key=unify_key,
        timeout_s=timeout_s,
        poll_interval_s=0.25,
        early_auth_hint_after_s=3.0,
    )


def diagnose_agent_service_setup(
    *,
    repo_root: Path,
    agent_server_url: str,
) -> AgentServiceDiagnosis:
    """
    Return a user-facing diagnosis + suggested next steps for agent-service.

    This function is intentionally "best-effort" and should never raise.
    """
    try:
        unify_key = os.environ.get("UNIFY_KEY") or ""
        orchestra_url = os.environ.get("ORCHESTRA_URL") or ""
        if not unify_key:
            return AgentServiceDiagnosis(
                ok=False,
                summary="UNIFY_KEY is not set",
                help_text=(
                    "Mode 3 requires `UNIFY_KEY` so the sandbox can authenticate to agent-service.\n\n"
                    "Fix:\n"
                    "- Add `UNIFY_KEY=...` to your repo `.env`, or export it in your shell.\n"
                    "- Then re-run the sandbox.\n\n"
                    "Notes:\n"
                    "- `agent-service` also verifies keys against Unify; ensure `ORCHESTRA_URL` is set if your setup requires it.\n\n"
                    "Docs:\n"
                    "- `sandboxes/conversation_manager/README.md` (Mode 3)\n"
                    "- `sandboxes/actor/README.md` (Magnitude agent-service setup)\n"
                ),
            )
        orchestra_url_hint = ""
        if not orchestra_url:
            orchestra_url_hint = (
                "Note: `ORCHESTRA_URL` is not set. agent-service verifies the key via "
                "`$ORCHESTRA_URL/user/basic-info`, so a missing/incorrect/unreachable "
                "`ORCHESTRA_URL` can cause authentication failures."
            )

        if _validate_agent_service(
            agent_server_url=agent_server_url,
            unify_key=unify_key,
        ):
            return AgentServiceDiagnosis(
                ok=True,
                summary="agent-service reachable",
                help_text="agent-service appears to be running and reachable.",
            )

        # Distinguish "service is down" vs "service is up but auth failed".
        noauth_status = _probe_agent_service_noauth_status(
            agent_server_url=agent_server_url,
        )
        service_up = noauth_status in {400, 401, 404}

        agent_dir = repo_root / "agent-service"
        magnitude_dir = repo_root / "magnitude"

        missing_bits: list[str] = []
        missing = _require_node_tooling()
        if missing:
            missing_bits.append(missing)

        lines: list[str] = []
        if service_up:
            lines.append(
                "agent-service appears to be running, but authentication failed. "
                "This usually means `UNIFY_KEY` is invalid, or `ORCHESTRA_URL` is missing/incorrect "
                "(agent-service verifies keys against Unify).",
            )
            lines.append("")
        else:
            lines.append(
                "Mode 3 needs the Magnitude `agent-service` running and reachable at "
                f"`{agent_server_url}`.",
            )
            lines.append("")
        if orchestra_url_hint:
            lines.append(orchestra_url_hint)
            lines.append("")

        if missing_bits:
            lines.append("Missing system dependencies:")
            lines.append(f"- {', '.join(missing_bits)}")
            lines.append("")
            lines.append("Fix:")
            lines.append("- Install Node.js (Node 22+ recommended for Unity).")
            lines.append("- Re-run the sandbox.")
            lines.append("")
        else:
            lines.append("System dependencies: OK (node/npm/npx found)")
            lines.append("")

        if not agent_dir.exists():
            lines.append("Repo layout issue:")
            lines.append(f"- Missing `agent-service/` at: {agent_dir}")
            lines.append("")
        else:
            lines.append(f"- Found `agent-service/`: {agent_dir}")
            lines.append("")

        if not magnitude_dir.exists():
            lines.append("Magnitude dependency:")
            lines.append(
                "- Missing `magnitude/` (Unity's Magnitude fork). This is required because "
                "`agent-service/package.json` depends on local `magnitude-core`.",
            )
            lines.append("")
        else:
            lines.append(f"- Found `magnitude/`: {magnitude_dir}")
            lines.append("")

        lines.append("Quick start (if you already have magnitude cloned/built):")
        port = _parse_port(agent_server_url)
        lines.append("```bash")
        lines.append("cd agent-service")
        lines.append("npm ci")
        lines.append(f"PORT={port} npx ts-node src/index.ts")
        lines.append("```")
        lines.append("")

        lines.append("Docs (step-by-step):")
        lines.append("- `sandboxes/actor/README.md` → “Magnitude Agent Service Setup”")
        lines.append("- Repo `README.md` (Node + agent-service prerequisites)")

        return AgentServiceDiagnosis(
            ok=False,
            summary=(
                "agent-service authentication failed"
                if service_up
                else "agent-service not running / unreachable"
            ),
            help_text="\n".join(lines),
        )
    except Exception as exc:
        return AgentServiceDiagnosis(
            ok=False,
            summary="Failed to diagnose agent-service setup",
            help_text=f"Unexpected error while diagnosing agent-service: {type(exc).__name__}: {exc}",
        )


def try_auto_bootstrap_agent_service(
    *,
    repo_root: Path,
    agent_server_url: str,
    progress: Optional[ProgressCallback] = None,
    timeout_s: float = 45.0,
) -> AgentServiceBootstrapResult:
    """Install/build deps (if needed) and start `agent-service`, then validate readiness."""
    progress = progress or (lambda _m: None)
    unify_key = os.environ.get("UNIFY_KEY") or ""
    if not unify_key:
        return AgentServiceBootstrapResult(
            ok=False,
            summary="UNIFY_KEY is not set (required for agent-service auth)",
            process=None,
        )
    if not (os.environ.get("ORCHESTRA_URL") or ""):
        progress(
            "[agent-service] Note: ORCHESTRA_URL is not set. "
            "agent-service verifies keys via Unify, so auth may fail until it is configured.",
        )

    # Already up?
    if _validate_agent_service(agent_server_url=agent_server_url, unify_key=unify_key):
        return AgentServiceBootstrapResult(
            ok=True,
            summary="agent-service already running",
            process=None,
        )

    missing = _require_node_tooling()
    if missing:
        return AgentServiceBootstrapResult(
            ok=False,
            summary=f"Missing `{missing}` (install Node.js / npm first)",
            process=None,
        )

    # If the port is already bound, attempt to free it (only if it's our agent-service).
    port_status = _probe_agent_service_noauth_status(agent_server_url=agent_server_url)
    if port_status in {400, 401, 404}:
        r = free_agent_service_port(
            repo_root=repo_root,
            agent_server_url=agent_server_url,
            progress=progress,
        )
        if not r.released:
            return AgentServiceBootstrapResult(
                ok=False,
                summary=r.summary,
                process=None,
            )

    agent_dir = repo_root / "agent-service"
    magnitude_dir = repo_root / "magnitude"
    if not agent_dir.exists():
        return AgentServiceBootstrapResult(
            ok=False,
            summary=f"Missing `agent-service/` at {agent_dir}",
            process=None,
        )
    if not magnitude_dir.exists():
        return AgentServiceBootstrapResult(
            ok=False,
            summary=(
                "Missing `magnitude/` (Unity's Magnitude fork). Cannot auto-bootstrap "
                "because agent-service depends on local magnitude-core."
            ),
            process=None,
        )

    # Install/build magnitude packages (best-effort, but required in most setups).
    for pkg in ("magnitude-core", "magnitude-extract"):
        pkg_dir = magnitude_dir / "packages" / pkg
        if not pkg_dir.exists():
            return AgentServiceBootstrapResult(
                ok=False,
                summary=f"Missing magnitude package directory: {pkg_dir}",
                process=None,
            )
        if not (pkg_dir / "node_modules").exists():
            progress(f"[agent-service] Installing {pkg} deps...")
            try:
                subprocess.run(["npm", "ci"], cwd=str(pkg_dir), check=True)
            except subprocess.CalledProcessError as exc:
                return AgentServiceBootstrapResult(
                    ok=False,
                    summary=(
                        f"Failed to install {pkg} deps (`npm ci` exited {exc.returncode})"
                    ),
                    process=None,
                )
        progress(f"[agent-service] Building {pkg}...")
        try:
            subprocess.run(["npm", "run", "build"], cwd=str(pkg_dir), check=True)
        except subprocess.CalledProcessError as exc:
            return AgentServiceBootstrapResult(
                ok=False,
                summary=f"Failed to build {pkg} (`npm run build` exited {exc.returncode})",
                process=None,
            )

    # Install agent-service deps
    if not (agent_dir / "node_modules").exists():
        progress("[agent-service] Installing agent-service deps...")
        try:
            subprocess.run(["npm", "ci"], cwd=str(agent_dir), check=True)
        except subprocess.CalledProcessError as exc:
            return AgentServiceBootstrapResult(
                ok=False,
                summary=(
                    f"Failed to install agent-service deps (`npm ci` exited {exc.returncode})"
                ),
                process=None,
            )

    # Start service on configured port.
    port = _parse_port(agent_server_url)
    env = os.environ.copy()
    env.setdefault("PORT", port)

    port_int = int(env.get("PORT") or port)
    log_fh, log_path = _open_agent_service_log(repo_root=repo_root, port=port_int)
    progress(
        f"[agent-service] Starting on port {env.get('PORT')} (bootstrap). "
        f"Logs: {log_path}",
    )
    proc = subprocess.Popen(
        ["npx", "ts-node", "src/index.ts"],
        cwd=str(agent_dir),
        env=env,
        stdout=log_fh,
        stderr=log_fh,
        text=True,
        start_new_session=True,
    )
    try:
        log_fh.close()
    except Exception:
        pass
    return _wait_for_ready_or_explain_auth(
        proc=proc,
        agent_server_url=agent_server_url,
        unify_key=unify_key,
        timeout_s=timeout_s,
        poll_interval_s=0.5,
        early_auth_hint_after_s=5.0,
    )
