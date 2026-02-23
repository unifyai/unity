"""
Live voice call support for the ConversationManager sandbox.

Spawns the production voice agent subprocess inside a LiveKit room. The
developer joins via a locally self-hosted copy of the LiveKit Agents
Playground (browser) and talks to the assistant through their microphone.

On first use the playground repo is cloned, dependencies installed, and a
small patch applied so that ``?url=…&token=…`` query params trigger an
automatic connection — zero manual steps after the browser tab opens.

Requires: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET env vars
          plus voice-provider credentials (Deepgram + Cartesia/ElevenLabs).
          Node.js (with pnpm or npm) is required for the playground server.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

from livekit import api

_VOICE_AGENT_LOG_NAME = ".logs_voice_agent.txt"
_CONNECTION_FILE = Path(".live_voice_connect.json")
_READINESS_POLL_INTERVAL_SECONDS = 0.25
_SANDBOX_LAUNCH_CWD_ENV = "UNITY_SANDBOX_LAUNCH_CWD"

_PLAYGROUND_DIR = Path(__file__).resolve().parents[2] / ".livekit-playground"
_PLAYGROUND_REPO = "https://github.com/livekit/agents-playground.git"
_PLAYGROUND_PORT = 3100
_PLAYGROUND_PATCH_MARKER = "/* SANDBOX_AUTO_CONNECT */"
_HOSTED_PLAYGROUND_URL = "https://agents-playground.livekit.io"

_playground_process: subprocess.Popen | None = None
_playground_atexit_registered = False


@dataclass
class LiveVoiceSession:
    """Tracks an active live voice call."""

    room_name: str
    agent_name: str
    user_token: str
    livekit_url: str
    playground_url: str = ""
    log_file: str = ""
    connection_file: str = ""
    browser_opened: bool = False
    ready: bool = False
    ready_source: str = ""
    ready_wait_seconds: float = 0.0
    ready_timeout_seconds: float = 0.0
    agent_joined_room: bool = False
    _log_fh: object = field(default=None, repr=False)


# ── LiveKit helpers ───────────────────────────────────────────────────────


def _require_env(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        raise RuntimeError(f"{name} is required for --live-voice mode.")
    return val


def _generate_user_token(room_name: str) -> str:
    return (
        api.AccessToken()
        .with_identity("developer")
        .with_name("Developer")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )


async def _create_room(room_name: str) -> None:
    lk = api.LiveKitAPI()
    try:
        await lk.room.create_room(api.CreateRoomRequest(name=room_name))
    finally:
        await lk.aclose()


async def _delete_room(room_name: str) -> None:
    lk = api.LiveKitAPI()
    try:
        await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))
    finally:
        await lk.aclose()


# ── Output suppression ───────────────────────────────────────────────────


def _spawn_quiet(log_path: Path):
    """
    Redirect all output during voice-agent subprocess spawn to *log_path*.

    Patches both the *module-level* binding of ``run_script`` in
    ``call_manager`` (``from unity.helpers import run_script`` creates a
    local copy) and ``sys.stdout``/``sys.stderr`` so parent-process prints
    don't leak into the terminal either.

    Returns ``(log_fh, restore_fn)``.
    """
    import unity.conversation_manager.domains.call_manager as _cm_mod
    import unity.helpers as _helpers

    log_fh = open(log_path, "w")
    orig_helpers, orig_cm = _helpers.run_script, _cm_mod.run_script

    def _quiet_run_script(script, *args, terminal: bool = False) -> subprocess.Popen:
        py_cmd = [sys.executable, str(Path(script).expanduser().resolve()), *args]
        return subprocess.Popen(
            py_cmd,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )

    _helpers.run_script = _quiet_run_script  # type: ignore[assignment]
    _cm_mod.run_script = _quiet_run_script  # type: ignore[assignment]
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = log_fh  # type: ignore[assignment]
    sys.stderr = log_fh  # type: ignore[assignment]

    def restore() -> None:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        _helpers.run_script = orig_helpers  # type: ignore[assignment]
        _cm_mod.run_script = orig_cm  # type: ignore[assignment]

    return log_fh, restore


# ── Self-hosted LiveKit Agents Playground ─────────────────────────────────


def _log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _playground_is_bootstrapped() -> bool:
    return (_PLAYGROUND_DIR / "node_modules").is_dir()


def _bootstrap_playground() -> None:
    """Clone the LiveKit Agents Playground, install deps, and patch for auto-connect."""
    _log("🔧 Bootstrapping LiveKit Agents Playground (one-time setup)…")

    if _PLAYGROUND_DIR.exists():
        shutil.rmtree(_PLAYGROUND_DIR)

    _log("   Cloning repository…")
    subprocess.run(
        ["git", "clone", "--depth", "1", _PLAYGROUND_REPO, str(_PLAYGROUND_DIR)],
        check=True,
        capture_output=True,
    )

    _log("   Installing dependencies…")
    if shutil.which("pnpm"):
        subprocess.run(
            ["pnpm", "install"],
            cwd=_PLAYGROUND_DIR,
            check=True,
            capture_output=True,
        )
    elif shutil.which("npm"):
        subprocess.run(
            ["npm", "install"],
            cwd=_PLAYGROUND_DIR,
            check=True,
            capture_output=True,
        )
    else:
        raise RuntimeError(
            "Neither pnpm nor npm found. Install Node.js to use --live-voice with the local playground.",
        )

    _patch_playground_for_auto_connect()
    _log("   ✅ Playground ready.")


def _patch_playground_for_auto_connect() -> None:
    """
    Patch ``index.tsx`` so that ``?url=<ws>&token=<jwt>`` query params
    trigger an immediate auto-connect, bypassing the manual connect screen.

    Uses ``useEffect`` to read params *after* hydration to avoid SSR/client
    mismatch errors.  Falls back gracefully if upstream source has changed.
    """
    index_file = _PLAYGROUND_DIR / "src" / "pages" / "index.tsx"
    content = index_file.read_text()

    if _PLAYGROUND_PATCH_MARKER in content:
        return

    patched = content

    # ── patch 1: add useEffect to the React import ──
    old_import = 'import { useState } from "react";'
    new_import = 'import { useEffect, useState } from "react";'
    if old_import in patched:
        patched = patched.replace(old_import, new_import, 1)

    # ── patch 2: insert a useEffect after the tokenSource useState ──
    #
    # Reads url+token from query params on the client *after* hydration,
    # then sets tokenSource and autoConnect — no SSR/client mismatch.
    old_anchor = "    return undefined;\n  });\n\n  return ("
    new_anchor = (
        "    return undefined;\n"
        "  });\n"
        "\n"
        f"  {_PLAYGROUND_PATCH_MARKER}\n"
        "  useEffect(() => {\n"
        "    const p = new URLSearchParams(window.location.search);\n"
        '    const serverUrl = p.get("url");\n'
        '    const token = p.get("token");\n'
        "    if (serverUrl && token) {\n"
        "      setTokenSource(TokenSource.literal({ serverUrl, participantToken: token }));\n"
        "      setAutoConnect(true);\n"
        "    }\n"
        "  }, []);\n"
        "\n"
        "  return ("
    )
    if old_anchor in patched:
        patched = patched.replace(old_anchor, new_anchor, 1)

    if patched == content:
        _log(
            "⚠️  Could not patch playground for auto-connect (upstream source changed).\n"
            "   The playground will still work but you'll need to paste URL + token manually.",
        )
        return

    index_file.write_text(patched)


def _playground_server_is_ready() -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{_PLAYGROUND_PORT}", timeout=2)
        return True
    except Exception:
        return False


def _kill_playground_server() -> None:
    global _playground_process
    if _playground_process is not None and _playground_process.poll() is None:
        _playground_process.terminate()
        try:
            _playground_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _playground_process.kill()
    _playground_process = None


def _ensure_playground_server() -> str:
    """
    Bootstrap the playground if needed, start the dev server if it isn't
    already running, and return the base URL (``http://localhost:<port>``).

    The server is kept alive across calls and cleaned up on process exit.
    Falls back to the hosted playground URL if Node.js is unavailable.
    """
    global _playground_process, _playground_atexit_registered

    base_url = f"http://localhost:{_PLAYGROUND_PORT}"

    if _playground_server_is_ready():
        return base_url

    if not shutil.which("npx"):
        _log(
            "⚠️  Node.js not found — falling back to the hosted LiveKit playground.\n"
            "   Install Node.js for zero-click auto-connect.",
        )
        return _HOSTED_PLAYGROUND_URL

    if not _playground_is_bootstrapped():
        _bootstrap_playground()

    server_log = _PLAYGROUND_DIR / ".server.log"
    log_fh = open(server_log, "w")

    _playground_process = subprocess.Popen(
        [shutil.which("npx"), "next", "dev", "-p", str(_PLAYGROUND_PORT)],  # type: ignore[arg-type]
        cwd=_PLAYGROUND_DIR,
        stdout=log_fh,
        stderr=log_fh,
        env={**os.environ, "NODE_NO_WARNINGS": "1"},
    )

    if not _playground_atexit_registered:
        atexit.register(_kill_playground_server)
        _playground_atexit_registered = True

    _log(f"   Starting playground dev server on port {_PLAYGROUND_PORT}…")
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if _playground_process.poll() is not None:
            raise RuntimeError(
                f"Playground server exited unexpectedly (code {_playground_process.returncode}). "
                f"Check {server_log}",
            )
        if _playground_server_is_ready():
            _log("   ✅ Playground server ready.")
            return base_url
        time.sleep(0.5)

    raise RuntimeError(
        f"Playground server did not start within 45s. Check {server_log}",
    )


def _build_playground_url(base_url: str, livekit_url: str, token: str) -> str:
    """Build the full playground URL with embedded connection params."""
    if base_url == _HOSTED_PLAYGROUND_URL:
        return base_url
    params = urlencode({"url": livekit_url, "token": token})
    return f"{base_url}/?{params}"


def _write_connection_file(session: LiveVoiceSession) -> None:
    _CONNECTION_FILE.write_text(
        json.dumps(
            {
                "playground": session.playground_url,
                "url": session.livekit_url,
                "token": session.user_token,
                "room": session.room_name,
            },
            indent=2,
        )
        + "\n",
    )


def _voice_agent_log_path() -> Path:
    """Return the voice-agent log path anchored to sandbox launch cwd."""
    launch_cwd = os.environ.get(_SANDBOX_LAUNCH_CWD_ENV, "").strip()
    if launch_cwd:
        return Path(launch_cwd).expanduser().resolve() / _VOICE_AGENT_LOG_NAME
    return Path.cwd().resolve() / _VOICE_AGENT_LOG_NAME


# ── Call-manager defaults ────────────────────────────────────────────────


def _ensure_call_manager_config(cm) -> None:
    """Fill in sensible defaults when voice settings are empty (common in sandbox)."""
    mgr = cm.call_manager
    if not mgr.voice_provider:
        mgr.voice_provider = os.environ.get("VOICE_PROVIDER", "cartesia")
    if not mgr.voice_id:
        mgr.voice_id = os.environ.get("VOICE_ID", "")
    if not mgr.assistant_bio:
        mgr.assistant_bio = os.environ.get(
            "ASSISTANT_BIO",
            "A helpful AI assistant. I always speak English unless explicitly asked to use another language.",
        )


def _open_playground(url: str) -> bool:
    """Best-effort browser launch."""
    try:
        return bool(webbrowser.open(url, new=2, autoraise=True))
    except Exception:
        return False


def _cm_in_meet_mode(cm) -> bool:
    """True when CM has processed UnifyMeetStarted and switched to meet mode."""
    mode = getattr(cm, "mode", None)
    mode_value = getattr(mode, "value", mode)
    return str(mode_value or "").strip().lower() == "meet"


async def _agent_joined_room(room_name: str) -> bool:
    """
    Check whether a non-developer participant has joined the room.

    In sandbox live-voice sessions this is typically the production voice agent.
    """
    lk = api.LiveKitAPI()
    try:
        participants = await lk.room.list_participants(
            api.ListParticipantsRequest(room=room_name),
        )
    finally:
        await lk.aclose()

    for participant in getattr(participants, "participants", []) or []:
        identity = str(getattr(participant, "identity", "") or "").strip().lower()
        name = str(getattr(participant, "name", "") or "").strip().lower()
        if identity == "developer" or name == "developer":
            continue
        return True
    return False


async def _wait_for_readiness(
    *,
    cm,
    room_name: str,
    timeout_seconds: float,
) -> tuple[bool, str, float, bool]:
    """
    Wait until live voice is truly ready or timeout is reached.

    Primary signal:
      - CM mode becomes `meet` (after UnifyMeetStarted is handled).
    Secondary signal:
      - A non-developer participant has joined the room.
    """
    timeout_seconds = max(0.0, float(timeout_seconds))
    started_at = time.monotonic()
    agent_joined_room = False
    next_participant_check_at = 0.0

    while True:
        if _cm_in_meet_mode(cm):
            return (
                True,
                "unify-meet-started",
                max(0.0, time.monotonic() - started_at),
                agent_joined_room,
            )

        elapsed = max(0.0, time.monotonic() - started_at)
        if not agent_joined_room and elapsed >= next_participant_check_at:
            try:
                agent_joined_room = await _agent_joined_room(room_name)
            except Exception:
                pass
            next_participant_check_at = elapsed + 1.0

        if elapsed >= timeout_seconds:
            source = "agent-joined-room" if agent_joined_room else "timeout"
            return False, source, elapsed, agent_joined_room

        await asyncio.sleep(_READINESS_POLL_INTERVAL_SECONDS)


# ── Public API ───────────────────────────────────────────────────────────


async def start_session(
    cm,
    contact: dict,
    boss: dict,
    *,
    open_browser: bool = True,
    ready_timeout_seconds: float = 20.0,
) -> LiveVoiceSession:
    """Create a LiveKit room, spawn voice agent, and wait for readiness."""
    livekit_url = _require_env("LIVEKIT_URL")
    _require_env("LIVEKIT_API_KEY")
    _require_env("LIVEKIT_API_SECRET")

    room_name = f"sandbox_{uuid.uuid4().hex[:8]}"
    agent_name = f"sandbox_{uuid.uuid4().hex[:8]}"

    _ensure_call_manager_config(cm)
    cm.call_manager.call_contact = contact
    await _create_room(room_name)

    # The subprocess runs ``agents.cli.run_app()`` in "dev" mode which
    # auto-joins the room — no separate dispatch_agent() call needed.
    voice_agent_log = _voice_agent_log_path()
    log_fh, restore = _spawn_quiet(voice_agent_log)
    try:
        await cm.call_manager.start_unify_meet(
            contact=contact,
            boss=boss,
            room_name=room_name,
        )
    finally:
        restore()

    user_token = _generate_user_token(room_name)
    playground_base = _ensure_playground_server()
    playground_url = _build_playground_url(playground_base, livekit_url, user_token)

    session = LiveVoiceSession(
        room_name=room_name,
        agent_name=agent_name,
        user_token=user_token,
        livekit_url=livekit_url,
        playground_url=playground_url,
        log_file=str(voice_agent_log.resolve()),
        connection_file=str(_CONNECTION_FILE.resolve()),
        _log_fh=log_fh,
    )
    _write_connection_file(session)
    session.browser_opened = _open_playground(playground_url) if open_browser else False
    session.ready_timeout_seconds = max(0.0, float(ready_timeout_seconds))
    (
        session.ready,
        session.ready_source,
        session.ready_wait_seconds,
        session.agent_joined_room,
    ) = await _wait_for_readiness(
        cm=cm,
        room_name=room_name,
        timeout_seconds=session.ready_timeout_seconds,
    )
    return session


async def stop_session(cm, session: LiveVoiceSession) -> None:
    """Terminate voice agent, close log file, delete LiveKit room.

    The playground dev server stays running for subsequent calls.
    """
    await cm.call_manager.cleanup_call_proc()

    try:
        fh = session._log_fh
        if fh is not None and hasattr(fh, "close"):
            fh.close()
    except Exception:
        pass

    try:
        _CONNECTION_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    try:
        await _delete_room(session.room_name)
    except Exception:
        pass
