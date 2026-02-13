"""
Live voice call support for the ConversationManager sandbox.

Spawns the production voice agent subprocess inside a LiveKit room. The
developer joins via the Agents Playground (browser) and talks to the
assistant through their microphone.

Requires: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET env vars
          plus voice-provider credentials (Deepgram + Cartesia/ElevenLabs).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

from livekit import api

_VOICE_AGENT_LOG = Path(".logs_voice_agent.txt")
_CONNECTION_FILE = Path(".live_voice_connect.json")
_PLAYGROUND_URL = "https://agents-playground.livekit.io"
_READINESS_POLL_INTERVAL_SECONDS = 0.25


@dataclass
class LiveVoiceSession:
    """Tracks an active live voice call."""

    room_name: str
    agent_name: str
    user_token: str
    livekit_url: str
    playground_url: str = _PLAYGROUND_URL
    log_file: str = ""
    connection_file: str = ""
    clipboard_ok: bool = False
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


# ── Clipboard / connection file ──────────────────────────────────────────


def _copy_to_clipboard(text: str) -> bool:
    """Best-effort copy to system clipboard."""
    try:
        if sys.platform == "darwin":
            return (
                subprocess.run(
                    ["pbcopy"],
                    input=text.encode(),
                    check=True,
                    timeout=2,
                ).returncode
                == 0
            )
        if sys.platform.startswith("linux"):
            for cmd in (
                ["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard"],
            ):
                try:
                    subprocess.run(cmd, input=text.encode(), check=True, timeout=2)
                    return True
                except FileNotFoundError:
                    continue
    except Exception:
        pass
    return False


def _write_connection_file(session: LiveVoiceSession) -> None:
    _CONNECTION_FILE.write_text(
        json.dumps(
            {
                "playground": _PLAYGROUND_URL,
                "url": session.livekit_url,
                "token": session.user_token,
                "room": session.room_name,
            },
            indent=2,
        )
        + "\n",
    )


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


def _open_playground() -> bool:
    """Best-effort browser launch for the hosted LiveKit playground."""
    try:
        return bool(webbrowser.open(_PLAYGROUND_URL, new=2, autoraise=True))
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
    log_fh, restore = _spawn_quiet(_VOICE_AGENT_LOG)
    try:
        await cm.call_manager.start_unify_meet(
            contact=contact,
            boss=boss,
            livekit_agent_name=agent_name,
            room_name=room_name,
        )
    finally:
        restore()

    user_token = _generate_user_token(room_name)
    session = LiveVoiceSession(
        room_name=room_name,
        agent_name=agent_name,
        user_token=user_token,
        livekit_url=livekit_url,
        log_file=str(_VOICE_AGENT_LOG.resolve()),
        connection_file=str(_CONNECTION_FILE.resolve()),
        _log_fh=log_fh,
    )
    _write_connection_file(session)
    session.clipboard_ok = _copy_to_clipboard(user_token)
    session.browser_opened = _open_playground() if open_browser else False
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
    """Terminate voice agent, close log file, delete LiveKit room."""
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
