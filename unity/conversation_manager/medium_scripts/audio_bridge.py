"""PulseAudio <-> LiveKit audio bridge for Google Meet integration.

Bridges audio between the browser's PulseAudio virtual devices and a local
LiveKit room, enabling the existing fast brain (STT/TTS/VAD) to hear meeting
participants and speak into the call.

Audio routing (per deploy/desktop/device.sh defaults):

    Participants → Meet → browser plays to agent_sink (default sink)
        → agent_sink.monitor → parec reads → publishes to LiveKit room
        → fast brain STT (Deepgram)

    Fast brain TTS → LiveKit audio track
        → bridge subscribes → writes to pacat → meet_sink
        → meet_mic (remap of meet_sink.monitor) = browser mic (default source)
        → Meet sends to participants

Uses parec/pacat (PulseAudio CLI tools from pulseaudio-utils) instead of
PyAudio/PortAudio, bypassing the libportaudio2 PulseAudio backend requirement.

Usage:
    bridge = AudioBridge(room_name="unity_25_gmeet")
    await bridge.start()
    ...
    await bridge.stop()
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Optional

from livekit import api as lk_api, rtc as lk_rtc

from unity.logger import LOGGER
from unity.common.hierarchical_logger import ICONS

SAMPLE_RATE = 48000
NUM_CHANNELS = 1
FRAME_DURATION_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_DURATION_MS // 1000
BYTES_PER_FRAME = SAMPLES_PER_FRAME * NUM_CHANNELS * 2  # s16le = 2 bytes/sample

INGEST_SOURCE = "agent_sink.monitor"
PLAYBACK_SINK = "meet_sink"

_TAG = "[AudioBridge]"


def _ensure_pulse_env() -> None:
    """Set PulseAudio env vars so parec/pacat find PipeWire-PulseAudio.

    PipeWire's PulseAudio compatibility layer creates its socket at
    ``$XDG_RUNTIME_DIR/pulse/native``.  If ``PULSE_SERVER`` isn't set,
    libpulse may not find the server in container environments where
    the XDG runtime dir isn't the conventional ``/run/user/<uid>``.
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR", "")
    if not xdg:
        xdg = "/tmp/runtime-unity"
        os.environ["XDG_RUNTIME_DIR"] = xdg

    if not os.environ.get("PULSE_SERVER"):
        sock = Path(xdg) / "pulse" / "native"
        if sock.exists():
            os.environ["PULSE_SERVER"] = f"unix:{sock}"
            LOGGER.info(
                f"{ICONS['ipc']} {_TAG} Set PULSE_SERVER={os.environ['PULSE_SERVER']}",
            )
        else:
            LOGGER.warning(
                f"{ICONS['ipc']} {_TAG} PulseAudio socket not found at {sock}",
            )

    LOGGER.info(
        f"{ICONS['ipc']} {_TAG} PulseAudio env: "
        f"XDG_RUNTIME_DIR={os.environ.get('XDG_RUNTIME_DIR', '<unset>')}, "
        f"PULSE_SERVER={os.environ.get('PULSE_SERVER', '<unset>')}, "
        f"DBUS_SESSION_BUS_ADDRESS={os.environ.get('DBUS_SESSION_BUS_ADDRESS', '<unset>')}",
    )


def _verify_pulse_server() -> bool:
    """Quick check: can we talk to PulseAudio via pactl?"""
    try:
        result = subprocess.run(
            ["pactl", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if (
                    "Server Name" in line
                    or "Default Sink" in line
                    or "Default Source" in line
                ):
                    LOGGER.info(f"{ICONS['ipc']} {_TAG} pactl: {line.strip()}")
            return True
        LOGGER.warning(
            f"{ICONS['ipc']} {_TAG} pactl info failed (rc={result.returncode}): "
            f"{result.stderr.strip()}",
        )
        return False
    except FileNotFoundError:
        LOGGER.warning(
            f"{ICONS['ipc']} {_TAG} pactl not found — pulseaudio-utils not installed?",
        )
        return False
    except Exception as exc:
        LOGGER.warning(f"{ICONS['ipc']} {_TAG} pactl check failed: {exc}")
        return False


def _verify_pulse_device(device: str, *, is_source: bool) -> bool:
    """Verify a PulseAudio source or sink exists."""
    kind = "sources" if is_source else "sinks"
    try:
        result = subprocess.run(
            ["pactl", "list", "short", kind],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            LOGGER.warning(
                f"{ICONS['ipc']} {_TAG} pactl list {kind} failed: "
                f"{result.stderr.strip()}",
            )
            return False
        if device in result.stdout:
            LOGGER.info(f"{ICONS['ipc']} {_TAG} Verified {kind[:-1]}: {device}")
            return True
        LOGGER.warning(
            f"{ICONS['ipc']} {_TAG} {kind[:-1]} '{device}' not found. "
            f"Available:\n{result.stdout.strip()}",
        )
        return False
    except Exception as exc:
        LOGGER.warning(f"{ICONS['ipc']} {_TAG} Device check failed: {exc}")
        return False


class AudioBridge:
    """Bridges PulseAudio virtual devices to a LiveKit room via parec/pacat.

    Uses PulseAudio CLI tools (from pulseaudio-utils) directly, bypassing
    PortAudio entirely. This avoids the requirement for libportaudio2 to be
    compiled with PulseAudio backend support.
    """

    def __init__(self, room_name: str) -> None:
        self._room_name = room_name
        self._room: Optional[lk_rtc.Room] = None
        self._ingest_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None
        self._audio_source: Optional[lk_rtc.AudioSource] = None
        self._parec_proc: Optional[asyncio.subprocess.Process] = None
        self._pacat_proc: Optional[asyncio.subprocess.Process] = None
        self._running = False

    async def start(self) -> None:
        """Connect to the LiveKit room and start audio bridging."""
        self._running = True

        _ensure_pulse_env()
        if not _verify_pulse_server():
            LOGGER.error(
                f"{ICONS['ipc']} {_TAG} PulseAudio server unreachable — "
                "cannot start bridge",
            )
            return

        _verify_pulse_device(INGEST_SOURCE, is_source=True)
        _verify_pulse_device(PLAYBACK_SINK, is_source=False)

        lk_url = os.environ.get("LIVEKIT_URL", "")
        lk_key = os.environ.get("LIVEKIT_API_KEY", "")
        lk_secret = os.environ.get("LIVEKIT_API_SECRET", "")

        if not lk_url:
            LOGGER.error(f"{ICONS['ipc']} {_TAG} LIVEKIT_URL not set, cannot start")
            return

        token = (
            lk_api.AccessToken(lk_key, lk_secret)
            .with_identity("browser-audio-bridge")
            .with_grants(
                lk_api.VideoGrants(
                    room_join=True,
                    room=self._room_name,
                ),
            )
            .to_jwt()
        )

        self._room = lk_rtc.Room()

        @self._room.on("track_subscribed")
        def _on_track_subscribed(
            track: lk_rtc.Track,
            publication: lk_rtc.RemoteTrackPublication,
            participant: lk_rtc.RemoteParticipant,
        ) -> None:
            if track.kind == lk_rtc.TrackKind.KIND_AUDIO:
                if self._playback_task is None or self._playback_task.done():
                    self._playback_task = asyncio.create_task(
                        self._playback_loop(lk_rtc.AudioStream(track)),
                    )

        await self._room.connect(lk_url, token)
        LOGGER.info(f"{ICONS['ipc']} {_TAG} Connected to room {self._room_name}")

        self._audio_source = lk_rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
        track = lk_rtc.LocalAudioTrack.create_audio_track(
            "browser-audio",
            self._audio_source,
        )
        await self._room.local_participant.publish_track(track)

        self._ingest_task = asyncio.create_task(self._ingest_loop())

    async def stop(self) -> None:
        """Disconnect from the LiveKit room and stop all audio tasks."""
        self._running = False

        for proc in (self._parec_proc, self._pacat_proc):
            if proc and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3)
                except asyncio.TimeoutError:
                    proc.kill()

        self._parec_proc = None
        self._pacat_proc = None

        for task in (self._ingest_task, self._playback_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._ingest_task = None
        self._playback_task = None

        if self._room:
            await self._room.disconnect()
            self._room = None

        LOGGER.info(f"{ICONS['ipc']} {_TAG} Disconnected from room {self._room_name}")

    async def _ingest_loop(self) -> None:
        """Read PCM audio from agent_sink.monitor via parec and publish to LiveKit."""
        cmd = [
            "parec",
            f"--device={INGEST_SOURCE}",
            "--format=s16le",
            f"--rate={SAMPLE_RATE}",
            f"--channels={NUM_CHANNELS}",
            "--latency-msec=20",
        ]
        LOGGER.info(f"{ICONS['ipc']} {_TAG} Starting ingest: {' '.join(cmd)}")

        try:
            self._parec_proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert self._parec_proc.stdout is not None
            LOGGER.info(
                f"{ICONS['ipc']} {_TAG} parec started (pid={self._parec_proc.pid})",
            )

            while self._running:
                data = await self._parec_proc.stdout.readexactly(BYTES_PER_FRAME)
                frame = lk_rtc.AudioFrame(
                    data=data,
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    samples_per_channel=SAMPLES_PER_FRAME,
                )
                await self._audio_source.capture_frame(frame)

        except asyncio.IncompleteReadError:
            LOGGER.warning(f"{ICONS['ipc']} {_TAG} parec stream ended unexpectedly")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            LOGGER.error(f"{ICONS['ipc']} {_TAG} Ingest error: {exc}")
        finally:
            await self._drain_proc_stderr(self._parec_proc, "parec")
            if self._parec_proc and self._parec_proc.returncode is None:
                self._parec_proc.terminate()

    async def _playback_loop(self, audio_stream: lk_rtc.AudioStream) -> None:
        """Subscribe to fast brain's audio and write to meet_sink via pacat."""
        cmd = [
            "pacat",
            f"--device={PLAYBACK_SINK}",
            "--format=s16le",
            f"--rate={SAMPLE_RATE}",
            f"--channels={NUM_CHANNELS}",
            "--latency-msec=20",
        ]
        LOGGER.info(f"{ICONS['ipc']} {_TAG} Starting playback: {' '.join(cmd)}")

        try:
            self._pacat_proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert self._pacat_proc.stdin is not None
            LOGGER.info(
                f"{ICONS['ipc']} {_TAG} pacat started (pid={self._pacat_proc.pid})",
            )

            async for event in audio_stream:
                if not self._running:
                    break
                frame: lk_rtc.AudioFrame = event.frame
                self._pacat_proc.stdin.write(bytes(frame.data))
                await self._pacat_proc.stdin.drain()

        except (BrokenPipeError, ConnectionResetError):
            LOGGER.warning(
                f"{ICONS['ipc']} {_TAG} pacat pipe broken — process may have exited",
            )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            LOGGER.error(f"{ICONS['ipc']} {_TAG} Playback error: {exc}")
        finally:
            await self._drain_proc_stderr(self._pacat_proc, "pacat")
            if self._pacat_proc and self._pacat_proc.returncode is None:
                self._pacat_proc.terminate()

    @staticmethod
    async def _drain_proc_stderr(
        proc: Optional[asyncio.subprocess.Process],
        label: str,
    ) -> None:
        """Read and log any stderr from a subprocess for diagnostics."""
        if proc and proc.stderr:
            try:
                stderr = await asyncio.wait_for(proc.stderr.read(), timeout=2)
                if stderr:
                    LOGGER.warning(
                        f"{ICONS['ipc']} {_TAG} {label} stderr: "
                        f"{stderr.decode(errors='replace').strip()}",
                    )
            except (asyncio.TimeoutError, Exception):
                pass
