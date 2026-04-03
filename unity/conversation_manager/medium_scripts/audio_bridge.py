"""PulseAudio <-> LiveKit audio bridge for Google Meet integration.

Bridges audio between the browser's PulseAudio virtual devices and a local
LiveKit room, enabling the existing fast brain (STT/TTS/VAD) to hear meeting
participants and speak into the call.

Audio routing (per deploy/desktop/device.sh defaults):

    Participants → Meet → browser plays to agent_sink (default sink)
        → agent_sink.monitor → bridge reads → publishes to LiveKit room
        → fast brain STT (Deepgram)

    Fast brain TTS → LiveKit audio track
        → bridge subscribes → writes to meet_sink
        → meet_mic (remap of meet_sink.monitor) = browser mic (default source)
        → Meet sends to participants

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

_TAG = "[AudioBridge]"


def _ensure_pulse_env() -> None:
    """Set PulseAudio env vars so PortAudio discovers PipeWire-PulseAudio.

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
                f"{ICONS['ipc']} {_TAG} PulseAudio socket not found at {sock} — "
                "PortAudio may fall back to ALSA (which has no hardware in containers)",
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
            f"{ICONS['ipc']} {_TAG} pactl info failed (rc={result.returncode}): {result.stderr.strip()}",
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


class AudioBridge:
    """Bridges PulseAudio virtual devices to a LiveKit room.

    Connects to the LiveKit room as a hidden "browser-audio" participant and:
      - Publishes audio captured from ``agent_sink.monitor`` (participant voices)
        so the fast brain's STT can transcribe it.
      - Subscribes to the fast brain's published audio track and writes it to
        ``meet_sink`` so it flows into the browser's microphone input.
    """

    def __init__(self, room_name: str) -> None:
        self._room_name = room_name
        self._room: Optional[lk_rtc.Room] = None
        self._ingest_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None
        self._audio_source: Optional[lk_rtc.AudioSource] = None
        self._running = False

    async def start(self) -> None:
        """Connect to the LiveKit room and start audio bridging."""
        self._running = True

        _ensure_pulse_env()
        _verify_pulse_server()

        lk_url = os.environ.get("LIVEKIT_URL", "")
        lk_key = os.environ.get("LIVEKIT_API_KEY", "")
        lk_secret = os.environ.get("LIVEKIT_API_SECRET", "")

        if not lk_url:
            LOGGER.error(
                f"{ICONS['ipc']} [AudioBridge] LIVEKIT_URL not set, cannot start",
            )
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
        LOGGER.info(
            f"{ICONS['ipc']} [AudioBridge] Connected to room {self._room_name}",
        )

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

        LOGGER.info(
            f"{ICONS['ipc']} [AudioBridge] Disconnected from room {self._room_name}",
        )

    async def _ingest_loop(self) -> None:
        """Read PCM audio from agent_sink.monitor and publish to LiveKit.

        Uses PyAudio to capture from the PulseAudio monitor source that
        receives the browser's audio output (meeting participants' voices).
        """
        import pyaudio

        pa = pyaudio.PyAudio()
        self._log_portaudio_host_apis(pa)
        stream = None
        try:
            monitor_index = self._find_pulse_source(pa, "agent_sink.monitor")
            if monitor_index is None:
                LOGGER.error(
                    f"{ICONS['ipc']} {_TAG} Cannot find 'agent_sink.monitor' — "
                    "ingest loop cannot start (check PipeWire/PulseAudio)",
                )
                return
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=NUM_CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=monitor_index,
                frames_per_buffer=SAMPLES_PER_FRAME,
            )
            LOGGER.info(
                f"{ICONS['ipc']} {_TAG} Ingest stream opened (device={monitor_index})",
            )

            while self._running:
                data = await asyncio.to_thread(
                    stream.read,
                    SAMPLES_PER_FRAME,
                    exception_on_overflow=False,
                )
                frame = lk_rtc.AudioFrame(
                    data=data,
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    samples_per_channel=SAMPLES_PER_FRAME,
                )
                await self._audio_source.capture_frame(frame)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            LOGGER.error(f"{ICONS['ipc']} {_TAG} Ingest error: {exc}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            pa.terminate()

    async def _playback_loop(self, audio_stream: lk_rtc.AudioStream) -> None:
        """Subscribe to the fast brain's audio track and write to meet_sink.

        Writes PCM frames to the PulseAudio null sink whose monitor is mapped
        as the browser's microphone input, so TTS audio reaches the meeting.
        """
        import pyaudio

        pa = pyaudio.PyAudio()
        stream = None
        try:
            sink_index = self._find_pulse_sink(pa, "meet_sink")
            if sink_index is None:
                LOGGER.error(
                    f"{ICONS['ipc']} {_TAG} Cannot find 'meet_sink' — "
                    "playback loop cannot start (check PipeWire/PulseAudio)",
                )
                return
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=NUM_CHANNELS,
                rate=SAMPLE_RATE,
                output=True,
                output_device_index=sink_index,
                frames_per_buffer=SAMPLES_PER_FRAME,
            )
            LOGGER.info(
                f"{ICONS['ipc']} {_TAG} Playback stream opened (device={sink_index})",
            )

            async for event in audio_stream:
                if not self._running:
                    break
                frame: lk_rtc.AudioFrame = event.frame
                await asyncio.to_thread(stream.write, bytes(frame.data))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            LOGGER.error(f"{ICONS['ipc']} {_TAG} Playback error: {exc}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            pa.terminate()

    @staticmethod
    def _log_portaudio_host_apis(pa) -> None:
        """Log all PortAudio host APIs and devices for diagnostics."""
        api_count = pa.get_host_api_count()
        apis = []
        for i in range(api_count):
            info = pa.get_host_api_info_by_index(i)
            apis.append(f"{info['name']}(devices={info['deviceCount']})")
        LOGGER.info(f"{ICONS['ipc']} {_TAG} PortAudio host APIs: {', '.join(apis)}")

        has_pulse = any("pulse" in a.lower() for a in apis)
        if not has_pulse:
            LOGGER.warning(
                f"{ICONS['ipc']} {_TAG} PortAudio has NO PulseAudio backend! "
                "Audio bridge requires PulseAudio. Ensure libportaudio2 is built "
                "with PulseAudio support (apt install libportaudio2).",
            )

    @staticmethod
    def _find_pulse_source(pa, name: str) -> int | None:
        """Find a PulseAudio source (input device) by name substring."""
        sources = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                dev_name = info.get("name", "")
                sources.append(f"  [{i}] {dev_name} (in={info['maxInputChannels']})")
                if name in dev_name:
                    LOGGER.info(
                        f"{ICONS['ipc']} {_TAG} Found source '{name}' at index {i}",
                    )
                    return i
        LOGGER.warning(
            f"{ICONS['ipc']} {_TAG} Source '{name}' NOT FOUND. "
            f"Available input devices ({len(sources)}):\n" + "\n".join(sources),
        )
        return None

    @staticmethod
    def _find_pulse_sink(pa, name: str) -> int | None:
        """Find a PulseAudio sink (output device) by name substring."""
        sinks = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxOutputChannels", 0) > 0:
                dev_name = info.get("name", "")
                sinks.append(f"  [{i}] {dev_name} (out={info['maxOutputChannels']})")
                if name in dev_name:
                    LOGGER.info(
                        f"{ICONS['ipc']} {_TAG} Found sink '{name}' at index {i}",
                    )
                    return i
        LOGGER.warning(
            f"{ICONS['ipc']} {_TAG} Sink '{name}' NOT FOUND. "
            f"Available output devices ({len(sinks)}):\n" + "\n".join(sinks),
        )
        return None
