'''utils.py
Shared voice‑mode helpers for sandbox scripts: audio capture, Deepgram STT
and Cartesia TTS.  Extracted from the original sandbox implementations so
both transcript_sandbox.py and tasklist_sandbox.py can import them.
'''

from __future__ import annotations

import asyncio
import os
import threading
import aiohttp
import sys
import wave
from contextlib import contextmanager
from ctypes import CFUNCTYPE, c_char_p, c_int, cdll
from typing import List

import pyaudio
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from livekit.plugins import cartesia

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Audio / PortAudio boilerplate
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)


def _py_error_handler(filename, line, function, err, fmt):  # noqa: D401 – C callback sig
    pass


c_error_handler = ERROR_HANDLER_FUNC(_py_error_handler)


@contextmanager
def noalsaerr():
    'Temporarily suppress ALSA warnings (common on Linux CI containers).'
    try:
        asound = cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(c_error_handler)
        yield
        asound.snd_lib_error_set_handler(None)
    except Exception:
        yield


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def record_until_enter() -> bytes:
    'Record audio between two ENTER presses and return WAV bytes.'
    with noalsaerr():
        pa = pyaudio.PyAudio()

    frames: List[bytes] = []
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    def _capture():
        while not stop.is_set():
            frames.append(stream.read(CHUNK, exception_on_overflow=False))

    stop = threading.Event()
    thr = threading.Thread(target=_capture, daemon=True)

    input('\nPress ↵ to start recording…')
    thr.start()
    input('🎙️  Recording… press ↵ again to stop.')
    stop.set()
    thr.join()

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print('✅ Recording captured.')

    wav_path = '/tmp/voice_input.wav'
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
    with open(wav_path, 'rb') as f:
        return f.read()


def transcribe_deepgram(audio_bytes: bytes) -> str:
    'Send *audio_bytes* to Deepgram SDK v4 and return the transcript.'
    key = os.getenv('DEEPGRAM_API_KEY')
    if not key:
        print('[Voice] Deepgram key missing – fallback to CLI input.')
        return input('> ')

    dg = DeepgramClient(api_key=key)
    payload: FileSource = {'buffer': audio_bytes}
    opts = PrerecordedOptions(model='nova-3', smart_format=True, punctuate=True)

    try:
        response = dg.listen.rest.v('1').transcribe_file(payload, opts)
        return response.results.channels[0].alternatives[0].transcript.strip()
    except Exception as exc:
        print(f'[Voice] Deepgram error ({exc}) – fallback to CLI input.')
        return input('> ')


def speak(text: str):
    """Speak *text* via Cartesia TTS – press ↵ to interrupt playback."""
    if "CARTESIA_API_KEY" not in os.environ:
        return
    
    print("🗣️ Assistant speaking… press ↵ to skip.")

    async def _gen() -> bytes:
        async with aiohttp.ClientSession() as s:
            tts = cartesia.TTS(http_session=s)
            stream = tts.synthesize(text)

            # Collect the entire synthesis into one object
            frame = await stream.collect()

            # Newest SDK exposes to_pcm_bytes()
            if hasattr(frame, "to_pcm_bytes"):
                return frame.to_pcm_bytes()

            # Older SDK exposes data attr
            if hasattr(frame, "data"):
                return bytes(frame.data)

            # Fallback: to_wav_bytes() → strip 44-byte header
            if hasattr(frame, "to_wav_bytes"):
                wav = frame.to_wav_bytes()
                return wav[44:]

            # Last resort – try bytes() conversion
            return bytes(frame)

    pcm = asyncio.run(_gen())
    if not pcm:
        return

    stop = threading.Event()

    def _wait_enter():
        sys.stdin.readline()
        stop.set()

    threading.Thread(target=_wait_enter, daemon=True).start()

    with noalsaerr():
        pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True,
    )

    chunk = 4800  # ≈0.1 s of audio
    i = 0
    while i < len(pcm) and not stop.is_set():
        stream.write(pcm[i : i + chunk])
        i += chunk

    stream.stop_stream()
    stream.close()
    pa.terminate()

    if stop.is_set():  # flush the newline the user pressed
        try:
            import termios

            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass
