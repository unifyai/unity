from dotenv import load_dotenv

load_dotenv()

from argparse import ArgumentParser
import asyncio
from datetime import datetime, timezone
from google.cloud import storage
import os
import pyaudio
from pydub import AudioSegment
import requests
import wave
import unify
import threading
import time
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, Static, Header, Footer
from textual.reactive import reactive

unify.activate("Assistants")


def fetch_nearest_recording(assistant_id: str, timestamp: datetime):
    response = requests.get(
        f"{os.environ['UNIFY_BASE_URL']}/assistant/{assistant_id}/recordings",
        headers={"Authorization": f"Bearer {os.environ['UNIFY_KEY']}"},
    )
    recordings = response.json()["info"]
    distances = [
        abs(
            (
                datetime.strptime(
                    recording["created_at"], "%Y-%m-%dT%H:%M:%S.%f"
                ).replace(tzinfo=timezone.utc)
                - timestamp
            ).total_seconds()
        )
        for recording in recordings
    ]
    min_index = distances.index(min(distances))
    nearest_recording = recordings[min_index]
    return nearest_recording


def fetch_audio_file(recording_url: str):
    client = storage.Client(project="saas-368716")
    bucket = client.bucket("log-images-bucket")
    audio_folder = (
        "assistant_call_recording"
        if not os.environ.get("STAGING")
        else "assistant_call_recording_staging"
    )
    blob = bucket.blob(f"{audio_folder}/{recording_url.split('/')[-1]}")
    file_name = f"audio/{recording_url.split('/')[-1]}.mp3"
    blob.download_to_filename(file_name)
    return file_name


class TranscriptDisplay(Static):
    """Widget to display transcripts with timestamps"""

    def __init__(self, phone_utterance_seconds):
        super().__init__("")
        self.phone_utterance_seconds = phone_utterance_seconds
        self.current_second = 0
        self.audio_offset = 1
        self.displayed_utterances = set()  # Store indices instead of dicts
        self.all_content = []  # Store all content as a list

    def update_time(self, seconds):
        """Update the display based on current time"""
        self.current_second = seconds
        new_content = []

        for i, utterance in enumerate(self.phone_utterance_seconds):
            if (
                utterance["second"] + self.audio_offset < seconds
                and i not in self.displayed_utterances
            ):
                new_content.append(
                    f"[{utterance['second']:02d}s] {utterance['role']}: {utterance['content']}"
                )
                self.displayed_utterances.add(i)

        if new_content:
            self.all_content.extend(new_content)
            self.update("\n".join(self.all_content))
            # Auto-scroll to the bottom to show latest content
            self.scroll_end()


class AudioPlayer(App):
    """Main application for audio playback with transcripts"""

    CSS = """
    AudioPlayer {
        layout: grid;
        grid-size: 1;
        grid-rows: 1fr auto;
    }
    
    .main-container {
        layout: vertical;
        height: 100%;
    }
    
    .controls {
        layout: horizontal;
        height: auto;
        padding: 1;
        border-bottom: solid green;
    }
    
    .transcript-container {
        height: 1fr;
        border: solid blue;
        padding: 1;
    }
    
    TranscriptDisplay {
        background: $surface;
        color: $text;
        min-height: 100%;
    }
    
    Button {
        margin: 0 1;
    }
    """

    def __init__(self, assistant_id: str, assistant_name: str, timestamp_str: str):
        super().__init__()
        self.assistant_id = assistant_id
        self.assistant_name = assistant_name
        self.timestamp_str = timestamp_str
        self.audio_file = None
        self.phone_utterance_seconds = []
        self.is_playing = False
        self.current_second = 0
        self.audio_thread = None
        self.pause_event = threading.Event()
        self.audio_position = 0  # Track audio position in frames
        self.audio_stream = None
        self.audio_wave = None
        self.audio_pyaudio = None
        self.audio_initialized = False

    def compose(self) -> ComposeResult:
        """Create child widgets for the app"""
        yield Header()

        with Container(classes="main-container"):
            with Horizontal(classes="controls"):
                yield Button("Play", id="play-pause", variant="primary")
                yield Button("Reset", id="reset", variant="error")
                yield Button("⏮ -5s", id="backward")
                yield Button("⏭ +5s", id="forward")
                yield Static(f"Time: {self.current_second:02d}s", id="time-display")

            with VerticalScroll(classes="transcript-container"):
                yield TranscriptDisplay(self.phone_utterance_seconds)

        yield Footer()

    def on_mount(self) -> None:
        """Set up the app when it starts"""
        self.load_data()
        self.update_time_display()
        # Set up timer for updating playback time
        self.set_interval(1.0, self.update_playback_time)

    def load_data(self):
        """Load audio file and transcript data"""
        try:
            # Fetch data
            timestamp = datetime.strptime(self.timestamp_str, "%Y-%m-%dT%H:%M:%S.%f%z")
            recording = fetch_nearest_recording(self.assistant_id, timestamp)
            self.audio_file = fetch_audio_file(recording["url"])
            context = f"{self.assistant_name.replace(' ', '')}/Events/Comms"

            # Get transcript events
            phone_utterance_events = unify.get_logs(
                project="Assistants",
                context=context,
                filter=(
                    "payload_cls == 'PhoneUtteranceEvent' "
                    f"and timestamp > '{self.timestamp_str}'"
                ),
                limit=1000,
                sorting={"timestamp": "ascending"},
            )

            if phone_utterance_events:
                phone_utterance_events = list(
                    map(lambda x: x.entries, phone_utterance_events)
                )

                # Calculate timing
                current_second = 0
                current_timestamp = datetime.strptime(
                    phone_utterance_events[0]["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z"
                )

                for event in phone_utterance_events:
                    new_timestamp = datetime.strptime(
                        event["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z"
                    )
                    current_second = int(
                        current_second
                        + (new_timestamp - current_timestamp).total_seconds()
                    )
                    current_timestamp = new_timestamp
                    self.phone_utterance_seconds.append(
                        {
                            "second": current_second,
                            "role": event["role"],
                            "content": event["content"],
                        }
                    )

                # Update transcript display
                transcript_display = self.query_one(TranscriptDisplay)
                transcript_display.phone_utterance_seconds = (
                    self.phone_utterance_seconds
                )

        except Exception as e:
            self.notify(f"Error loading data: {e}", severity="error")

    def initialize_audio(self):
        """Initialize audio resources once"""
        if self.audio_initialized:
            return

        try:
            # Convert MP3 to WAV if not already done
            song = AudioSegment.from_mp3(self.audio_file)
            song.export("audio/output.wav", format="wav")

            self.audio_pyaudio = pyaudio.PyAudio()
            self.audio_stream = self.audio_pyaudio.open(
                format=self.audio_pyaudio.get_format_from_width(song.sample_width),
                channels=song.channels,
                rate=song.frame_rate,
                output=True,
            )

            self.audio_wave = wave.open("audio/output.wav", "rb")
            self.audio_initialized = True

        except Exception as e:
            self.notify(f"Audio initialization error: {e}", severity="error")

    def play_audio(self):
        """Play audio in a separate thread"""

        def audio_worker():
            try:
                # Initialize audio if not done
                if not self.audio_initialized:
                    self.call_from_thread(self.initialize_audio)
                    time.sleep(0.1)  # Give time for initialization

                if not self.audio_initialized:
                    return

                chunk = 1024

                # Set position if resuming
                if self.audio_position > 0:
                    self.audio_wave.setpos(self.audio_position)

                data = self.audio_wave.readframes(chunk)
                while len(data) > 0 and self.is_playing:
                    if self.pause_event.is_set():
                        # Save current position before pausing
                        self.audio_position = self.audio_wave.tell()
                        time.sleep(0.1)
                        continue

                    self.audio_stream.write(data)
                    data = self.audio_wave.readframes(chunk)

                # If we reach the end, mark as not playing
                if not data:
                    self.call_from_thread(self.pause_playback)
                    self.audio_position = 0  # Reset position

            except Exception as e:
                self.call_from_thread(
                    self.notify, f"Audio error: {e}", severity="error"
                )

        self.audio_thread = threading.Thread(target=audio_worker, daemon=True)
        self.audio_thread.start()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "play-pause":
            if not self.is_playing:
                self.start_playback()
            else:
                self.pause_playback()
        elif event.button.id == "forward":
            self.jump_forward(5)
        elif event.button.id == "backward":
            self.jump_backward(5)
        elif event.button.id == "reset":
            self.reset_playback()

    def start_playback(self):
        """Start audio playback"""
        self.is_playing = True
        self.pause_event.clear()

        # Only start new audio thread if not already running
        if not self.audio_thread or not self.audio_thread.is_alive():
            self.play_audio()

        self.query_one("#play-pause").label = "Pause"
        self.query_one("#play-pause").variant = "warning"

    def pause_playback(self):
        """Pause audio playback"""
        self.is_playing = False
        self.pause_event.set()
        self.query_one("#play-pause").label = "Play"
        self.query_one("#play-pause").variant = "primary"

    def jump_forward(self, seconds: int):
        """Jump forward by specified seconds"""
        if self.is_playing:
            self.pause_playback()

        self.current_second += seconds
        self.update_time_display()

        # Update transcript display to show all content up to current time
        transcript_display = self.query_one(TranscriptDisplay)
        transcript_display.update_time(self.current_second)

        # Update audio position if audio is initialized
        if self.audio_initialized and self.audio_wave:
            self.update_audio_position()

    def jump_backward(self, seconds: int):
        """Jump backward by specified seconds"""
        if self.is_playing:
            self.pause_playback()

        self.current_second = max(0, self.current_second - seconds)
        self.update_time_display()

        # Reset transcript display and rebuild up to current time
        transcript_display = self.query_one(TranscriptDisplay)
        transcript_display.displayed_utterances.clear()
        transcript_display.all_content.clear()
        transcript_display.update_time(self.current_second)

        # Update audio position if audio is initialized
        if self.audio_initialized and self.audio_wave:
            self.update_audio_position()

    def update_audio_position(self):
        """Update audio position based on current time using actual audio properties"""
        if not self.audio_initialized or not self.audio_wave:
            return

        try:
            # Get actual audio properties
            sample_rate = self.audio_wave.getframerate()
            channels = self.audio_wave.getnchannels()
            sample_width = self.audio_wave.getsampwidth()

            # Calculate frames per second
            frames_per_second = sample_rate

            # Calculate frame position
            new_position = int(self.current_second * frames_per_second)

            # Ensure position is within bounds
            total_frames = self.audio_wave.getnframes()
            self.audio_position = max(0, min(new_position, total_frames))

        except Exception as e:
            self.notify(f"Error updating audio position: {e}", severity="error")

    def reset_playback(self):
        """Reset playback to beginning"""
        self.pause_playback()
        self.current_second = 0
        self.audio_position = 0  # Reset audio position

        # Reset audio file position
        if self.audio_wave:
            self.audio_wave.rewind()

        self.update_time_display()

        # Reset transcript display
        transcript_display = self.query_one(TranscriptDisplay)
        transcript_display.displayed_utterances.clear()
        transcript_display.all_content.clear()
        transcript_display.update("")

    def on_unmount(self) -> None:
        """Clean up audio resources when app closes"""
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.audio_pyaudio:
            self.audio_pyaudio.terminate()
        if self.audio_wave:
            self.audio_wave.close()

    def update_time_display(self):
        """Update the time display"""
        time_widget = self.query_one("#time-display")
        time_widget.update(f"Time: {self.current_second:02d}s")

    def update_playback_time(self) -> None:
        """Update playback time and transcripts"""
        if self.is_playing and not self.pause_event.is_set():
            self.current_second += 1
            self.update_time_display()

            # Update transcript display
            transcript_display = self.query_one(TranscriptDisplay)
            transcript_display.update_time(self.current_second)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--assistant-id", type=str, required=True)
    parser.add_argument("--assistant-name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()

    app = AudioPlayer(args.assistant_id, args.assistant_name, args.timestamp)
    app.run()
