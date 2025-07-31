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

unify.activate("Assistants")


def fetch_nearest_recording(assistant_id: str, timestamp: datetime):
    response = requests.get(
        f"{os.environ['UNIFY_BASE_URL']}/assistant/{assistant_id}/recordings",
        headers={
            "Authorization": f"Bearer {os.environ['UNIFY_KEY']}"
        }
    )
    recordings = response.json()["info"]
    distances = [
        abs((datetime.strptime(
            recording["created_at"], "%Y-%m-%dT%H:%M:%S.%f"
        ).replace(tzinfo=timezone.utc) - timestamp).total_seconds())
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


def play_audio(filename):
    """Audio playback function"""
    song = AudioSegment.from_mp3(filename)
    song.export("audio/output.wav", format="wav")
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(song.sample_width),
        channels=song.channels,
        rate=song.frame_rate,
        output=True
    )

    # Define chunk size of audio frames
    chunk = 1024

    # Open the audio file
    wf = wave.open("audio/output.wav", "rb")

    # Read and write audio frames
    data = wf.readframes(chunk)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk)
    
    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()


async def main(assistant_id: str, assistant_name: str, timestamp_str: str):
    # fetch audio file
    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f%z")
    recording = fetch_nearest_recording(assistant_id, timestamp)
    file_name = fetch_audio_file(recording["url"])
    context = f"{assistant_name.replace(' ', '')}/Events/Comms"

    # fetch startup event corresponding to the timestamp
    logs = unify.get_logs(
        project="Assistants",
        context=context,
        filter=f"payload_cls == 'StartupEvent' and timestamp == '{timestamp_str}'",
        limit=1,
    )
    if len(logs) == 0:
        print("No startup event found")
        return
    startup_event = logs[0].entries
    print("Startup Event:")
    print(startup_event)
    print()

    # fetch the phone call ended event corresponding to the startup
    phone_call_ended_events = unify.get_logs(
        project="Assistants",
        context=context,
        filter=f"payload_cls == 'PhoneCallEndedEvent' and timestamp > '{timestamp_str}'",
        limit=1,
        sorting={"timestamp": "ascending"},
    )
    if len(phone_call_ended_events) == 0:
        print("No phone call ended event found")
        return
    phone_call_ended_event = phone_call_ended_events[0].entries
    print("Phone Call Ended Event:")
    print(phone_call_ended_event)
    print()

    # get phone utterance events between the startup and phone call ended events
    phone_utterance_events = unify.get_logs(
        project="Assistants",
        context=context,
        filter=(
            "payload_cls == 'PhoneUtteranceEvent' "
            f"and timestamp > '{startup_event['timestamp']}' "
            f"and timestamp < '{phone_call_ended_event['timestamp']}'"
        ),
        limit=1000,
        sorting={"timestamp": "ascending"},
    )
    if len(phone_utterance_events) == 0:
        print("No phone call ended event found")
        return
    phone_utterance_events = list(map(lambda x: x.entries, phone_utterance_events))
    print("Phone Utterance Events:")
    phone_utterance_seconds = []
    current_second = 0
    current_timestamp = datetime.strptime(phone_utterance_events[0]["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z")
    for event in phone_utterance_events:
        new_timestamp = datetime.strptime(event["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z")
        current_second = int(current_second + (new_timestamp - current_timestamp).total_seconds())
        current_timestamp = new_timestamp
        phone_utterance_seconds.append({
            "second": current_second,
            "role": event["role"],
            "content": event["content"]
        })
        print(current_second, "\t|", event["role"], ":", event["content"])
    print()

    # play audio
    audio_task = asyncio.create_task(asyncio.to_thread(play_audio, file_name))
    
    # do other work
    seconds, idx = 0, 0
    print("Playing audio...")
    while not audio_task.done():
        if idx < len(phone_utterance_seconds) and seconds >= phone_utterance_seconds[idx]["second"]:
            print(f"{phone_utterance_seconds[idx]['role']} : {phone_utterance_seconds[idx]['content']}")
            idx += 1
        seconds += 1
        await asyncio.sleep(1)
    
    print("Audio playback completed!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--assistant-id", type=str, required=True)
    parser.add_argument("--assistant-name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()

    asyncio.run(main(args.assistant_id, args.assistant_name, args.timestamp))
