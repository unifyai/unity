from langchain_openai import ChatOpenAI
from browser_use import Agent, BrowserSession
import asyncio
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
import cv2
import time
import threading

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

load_dotenv()

# Set up for virtual audio
devices = sd.query_devices()
virtual_sink = None
for idx, dev in enumerate(devices):
    if "pipewire" in dev["name"] and dev["max_output_channels"] > 0:
        virtual_sink = idx
        break
print("virtual mic index:", virtual_sink)


# Set up for virtual camera
Gst.init(None)

video_path = "avatar.mov"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

pipeline_str = (
    f"appsrc name=src is-live=true block=true format=TIME "
    f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={int(fps)}/1 "
    "! videoconvert "
    "! video/x-raw,format=YUY2 "
    "! v4l2sink device=/dev/video10"
)

pipeline = Gst.parse_launch(pipeline_str)
appsrc = pipeline.get_by_name("src")


def push_frame(frame):
    # frame is a BGR numpy array from OpenCV
    data = frame.tobytes()
    buf = Gst.Buffer.new_allocate(None, len(data), None)
    buf.fill(0, data)
    buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, int(fps))

    timestamp = getattr(push_frame, "timestamp", 0)
    buf.pts = buf.dts = timestamp
    push_frame.timestamp = timestamp + buf.duration

    ret = appsrc.emit("push-buffer", buf)
    if ret != Gst.FlowReturn.OK:
        print("Error pushing buffer:", ret)
        return False
    return True


pipeline.set_state(Gst.State.PLAYING)


def start_camera_loop():
    def loop():
        print("Streaming video to /dev/video10...")
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if not push_frame(frame):
                break
            time.sleep(1.0 / fps)

    threading.Thread(target=loop, daemon=True).start()


start_camera_loop()


llm = ChatOpenAI(model="gpt-4o")


async def main():
    browser = BrowserSession(
        executable_path="/usr/bin/chromium",
        chromium_sandbox=False,
        keep_alive=True,
        args=[
            "--start-maximized",
            "--no-sandbox",
            "--window-position=0,0",
            "--window-size=1920,1080",
            "--start-fullscreen",
            "--use-fake-ui-for-media-stream",
            "--enable-features=WebRtcV4L2VideoCapture",
        ],
        permissions=["microphone", "camera"],
    )

    agent = Agent(
        task="You're a helpful assistant. Open google search page and wait for the user to give you a task.",
        llm=llm,
        browser_session=browser,
    )
    result = await agent.run()

    while True:
        print("mic", sd.query_devices())
        action = input("Actions: ")
        if action == "close":
            break
        elif action == "play audio":
            # Play sample audio
            data, samplerate = sf.read("audio.wav")
            sd.play(data, samplerate, device=virtual_sink)
            sd.wait()
            continue

        agent.add_new_task(action)
        result = await agent.run()
        print(result)

    await agent.close()
    await browser.close()
    pipeline.set_state(Gst.State.NULL)
    cap.release()


if __name__ == "__main__":
    asyncio.run(main())
