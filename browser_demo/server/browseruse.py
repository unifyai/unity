from langchain_openai import ChatOpenAI
from browser_use import Agent, BrowserSession
import asyncio
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf

load_dotenv()

devices = sd.query_devices()
virtual_sink = None
for idx, dev in enumerate(devices):
    if "pipewire" in dev["name"] and dev["max_output_channels"] > 0:
        virtual_sink = idx
        break
print("virtual mic index:", virtual_sink)

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
            # "--use-fake-device-for-media-stream",
            # "--use-file-for-fake-video-capture=avatar.y4m",
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


if __name__ == "__main__":
    asyncio.run(main())
