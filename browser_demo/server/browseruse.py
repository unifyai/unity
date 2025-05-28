from langchain_openai import ChatOpenAI
from browser_use import Agent, BrowserSession
from dotenv import load_dotenv

load_dotenv()

import asyncio

llm = ChatOpenAI(model="gpt-4o")


async def main():
    browser = BrowserSession(
        # executable_path="/usr/bin/chromium",
        chromium_sandbox=False,
        keep_alive=True,
        args=[
            "--start-maximized",
            "--no-sandbox",
            "--window-position=0,0",
            "--window-size=1920,1080",
            "--start-fullscreen",
            # "--use-fake-ui-for-media-stream",
            # "--use-fake-device-for-media-stream",
        ],
        # permissions=["microphone", "camera"],
    )

    agent = Agent(
        task="You're a helpful assistant. Open google search page and wait for the user to give you a task.",
        llm=llm,
        browser_session=browser,
    )
    result = await agent.run()

    while True:
        # await asyncio.sleep(1)
        action = input("Actions: ")
        if action == "close":
            break

        agent.add_new_task(action)
        result = await agent.run()
        print(result)
    
    await agent.close()
    # await context.close()
    await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
