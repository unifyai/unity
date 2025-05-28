from langchain_openai import ChatOpenAI
from browser_use import Agent, BrowserSession
from dotenv import load_dotenv

load_dotenv()

import asyncio

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
            "--start-fullscreen"
        ],
    )
    # context = await browser.new_context()
    agent = Agent(
        task="You're a helpful assistant. Open google search page and wait for the user to give you a task.",
        llm=llm,
        browser_session=browser,
        # browser_context=context,
    )
    result = await agent.run()
    # print(browser.config)

    while True:
        await asyncio.sleep(1)
        # action = input("Actions: ")
        # if action == "close":
        #     break

        # # print(browser.config)
        # agent.add_new_task(action)
        # result = await agent.run()
        # print(result)
    
    await agent.close()
    # await context.close()
    await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
