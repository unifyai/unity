from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser
from dotenv import load_dotenv

load_dotenv()

import asyncio

llm = ChatOpenAI(model="gpt-4o")


async def main():
    browser = Browser()
    context = await browser.new_context()
    agent = Agent(
        task="You're a helpful assistant. Wait for the user to give you a task.",
        llm=llm,
        browser=browser,
        browser_context=context,
    )

    while True:
        action = input("Actions: ")
        if action == "close":
            break

        # print(browser.config)

        agent.add_new_task(action)
        result = await agent.run()
        print(result)
    
    await agent.close()
    await context.close()
    await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
