from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser
from dotenv import load_dotenv
load_dotenv()

import asyncio

llm = ChatOpenAI(model="gpt-4o")

async def main():
    browser = Browser()
    context = await browser.new_context()

    while True:
        action = input("Actions: ")
        if action == "close":
            break

        print(browser.config)

        agent = Agent(
            task=action,
            llm=llm,
            browser=browser,
            browser_context=context,
        )
        result = await agent.run()
        print(result)

    await context.close()
    await browser.close()

if __name__ == "__main__":
    asyncio.run(main())