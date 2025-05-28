import sys, pathlib
import time
import asyncio

# Ensure repository root is on PYTHONPATH so `import unity` works when this
# script is executed directly from inside the "sandboxes" folder.
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from unity.controller.session import create_session, get_live_view_urls
from langchain_openai import ChatOpenAI
from browser_use import Agent, BrowserSession
from dotenv import load_dotenv

load_dotenv()


session = create_session()

print(session.connect_url)
print(session.id)


llm = ChatOpenAI(model="gpt-4o")


async def main():
    browser = BrowserSession(
        cdp_url=session.connect_url,
    )

    agent = Agent(
        task="You're a helpful assistant. Open google search page and wait for the user to give you a task.",
        llm=llm,
        browser_session=browser,
    )
    result = await agent.run()
    # print(get_live_view_urls(session.id))

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



