import asyncio
from dotenv import load_dotenv
import os
import threading
import queue
from typing import List

from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from langchain_openai import ChatOpenAI
from constants import LOGGER

load_dotenv()


class OffTheShelf(threading.Thread):
    def __init__(
        self,
        transcript_q: "queue.Queue[List[str]]",
        task_completion_q: "asyncio.Queue[str]",
        action_completion_q: "queue.Queue[str]",
        coms_asyncio_loop: asyncio.AbstractEventLoop,
        *,
        daemon: bool = True,
    ) -> None:
        super().__init__(daemon=daemon)
        self._transcript_q = transcript_q
        self._task_completion_q = task_completion_q
        self._action_completion_q = action_completion_q
        self._coms_asyncio_loop = coms_asyncio_loop
        self._browser = Browser(config=BrowserConfig(disable_security=True))
        self._browser_context = BrowserContext(browser=self._browser)
        self._agent = Agent(
            task="You're a web assistant. Wait for user instructions.",
            llm=ChatOpenAI(
                model="gpt-4.1@openai",
                base_url=os.getenv("UNIFY_BASE_URL"),
                api_key=os.getenv("UNIFY_KEY"),
            ),
            browser=self._browser,
            browser_context=self._browser_context,
        )

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_run())
        finally:
            loop.close()

    async def _async_run(self) -> None:
        while True:
            messages = self._transcript_q.get()
            if messages is None:
                break
            self._agent.add_new_task(messages)
            result = await self._agent.run()
            LOGGER.info(result)
            self._action_completion_q.put(result)
            self._coms_asyncio_loop.call_soon_threadsafe(
                self._task_completion_q.put_nowait,
                result,
            )
