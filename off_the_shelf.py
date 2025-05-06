import asyncio
import json
from dotenv import load_dotenv
import os
import threading
import queue
from typing import List

from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from langchain_openai import ChatOpenAI
import redis
from constants import LOGGER

load_dotenv()


class OffTheShelf(threading.Thread):
    def __init__(self, *, daemon: bool = True) -> None:
        super().__init__(daemon=daemon)
        self._redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self._pubsub = self._redis_client.pubsub()
        self._pubsub.subscribe("transcript")
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
        for transcript in self._pubsub.listen():
            if transcript["type"] != "message":
                continue
            messages = json.loads(transcript["data"])
            self._agent.add_new_task(messages)
            result = await self._agent.run()
            result = json.loads(result.model_dump_json())
            history_list = []
            for history in result["history"]:
                history.pop("state")
                history.pop("metadata")
                history_list.append(history)
            result = json.dumps({"result": history_list})
            LOGGER.info(result)
            self._redis_client.publish("action_completion", result)
            self._redis_client.publish("task_completion", result)
