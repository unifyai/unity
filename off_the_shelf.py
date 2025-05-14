import asyncio
import json
from dotenv import load_dotenv
import os
import threading

from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from langchain_openai import ChatOpenAI
import redis
from constants import LOGGER
from state import State

load_dotenv()

# Shared runtime state (safe to create eagerly – no event-loop needed)
_state = State()


# Browser assistant
class BrowserAssistant(threading.Thread):
    def __init__(self, *, daemon: bool = True) -> None:
        super().__init__(daemon=daemon)

        # Create a dedicated event loop for the browser
        self._browser_loop = asyncio.new_event_loop()

        # Create Redis connection
        self._redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self._pubsub = self._redis_client.pubsub()
        self._pubsub.subscribe("transcript")

        # Initialize state
        _state.set_task_running(False)
        _state.set_task_paused(False)
        _state.set_last_task_result(None)
        _state.set_last_step_results([])
        self._last_step_results = []

        # Start a daemon thread for the browser loop
        self._browser_thread = threading.Thread(
            target=self._run_browser_loop,
            daemon=True,
        )
        self._browser_thread.start()

        # Initialize browser objects in the browser thread
        future = asyncio.run_coroutine_threadsafe(
            self._initialize_browser(),
            self._browser_loop,
        )
        # Wait for browser initialization to complete
        self._browser, self._browser_context, self._browser_agent = future.result()
        print(f"Browser agent initialized: {self._browser_agent}")

    def _run_browser_loop(self):
        """Run the browser event loop in a dedicated thread."""
        asyncio.set_event_loop(self._browser_loop)
        self._browser_loop.run_forever()

    async def _initialize_browser(self):
        """Initialize browser objects on the browser thread."""
        print("Initializing browser agent...")
        browser = Browser(config=BrowserConfig(disable_security=True))
        browser_context = BrowserContext(browser=browser)
        browser_agent = Agent(
            task="You're a web assistant. Wait for user instructions.",
            llm=ChatOpenAI(
                model="gpt-4.1@openai",
                base_url=os.getenv("UNIFY_BASE_URL"),
                api_key=os.getenv("UNIFY_KEY"),
            ),
            browser=browser,
            browser_context=browser_context,
        )
        return browser, browser_context, browser_agent

    def run(self) -> None:
        """Main thread: listen for Redis messages and dispatch browser tasks."""
        for transcript in self._pubsub.listen():
            if transcript["type"] != "message":
                continue
            try:
                messages = json.loads(transcript["data"])
                print(f"Received task: {messages}")

                # Send the task to the browser agent
                future = asyncio.run_coroutine_threadsafe(
                    self._add_new_task(messages),
                    self._browser_loop,
                )
                future.result()  # Wait for task to be added

                if not _state.task_running:
                    # Reset state for a new task
                    future = asyncio.run_coroutine_threadsafe(
                        self._reset_agent_history(),
                        self._browser_loop,
                    )
                    future.result()

                    _state.set_task_running(True)
                    _state.set_task_paused(False)
                    _state.set_last_task_result(None)
                    _state.set_last_step_results([])
                    self._last_step_results = []

                    # Run the browser task on the browser thread
                    print(f"Starting browser task with agent: {self._browser_agent}")
                    future = asyncio.run_coroutine_threadsafe(
                        self._browser_run(),
                        self._browser_loop,
                    )

                    # Set up callback for when task completes
                    def _handle_task_completion(fut):
                        try:
                            result = fut.result()
                            print("Browser task completed successfully")
                        except Exception as e:
                            print(f"Error executing browser task: {e}")
                            import traceback

                            traceback.print_exc()
                        finally:
                            _state.set_task_running(False)

                    future.add_done_callback(_handle_task_completion)
                else:
                    print("Task already running, cannot start new task")
            except Exception as e:
                print(f"Error in main thread: {e}")
                import traceback

                traceback.print_exc()

    async def _add_new_task(self, messages):
        """Add a new task to the browser agent (runs on browser thread)."""
        self._browser_agent.add_new_task(messages)

    async def _reset_agent_history(self):
        """Reset the agent history (runs on browser thread)."""
        self._browser_agent.state.history.history = []

    async def set_last_step_result(self, result: Agent):
        """Callback used by browser agent after each step."""
        last_action = result.state.history.last_action()
        self._last_step_results.append(
            json.dumps({} if last_action is None else last_action),
        )
        _state.set_last_step_results(self._last_step_results)

    async def _browser_run(self):
        """Execute a browser task (runs on browser thread)."""
        try:
            # This is called on the browser thread, so we can await directly
            result = await self._browser_agent.run(
                on_step_end=self.set_last_step_result,
            )

            # Process the result
            result_dict = json.loads(result.model_dump_json())
            history_list = []
            for history in result_dict["history"]:
                history.pop("state")
                history.pop("metadata")
                history_list.append(history)

            result_json = json.dumps({"result": history_list}, indent=4)
            LOGGER.info(result_json)
            self._redis_client.publish("task_completion", result_json)
            return result_json
        except Exception as e:
            print(f"Error in browser run: {e}")
            import traceback

            traceback.print_exc()
            raise


# Browser controller
class BrowserController(threading.Thread):
    def __init__(self, *, daemon: bool = True) -> None:
        super().__init__(daemon=daemon)
        self._redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self._pubsub = self._redis_client.pubsub()
        self._pubsub.subscribe("command")

        # Reference to BrowserAssistant to access browser agent
        self._browser_assistant = None

    def set_browser_assistant(self, assistant: BrowserAssistant):
        """Set reference to the BrowserAssistant instance."""
        self._browser_assistant = assistant

    def run(self) -> None:
        """Process command messages to control browser agent."""
        # Wait for browser assistant to be set
        while self._browser_assistant is None:
            import time

            time.sleep(0.1)

        print("BrowserController ready")

        for command in self._pubsub.listen():
            if command["type"] != "message":
                continue

            print(f"Received command: {command.get('action')}")

            try:
                browser_agent = self._browser_assistant._browser_agent
                browser_loop = self._browser_assistant._browser_loop

                if command.get("action") == "pause_task":
                    # Execute command on browser thread
                    future = asyncio.run_coroutine_threadsafe(
                        self._pause_task(browser_agent),
                        browser_loop,
                    )
                    future.result()
                    _state.set_task_running(False)
                    _state.set_task_paused(True)
                    print("Task paused")

                elif command.get("action") == "resume_task":
                    future = asyncio.run_coroutine_threadsafe(
                        self._resume_task(browser_agent),
                        browser_loop,
                    )
                    future.result()
                    _state.set_task_running(True)
                    _state.set_task_paused(False)
                    print("Task resumed")

                elif command.get("action") == "cancel_task":
                    future = asyncio.run_coroutine_threadsafe(
                        self._cancel_task(browser_agent),
                        browser_loop,
                    )
                    future.result()
                    _state.set_task_running(False)
                    print("Task canceled")
            except Exception as e:
                print(f"Error handling command: {e}")
                import traceback

                traceback.print_exc()

    async def _pause_task(self, browser_agent):
        """Pause the task (runs on browser thread)."""
        browser_agent.pause()

    async def _resume_task(self, browser_agent):
        """Resume the task (runs on browser thread)."""
        browser_agent.resume()

    async def _cancel_task(self, browser_agent):
        """Cancel the task (runs on browser thread)."""
        browser_agent.stop()
