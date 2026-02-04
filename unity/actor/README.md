# Actor Module

This directory contains different "actor" implementations, which are responsible for taking a high-level user goal and breaking it down into a series of actions to be executed.

## Available Actors

### HierarchicalActor

The primary actor implementation. It takes a proactive, code-first approach by generating a complete, executable Python script that represents the entire plan. This script is then executed, and it has the ability to debug, verify, and modify itself at runtime.

-   **Plan Representation**: The plan is an explicit, well-structured Python script containing functions decorated with `@verify`.
-   **State Management**: State is managed explicitly within the Python execution context, including a function call stack.
-   **Correction**: Self-correction is built-in. The `@verify` decorator checks the outcome of every step. Failures can trigger tactical replans (rewriting a single function) or strategic replans (escalating to the parent function to rethink the approach).
-   **Execution Model**: `Generate Code -> Execute Code -> (Verify -> Self-Correct/Re-implement -> Resume Execution)`

### CodeActActor

A conversational actor that uses a stateful code execution sandbox. It operates in a reactive, turn-based loop, maintaining a chat history with an LLM and executing Python code blocks to accomplish tasks.

-   **Plan Representation**: Implicit in conversation history.
-   **State Management**: Managed by the code execution sandbox which preserves variables between calls.
-   **Correction**: Reactive (requires user/LLM interjection).
-   **Execution Model**: `LLM -> Execute Python Code -> LLM -> Execute Python Code ...`

## How to Run an Actor

First, ensure you have the necessary setup:

1.  **Environment Variables**: Create a `.env` file in the project root with your `OPENAI_API_KEY`. Optionally set `UNIFY_MODEL` to override the default LLM model.
2.  **Redis**: Make sure a Redis server is running on `localhost:6379`.
3.  **Dependencies**: Install all required packages (`pip install -r requirements.txt`).
4.  **Playwright**: Install automation binaries with `playwright install`.

### Run the HierarchicalActor

Create `run_hp.py` and execute it with `python run_hp.py`.

```python
# run_hp.py
import asyncio
import unify
from unity.actor.hierarchical_actor import HierarchicalActor
import logging
import sys
from dotenv import load_dotenv

load_dotenv()

# Setup basic logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)-8s [%(name)s] %(message)s"))
    root_logger.addHandler(handler)


async def main():
    """Initializes and runs the HierarchicalActor for a web task."""
    unify.activate("hp_demo")

    actor = HierarchicalActor(headless=True)

    task = "Go to google.com, search for 'latest news on AI agents', and return the title of the first result."
    print(f"Executing task: {task}")

    active_task = await actor.act(task)

    print("\n=== FINAL RESULT ===")
    print(await active_task.result())

    await actor.close()


if __name__ == "__main__":
    asyncio.run(main())
```
