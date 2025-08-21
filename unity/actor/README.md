# Actor Module

This directory contains different "actor" implementations, which are responsible for taking a high-level user goal and breaking it down into a series of actions to be executed. Each actor offers a different approach to task decomposition and execution.

## Actor Design Philosophies

There are two primary design philosophies represented here: **Conversational Actors** and a **Programmatic Actor**.

### 1. Conversational Actors (`ToolLoopActor`, `BrowserUseActor`)

These actors operate in a reactive, turn-based loop. They maintain a chat history with an LLM and, at each step, decide which "tool" (a Python function) to call next based on the user's goal and the history of previous actions.

-   **Plan Representation**: The "plan" is implicit and exists only as the conversation history. There is no persistent, machine-readable plan object.
-   **State Management**: State is managed by the sequence of tool calls and their results in the chat history.
-   **Correction**: To correct a mistake, one typically "interjects" with a message like "That was wrong, try doing this instead," which influences the LLM's next tool choice.
-   **Execution Model**: `LLM -> Tool Call -> LLM -> Tool Call ...`

### 2. Programmatic Actor (`HierarchicalActor`)

This actor takes a more proactive, code-first approach. It generates a complete, executable Python script that represents the entire plan. This script is then executed, and it has the ability to debug, verify, and modify itself at runtime.

-   **Plan Representation**: The plan is an explicit, well-structured Python script containing functions decorated with `@verify`.
-   **State Management**: State is managed explicitly within the Python execution context, including a function call stack.
-   **Correction**: Self-correction is built-in. The `@verify` decorator checks the outcome of every step. Failures can trigger tactical replans (rewriting a single function) or strategic replans (escalating to the parent function to rethink the approach).
-   **Execution Model**: `Generate Code -> Execute Code -> (Verify -> Self-Correct/Re-implement -> Resume Execution)`

## Key Differences

| Feature                 | ToolLoopActor (TLP)                                      | BrowserUseActor (BUP)                                  | HierarchicalActor (HP)                                                                                              |
| ----------------------- | ---------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Planning Paradigm** | Conversational / Tool-Use                                  | Conversational / Tool-Use                                | Programmatic / Code-First                                                                                             |
| **Underlying Controller** | `unity.controller.Controller`                              | `browser_use.controller.Service`                         | `unity.controller.Controller`                                                                                         |
| **Action Primitives** | High-level `act(str)` and `observe(str)` primitives.       | A rich set of structured tools from the `browser_use` library (e.g., `Maps_to_url`, `click_element`). | High-level `act(str)` and `observe(str)` primitives, same as TLP.                                                   |
| **Plan Representation** | Implicit (Chat History)                                    | Implicit (Chat History)                                  | Explicit (Executable Python Script)                                                                                   |
| **State Management** | Implicit in conversation.                                  | Implicit in conversation.                                | Explicit via Python call stack and cached function results.                                                           |
| **Self-Correction** | Reactive (requires user/LLM interjection).                 | Reactive (requires user/LLM interjection).               | Proactive (Built-in verification, tactical/strategic replanning, and dynamic implementation of stubbed functions). |
| **Modifiability** | Can be steered by interjection.                            | Can be steered by interjection.                          | The entire plan source code can be surgically modified at runtime, with automated "course correction."              |

## How to Run a Actor

First, ensure you have the necessary setup:

1.  **Environment Variables**: Create a `.env` file in the project root with your `OPENAI_API_KEY` and desired `UNIFY_MODEL`.
2.  **Redis**: Make sure a Redis server is running on `localhost:6379`.
3.  **Dependencies**: Install all required packages (`pip install -r requirements.txt`).
4.  **Playwright**: Install browser binaries with `playwright install`.

You can run any of the actors using the simple scripts below.

### 1. Run the HierarchicalActor (HP)

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
    """Initializes and runs the HierarchicalActor for a browser task."""
    unify.activate("hp_demo")

    actor = HierarchicalActor(
        headless=True,
    )

    DEMO_TASK_BROWSER = "Go to google.com, search for 'latest news on AI agents', and return the title of the first result."
    DEMO_TASK_EMAIL = "Send an email to contact ID 42 letting them know that their invoice is ready and a payment is due."
    DEMO_TASK_PHONE = "Call contact ID 77 and ask them what they would like for dinner. The options are pizza, pasta, or sushi. After they choose, ask if they want it delivered right away."

    print(f"Executing task: {task}")

    active_task = await actor.execute(task)

    print("\n=== FINAL RESULT ===")
    print(await active_task.result())

    await actor.close()
    controller.join(timeout=2)


if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Run the ToolLoopActor (TLP)

Create `run_tlp.py` and execute it with `python run_tlp.py`.

```python
# run_tlp.py
import asyncio
import unify
from unity.actor.tool_loop_actor import ToolLoopActor
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
    """Initializes and runs the ToolLoopActor for a browser task."""
    unify.activate("tlp_demo")

    actor = ToolLoopActor(headless=False)

    task = "Go to google.com, search for 'latest news on AI agents', and return the title of the first result."
    print(f"Executing task: {task}")

    active_task = await actor.execute(task)

    print("\n=== FINAL RESULT ===")
    print(await active_task.result())

    await actor.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Run the BrowserUseActor (BUP)

Create `run_bup.py` and execute it with `python run_bup.py`.

```python
# run_bup.py
import asyncio
import unify
from unity.actor.browser_use_actor import BrowserUseActor
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
    """Initializes and runs the BrowserUseActor for a browser task."""
    unify.activate("bup_demo")

    actor = BrowserUseActor(headless=False)

    task = "Go to google.com, search for 'latest news on AI agents', and return the title of the first result."
    print(f"Executing task: {task}")

    active_task = await actor.execute(task)

    print("\n=== FINAL RESULT ===")
    print(await active_task.result())

    await actor.close()


if __name__ == "__main__":
    asyncio.run(main())
```
