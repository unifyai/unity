# Actor Module

This directory contains different "actor" implementations, which are responsible for taking a high-level user goal and breaking it down into a series of actions to be executed.

## Available Actors

### CodeActActor

A conversational actor that uses a stateful code execution sandbox. It operates in a reactive, turn-based loop, maintaining a chat history with an LLM and executing Python code blocks to accomplish tasks.

-   **Plan Representation**: Implicit in conversation history.
-   **State Management**: Managed by the code execution sandbox which preserves variables between calls.
-   **Correction**: Reactive (requires user/LLM interjection).
-   **Execution Model**: `LLM -> Execute Python Code -> LLM -> Execute Python Code ...`

### SingleFunctionActor

A minimal actor that executes a single function or primitive. Useful for testing stored functions, deploying rigid pre-defined workflows, and executing action primitives (state manager methods) directly. Supports steerable forwarding when the executed function returns a `SteerableToolHandle`.

## How to Run an Actor

First, ensure you have the necessary setup:

1.  **Environment Variables**: Create a `.env` file in the project root with your `OPENAI_API_KEY`. Optionally set `UNIFY_MODEL` to override the default LLM model.
2.  **Dependencies**: Install all required packages (`pip install -r requirements.txt`).
4.  **Playwright**: Install automation binaries with `playwright install`.
