Conversation Manager Sandbox
===========================

This folder contains an **interactive playground** for the `ConversationManager` component that lives in `unity/conversation_manager/`. The goal of the sandbox is to let you experiment with the manager in isolation – simulate user interactions (phone calls, Meet sessions), observe how events flow through the EventBus, prompt construction, STT/LLM/TTS chains, and tool loops before integrating into a larger system.

What is the `ConversationManager`?
-------------------------------
`ConversationManager` orchestrates real-time conversational flows, handling speech-to-text, LLM interactions, text-to-speech, and event dispatch (e.g. phone calls, SMS, email, WhatsApp, Google Meet). It wires up the shared `EventBus`, manages prompt builders, and auto-pins relevant events while streaming audio.

Running the sandbox
-------------------
The entry-point lives at `sandboxes/conversation_manager/sandbox.py` and can be executed directly or via Python’s `-m` switch:

```bash
# Default (local GUI mode)
python -m sandboxes.conversation_manager.sandbox

# Full non-GUI mode (real comms only)
python -m sandboxes.conversation_manager.sandbox --full

# Specify which tools to enable (choices: conductor, contact, transcript, knowledge, scheduler, comms)
python -m sandboxes.conversation_manager.sandbox --enabled-tools comms,conductor
```

CLI flags
~~~~~~~~~
* `--local` (default): Enable local GUI mode.
* `--full`         : Disable local GUI mode (real comms and no GUI).
* `--enabled-tools`: Comma-separated list of enabled tools (choices: conductor, contact, transcript, knowledge, scheduler, comms). Default: None (all tools enabled).

Interactive commands inside the REPL
-----------------------------------
Once the sandbox starts you will see a prompt. Any free-form text you type is routed as a user utterance to the `ConversationManager`, driving the STT→LLM→TTS flow and event dispatch. Use:

* `<free text>`: Send a user message and observe the system’s response.
* `help`         : Show available commands.
* `quit`         : Exit the sandbox.

Troubleshooting
---------------
* **OpenAI key**: Ensure `OPENAI_API_KEY` is set for LLM calls.
* **LiveKit credentials**: If using Meet or phone-call sandbox, set `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`.
* **TTS provider keys**: For voice I/O, set your `CARTESIA_API_KEY` or appropriate TTS provider credentials.
