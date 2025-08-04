Conversation Manager Sandbox
===========================

This folder contains an **interactive playground** for the `ConversationManager` component that lives in `unity/conversation_manager/`. The goal of the sandbox is to let you experiment with the manager in isolation – simulate user interactions (phone calls, sms, whatsapp, email), observe how events flow through the EventBus, prompt construction, STT/LLM/TTS chains, and tool loops before integrating into a larger system.

What is the `ConversationManager`?
-------------------------------
`ConversationManager` orchestrates real-time conversational flows, handling speech-to-text, LLM interactions, text-to-speech, and event dispatch (e.g. phone calls, SMS, email, WhatsApp). It wires up the shared `EventBus`, manages prompt builders, and auto-pins relevant events while streaming audio.

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

Local GUI usage (default)
-------------------------
When you run the sandbox with no flags (or with `--local`), a local Textual-based GUI opens. Use:

* Arrow keys or Tab to navigate the menu.
* Press Enter on a menu item to choose:
  - **Send SMS**: Enter a text message and press Enter to simulate an incoming SMS.
  - **Send WhatsApp**: Enter a text message and press Enter to simulate a WhatsApp message.
  - **Send Email**: Enter a text message and press Enter to simulate an email.
  - **Send Call**: Fill in task name, description, and purpose, then press **Call** to initiate a phone call; use **End Call** to stop.
  - **Quit**: Exit the sandbox.

Full comms mode (`--full`)
---------------------------
This mode starts a local LiveKit server and runs the sandbox without GUI, listening for incoming phone calls.

Prerequisites:
* Ensure you have a LiveKit server running locally (e.g., via Docker or `livekit-server --config livekit.yaml`).
* Set the `CALL_FROM_NUMBER` environment variable to the phone number you want the agent to answer.

Then start:
```bash
python -m sandboxes.conversation_manager.sandbox --full
```

Dial the agent at the configured `CALL_FROM_NUMBER`. The agent will answer and you can speak naturally; it handles STT→LLM→TTS over the call. Hang up to end the call.

Troubleshooting
---------------
Required environment variables (in addition to those in the root `README.md`):
* UNIFY_KEY
* USER_NAME
* USER_EMAIL
* USER_PHONE_NUMBER
* ASSISTANT_NAME
* ASSISTANT_NUMBER
* ASSISTANT_EMAIL
* OPENAI_API_KEY
