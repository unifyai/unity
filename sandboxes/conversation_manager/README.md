Conversation Manager Sandbox
===========================

This folder contains an **interactive playground** for the `ConversationManager` component that lives in `unity/conversation_manager/`. The goal of the sandbox is to let you experiment with the manager in isolation – simulate user interactions (phone calls, sms, whatsapp, email), observe how events flow through the EventBus, prompt construction, STT/LLM/TTS chains, and tool loops before integrating into a larger system.

Prefer a quick demo? Watch this [video walkthrough](https://www.loom.com/share/f8a87d725e074eaa960c5021164dc3cd?sid=98fc555f-5ca4-47f0-838c-50eadba08d48)

For understanding how natural conversations are, users can also be simulated. Watch this [video walkthrough](https://www.loom.com/share/9a2f1c0655af46c883bf5edb19f90dca?sid=74b9b39d-88e0-4748-b036-b4577144e84e)

This [video walkthrough](https://www.loom.com/share/608a025efdb64e82b48ba3e03b34a055?sid=a1879903-2ec3-46fa-aa67-9555aa132093) also gives a general overview on how assistant is capable of joining Google Meet sessions.

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
python -m sandboxes.conversation_manager.sandbox --enabled_tools comms,conductor
```

CLI flags
~~~~~~~~~
Run `python -m sandboxes.conversation_manager.sandbox --help` to see additional flags:
* `--local` (default): Enable local GUI mode.
* `--full`         : Disable local GUI mode (real comms and no GUI).
* `--enabled_tools`: Comma-separated list of enabled tools (choices: conductor, contact, transcript, knowledge, scheduler, comms). Default: None (all tools enabled).

Standard flags:
* `--voice` / `-v`            – enable voice input/output (scenario seeding and TTS)
* `--project_name` / `-p`     – Unify project/context name
* `--overwrite` / `-o`        – delete existing data for the project before start
* `--project_version`         – load a specific saved version (index)
* `--traced` / `-t`           – wrap manager calls with Unify tracing
* `--debug` / `-d`            – show verbose tool logs (reasoning steps)

Local GUI usage (default)
-------------------------
When you run the sandbox with no flags (or with `--local`), a local Textual-based GUI opens  in a separate terminal. Use:

* Arrow keys or Tab to navigate the menu.
* Press Enter on a menu item to choose:
  - **Send SMS**: Enter a text message and press Enter to simulate an incoming SMS.
  - **Send WhatsApp**: Enter a text message and press Enter to simulate a WhatsApp message.
  - **Send Email**: Enter a text message and press Enter to simulate an email.
  - **Send Call**: Fill in task name, description, and purpose, then press **Call** to initiate a phone call; use **End Call** to stop.
  - **Quit**: Exit the sandbox.

Full comms mode (`--full`)
---------------------------
In this mode the ConversationManager service runs without the GUI, handling real incoming SMS, WhatsApp, Email, and phone calls. It starts up a LiveKit server locally, thus no scenario seeding in this mode.

To start:
```bash
python -m sandboxes.conversation_manager.sandbox --full
```

Real incoming events and calls are processed by the live service; phone calls require a running LiveKit server and properly configured environment variables.

Two-agent simulation mode
-------------------------
Alternatively, you can simulate two agents (user & assistant) interacting in-process without spinning up LiveKit or handling real events.

Entry-point: `sandboxes/conversation_manager/simulated.py`

To start the simulation:
```bash
python -m sandboxes.conversation_manager.simulated [--voice] [--project_name NAME] [--overwrite] [--num_turns N]
```

Accepted commands in simulation:
- `start` / `s`     : begin the first user turn.
- `continue` / `c`  : run the next round of back-and-forth (default N turns).
- `medium` / `m`    : change communication medium (`phone`, `sms`, `email`), resets history.
- `help` / `h`      : show this help menu.
- `exit` / `quit`   : terminate the simulation.

CLI flags for simulation (in addition to sandbox flags above):
- `--num_turns` / `-n` : number of back-and-forth turns per cycle (default: 5).

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
