ConversationManager Sandbox
==========================

This folder contains an **interactive playground** for the `ConversationManager` component (`unity/conversation_manager/`). The sandbox is designed to let you test the CM “brain” locally, in isolation, with simulated comms by default — plus an optional **real-comms mode** with explicit safety confirmations.

## Quick start

```bash
# REPL (default) — simulated comms, SimulatedActor, no external infrastructure needed
python -m sandboxes.conversation_manager.sandbox --project_name Sandbox --overwrite

# REPL + voice (optional) — enables `sayv` (voice phone utterances) and TTS for phone responses
python -m sandboxes.conversation_manager.sandbox --voice --project_name Sandbox --overwrite

# GUI (optional) — Textual, same process/event loop as CM
python -m sandboxes.conversation_manager.sandbox --gui --project_name Sandbox --overwrite

# Real-comms (optional, requires infra) — REPL only
python -m sandboxes.conversation_manager.sandbox --real-comms --project_name Sandbox --overwrite
```

## Modes

### Simulated mode (default)
- No real SMS/emails/calls
- Uses `SimulatedActor` (no real Actor/browser)
- No external comms infrastructure required

### Real-comms mode (`--real-comms`)
- Comms are **real** (SMS/email/calls) via `CommsManager`
- Sandbox applies a **confirmation prompt** before any outbound action
- Requires backend infrastructure + correct session/env configuration
- **REPL only** (GUI is simulated-only)

## Command reference (REPL + GUI command bar)

### Meta
- `help` / `h` / `?`: show help
- `quit` / `exit`: exit sandbox
- `reset`: clear sandbox + CM state (best-effort)
- `save_project` / `sp`: snapshot the Unify project

### Event simulation (inbound → CM)
- `sms <message>`
- `email <subject> | <body>`
- `call`
- `say <text>` (during a call)
- `sayv` (during a call, requires `--voice`)
- `end_call`

During a call, any non-command text is treated as an utterance.

### Steering (only while active)
Steering is available whenever **either**:
- an Actor handle exists (full steering), **or**
- a brain run is in-flight (best-effort steering)

Commands:
- `/pause`
- `/resume`
- `/i <msg>`
- `/ask <q>`
- `/stop [reason]`

**Actor handle mode**: forwards to `SteerableToolHandle` methods.

**Brain-run mode (best-effort)**:
- `/pause` queues events until `/resume`
- `/resume` flushes queued events
- `/i <msg>` publishes an inbound event to trigger a fresh brain run
- `/ask <q>` prints a state snapshot
- `/stop` returns to idle immediately (does not cancel mid-generation)

### Scenario seeding (idle-only)
- `us <description>`: generate a synthetic transcript and publish inbound events into CM
- `usv`: voice scenario seeding (requires `--voice`)

Scenario seeding is disabled while active; use `/stop` or wait.

## Voice mode (`--voice`)

When enabled:
- `sayv` records microphone audio, transcribes via Deepgram (STT), and sends the transcript as a phone utterance.
- Assistant phone-call responses (`[Phone → User] ...`) are also spoken via TTS (Cartesia) on a best-effort basis.

## Real-comms safety confirmations

In `--real-comms` mode, outbound actions are intercepted and require confirmation (default: **N**):
- SMS (`send_sms_message_via_number`)
- Email (`send_email_via_address`)
- Unify message (`send_unify_message`)
- Phone call (`start_call`)

You can bypass prompts with `--auto-confirm` (dangerous; use only for controlled testing).

## Examples

See `sandboxes/conversation_manager/examples.md`.

## Troubleshooting

### “(no active conversation) Steering commands…”
Steering commands only work while CM is processing (active handle or brain run in-flight). Send an event (`sms ...`, `call`, etc.) first.

### “Scenario seeding is disabled while active”
Scenario seeding is idle-only. Use `/stop` or wait for the active action to complete.

### Real-comms mode fails to start
Real-comms requires backend infrastructure and correct env/session configuration. Check your `.env` / `SESSION_DETAILS` settings and comms deployment.

### “Provider List: https://docs.litellm.ai/docs/providers”
This is emitted by Litellm when provider config is missing/mismatched. Ensure your LLM credentials/config are set for local runs.

## Other entrypoints

**Alternate (GUI-only module)**:
```bash
python -m sandboxes.conversation_manager.gui
```

**Recommended (GUI mode)**:
```bash
python -m sandboxes.conversation_manager.sandbox --gui
```

This sandbox provides:
- A unified entrypoint (`sandbox.py`)
- REPL-first UX + optional in-process GUI
- Dual-mode steering (Actor handle + brain-run best-effort)
- Scenario seeding (`us`, `usv`)
- Real-comms mode with safety prompts (`--real-comms`)
