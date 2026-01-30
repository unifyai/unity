ConversationManager Sandbox
==========================

This folder contains an **interactive playground** for the `ConversationManager` component (`unity/conversation_manager/`). The sandbox is designed to let you test the CM “brain” locally, in isolation, with simulated comms by default — plus an optional **real-comms mode** with explicit safety confirmations.

## Quick start

```bash
# REPL (default) — prompts for actor config; no external infrastructure needed for mode 1/2
python -m sandboxes.conversation_manager.sandbox --project_name Sandbox --overwrite

# REPL + voice (optional) — enables `sayv` (voice phone utterances) and TTS for phone responses
python -m sandboxes.conversation_manager.sandbox --voice --project_name Sandbox --overwrite

# GUI (optional) — Textual, same process/event loop as CM
python -m sandboxes.conversation_manager.sandbox --gui --project_name Sandbox --overwrite

# Real-comms (optional, requires infra) — REPL only
python -m sandboxes.conversation_manager.sandbox --real-comms --project_name Sandbox --overwrite
```

## Loom walkthrough

See the recorded walkthrough here: [Loom demo](https://www.loom.com/share/44171c4c1aa2475abd539d1251e1baab).

## Actor configurations (modes)

On startup, the sandbox prompts you to select one of three configurations (and remembers the last-used choice in a project-local file):

- **Mode 1 — `SandboxSimulatedActor`**: simulated managers, **no computer interface**
- **Mode 2 — `CodeActActor + simulated managers`**: mock computer backend (no agent-service)
- **Mode 3 — `CodeActActor + real managers + real computer interface`**: uses **agent-service** (Magnitude) + real state managers

Configuration persistence:
- Saved to **`.cm_sandbox_config`** in the repo root (gitignored).
- You can switch configs at runtime via the `config` command (restarts sandbox).

### Real-comms mode (`--real-comms`)
- Comms are **real** (SMS/email/calls) via `CommsManager`
- Sandbox applies a **confirmation prompt** before any outbound action
- Requires backend infrastructure + correct session/env configuration
- **REPL only** (GUI is simulated-only)

## Computer integration (Mode 3)

- The Magnitude agent runs in a **separate Chromium instance** (agent-service).
- The GUI “Computer” tab shows:
  - last known URL (best-effort)
  - recent computer actions (navigate/act/observe/query)

Relevant flags:
- `--agent-server-url http://localhost:3000`
- `--agent-mode web` (or `desktop`)
- `--headless` (launch Chromium headless)

## Command reference (REPL + GUI command bar)

### Meta
- `help` / `h` / `?`: show help
- `quit` / `exit`: exit sandbox
- `reset`: clear sandbox + CM state (best-effort)
- `save_project` / `sp`: snapshot the Unify project
- `config`: switch actor configuration (restarts sandbox; state is reset)

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

## Trace / tree / logs (CodeAct UX helpers)

These are intended for debugging and “execution visibility” in Mode 2/3:

- `trace [N]`: show last N CodeAct execution turns (default 3)
- `tree`: show the current manager call hierarchy (EventBus `ManagerMethod` events)
- `show_logs <cm|actor|manager|all>` / `collapse_logs <...>`: expand/collapse log categories

CLI:
- `--show-trace`: auto-print trace after each CodeAct code turn (REPL only)

## Troubleshooting

### “(no active conversation) Steering commands…”
Steering commands only work while CM is processing (active handle or brain run in-flight). Send an event (`sms ...`, `call`, etc.) first.

### “Scenario seeding is disabled while active”
Scenario seeding is idle-only. Use `/stop` or wait for the active action to complete.

### Real-comms mode fails to start
Real-comms requires backend infrastructure and correct env/session configuration. Check your `.env` / `SESSION_DETAILS` settings and comms deployment.

### Mode 3 fails validation (“agent-service is not running or unreachable”)
Mode 3 requires:
- `agent-service` running and reachable at `--agent-server-url`
- `UNIFY_KEY` set (agent-service uses it for auth)

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

## Example: CodeAct + real managers + browser (GUI)

Start the sandbox in GUI mode with agent-service wired up:

```bash
python -m sandboxes.conversation_manager.sandbox \
  --gui \
  --voice \
  --project_name Sandbox \
  --overwrite \
  --agent-server-url http://localhost:3000 \
  --agent-mode web
```

Then, in the command bar, try an end-to-end request (Mode 3):

```text
sms Can you find OpenAI's careers page, check if there’s a “Backend Engineer” role open, and if so create a task for me called “Apply to OpenAI” with the role URL in the description?
```

Tip: add `--headless` if you don’t want a visible Chromium window.
