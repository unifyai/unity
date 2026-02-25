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

https://www.loom.com/share/44171c4c1aa2475abd539d1251e1baab

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
- **REPL**: prompts for Y/N confirmation before each outbound action
- **GUI**: auto-confirms (the GUI provides its own compose-and-send UX)
- Requires backend infrastructure + correct session/env configuration
- Use `--auto-confirm` in REPL mode to skip confirmation prompts

## Computer integration (Mode 3)

- The Magnitude agent runs in a **separate Chromium instance** (agent-service).
- The GUI “Computer” tab shows:
  - last known URL (best-effort)
  - recent computer actions (navigate/act/observe/query)

Relevant flags:
- `--agent-server-url http://localhost:3000`
- `--agent-mode web` (or `desktop`)
- `--headless` (launch Chromium headless)
- `--agent-service-bootstrap guide|auto` (help with setup; `auto` can install/build/start best-effort)

## Command reference (REPL + GUI command bar)

### Meta
- `help` / `h` / `?`: show help
- `quit` / `exit`: exit sandbox
- `reset`: clear sandbox + CM state (best-effort)
- `save_project` / `sp`: snapshot the Unify project
- `save_state [path]`: save structured state snapshot (logs, tree, traces) to JSON file
- `config`: switch actor configuration (restarts sandbox; state is reset)

### Event simulation (inbound → CM)
- `sms <message>`
- `email <subject> | <body>`
- `call`
- `say <text>` (during a call)
- `sayv` (during a call, requires `--voice`)
- `end_call`

During a call, any non-command text is treated as an utterance.

#### Meet interaction events (Unify Meet session simulation)

These simulate frontend events from a Unify Meet session. All are freely triggerable at any time (no active call required). An optional `[reason]` string is passed through to logs for debugging.

- `assistant_screen_share_start [reason]` — User enables viewing the assistant's desktop
- `assistant_screen_share_stop [reason]` — User disables viewing the assistant's desktop
- `user_screen_share_start [reason]` — User starts sharing their screen with the assistant
- `user_screen_share_stop [reason]` — User stops sharing their screen
- `user_webcam_start [reason]` — User enables their webcam
- `user_webcam_stop [reason]` — User disables their webcam
- `user_remote_control_start [reason]` — User takes remote control of the assistant's desktop
- `user_remote_control_stop [reason]` — User releases remote control

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

## Live voice mode (`--live-voice`)

Enables **real voice calls** through the sandbox. When `--live-voice` is active, the `call` command:

1. Creates a LiveKit room
2. Spawns the **production voice agent subprocess** (the same `call.py` used in production)
3. Bootstraps a local copy of the LiveKit Agents Playground (one-time; requires Node.js)
4. Opens the playground in your browser with URL + token embedded as query params (auto-connects)
5. Waits for a readiness signal (`UnifyMeetStarted`) before reporting the call as ready (with timeout fallback)

The full fast-brain (voice agent) + slow-brain (ConversationManager) loop runs exactly as it does in production. You talk through your browser, and the sandbox displays all events (utterances, call guidance, actor actions) in real-time.

```bash
# Start with live voice
python -m sandboxes.conversation_manager.sandbox --live-voice --project_name Sandbox --overwrite

# Then in the sandbox:
cm> call
#  => opens LiveKit Playground (best-effort) and prints readiness status
#  => speak to the assistant through your mic
cm> end_call
#  => tears down voice agent, room, and IPC socket
```

### Requirements

Requires voice-related env vars (`LIVEKIT_*`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY` or `ELEVEN_API_KEY`). See the **Voice / Live-Voice** section under Environment Variables for the full list.

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
- `save_state [path]`: save structured state snapshot to file (see below)

CLI:
- `--show-trace`: auto-print trace after each CodeAct code turn (REPL only)

### Structured State Snapshots (`save_state`)

The `save_state` command captures the current sandbox display state and saves it to a file. This is useful for:
- Debugging concurrent Actor calls
- Sharing session state with teammates
- Post-mortem analysis of complex runs

```bash
# Auto-generate filename with timestamp
save_state

# Specify custom path
save_state my_debug_session.json
```

This creates two files:
- **JSON file**: Machine-readable structured data (logs grouped by handle, event trees, traces)
- **Text file**: Human-readable formatted output similar to the GUI layout

The snapshot includes:
- **CM Logs**: Grouped by Actor handle ID
- **Actor Logs**: Grouped by Actor handle ID
- **Manager Logs**: Grouped by Actor handle ID (requires `EVENTBUS_PUBLISHING_ENABLED=true`)
- **Event Trees**: All execution trees with handle IDs
- **CodeAct Traces**: All traces grouped by Actor handle ID

## Environment Variables

All env vars are read from `.env` at the repo root. The tables below group them by feature; only "Core" is needed for basic simulated mode.

### Core (all modes)

| Variable | Required | Description |
|---|---|---|
| `UNIFY_KEY` | **Yes** | Unify API key (auth for Orchestra) |
| `ORCHESTRA_URL` | **Yes** | Orchestra API base URL (e.g. `http://localhost:8000/v0`) |
| `ORCHESTRA_ADMIN_KEY` | **Yes** (real-comms) | Admin key for comms service auth |
| `UNIFY_ENDPOINT` | No | Default LLM endpoint (e.g. `gpt-5@openai`). Falls back to unillm defaults |
| `OPENAI_API_KEY` | Depends | Required if using OpenAI models |
| `ANTHROPIC_API_KEY` | Depends | Required if using Anthropic models |

### Identity (user / assistant)

These populate `SESSION_DETAILS` and the boss contact record. Without them the sandbox falls back to placeholder values, which breaks real-comms.

| Variable | Default | Description |
|---|---|---|
| `USER_FIRST_NAME` | `"User"` | Boss's first name |
| `USER_SURNAME` | `""` | Boss's surname |
| `USER_NUMBER` | `"+15550001234"` | Boss's phone number — **must be set for real SMS** |
| `USER_EMAIL` | `"user@example.com"` | Boss's email — **must be set for real email** |
| `USER_ID` | (auto) | Boss's user ID |
| `ASSISTANT_FIRST_NAME` | `"Default"` | Assistant's first name |
| `ASSISTANT_SURNAME` | `""` | Assistant's surname |
| `ASSISTANT_NUMBER` | `"+10000000000"` | Assistant's outbound phone number (Twilio) — **must be set for real SMS/calls** |
| `ASSISTANT_EMAIL` | `"assistant@unify.ai"` | Assistant's outbound email address — **must be set for real email** |
| `ASSISTANT_ID` | (auto) | Assistant ID |
| `ASSISTANT_AGE` | `""` | Assistant's age (used in prompts) |

### Real-Comms mode (`--real-comms`)

| Variable | Required | Description |
|---|---|---|
| `UNITY_COMMS_URL` | **Yes** | Communication service URL (e.g. `https://unity-comms-app-staging-....run.app`) |
| `ORCHESTRA_ADMIN_KEY` | **Yes** | Admin key used by comms service for auth headers |
| `ASSISTANT_NUMBER` | **Yes** | Twilio-provisioned number for outbound SMS and calls |
| `ASSISTANT_EMAIL` | **Yes** | Email address for outbound email |
| `USER_NUMBER` | **Yes** | Boss's real phone number (SMS replies go here) |
| `USER_EMAIL` | **Yes** | Boss's real email (email replies go here) |

### Voice / Live-Voice (`--voice`, `--live-voice`)

| Variable | Required | Description |
|---|---|---|
| `LIVEKIT_URL` | **Yes** | LiveKit server URL (e.g. `wss://your-project.livekit.cloud`) |
| `LIVEKIT_API_KEY` | **Yes** | LiveKit API key |
| `LIVEKIT_API_SECRET` | **Yes** | LiveKit API secret |
| `LIVEKIT_SIP_URI` | For SIP calls | LiveKit SIP trunk URI |
| `DEEPGRAM_API_KEY` | **Yes** | Speech-to-text (Deepgram) |
| `CARTESIA_API_KEY` | Depends | Text-to-speech (Cartesia) — required if `VOICE_PROVIDER=cartesia` |
| `ELEVEN_API_KEY` | Depends | Text-to-speech (ElevenLabs) — required if `VOICE_PROVIDER=elevenlabs` |
| `VOICE_PROVIDER` | No | `cartesia`, `elevenlabs`, or `gpt-realtime` (default: `cartesia`) |
| `VOICE_ID` | No | Voice ID for the selected TTS provider |
| `VOICE_MODE` | No | `tts` or `sts` (speech-to-speech) |

### Debugging / Observability

| Variable | Default | Description |
|---|---|---|
| `EVENTBUS_PUBLISHING_ENABLED` | `false` | **Required for Event Tree and Manager Logs** in the GUI. Without this, Event Tree is empty and Manager Logs show "(no logs)" |
| `DEBUG_TOOL_RESULTS` | `false` | Log full tool call results |
| `DEBUG_LLM_TURN` | `false` | Log each LLM turn |
| `LLM_IO_DEBUG` | `false` | Log raw LLM request/response I/O |
| `STAGING` | `false` | Enable staging-mode behaviors |

### Mode 3 (agent-service / Magnitude)

| Variable | Required | Description |
|---|---|---|
| `UNIFY_KEY` | **Yes** | Used by agent-service for auth |
| `ORCHESTRA_URL` | **Yes** | Agent-service connects to Orchestra |

### Vertex AI (optional provider)

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | Depends | Path to service account JSON |
| `VERTEXAI_PROJECT` | Depends | GCP project ID |
| `VERTEXAI_LOCATION` | Depends | GCP region (e.g. `europe-west1`) |

### Example `.env` (real-comms + live-voice)

```bash
# Core
UNIFY_KEY=your-unify-key
ORCHESTRA_URL=http://localhost:8000/v0
ORCHESTRA_ADMIN_KEY=your-admin-key

# LLM
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
UNIFY_ENDPOINT=gpt-5@openai

# Identity
USER_FIRST_NAME=Yusha
USER_SURNAME=Arif
USER_NUMBER=+19294608302
USER_EMAIL=yusha@unify.ai
ASSISTANT_FIRST_NAME=Liz
ASSISTANT_SURNAME=
ASSISTANT_NUMBER=+19134048493
ASSISTANT_EMAIL=default-assistant-4@unify.ai
ASSISTANT_ID=4

# Comms
UNITY_COMMS_URL=https://unity-comms-app-staging-....run.app

# Voice
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-key
LIVEKIT_API_SECRET=your-secret
DEEPGRAM_API_KEY=your-key
CARTESIA_API_KEY=your-key
VOICE_PROVIDER=cartesia

# Debugging
EVENTBUS_PUBLISHING_ENABLED=true
DEBUG_TOOL_RESULTS=true
DEBUG_LLM_TURN=true
```

## Troubleshooting

### Event Tree is empty / Manager Logs show "(no logs)"
Set `EVENTBUS_PUBLISHING_ENABLED=true` (see Debugging / Observability in the Environment Variables section above).

### SMS replies fail with "Failed to send sms to +15550001234"
The placeholder number means `USER_NUMBER` is not set in `.env`. Set it to the boss's real phone number (see Identity section above).

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

If you’re on a fresh install and don’t have Magnitude set up yet:
- See `sandboxes/actor/README.md` → “Magnitude Agent Service Setup” (step-by-step)
- The sandbox can also print setup instructions (default) and can *attempt* auto-bootstrap with:
  - `--agent-service-bootstrap auto`

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
