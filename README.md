<p align="center">
  <img src="https://raw.githubusercontent.com/unifyai/.github/main/public_images/unify_github_banner.png" alt="Unity" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
  <a href="https://github.com/unifyai/unity/actions"><img src="https://img.shields.io/github/actions/workflow/status/unifyai/unity/tests.yml?branch=staging&style=for-the-badge" alt="CI"></a>
  <a href="https://discord.com/invite/sXyFF8tDtm"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://unify.ai"><img src="https://img.shields.io/badge/Built%20by-Unify-black?style=for-the-badge" alt="Built by Unify"></a>
</p>

# Unity

Unity is the cognitive architecture behind [Unify's](https://unify.ai) persistent AI colleagues. It implements steerable nested execution, code-first planning, dual-brain voice, and distributed state managers, and runs in production as the agent runtime inside every assistant Unify hosts.

> **Demo:** [Launch video](https://youtu.be/qjSWiCd8Bq8?si=8eM0XnHH842_pbgo) • [Longer-form screenshares](https://youtube.com/playlist?list=PLwNuX3xB_tv-AsywAKYnGVv8X5AaEUc2Z)
>
> **Technical overview:** [ARCHITECTURE.md](ARCHITECTURE.md)

## What's open and what isn't

Unity is the **open core** of the Unify platform. This repository contains the full agent runtime — the managers, tool loops, CodeAct, dual-brain voice coordination, event backbone, memory consolidation. All MIT-licensed.

The persistence backend is also open-source: [Orchestra](https://github.com/unifyai/orchestra) (FastAPI + Postgres + pgvector) runs as a Docker container on your machine by default. The [Quick Start](#quick-start) `curl | bash` installer spins it up for you. **No Unify account required** to run the full open core locally.

**Not open-sourced** — the managed platform layer. External communication routing, the hosted communication edge (telephony, WhatsApp Business Solution Provider, Microsoft 365 tenant integration, SIP trunking), the assistant session control plane, the billing layer, and the identity layer run as part of the hosted service at [unify.ai](https://unify.ai). You can point the runtime at Unify's hosted Orchestra instead of a local one, but features that depend on the managed platform layer only work against the hosted backend.

If you're here to **study the runtime**, start with [ARCHITECTURE.md](ARCHITECTURE.md). If you're here to **run it**, the [Quick Start](#quick-start) gets you a full local install (runtime + Orchestra) in under 5 minutes.

## What the runtime does

<table>
<tr><td><b>Steerable nested execution</b></td><td>Every operation returns a live handle — pause, resume, interject, or query at any depth without restarting. Handles nest: steering at one level propagates through the whole tree.</td></tr>
<tr><td><b>Code plans, not tool menus</b></td><td>The Actor writes Python programs over typed primitives with variables, loops, and real control flow — not one JSON tool call at a time.</td></tr>
<tr><td><b>Dual-brain voice</b></td><td>A real-time voice process (sub-second latency) runs alongside a slower orchestration layer that continues tool use and planning in the background. They coordinate over IPC.</td></tr>
<tr><td><b>Distributed state managers</b></td><td>Contacts, knowledge, tasks, transcripts, guidance, files, and more — each owned by a specialized manager running its own async LLM tool loop, composed via English-language APIs.</td></tr>
<tr><td><b>Structured memory consolidation</b></td><td>Documents, screenshares, calls, tasks, and follow-up corrections get consolidated into typed, queryable state.</td></tr>
<tr><td><b>Concurrent steerable actions</b></td><td>Multiple tasks run at once. Each gets its own steering surface for inspection, interruption, and redirection.</td></tr>
<tr><td><b>Persistent identity across channels</b></td><td>Messages, SMS, email, phone calls, and meetings all update the same identity, memory, and task state.</td></tr>
</table>

## Quick Start

Get a fully local sandbox running in under 5 minutes. The runtime, the LLM client, and the persistence backend ([Orchestra](https://github.com/unifyai/orchestra), via Docker) all run on your machine. Hosted backend at [unify.ai](https://unify.ai) is an opt-in alternative.

### Prerequisites

- **Python 3.12+** (the installer will fetch it via `uv` if you don't have it)
- **Docker** (runs the local Orchestra backend — Postgres + pgvector in a container)
- **PortAudio** (audio support)
  - macOS: `brew install portaudio`
  - Ubuntu/Debian: `sudo apt-get install portaudio19-dev python3-dev`
- **An LLM provider key** — [OpenAI](https://platform.openai.com/api-keys) or [Anthropic](https://console.anthropic.com/) are the simplest paths from this README

> If you don't want a local Orchestra and would rather point at Unify's hosted backend, install with `--skip-setup` and fill in `UNIFY_KEY` / `ORCHESTRA_URL` manually.

### Install

One command:

```bash
curl -fsSL https://raw.githubusercontent.com/unifyai/unity/main/scripts/install.sh | bash
```

This clones `unity`, `unify`, `unillm`, and `orchestra` as siblings under `~/.unity/`, installs `uv` and `poetry` if you don't have them, syncs dependencies, drops a `unity` command into `~/.local/bin/`, then spins up a local Orchestra (Docker Postgres + FastAPI) and wires your `.env` with the local `UNIFY_KEY` and `ORCHESTRA_URL`. Works on macOS, Linux, and WSL2.

Flags: `--skip-setup` (don't spin up Orchestra — just install the code), `--no-cli` (don't install the `unity` shim), `--dir PATH` (install somewhere other than `~/.unity`), `--branch NAME`.

<details>
<summary>Manual install</summary>

```bash
git clone https://github.com/unifyai/unity.git      ~/.unity/unity
git clone https://github.com/unifyai/unify.git      ~/.unity/unify
git clone https://github.com/unifyai/unillm.git     ~/.unity/unillm
git clone https://github.com/unifyai/orchestra.git  ~/.unity/orchestra

cd ~/.unity/unity
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

cd ~/.unity/orchestra
poetry install
ORCHESTRA_INACTIVITY_TIMEOUT_SECONDS=0 scripts/local.sh start
# Copy the UNIFY_BASE_URL and UNIFY_KEY it prints into ~/.unity/unity/.env
```

</details>

### Configure

After `install.sh`, `~/.unity/unity/.env` already has `ORCHESTRA_URL` and `UNIFY_KEY` set to your local Orchestra. All you need to add is an LLM provider key:

```bash
OPENAI_API_KEY=sk-...        # or ANTHROPIC_API_KEY=...
```

`unillm` can also be pointed at other supported providers and compatible local endpoints.

### Run

```bash
unity --project_name Sandbox --overwrite
```

Other `unity` subcommands:
- `unity setup` — re-bootstrap local Orchestra (useful if Docker wasn't running the first time)
- `unity status` — show local Orchestra status
- `unity stop` — stop local Orchestra
- `unity restart` — stop + start (wipes DB)
- `unity help`

<details>
<summary>Without the CLI shim</summary>

```bash
cd ~/.unity/unity
source .venv/bin/activate
python -m sandboxes.conversation_manager.sandbox --project_name Sandbox --overwrite
```

</details>

At the configuration prompt, **select option 2** (CodeAct + Simulated Managers). This runs the full architecture: ConversationManager orchestrates CodeActActor, which writes and executes Python plans against the manager APIs, with simulated backends for the managers themselves.

Option 1 is a simpler view that shows ConversationManager's orchestration without CodeAct — useful to focus on the brain and steering layer in isolation.

### Interact

```
> msg Hey, can you help me organize my upcoming week?
> sms I need to reschedule my meeting with Sarah to Thursday
> email Project Update | Here are the Q3 numbers you asked for...
```

Commands: `msg` (Unify message), `sms`, `email`, `call`, `meet`. Type `help` for the full list.

### Going deeper

Option 3 at the configuration prompt adds a real computer interface (virtual desktop + browser via agent-service) on top of the CodeAct architecture. See [`sandboxes/conversation_manager/README.md`](sandboxes/conversation_manager/README.md) for the full matrix — voice mode, live voice calls, local comms, hosted comms, GUI mode.

---

## How it works

Every operation in Unity returns a **live handle** you can steer. These handles nest: the user steers the ConversationManager, the ConversationManager steers the Actor, the Actor steers the managers. Corrections, pauses, and queries propagate through the full depth.

In practice:
- "Also include Q2 numbers" mid-way through a report → the agent adjusts without restarting
- "Pause that, something urgent" → work freezes and resumes exactly where it left off
- "How's the flight search going?" → you get a status update without disrupting the work
- Three tasks running at once, each independently steerable

### Steerable handles

The universal return type. Every manager's `ask`, `update`, and `execute` methods return one.

```python
handle = await actor.act("Research flights to Tokyo and draft an itinerary")

# Twenty seconds later, while it's still working:
await handle.interject("Also check train options from Tokyo to Osaka")

# Or if something urgent comes up:
await handle.pause()
# ... deal with the urgent thing ...
await handle.resume()
```

When the Actor calls `primitives.contacts.ask(...)`, the ContactManager starts its own tool loop and returns its own handle — nested inside the Actor's handle, which is nested inside the ConversationManager's. Steering at any level propagates.

### CodeAct — the Actor writes programs

```python
contacts = await primitives.contacts.ask(
    "Who was involved in the Henderson project?"
)
for contact in contacts:
    history = await primitives.knowledge.ask(
        f"What was {contact} last working on?"
    )
    await primitives.contacts.update(
        f"Send {contact} a catch-up email referencing {history}"
    )
```

This runs in a sandboxed execution session with the full `primitives.*` API available — the same typed interfaces the rest of the system uses. One program per turn, with variables, loops, and real control flow. Contact lookup → knowledge retrieval → outbound communication becomes one plan, not three separate tool-selection turns.

### Dual-brain voice

**Slow brain** — the ConversationManager. Sees the full picture: all conversations, notifications, in-flight actions. Makes deliberate decisions. Runs in the main process.

**Fast brain** — a real-time voice agent on LiveKit, running as a separate subprocess. Sub-second latency. Handles the conversation autonomously.

They talk over IPC. When the slow brain wants to guide the conversation, it sends:
- **SPEAK** — "say exactly this" (bypasses the fast brain's LLM entirely)
- **NOTIFY** — "here's some context, decide what to do with it"
- **BLOCK** — nothing; the fast brain keeps going on its own

A speech urgency evaluator can preempt the slow brain when the user says something that needs immediate attention.

### Memory consolidation

Every 50 messages, the MemoryManager runs a background extraction pass. It pulls out:

- Contact profiles — who people are, their roles, relationships
- Per-contact summaries — what you've been discussing, sentiment, themes
- Response policies — how each person prefers to communicate
- Domain knowledge — project details, preferences, long-term facts
- Tasks — things you committed to, deadlines, follow-ups

Structured, queryable state in typed tables rather than freeform transcript summaries.

### Concurrent actions

```
┌─ In-Flight Actions ────────────────────────────────┐
│                                                     │
│  [0] research_flights  ██████████░░░  In progress   │
│      → ask, interject, stop, pause                  │
│                                                     │
│  [1] draft_summary     ████████████░  In progress   │
│      → ask, interject, stop, pause                  │
│                                                     │
│  [2] find_restaurants   ██░░░░░░░░░░  Starting      │
│      → ask, interject, stop, pause                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

Each action gets its own dynamically-generated steering tools. You can inspect, interject into, pause, resume, or stop one action without affecting the others.

---

## Architecture

```
ConversationManager (dual-brain orchestration, event-driven scheduling)
    │
    │   Slow Brain ◄── IPC ──► Fast Brain (real-time voice, LiveKit)
    │
    ▼
CodeActActor (generates Python plans, calls primitives.* APIs)
    │
    ▼
State Managers (each runs its own async LLM tool loop)
    │
    ├── ContactManager        — people and relationships
    ├── KnowledgeManager      — domain facts, structured knowledge
    ├── TaskScheduler         — durable tasks, execution with live handles
    ├── TranscriptManager     — conversation history and search
    ├── GuidanceManager       — procedures, SOPs, how-to knowledge
    ├── FileManager           — file parsing and registry
    ├── ImageManager          — image storage, vision queries
    ├── FunctionManager       — user-defined functions, primitives registry
    ├── WebSearcher           — web research orchestration
    ├── SecretManager         — encrypted secret storage
    ├── BlacklistManager      — blocked contact details
    └── DataManager           — low-level data operations
    │
    ├── EventBus              — typed pub/sub backbone (Pydantic events)
    └── MemoryManager         — offline consolidation every 50 messages
```

### How a request flows

1. User message arrives. The slow brain renders a full state snapshot and makes a single-shot tool decision.
2. It starts an action via `actor.act(...)` → gets back a `SteerableToolHandle`, registered in `in_flight_actions`.
3. The Actor generates a Python plan calling typed primitives. Each primitive dispatches to a manager running its own LLM tool loop, returning its own steerable handle.
4. Meanwhile, the slow brain can start more work, steer existing work, or guide the fast brain during voice calls.
5. The MemoryManager observes message events and periodically distills conversations into structured knowledge.
6. The EventBus carries typed events with hierarchy labels aligned to tool-loop lineage, making everything observable.

## Open-sourced alongside Unity

| Repo | Role |
|------|------|
| **unity** (this) | The agent runtime — managers, tool loops, CodeAct, voice, orchestration |
| **[orchestra](https://github.com/unifyai/orchestra)** | Persistence backend — FastAPI + Postgres + pgvector. Installer spins it up locally in Docker |
| **[unify](https://github.com/unifyai/unify)** | Python SDK — the client Unity uses to talk to Orchestra |
| **[unillm](https://github.com/unifyai/unillm)** | LLM access layer — OpenAI, Anthropic, or any compatible endpoint |

All MIT-licensed. The managed product layer — communication routing, telephony, the assistant session control plane, the web dashboard, billing, identity — runs on [Unify's platform](https://unify.ai) and is not part of this open core. You can point Unity at Unify's hosted Orchestra instead of a local one, but managed-service features only work against the hosted backend.

## Running the tests

Tests exercise the real system (steerable handles, CodeAct, manager composition, nested tool loops) against simulated backends with cached LLM responses:

```bash
uv sync --all-groups
source .venv/bin/activate

tests/parallel_run.sh tests/                    # everything
tests/parallel_run.sh tests/actor/              # one module
tests/parallel_run.sh tests/contact_manager/    # another
```

See [tests/README.md](tests/README.md) for the full philosophy — responses are cached, not mocked.

## Where to start reading

| File | What's there |
|------|-------------|
| `unity/common/async_tool_loop.py` | `SteerableToolHandle` — the protocol everything returns |
| `unity/common/_async_tool/loop.py` | The async tool loop engine — nesting, steering, context propagation |
| `unity/actor/code_act_actor.py` | CodeAct — plan generation, sandbox, primitives |
| `unity/conversation_manager/conversation_manager.py` | Dual-brain orchestration, debouncing, in-flight actions |
| `unity/conversation_manager/domains/brain_action_tools.py` | How the brain starts, steers, and tracks concurrent work |
| `unity/function_manager/primitives/registry.py` | How primitives are assembled into the typed API surface |
| `unity/events/event_bus.py` | Typed event backbone |
| `unity/memory_manager/memory_manager.py` | Offline consolidation pipeline |

## Project structure

```
unity/
├── unity/
│   ├── actor/                    # CodeActActor
│   ├── conversation_manager/     # Dual-brain orchestration
│   │   └── domains/              # Brain tools, action tracking, rendering
│   ├── common/
│   │   ├── async_tool_loop.py    # SteerableToolHandle
│   │   └── _async_tool/          # Tool loop internals
│   ├── contact_manager/
│   ├── knowledge_manager/
│   ├── task_scheduler/
│   ├── transcript_manager/
│   ├── guidance_manager/
│   ├── memory_manager/
│   ├── function_manager/
│   ├── file_manager/
│   ├── image_manager/
│   ├── web_searcher/
│   ├── secret_manager/
│   ├── events/
│   └── manager_registry.py
├── sandboxes/                    # Interactive playgrounds
│   └── conversation_manager/     # Full ConversationManager sandbox (start here)
├── tests/
├── agent-service/                # Node.js desktop/browser automation
└── deploy/                       # Dockerfile, Cloud Build, virtual desktop
```

## Design principles

**No regex or substring matching for routing user intent.** Everything goes through LLM reasoning, guided by prompts and tool docstrings. If the system handles something wrong, we fix the prompt, not add a hardcoded rule.

**No mocked LLMs in tests.** Every test uses real inference, cached for speed. Delete the cache and you're re-evaluating against live models.

**No defensive coding.** No try/except around things that shouldn't fail. No null checks for things that shouldn't be null. The system fails loud when assumptions break.

**English as an API.** Managers communicate through natural-language interfaces. The Actor orchestrates through English-language primitives. The whole system stays inspectable without reading implementation code.

---

## License

MIT — see [LICENSE](LICENSE).

Built by the team at [Unify](https://unify.ai).
