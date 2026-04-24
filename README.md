<p align="center">
  <img src="https://raw.githubusercontent.com/unifyai/.github/main/public_images/unify_github_banner.png" alt="Unity" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License"></a>
  <a href="https://docs.unify.ai/basics/overview"><img src="https://img.shields.io/badge/Docs-docs.unify.ai-4A67FF?style=for-the-badge" alt="Docs"></a>
  <a href="https://github.com/unifyai/unity/actions"><img src="https://img.shields.io/github/actions/workflow/status/unifyai/unity/tests.yml?branch=staging&style=for-the-badge" alt="CI"></a>
  <a href="https://discord.com/invite/sXyFF8tDtm"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://unify.ai"><img src="https://img.shields.io/badge/Built%20by-Unify-black?style=for-the-badge" alt="Built by Unify"></a>
</p>

# Unity

Unity is an open-source agent behind [Unify's](https://unify.ai) persistent AI colleagues. It is built for assistants you can onboard like teammates, interrupt and redirect at any depth, talk to in real time, run fully locally by default, and grow over time through typed, long-lived memory.

Instead of one flat tool loop, Unity separates live conversation from action execution. The `ConversationManager` owns the realtime interaction layer and voice orchestration above the `Actor`, which writes code-first plans. Specialized managers own typed state for contacts, transcripts, knowledge, tasks, functions, guidance, secrets, files, images, and more. That state lives in Postgres with pgvector support via [Orchestra](https://github.com/unifyai/orchestra), so memory keeps getting richer without collapsing into one opaque prompt thread. Agents in the same Hive can also share one typed memory layer across all of those managers, so teams learn together instead of starting from scratch body by body.

> **Start here:** [Overview](https://docs.unify.ai/basics/overview) • [Quickstart](https://docs.unify.ai/basics/quickstart) • [Demos](https://docs.unify.ai/basics/demos) • [ARCHITECTURE.md](ARCHITECTURE.md)

## What Makes Unity Different

- **`ConversationManager` stays above the `Actor`.** Live conversation, voice, and in-flight coordination stay separate from code-first action execution.
- **Memory is typed and self-evolving.** Contacts, transcripts, knowledge, tasks, functions, guidance, secrets, files, and images live in dedicated managers backed by Postgres + pgvector.
- **Teams can share a Hive mind.** Agents in the same Hive build up the same typed memory across state managers, so a team of agents can learn together instead of each one relearning the same context alone.
- **Steering is recursive and bidirectional.** Top-down control (`ask`, `interject`, `pause`, `resume`, `stop`) propagates downward, while bottom-up `next_notification` and `next_clarification` flow back upward.
- **Voice is first-class.** A fast realtime process handles sub-second conversation while a slower orchestration layer keeps planning, using tools, and steering concurrent work.
- **It is easy to try.** Unity starts from a one-command install, takes seconds to kick off, and runs fully locally by default.

Unity is designed to feel less like "prompt, then execute" and more like working with a colleague that can stay in the loop, ask for clarification, be redirected mid-flight, and continuously consolidate experience into reusable structure.

## Core Capabilities

<table>
<tr><td><b>Typed, self-evolving memory</b></td><td>Contacts, transcripts, knowledge, tasks, functions, guidance, secrets, files, images, and more live in dedicated typed stores, backed by Postgres + pgvector through Orchestra rather than one freeform memory blob.</td></tr>
<tr><td><b>Hive-shared team memory</b></td><td>Agents in the same Hive share one typed, self-evolving memory layer across state managers, so teams can coordinate around the same growing knowledge instead of maintaining separate memories per agent.</td></tr>
<tr><td><b>Fully nested steering</b></td><td>Every operation returns a live handle. Top-down control (<code>ask</code>, <code>interject</code>, <code>pause</code>, <code>resume</code>, <code>stop</code>) propagates downward through the full tree, while bottom-up notifications and clarification requests flow back upward.</td></tr>
<tr><td><b>Clear realtime boundary</b></td><td><code>ConversationManager</code> owns live conversation, voice, and in-flight coordination above the <code>Actor</code>, which focuses on code-first execution over typed primitives.</td></tr>
<tr><td><b>Realtime voice, not just chat</b></td><td>A fast voice process handles sub-second conversation while a slower orchestration layer keeps planning, using tools, and steering concurrent work in the background.</td></tr>
<tr><td><b>Code-first execution</b></td><td>The Actor writes Python programs with variables, loops, and real control flow instead of emitting one JSON tool call at a time.</td></tr>
<tr><td><b>Distributed state managers</b></td><td>Each domain manager owns its slice of state and runs its own async LLM tool loop, so memory and capabilities stay inspectable instead of melting into one monolithic agent.</td></tr>
<tr><td><b>Concurrent multi-task execution</b></td><td>Multiple actions can run at once, each with its own steering surface, notification flow, and transcript lineage.</td></tr>
<tr><td><b>Computer-native workflows</b></td><td>Unity can layer desktop and browser control on top of the same steering, memory, and orchestration model rather than treating GUI work as a separate subsystem.</td></tr>
</table>

## Quick Start

By default, Unity's open-core quickstart is fully local: the agent, the LLM client, and the persistence backend ([Orchestra](https://github.com/unifyai/orchestra), via Docker) all run on your machine. Hosted backend at [unify.ai](https://unify.ai) is optional.

### Fastest path

Prerequisites:

- **Python 3.12+** (the installer will fetch it with `uv` if needed)
- **Docker** (runs the local Orchestra backend)
- **PortAudio** for audio support
  - macOS: `brew install portaudio`
  - Ubuntu/Debian: `sudo apt-get install portaudio19-dev python3-dev`
- **One LLM provider key** — [OpenAI](https://platform.openai.com/api-keys) or [Anthropic](https://console.anthropic.com/) are the simplest paths from this README

Install:

```bash
curl -fsSL https://raw.githubusercontent.com/unifyai/unity/main/scripts/install.sh | bash
```

The install is a single shell command and takes seconds to kick off. On first run, Unity then clones `unity`, `unify`, `unillm`, and `orchestra` as siblings under `~/.unity/`, installs dependencies, creates a `unity` CLI shim in `~/.local/bin/`, boots a local Orchestra in Docker, and writes the local `UNIFY_KEY` / `ORCHESTRA_URL` into `~/.unity/unity/.env`.

Then add one model provider key to `~/.unity/unity/.env`:

```bash
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=...
```

Run the sandbox:

```bash
unity --project_name Sandbox --overwrite
```

At the configuration prompt:

| Option | What it gives you |
|------|------|
| `1` | ConversationManager orchestration without CodeAct — useful if you want to isolate the top-level brain |
| `2` | The full agent: ConversationManager + CodeAct + simulated managers |
| `3` | Option 2 plus desktop/browser control through `agent-service` |

If you're evaluating Unity as an agent, start with **option 2**.

Example session:

```text
> msg Hey, can you help me organize my upcoming week?
> sms I need to reschedule my meeting with Sarah to Thursday
> email Project Update | Here are the Q3 numbers you asked for...
```

Other `unity` subcommands:

- `unity setup` — re-bootstrap local Orchestra
- `unity status` — show local Orchestra status
- `unity stop` — stop local Orchestra
- `unity restart` — stop + start (wipes the local DB)
- `unity help`

### Full local open-core path

The local open-core path is the default. You do **not** need a Unify account to run Unity locally with Orchestra in Docker.

If you want to install the code without starting a local Orchestra, use:

```bash
curl -fsSL https://raw.githubusercontent.com/unifyai/unity/main/scripts/install.sh | bash -s -- --skip-setup
```

That leaves the code installed but expects you to point Unity at either:

- your own Orchestra deployment via `ORCHESTRA_URL`, or
- Unify's hosted backend via `UNIFY_KEY` + `ORCHESTRA_URL`

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
# Copy the ORCHESTRA_URL and UNIFY_KEY it prints into ~/.unity/unity/.env
```

</details>

### Configuration files

The installer copies `.env.example` to `.env`. That file is intentionally minimal for the public quickstart.

- Use `.env.example` if you want the smallest working sandbox config.
- Use `.env.advanced.example` if you want local comms, hosted comms, LiveKit, Tavily, voice integrations, visual caching, or test-infra settings.

For the full sandbox matrix — voice mode, live voice calls, local comms, hosted comms, and GUI mode — see [`sandboxes/conversation_manager/README.md`](sandboxes/conversation_manager/README.md).

## Migrating from Hermes Agent or OpenClaw

Unity does not have a one-shot importer yet. The clean migration path is to move data into Unity's native surfaces instead of trying to preserve another agent's prompt layout verbatim.

The rough mapping is:

- long-lived people data → `ContactManager`
- past conversations and raw chat history → `TranscriptManager`
- durable facts and semantic memory → `KnowledgeManager`
- active commitments and background work → `TaskScheduler`
- reusable procedures and learned workflows → `GuidanceManager`
- callable code utilities → `FunctionManager`
- credentials and tokens → `SecretManager`
- supporting documents and screenshots → `FileManager` / `ImageManager`

A practical migration flow looks like:

1. Export the parts of the old system that are actually durable: contacts, key transcripts, docs, procedures, tasks, and reusable code.
2. Move procedures into guidance and code helpers into functions rather than keeping everything in one undifferentiated memory or skill layer.
3. Recreate channel and agent configuration in Unity natively instead of trying to translate every config field 1:1.
4. Start with the option 2 sandbox, re-onboard the assistant with the same docs, screenshares, calls, and voice notes you would use for a human teammate, and let Unity consolidate new experience back into typed memory over time.

## What's open and what isn't

Unity is the **open core** of the Unify platform. This repository contains the agent itself: the managers, async tool loops, CodeAct actor, dual-brain voice coordination, event backbone, and memory consolidation.

The persistence backend is open-source too: [Orchestra](https://github.com/unifyai/orchestra) runs locally by default in the quickstart. The supporting client libraries [Unify](https://github.com/unifyai/unify) and [UniLLM](https://github.com/unifyai/unillm) are open-source as well.

**Not open-sourced** is the managed platform layer around the agent: hosted communication routing, telephony and SIP infrastructure, Microsoft 365 tenant integration, assistant session control plane, billing, and identity. You can point Unity at Unify's hosted backend instead of a local Orchestra, but features that depend on the managed platform layer only work against the hosted service.

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
| **unity** (this) | The agent — managers, tool loops, CodeAct, voice, orchestration |
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
