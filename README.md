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

**An AI agent you can steer while it works.** Interrupt it mid-task, ask what it's doing, give it new instructions, or run five things at once — without restarting anything. Voice-native, with memory that compounds over time.

<table>
<tr><td><b>Steerable at every layer</b></td><td>Every operation returns a live handle you can pause, resume, interject into, or query — at any depth. Redirect a running task without losing progress.</td></tr>
<tr><td><b>Talks while it thinks</b></td><td>A fast real-time voice agent keeps the conversation alive while a slower deliberation brain does the actual work in the background. No awkward silence.</td></tr>
<tr><td><b>Programs, not tool calls</b></td><td>The agent writes Python plans over typed primitives — with variables, loops, and real control flow — instead of picking from a JSON tool menu one step at a time.</td></tr>
<tr><td><b>Memory that compounds</b></td><td>Contacts, knowledge, tasks, and communication preferences are continuously extracted from conversations into structured, queryable tables. After a month, the agent knows your world.</td></tr>
<tr><td><b>Concurrent by default</b></td><td>Run several tasks at once. Steer each one independently — ask for progress on one while redirecting another.</td></tr>
</table>

## Quick Start

Get the agent running in your terminal in under 5 minutes. Unity runs locally, uses the model provider you choose, and connects to Unify's hosted persistence layer — no local database or Docker for the first run.

### Prerequisites

- **Python 3.12+**
- **PortAudio** (system dependency for audio support)
  - macOS: `brew install portaudio`
  - Ubuntu/Debian: `sudo apt-get install portaudio19-dev python3-dev`
- **A [Unify](https://unify.ai) account** — the default quickstart uses Unify's hosted persistence plane for projects, logs, and manager state
- **An LLM provider key** — [OpenAI](https://platform.openai.com/api-keys) or [Anthropic](https://console.anthropic.com/) are the simplest paths from this README

### Install

```bash
# Clone all three repos as siblings
git clone https://github.com/unifyai/unity.git
git clone https://github.com/unifyai/unify.git
git clone https://github.com/unifyai/unillm.git

cd unity

# Install uv (skip if already installed: https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
```

### Configure

```bash
cp .env.example .env
```

Open `.env` and fill in the required credentials for the default quickstart:

```bash
UNIFY_KEY=your-unify-key
OPENAI_API_KEY=sk-...        # simplest path
# or: ANTHROPIC_API_KEY=...
```

You bring the model provider. OpenAI and Anthropic are the most direct options from this README, while `unillm` can also be configured for other supported providers and compatible local endpoints.

Everything else has sensible defaults. The default sandbox connects to Unify's hosted backend (`api.unify.ai`) for persistence, so you can explore the runtime without first standing up a local database or event stack.

### Run

```bash
source .venv/bin/activate
python -m sandboxes.conversation_manager.sandbox --project_name Sandbox --overwrite
```

You'll see a configuration prompt — select **Mode 1** (`SandboxSimulatedActor`). This runs the full ConversationManager brain with simulated backends. No external infrastructure required.

### Interact

Once the REPL starts, try:

```
> msg Hey, can you help me organize my upcoming week?
> sms I need to reschedule my meeting with Sarah to Thursday
> email Project Update | Here are the Q3 numbers you asked for...
```

Commands: `msg` (Unify message), `sms` (SMS), `email` (email), `call` (phone call), `meet` (video meeting). Type `help` for the full list.

**Video walkthrough**: https://www.loom.com/share/44171c4c1aa2475abd539d1251e1baab

### Going deeper

Mode 1 simulates everything to show the ConversationManager's orchestration. For the real CodeAct architecture (where the Actor writes and executes Python plans against the manager APIs), select **Mode 2** at the configuration prompt.

See the full sandbox docs at [`sandboxes/conversation_manager/README.md`](sandboxes/conversation_manager/README.md) — it covers Mode 3 (real computer interface), voice mode, live voice calls, real comms, GUI mode, and more.

---

## Quick answers

- **Is Unity fully local today?** Not end-to-end. The supported quickstart runs the brain locally but uses Unify's hosted backend for persistence and state.
- **Do I have to use OpenAI or Anthropic?** No. Those are the simplest documented paths here. `unillm` can be pointed at other supported providers and compatible local endpoints.
- **Do I have to use the hosted backend?** The default quickstart does. A broader self-hosted path is on the roadmap below.

---

## How it works

Most agent frameworks give you one loop: the model picks a tool, calls it, reads the result, picks the next. If you want to change course, you cancel and start over.

Unity gives every operation its own loop and returns a **live handle** you can steer. These handles nest: the user steers the ConversationManager, the ConversationManager steers the Actor, the Actor steers the managers. Corrections, pauses, and queries propagate through the full depth.

In practice, this means:
- "Also include Q2 numbers" mid-way through a report → the agent adjusts without restarting
- "Pause that, something urgent" → work freezes and resumes exactly where it left off
- "How's the flight search going?" → you get a status update without disrupting the work
- Three tasks running at once, each independently steerable

## Steerable handles

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

When the Actor calls `primitives.contacts.ask(...)`, the ContactManager starts its own tool loop and returns its own handle — nested inside the Actor's handle, which is nested inside the ConversationManager's. You can steer at any level and it propagates correctly.

The goal: you should be able to talk to your assistant while it's doing things for you, not wait in silence.

## CodeAct — the Actor writes programs, not tool calls

The Actor doesn't pick from a menu of JSON tools. It writes Python:

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

This runs in a sandboxed execution session with the full `primitives.*` API available — the same typed interfaces the rest of the system uses. One program per turn, with variables, loops, and real control flow. This matters because complex tasks that require composing several managers (look up contacts → query knowledge → send communications) can be expressed as a single coherent plan instead of 5+ round-trips where the model re-reads everything each time.

## Dual-brain voice

**Slow brain**: the ConversationManager. Sees the full picture — all conversations, notifications, in-flight actions. Makes deliberate decisions about what to do. Runs in the main process.

**Fast brain**: a real-time voice agent on LiveKit, running as a separate subprocess. Sub-second latency. Handles the conversation autonomously.

They talk over IPC. When the slow brain finishes a task or wants to guide the conversation, it sends the fast brain a notification:
- **SPEAK** — "say exactly this" (bypasses the fast brain's LLM entirely)
- **NOTIFY** — "here's some context, decide what to do with it"
- **BLOCK** — nothing; the fast brain keeps going on its own

So the assistant keeps talking to you while researching flights in the background. When the results come in, it naturally weaves them into whatever you're discussing. There's also a speech urgency evaluator that can preempt the slow brain if you say something that needs immediate attention.

Most agent frameworks go quiet while working. This architecture keeps the conversation alive.

## Memory that actually consolidates

Every 50 messages, the MemoryManager kicks in and runs a background extraction pass. It pulls out:

- Contact profiles — who people are, their roles, relationships
- Per-contact summaries — what you've been discussing, sentiment, themes
- Response policies — how each person prefers to communicate
- Domain knowledge — project details, preferences, long-term facts
- Tasks — things you committed to, deadlines, follow-ups

This is structured, queryable, and continuous. After a month of use, the system has a working model of your world — who the people are, what matters, what's in progress — stored in typed tables, not freeform text.

## Concurrent actions

The ConversationManager tracks everything that's running:

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

Each action gets its own dynamically generated steering tools. You can ask "how's the flight search going?" or "stop the summary, I'll do that myself" or "for the restaurants, make sure one has a private room" — and only the targeted action is affected.

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

## Repos

| Repo | What it does |
|------|-------------|
| **unity** (this) | The agent runtime — managers, tool loops, CodeAct, voice, orchestration |
| **[unify](https://github.com/unifyai/unify)** | Python SDK for persistence and logging (connects to Unify's hosted backend) |
| **[unillm](https://github.com/unifyai/unillm)** | LLM access layer — routes to OpenAI, Anthropic, or any compatible endpoint |

All three are MIT-licensed. The full product — with voice calls, messaging channels, and a management dashboard — runs on [Unify's platform](https://unify.ai).

---

## Running the test suite

Tests exercise the real system (steerable handles, CodeAct, manager composition, nested tool loops) against simulated backends with cached LLM responses:

```bash
# Install dev dependencies
uv sync --all-groups
source .venv/bin/activate

# Run tests (cached LLM responses — fast after first run)
tests/parallel_run.sh tests/                    # everything
tests/parallel_run.sh tests/actor/              # one module
tests/parallel_run.sh tests/contact_manager/    # another
```

See [tests/README.md](tests/README.md) for the full philosophy (we never mock the LLM — responses are cached, not faked).

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

## Design convictions

We don't use regex or substring matching to route user intent. Everything goes through LLM reasoning, guided by prompts and tool docstrings. If the system handles something wrong, we fix the prompt, not add a hardcoded rule.

We don't mock LLMs in tests. Every test uses real inference, cached for speed. Delete the cache and you're re-evaluating against live models.

We don't do defensive coding. No try/except around things that shouldn't fail. No null checks for things that shouldn't be null. The system fails loud when assumptions break.

We think of English as an API. Managers communicate through natural-language interfaces. The Actor orchestrates through English-language primitives. This makes the whole system inspectable without reading implementation code.

---

## Roadmap

Unity is under active development. Here's what we're working on:

- [ ] **Local deployment** — `docker compose up` for the full system (brain + persistence + communication) on your machine. This is the top priority.
- [ ] **Decouple Pub/Sub** — replace GCP Pub/Sub with a portable event delivery layer (local broker or direct async queues)
- [ ] **Local webhook adapters** — REST endpoints that replace Cloud Function webhook handlers for Twilio, Gmail, and other channel integrations
- [ ] **Local storage** — filesystem or S3-compatible alternative to GCS signed URLs
- [ ] **Simplified Orchestra** — lightweight local mode for the persistence layer, reducing external dependencies
- [ ] **One-command onboarding** — guided setup that provisions API keys, configures channels, and starts the assistant

We'll update this list as milestones are completed. Follow [GitHub Issues](https://github.com/unifyai/unity/issues) for detailed progress.

---

## Community

- [Documentation](https://docs.unify.ai) — architecture deep-dives, quickstart, guides
- [Discord](https://discord.com/invite/sXyFF8tDtm)
- [Issues](https://github.com/unifyai/unity/issues)
- [Discussions](https://github.com/unifyai/unity/discussions)

## License

MIT — see [LICENSE](LICENSE).

Built by the team at [Unify](https://unify.ai).
