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

Unity is the brain of an AI assistant. It's a distributed system where specialized managers (contacts, knowledge, tasks, transcripts, guidance, memory…) each run their own LLM-powered reasoning, coordinate through a typed event bus, and expose live handles you can pause, resume, interject into, or stop at any time.

The key abstraction is the **steerable handle**: every operation — from a simple contact lookup to a multi-step task execution — returns a handle that composes and nests to arbitrary depth. This README walks through the architecture.

---

## The core idea

Most agent frameworks work like this: one LLM, one loop, one tool call at a time. The model picks a tool, calls it, reads the result, picks the next tool. If you want to interrupt, you cancel and start over.

Unity works differently. Every operation — whether it's searching contacts, updating knowledge, or executing a multi-step task — runs inside its own async LLM tool loop and returns a **steerable handle**. These handles compose: the ConversationManager steers the Actor, the Actor steers the managers, and the user steers the ConversationManager. Steering propagates through the full depth.

This means the assistant can:
- Run several things at once and let you steer each one independently
- Accept corrections mid-task without restarting ("actually, also include the Q2 numbers")
- Pause work, handle something urgent, and resume where it left off
- Hold a real-time voice conversation while doing background work

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

This is the one we're most proud of.

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

## System dependencies

Unity is the "brain" in a larger system. It persists state through a backend API (via a Python SDK) and makes LLM calls through a caching/tracing layer:

| Repo | Open? | What it does |
|------|-------|-------------|
| **unity** (this) | ✅ MIT | The brain — managers, tool loops, CodeAct, orchestration |
| **[unify](https://github.com/unifyai/unify)** | ✅ MIT | Python SDK for persistence and logging |
| **[unillm](https://github.com/unifyai/unillm)** | ✅ MIT | LLM abstraction — caching, tracing, cost tracking |

The backend API, communication gateway (voice/SMS/email), and web console are hosted services. The full product — with voice calls, messaging channels, and a management dashboard — runs on [Unify's platform](https://unify.ai).

---

## Running locally

> **Status: coming soon.** Unity was built inside a GCP-hosted environment and we're actively decoupling it for standalone local deployment. The core architecture works today in tests against simulated backends, and full local deployment is on the [roadmap](#roadmap). Here's what it will look like — and what works right now.

### What works today

You can clone the repo, run the full test suite, and explore the architecture. Tests exercise the real system (steerable handles, CodeAct, manager composition, nested tool loops) against simulated backends with cached LLM responses:

```bash
git clone https://github.com/unifyai/unity.git
git clone https://github.com/unifyai/unify.git
git clone https://github.com/unifyai/unillm.git

cd unity
pip install uv && uv sync --all-groups
source .venv/bin/activate

cp .env.example .env
# Add your OPENAI_API_KEY and/or ANTHROPIC_API_KEY

# Run tests (cached LLM responses — fast after first run)
tests/parallel_run.sh tests/                    # everything
tests/parallel_run.sh tests/actor/              # one module
tests/parallel_run.sh tests/contact_manager/    # another
```

See [tests/README.md](tests/README.md) for the full philosophy (we never mock the LLM — responses are cached, not faked).

### Full local deployment (coming soon)

We're working toward a single `docker compose up` that runs the complete system on your machine:

```bash
# Coming soon
git clone https://github.com/unifyai/unity.git
cd unity
cp .env.example .env
# Add your API keys

docker compose up
# Unity brain + Orchestra (persistence) + Communication (channels)
# Available at localhost:8080
```

This will give you:

- **The full brain** — all managers, CodeAct actor, steerable handles, dual-brain voice
- **Local persistence** — Orchestra running locally with PostgreSQL, replacing the hosted API
- **Communication channels** — REST endpoints for sending messages, triggering calls, and receiving webhooks
- **Web console** — local dashboard for managing your assistant

### What's being decoupled

The main work is replacing GCP-specific infrastructure with portable alternatives:

| Component | Current (GCP) | Local target |
|-----------|---------------|-------------|
| Event delivery | Pub/Sub topics + subscriptions | Local message broker or direct async queues |
| Webhook ingress | Cloud Functions (Twilio, Gmail) | Local REST endpoints with adapter layer |
| Storage | GCS signed URLs | Local filesystem or S3-compatible store |
| Persistence | Hosted Orchestra API | Local Orchestra instance (PostgreSQL) |
| Voice | LiveKit Cloud | Local LiveKit server (`brew install livekit`) |

The brain itself (everything under `unity/`) is already infrastructure-agnostic — it talks to Orchestra through the Unify SDK and makes LLM calls through UniLLM. The decoupling work is in the communication layer and deployment glue.

Track progress on the [roadmap](#roadmap) below and in [GitHub Issues](https://github.com/unifyai/unity/issues).

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
├── tests/
├── agent-service/                # Node.js desktop/browser automation
└── deploy/                       # Dockerfiles, Kubernetes, virtual desktop
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

- 💬 [Discord](https://discord.com/invite/sXyFF8tDtm)
- 🐛 [Issues](https://github.com/unifyai/unity/issues)
- 💡 [Discussions](https://github.com/unifyai/unity/discussions)

## License

MIT — see [LICENSE](LICENSE).

Built by the team at [Unify](https://unify.ai).
