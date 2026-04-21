# Architecture

This document describes Unity's internal architecture for developers who want to understand how the system works, contribute, or evaluate the design decisions.

## Mental model

Unity implements an AI assistant's brain as a **distributed back office**. Rather than one monolithic agent loop, there are specialized **state managers** — each owning a slice of the assistant's persistent state (contacts, knowledge, tasks, transcripts, etc.) — coordinated by a central **Actor** that writes Python programs to compose them.

Every public operation in the system, from searching contacts to executing a multi-step task, runs inside its own **async LLM tool loop** and returns a **steerable handle**. These handles are the universal interface: you can pause, resume, interject into, ask questions about, or stop any operation — at any nesting depth — while it's running.

```
User
 │
 ▼
ConversationManager ◄── voice IPC ──► Fast Brain (LiveKit)
 │
 │  starts actions, steers in-flight work
 ▼
CodeActActor ── generates Python plans ──► primitives.* API
 │
 │  each primitive call starts its own LLM tool loop
 ▼
┌───────────────────────────────────────────────────────┐
│  State Managers (each returns a SteerableToolHandle)  │
│                                                       │
│  ContactManager    KnowledgeManager   TaskScheduler   │
│  TranscriptManager GuidanceManager    FileManager     │
│  ImageManager      FunctionManager    WebSearcher     │
│  SecretManager     BlacklistManager   DataManager     │
│                                                       │
│  EventBus ─── typed pub/sub backbone                  │
│  MemoryManager ─── offline consolidation              │
└───────────────────────────────────────────────────────┘
```

Steering propagates through the full tree: stopping the Actor stops its inner manager loops; interjecting into the ConversationManager can reach a deeply nested knowledge query.

Hosted deployment concerns are intentionally optional at this boundary. The
public repo exposes a small `unity.deploy_runtime` SPI for session assignment,
job lifecycle hooks, metrics export, and shutdown log archival, with local/no-op
defaults when no private hosted backend is installed.

---

## The async tool loop

**Files:** `unity/common/async_tool_loop.py`, `unity/common/_async_tool/loop.py`

The async tool loop is the universal runtime. Nearly every public manager method is implemented as: create an LLM client, register domain-specific tools, start a loop, return a handle.

### How it works

```
┌──────────────────────────────────────────────┐
│             async_tool_loop_inner            │
│                                              │
│  1. Send messages to LLM                     │
│  2. LLM returns tool calls                   │
│  3. Execute tools in parallel (with limits)  │
│  4. Collect results, append to transcript    │
│  5. Check for interjections, pauses, stops   │
│  6. Repeat until LLM produces final answer   │
│                                              │
│  At any point between steps:                 │
│   - Interjections are drained from the queue │
│   - Pause events block until resumed         │
│   - Stop events trigger graceful exit        │
│   - Context compression fires if too long    │
└──────────────────────────────────────────────┘
```

The loop handles:

- **Parallel tool execution** with configurable concurrency limits
- **Interjection queue** — new instructions injected mid-flight without restarting
- **Pause/resume** via asyncio events
- **Context compression** — when the conversation exceeds the model's context window, the loop compresses and restarts transparently
- **Dynamic tools** — tools that are generated at runtime based on the current state (e.g., per-action steering tools)
- **Tool policies** — gating which tools are available at which step (used for discovery-first patterns)
- **Time awareness** — optional wall-clock context injected after each tool completion
- **Prompt caching** — cooperative cache-control headers for providers that support it

### The interjection mechanism

When `handle.interject("also check trains")` is called, the message is placed in an asyncio queue. Between LLM turns, the loop drains this queue and injects the messages as new user turns. If `interrupt_llm_with_interjections` is enabled, an in-flight LLM call is cancelled and restarted with the interjection included.

This is how the user can redirect an agent mid-task without the overhead of stopping and restarting from scratch.

---

## Steerable handles

**File:** `unity/common/async_tool_loop.py`

`SteerableToolHandle` is the abstract protocol. `AsyncToolLoopHandle` is the concrete implementation backed by an asyncio task. Every public manager method returns one.

```python
class SteerableToolHandle(ABC):
    async def ask(self, question) -> SteerableToolHandle  # inspect the running work
    async def interject(self, message)                     # inject new instructions
    async def stop(self, reason=None)                      # cancel
    async def pause(self)                                  # freeze without cancelling
    async def resume(self)                                 # unfreeze
    def done(self) -> bool                                 # completion check
    async def result(self) -> str                          # await final answer
    async def next_clarification(self) -> dict             # bottom-up question from tool
    async def next_notification(self) -> dict              # bottom-up status update
    async def answer_clarification(self, call_id, answer)  # respond to tool's question
```

### Nested steering

When the Actor calls `primitives.contacts.ask(...)`, the ContactManager starts its own tool loop and returns its own `SteerableToolHandle`. This inner handle is tracked by the Actor's loop. When the user calls `handle.pause()` on the Actor's handle, the pause propagates to all active inner handles via a **mirror queue** mechanism:

1. The outer handle receives `pause()`
2. It sets its own pause event and enqueues a `_mirror` sentinel
3. The inner loop drains the mirror, synthesizes a helper tool call in the transcript (so the LLM sees it happened), and dispatches `pause()` to each child handle
4. Child handles repeat the process recursively

This gives full-depth steering with transcript visibility at every level.

### `ask()` — inspecting a running loop

`ask()` is not a simple state peek. It:

1. Snapshots the running loop's full transcript
2. Transforms roles to `inner_user`/`inner_assistant` (so the inspection LLM distinguishes the inspected conversation from its own)
3. Creates a fresh LLM client with the snapshot as system context
4. Starts a **new, read-only tool loop** to answer the question
5. If the inspected loop has its own inner handles, their `ask_*` tools are forwarded to the inspection loop, enabling recursive drill-down

This means you can ask "what's the flight search doing?" and the inspection loop can, if needed, call `ask_flight_search()` to query the inner handle's own transcript.

### `forward_handle_call` — signature adaptation

Different handle implementations extend the base signature with domain-specific kwargs (e.g., `BaseActiveTask.stop` adds `cancel`, `ConversationManagerHandle.interject` adds `pinned`). `forward_handle_call` introspects the target method's actual signature, filters out unsupported kwargs, and applies positional fallbacks — so delegation boundaries work without hand-written adapter code.

---

## The CodeAct Actor

**File:** `unity/actor/code_act_actor.py`

The Actor doesn't pick from a JSON tool menu. It generates Python programs that call typed primitives:

```python
contacts = await primitives.contacts.ask("Who was at the Henderson meeting?")
for contact in contacts:
    history = await primitives.knowledge.ask(f"What is {contact} working on?")
    await primitives.contacts.update(f"Send {contact} a status email about {history}")
```

This runs in a `PythonExecutionSession` — a sandboxed environment where the `primitives` namespace is pre-populated with async methods that dispatch to the real managers.

### Why CodeAct over JSON tools

JSON tool calling forces every composition to be a separate round-trip. To look up contacts, query knowledge for each, and send emails, the LLM needs 3+ turns where it re-reads the entire context each time. With CodeAct, the same logic is a single program with variables, loops, and branching — one plan, one LLM turn for the plan, then execution.

The Actor still uses the async tool loop internally (the LLM generates code as a "tool call" that gets executed), so it inherits all the steering, compression, and observability infrastructure.

### Discovery-first tool policy

The Actor implements a **gating policy**: until the LLM has queried both `FunctionManager` (what custom functions exist?) and `GuidanceManager` (what procedures/SOPs apply?), the full tool surface is hidden. This forces an explore-then-act pattern that prevents the LLM from jumping to action before understanding what's available.

### Primitives registry

**File:** `unity/function_manager/primitives/registry.py`

`ToolSurfaceRegistry` is the single source of truth for how managers are exposed to the Actor. Each manager has a `ManagerSpec` that defines:

- Which methods to expose (and which to exclude)
- The sandbox namespace (`primitives.<alias>.<method>`)
- Priority, domain, description, and usage hints for prompt construction
- Dependencies between managers

The registry auto-discovers methods from manager base classes, generates tool schemas, builds prompt context, and constructs the sandbox's global state — all from the spec definitions.

---

## State managers

Each manager follows the same pattern:

1. A **base class** (`base.py`) defines the public API as abstract methods with rich docstrings. These docstrings are the LLM-facing contract — they're attached to concrete implementations via `@functools.wraps`.

2. A **concrete implementation** that registers domain-specific tools in `__init__` and implements each public method as an async tool loop.

3. A **prompt builder** (`prompt_builders.py`) that constructs system prompts focusing on tool composition, contrastive guidance (when to use tool A vs. tool B), and high-level reasoning patterns. Tool-specific details stay in tool docstrings.

### Manager isolation

Managers communicate through their public APIs, not shared state. When the TranscriptManager needs contact information to resolve participant names, it calls `ContactManager.ask()` — which starts its own tool loop and returns a handle. This keeps each manager independently testable and replaceable.

The `_as_caller_description` class attribute on each manager tells nested loops who is calling: when KnowledgeManager calls FileManager, the FileManager's LLM sees "the KnowledgeManager, querying structured domain knowledge" as the caller context, not a raw user message.

### Key managers

**ContactManager** — People and relationships. CRUD over structured contact records with search, merge (deduplication), and relationship tracking.

**KnowledgeManager** — Domain facts. Structured knowledge tables with typed columns, vector search, and schema refactoring. Facts are queryable, not just dumped text.

**TaskScheduler** — Durable tasks. Create, edit, reorder, and execute tasks. `execute()` starts a task via `Actor.act()` and returns a live steerable handle, making tasks first-class concurrent operations.

**TranscriptManager** — Conversation history. Search, filter, and analyze past conversations. Can resolve participants via ContactManager.

**GuidanceManager** — Procedures and SOPs. Step-by-step instructions, software walkthroughs, and strategies for composing functions. Linked to FunctionManager entries.

**MemoryManager** — Offline consolidation. Runs periodically (every ~50 messages) to extract contacts, relationships, knowledge, tasks, and response policies from recent conversations into the structured managers.

---

## The ConversationManager and dual-brain voice

**File:** `unity/conversation_manager/conversation_manager.py`

The ConversationManager is the top-level orchestrator for live conversations. It has a fundamentally different design from the other managers because it handles real-time interaction.

### Slow brain / fast brain

**Slow brain** (ConversationManager): Runs in the main process. Sees the full picture — all conversations, notifications, in-flight actions, system state. Makes deliberate decisions about what to do. Uses a single-shot tool decision pattern (one LLM call → one action) rather than a multi-turn loop, because the user might send another message at any moment.

**Fast brain** (LiveKit voice agent): Runs as a separate subprocess. Sub-second latency for voice conversations. Handles the conversation autonomously using its own LLM.

They communicate over IPC with three signal types:
- **SPEAK** — "say exactly this" (bypasses the fast brain's LLM)
- **NOTIFY** — "here's context, decide how to use it"
- **BLOCK** — the fast brain continues autonomously

### In-flight action tracking

The ConversationManager maintains `in_flight_actions` — a dict of currently running steerable handles with metadata. For each action, it dynamically generates steering tools (`ask_<action>`, `interject_<action>`, `stop_<action>`, `pause_<action>`, `resume_<action>`) that the slow brain's LLM can call. This is how "how's the flight search going?" routes to the right handle.

### Event-driven scheduling

The ConversationManager uses a `Debouncer` that coalesces rapid-fire events (new messages, action completions, notifications) into batched brain invocations. This prevents thrashing when multiple things happen simultaneously.

---

## The event bus

**File:** `unity/events/event_bus.py`

The EventBus is an in-process, asyncio-friendly pub/sub system with:

- **Typed payloads** — all events are Pydantic models declared in `events/types/`. Invalid payloads are rejected at publish time.
- **Searchable history** — events are stored in a windowed deque per type, queryable with filters.
- **Callback registration** — subscribe to event types with async callbacks.
- **Callback cascade tracking** — `_CURRENT_ROOT_SEQ` (a context variable) tracks which callback triggered which, so `join_callbacks()` can await an entire cascade deterministically.
- **Unify log hydration** — the bus can be prefilled from persisted Unify logs, bridging in-process events with the durable backend.

Managers and tool loops publish structured events (tool calls, steering actions, method boundaries) via `to_event_bus()`. This feeds both runtime coordination (MemoryManager reacts to message events) and external observability.

### Lineage and hierarchy

Every tool loop has a **lineage** — a list of string segments tracking its position in the nesting tree, propagated via `TOOL_LOOP_LINEAGE` (a `ContextVar`). Each segment includes a random suffix for per-invocation identity:

```
["ConversationManager.act(a1b2)", "Actor.act(c3d4)", "ContactManager.ask(e5f6)"]
```

This lineage is attached to every event the loop publishes, enabling full parent-child correlation in logs and the frontend.

---

## Context propagation

**File:** `unity/common/_async_tool/propagation_mode.py`

When a tool loop calls a nested tool that starts its own loop, the parent conversation may need to be visible to the child (e.g., so the ContactManager knows what the user originally asked). Unity handles this with explicit role transformation:

- **`outer_user` / `outer_assistant`** — parent conversation roles, injected into child loops as system context
- **`inner_user` / `inner_assistant`** — child conversation roles, visible when the parent inspects via `ask()`
- **`user` / `assistant`** — current conversation roles (always the active loop)

This three-layer separation prevents prompt injection between nesting levels and lets each LLM clearly distinguish "what the user said" from "what the calling manager said" from "what I'm doing."

`ChatContextPropagation` controls the policy:
- `ALWAYS` — always pass parent context to child tools
- `NEVER` — never pass it
- `LLM_DECIDES` — expose a boolean parameter so the LLM can opt out per tool call

---

## Multi-request coordination

**File:** `unity/common/_async_tool/multi_handle.py`

A single tool loop can serve **multiple concurrent requests** through the `MultiHandleCoordinator`. Each request gets:

- A unique `request_id`
- Its own clarification and notification queues
- Independent completion/cancellation
- Tagged interjections so the LLM knows which request a message belongs to

The LLM calls `final_answer(request_id, answer)` to complete specific requests. The loop continues until all requests are done (or persists indefinitely if `persist=True`).

This is used by the ConversationManager to handle multiple user messages that arrive while the brain is already processing — rather than queuing them sequentially, they're multiplexed through a shared loop with shared context.

---

## Testing

**Directory:** `tests/`

Tests use real LLM calls, never mocked. Responses are cached by UniLLM so that:

- First run: real inference, responses stored
- Subsequent runs: cached responses replayed in milliseconds
- Cache key = exact LLM input (change a prompt → automatic cache miss → fresh inference)

This means the test suite is both **deterministic** (cached runs are byte-for-byte reproducible) and **honest** (every cached response was produced by a real model given that exact input).

Tests run in parallel via `tests/parallel_run.sh`, which spawns each test in an isolated tmux session with per-terminal tmux server isolation. Results stream inline as tests complete.

Tests fall on a spectrum between **symbolic** (infrastructure-focused: does steering work? does nesting propagate correctly?) and **eval** (capability-focused: did the assistant answer correctly?). The caching system makes both types fast after initial population.

**Synchronization** uses trigger-based helpers (`tests/async_helpers.py`) rather than sleeps, making tests robust to the 1000x timing difference between cached (milliseconds) and live (minutes) LLM calls.

---

## System dependencies

Unity persists state through **Orchestra** (a REST API backed by PostgreSQL) via the **Unify** Python SDK, and makes LLM calls through **UniLLM** (a caching/tracing/normalization layer).

```
Unity ──► Unify SDK ──► Orchestra API ──► PostgreSQL
  │
  └─────► UniLLM ──► OpenAI / Anthropic / etc.
```

For development and testing, the system runs against simulated backends. The core architecture (handles, loops, CodeAct, manager composition) is independent of the specific persistence layer.

---

## Directory layout

```
unity/
├── unity/
│   ├── common/
│   │   ├── async_tool_loop.py          # SteerableToolHandle, start_async_tool_loop
│   │   └── _async_tool/
│   │       ├── loop.py                 # async_tool_loop_inner (the engine)
│   │       ├── loop_config.py          # LoopConfig, TOOL_LOOP_LINEAGE
│   │       ├── multi_handle.py         # MultiHandleCoordinator
│   │       ├── propagation_mode.py     # ChatContextPropagation enum
│   │       ├── context_compression.py  # Transparent context compression
│   │       ├── dynamic_tools_factory.py # Runtime tool generation
│   │       └── messages.py             # forward_handle_call, mirror dispatch
│   ├── actor/
│   │   ├── base.py                     # BaseActor, BaseCodeActActor
│   │   ├── code_act_actor.py           # CodeActActor implementation
│   │   ├── execution.py                # PythonExecutionSession, sandbox
│   │   └── environments/               # Pluggable execution environments
│   ├── conversation_manager/
│   │   ├── conversation_manager.py     # ConversationManager (slow brain)
│   │   └── domains/
│   │       ├── brain.py                # Brain spec construction
│   │       ├── brain_action_tools.py   # Dynamic steering tools for in-flight actions
│   │       └── proactive_speech.py     # Fast brain IPC
│   ├── contact_manager/
│   ├── knowledge_manager/
│   ├── task_scheduler/
│   ├── transcript_manager/
│   ├── guidance_manager/
│   ├── memory_manager/
│   ├── function_manager/
│   │   └── primitives/
│   │       ├── registry.py             # ToolSurfaceRegistry (single source of truth)
│   │       └── scope.py                # PrimitiveScope
│   ├── file_manager/
│   ├── image_manager/
│   ├── web_searcher/
│   ├── secret_manager/
│   ├── data_manager/
│   ├── events/
│   │   ├── event_bus.py                # EventBus
│   │   └── types/                      # Pydantic event payloads
│   └── manager_registry.py             # Singleton factory for manager instances
├── tests/
│   ├── parallel_run.sh                 # Isolated parallel test runner
│   ├── async_helpers.py                # Trigger-based synchronization
│   └── <module>/                       # Tests mirror production structure
├── deploy/                             # Dockerfiles, Kubernetes, virtual desktop
└── agent-service/                      # Node.js browser automation agent
```

---

## Design principles

**English as an API.** Managers communicate through natural-language interfaces. The Actor orchestrates through English-language primitives. This makes the system inspectable without reading implementation code — you can read the LLM transcripts and understand what happened.

**No heuristics, no regex routing.** If the system needs to respond correctly to a type of user input, the fix is always a prompt or tool docstring improvement that nudges the LLM, never a hardcoded rule that pattern-matches on the input.

**Fail loud.** No defensive try/except around things that shouldn't fail. No null checks for things that shouldn't be null. When assumptions break, the system crashes visibly rather than silently degrading.

**Aggressive refactoring.** Zero backward compatibility. When requirements change, code is rewritten to optimally support the new requirements, not patched with compatibility shims.

**Real LLMs in tests.** Never mocked. Cached for speed, but every cached response was produced by a real model. This catches prompt regressions that mocks would hide.
