## Task Scheduler – Architecture and Guide

This package manages the creation, scheduling, execution, and re‑ordering of tasks. It provides:

- A public manager (`TaskScheduler`) that exposes ask/update/execute methods
- A simulated manager for demos/tests
- Strong types for tasks, schedules, and statuses
- An execution layer that runs a single task or a whole queue
- Queue manipulation primitives with strict invariants and safe re‑attachment
- A storage/view layer that centralizes Unify I/O and caching


### High‑level picture

- `TaskScheduler` is the orchestrator. It composes read/write “tools” and runs LLM loops for `ask`, `update`, and `execute`. It guarantees invariants and a single active task at a time.
- Execution starts a task either by detaching it from the queue (isolation) or by chaining the queue. Detachment records a `ReintegrationPlan` so a deferred task can be reinstated exactly where it came from. The public execution handle is always an `ActiveQueue`.
- All reads/writes go through `TasksStore` (Unify I/O) and `LocalTaskView` (cache of queue membership, head start_at, and ids). Queue edits use a single validated write funnel that enforces invariants and keeps neighbor pointers symmetric.
- For reorders, a small `queue_engine` computes minimal, invariant‑preserving updates (no direct DB logic inside planning).


### Key files and responsibilities

- `task_scheduler.py`
  - The core manager. Exposes public `ask`, `update`, `execute` and a comprehensive set of private tools (create/delete/cancel tasks; list/get/reorder/move/partition/set queues; bulk schedule edits; checkpoints; reinstatement).
  - Maintains in‑memory state: one active task pointer, a single primed task, reintegration plans, and queue checkpoints.
  - Uses `TasksStore` and `LocalTaskView` for I/O and caching. Exposes `ContactManager.ask` among its tools for cross‑domain flows.

- `active_task.py`
  - `ActiveTask`: a per‑task, steerable handle that wraps the actor’s live plan. Mirrors status into the Tasks row on pause/resume/stop/result. Classifies interjections (cancel/defer) using the scheduler and triggers reinstatement on defer. Returned internally; the public execution surface uses `ActiveQueue`.

- `active_queue.py`
  - `ActiveQueue`: a composite handle that sequentially executes a queue (head→tail), adopting each `ActiveTask` in turn. Tracks completions and supports queue‑aware `interject` and `ask`. Provides a passthrough mode for singleton or isolated executions.

- `activation_ops.py`
  - Low‑level, activation‑time link manipulation. Implements detachment semantics for isolation vs chained execution and records a `ReintegrationPlan`.

- `queue_utils.py`
  - Small helpers for symmetric neighbor updates and “attach between prev/next” with head‑only `start_at`. Used by the scheduler’s validated write funnel.

- `queue_engine.py`
  - Pure planning helpers. Given current rows and a desired order, returns minimal schedule/status updates that preserve invariants (head carries `start_at`, non‑heads at most `queued`, keep `active` unchanged).

- `reintegration.py`
  - `ReintegrationManager`: restores a deferred task to its precise previous position using the stored `ReintegrationPlan`, computing viable neighbors, validating invariants, and re‑attaching links.

- `storage.py`
  - `TasksStore`: centralized Unify I/O (reads/writes, normalization, checkpoint persistence).
  - `LocalTaskView`: best‑effort cache for queue membership, head timestamps, queue‑id allocation, and log‑id memoization; provides write helpers that keep caches coherent.

- `prompt_builders.py`
  - Builds dynamic system prompts for LLM loops (ask/update/execute) from the tools the scheduler actually exposes, with examples and safety guidance.

- `simulated.py`
  - `SimulatedTaskScheduler`: drop‑in replacement for demos/tests. Mirrors the real surface but does not touch storage. `_SimulatedTaskScheduleHandle` is a minimal steerable handle for simulated ask/update.

- `types/` (Pydantic models & enums)
  - `task.py`: canonical `Task` model (ids, name/description, status, schedule, trigger, deadline, repeat, priority, response_policy, activated_by). Enforces schedule/trigger exclusivity.
  - `schedule.py`: `Schedule` model (forbids `start_at` on non‑head).
  - `status.py`: lifecycle enum (`scheduled`, `queued`, `primed`, `active`, `triggerable`, `completed`, `cancelled`, `failed`).
  - `activated_by.py`: activation reasons (`schedule`, `queue`, `trigger`, `explicit`).
  - `repetition.py`: recurrence patterns. `priority.py`: priority enum. `trigger.py`: inbound trigger definition.


### Core flows

1) Ask (read‑only)
   - Builds a live toolset (filters, semantic search, queue readers, contact lookup), injects a dynamic system prompt, and runs a tool‑use loop. Must not mutate data.

2) Update (mutations)
   - Exposes creation, deletion, cancellation, and queue manipulation tools, plus atomic materialization (`set_queue`) and bulk schedule edits. Enforces queue/schedule invariants via a single validated write funnel. Auto‑checkpoints queue edits for easy revert.

3) Execute (run now)
   - Guards single‑active. If given a numeric id, can run in isolation (detach, followers keep schedule) or as a chain (preserve links). Otherwise uses an outer loop with explicit queue planning tools. Always checkpoints at session start. Returns an `ActiveQueue` handle in all cases (singleton passthrough for isolated/single‑task).


### Queue/schedule invariants (enforced centrally)

- Only the head may carry `start_at`; a head with `start_at` must be `scheduled`.
- Non‑head tasks cannot have `start_at` and cannot remain `scheduled` after reorders.
- `primed` is only valid at the head (and at most one primed overall).
- Trigger‑based tasks (`trigger`) cannot be placed in the runnable queue or scheduled.
- Writes go through `_validated_write(...)`, which checks invariants, prevents direct `active` writes, forbids cross‑queue adjacency, and ensures neighbor symmetry.


### Reinstatement (defer → restore)

- When a task is activated, detachment records a `ReintegrationPlan` with previous/next pointers, headness, head timestamp, original status, and queue_id.
- `ReintegrationManager` uses this plan to re‑attach safely: compute viable neighbors, set head timestamp if needed, derive desired status, validate invariants, and write links symmetrically.


### Storage and caching

- All I/O runs through `TasksStore`; caching and queue indexes via `LocalTaskView`.
- `LocalTaskView` optimizes: queue membership (forward/reverse), head `start_at`, log‑id memoization, queue‑id allocation. It is tolerant to cache misses and can refresh opportunistically.
- Environment toggle `UNITY_TS_LOCAL_VIEW_OFF` disables cache.


### Execution handles

- `ActiveTask`: internal steerable handle for a single running task; mirrors status and clears the scheduler’s active pointer when done.
- `ActiveQueue`: public execution handle that sequences tasks using persisted `next_task` links, supports interjection routing across the queue, and provides a completion summary. Uses passthrough when the queue is a singleton/isolated.


### Clarification and contacts

- The ask/update/execute loops can expose a `request_clarification` tool (when queues are provided) to ask the human questions without mixing clarifications into normal replies.
- The scheduler also exposes `ContactManager.ask` for cross‑domain context when tasks mention people/contact ids.


### Checkpoints & safety

- Queue‑affecting operations create checkpoints (`checkpoint_queue_state`) and expose helpers (`revert_to_checkpoint`, `get_latest_checkpoint`) so multi‑step plans are reversible.
- After any mutation (including execution start), the caller must refresh queue state before further edits (the execute prompt enforces this policy).


### Common environment variables

- `UNITY_TS_LOCAL_VIEW_OFF`: disable `LocalTaskView` caching.
- `UNITY_SIM_ACTOR_DURATION`: default simulated actor duration (seconds).


### Quick orientation for new contributors

- Start in `task_scheduler.py` to see public surface and tool wiring.
- Read `storage.py` to understand I/O & caching behaviors and how queue indexes are built.
- Inspect `activation_ops.py` and `queue_utils.py` to learn link semantics for activation and attachment.
- See `queue_engine.py` for pure planning of reorders and status derivation.
- `active_task.py` / `active_queue.py` define the live execution handles you’ll get back from `execute`.
- `reintegration.py` explains how deferred tasks are reinstated exactly.
