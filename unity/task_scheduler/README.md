Task Scheduler – Architecture and Guide

Overview
The Task Scheduler manages task rows stored in a Unify log context, exposing high‑level tools to create, search, update, execute, and orchestrate tasks in runnable queues. It enforces invariants around queue linkages, activation lifecycle, reintegration after isolated runs, and provides an interactive LLM loop for ask/update/execute flows.

Key concepts
- Task: A row in the Tasks context described by types in types/task.py. Each logical task may have multiple instances (instance_id increments per run). Core lifecycle is in types/status.py.
- Queue: A singly linked list of runnable tasks (schedule.prev_task/next_task) with a numeric queue_id at the task top level. Only the head can carry schedule.start_at. Head status is scheduled when start_at is present; non‑heads are at most queued.
- Activation: A task moves to active state and runs via an actor. Activation cause is recorded in types/activated_by.py.
- Isolation vs chain: Starting a task can detach it (isolated) or keep followers attached (chain). Detachment records a ReintegrationPlan to enable later restoration.
- Checkpoints: Execute/update flows can checkpoint queue state and optionally persist snapshots.

File-by-file tour

__init__.py
- Package marker.

base.py
- Defines abstract interfaces:
  - BaseActiveTask: contract for an active task handle (ask/interject/stop/pause/resume/result/done/valid_tools).
  - BaseTaskScheduler: abstract manager surface (ask/update/execute) that concrete TaskScheduler implements.

types/
- activated_by.py: StrEnum for why a task became active: schedule, queue, trigger, explicit.
- priority.py: StrEnum for priority (low/normal/high/urgent).
- repetition.py: Pydantic model RepeatPattern and helpers for recurrence semantics.
- schedule.py: Pydantic model Schedule with fields prev_task/next_task/start_at and a validator forbidding start_at on non‑head.
- status.py: StrEnum of lifecycle statuses.
- task.py: Pydantic model Task representing each row, including queue_id, task/instance ids, lifecycle, schedule/trigger, metadata, and to_post_json helper.
- trigger.py: Pydantic model Trigger describing inbound event conditions that start a task.
- reintegration_plan.py: Dataclass capturing the minimal info to reinstate an isolated task (prev/next, head_start_at, etc.).

storage.py
- Thin adapter around Unify for the Tasks context. Responsibilities:
  - ensure_context: creates the Tasks context with keys/fields.
  - get_rows, get_entries, get_logs_by_task_ids, get_minimal_rows_by_task_ids: optimized readers with projections.
  - update, log, create_many, delete: batched writes with normalization and None‑stripping rules consistent with Task model.
  - LocalTaskView centralizes I/O fast‑paths for runtime performance: maintains a fast queue index and membership map, memoizes task_id→log_id, exposes helpers for batch reads/writes, and marks caches stale on lifecycle changes. Reads/writes from tools should route through LocalTaskView.

queue_engine.py
- Stateless helpers for queue math:
  - _to_status: normalization to Status.
  - _sched_prev/_sched_start_at: light schedule extractors.
  - derive_status_after_queue_edit: status normalization rules (head+start_at → scheduled; others ≤ queued).
  - plan_reorder_queue: computes invariant‑preserving writes for a new_order.

_queue_utils.py
- Private helpers used by TaskScheduler for linkage symmetry:
  - sched_prev/sched_next: tolerant prev/next accessors over dict or Schedule.
  - sync_adjacent_links: given a task and its schedule, ensure neighbours’ reciprocal pointers are updated and non‑head start_at is stripped. Neighbour writes route via LocalTaskView to keep caches in sync.
  - attach_with_links: optimized attach that updates neighbours (via LocalTaskView), sets head start_at/status when applicable, carries queue_id, and writes the current task via the scheduler’s validated funnel; uses prefetched log objects where possible.

_queue_ops.py
- Higher‑level queue mutations used during activation and attachment:
  - detach_from_queue_for_activation(scheduler, task_id, detach):
    • Always records a ReintegrationPlan with prev/next/head_start_at/original_status/queue_id.
    • Isolation (detach=True): if detaching head, promote successor to head and carry head_start_at; if detaching middle, unlink neighbours; in both cases clear schedule on the detached task. Non‑heads never keep start_at.
    • Chain (detach=False): promote the activating task to head, keep followers attached, move head_start_at to new head, and strip start_at from its successor. Signals a per‑task asyncio.Event barrier to let ActiveQueue await linkage completion.
  - attach_with_links: thin wrapper that delegates to _queue_utils.attach_with_links (single source of attach logic).

reintegration.py
- ReintegrationManager applies a ReintegrationPlan to restore a task’s previous position. Handles edge cases: deleted neighbours, conflicting statuses (e.g., primed conflicts downgraded), and ensures head start_at/status semantics. Used when an isolated activation is deferred/stopped or requested explicitly by reinstate_to_previous_queue.

active_task.py
- ActiveTask wraps an actor’s run for a single task instance:
  - create: constructs the actor plan and returns an ActiveTask handle bound to (task_id, instance_id).
  - ask/interject: forwards calls to the actor with scheduler‑aware lifecycle controls. interject uses scheduler._classify_steering_intent and short‑circuits lifecycle requests (defer/cancel/pause/resume/stop) to avoid LLM routing.
  - stop/pause/resume/done/result: mirror actor behaviour while updating the backing task status and clearing the active pointer safely. On defer/stop, calls back into scheduler for reinstate flows.
  - valid_tools: exposes steerable methods dynamically (pause or resume depending on current state).

active_queue.py
- Composite handle that orchestrates a queue (chain) of tasks:
  - Creates with the first task’s ActiveTask handle and a background driver.
  - _drive: waits for current task result, records per‑task completions, awaits linkage barriers set by detach_from_queue_for_activation, and advances to the next queue member until exhausted. Produces final summary text listing completed tasks.
  - interject: routes user instructions to tasks via a lightweight LLM router that consumes a compact queue snapshot; supports bypass via UNITY_TS_DISABLE_LLM_ROUTER, passthrough for true singleton queues, and clarification when routing is ambiguous.
  - ask: prepends a human‑readable queue preamble (progress, next tasks, detailed rows) before delegating to the current task’s handle.
  - active_task_done: await the next completion; returns a JSON mapping {task_name: result} for completions since the last call (cursor semantics).
  - pause/resume/stop/done/result/valid_tools: steer the current task and reflect queue‑level completion.

simulated.py
- SimulatedTaskScheduler and a minimal _SimulatedTaskScheduleHandle used in tests and sandboxes:
  - Provides LLM‑backed but lightweight ask/update/execute semantics without exposing full toolsets.
  - Offers pause/resume/stop/clarification behaviours used by ActiveQueue/ActiveTask tests.

prompt_builders.py
- Builds dynamic system prompts for ask/update/execute with tool signatures and (optionally) rolling activity summaries. Used by TaskScheduler._new_llm_client calls.

task_scheduler.py
- Concrete implementation of BaseTaskScheduler. Major areas:
  - Construction: Ensures the Tasks context exists via TasksStore, seeds caches (queue index, primed/active pointers), and exposes tool dictionaries for ask/update/execute.
  - Public methods:
    • ask/update: start an LLM tool loop with read/update toolsets; integrate optional clarification queues; support optional reasoning/result wrapping.
    • execute: start task execution by numeric id (fast path) or via an LLM outer loop that can create/find and then execute by id. Honors isolated flag to choose detach semantics.
  - Internal execute helpers:
    • _execute_internal: validates runnable instance, detaches according to detach flag via _detach_from_queue_for_activation, constructs ActiveTask, sets activated_by, clones new instance for repeat/trigger, and updates status to active.
    • _execute_queue_internal: starts chain mode and returns an ActiveQueue composite handle.
  - Queue/read helpers:
    • _list_queues, _get_queue, _get_queue_for_task, _walk_queue_from_task: efficient readers that prefer a local queue index when fresh; fall back to storage scans only when required.
    • _reorder_queue, _move_tasks_to_queue, _set_queue, _set_schedules_atomic, _partition_queue: invariant‑preserving mutators that batch writes and maintain local caches, using queue_engine and _queue_utils to ensure symmetry and status normalization.
    • _validated_write: central funnel that enforces invariants (only heads may have start_at; scheduled requires either prev_task or start_at; no active via direct writes; no cross‑queue adjacency; trigger tasks cannot be queued/scheduled).
    • _detach_from_queue_for_activation and _attach_with_links: wrappers around _queue_ops helpers; mark queue index stale.
  - Status helpers and cloning: _update_task_status_instance, _clone_task_instance.
  - Reintegration: stores per‑instance ReintegrationPlan, exposes reinstate_to_previous_queue and _reinstate_task_to_previous_queue delegating to ReintegrationManager.
  - Checkpoints: checkpoint_queue_state/revert_to_checkpoint/get_latest_checkpoint for execute/update flows; optional persistence via UNITY_TS_PERSIST_CHECKPOINTS.
  - LLM loop: _start_loop orchestrates tool calls; prompt builders injected via prompt_builders.py.
  - Clarifications: _make_request_clarification_tool wires asyncio queues used by ActiveQueue and ActiveTask.
  - Misc: _new_llm_client, _best_effort, _normalize_filter_expr, and timing helpers for tests.

How things fit together
1) Storage and models: storage.py (TasksStore/LocalTaskView) mediates all reads/writes to the Unify context defined by the Pydantic models in types/.
2) TaskScheduler is the single entrypoint used by callers/tests. It exposes high‑level methods (ask/update/execute) and tool functions for queue and task manipulation.
3) Queue operations are centralized: high‑level mutators (_set_queue/_reorder_queue/_move_tasks_to_queue/_set_schedules_atomic/_partition_queue) compute invariant‑preserving payloads using queue_engine and enforce neighbour symmetry via _queue_utils.
4) Execution:
   - Isolated: _execute_internal(detach=True) uses _queue_ops.detach_from_queue_for_activation to detach the selected task and record a ReintegrationPlan. ActiveTask runs the actor and updates statuses; on stop/defer, reintegration restores the position using reintegration.ReintegrationManager.
   - Chain: _execute_queue_internal(detach=False) keeps followers linked. ActiveQueue manages sequential execution, awaits linkage barriers for deterministic traversal, routes interjections to current/future tasks, and composes a final summary or per‑task result stream via active_task_done.
5) Clarification: Both ActiveTask and ActiveQueue can request clarification via asyncio queues, allowing external drivers/tests to supply answers.
6) Caching/Indexing: LocalTaskView maintains a local queue index, head‑start cache, membership map, and task_id→log_id memoization. High‑level tools update/mark it stale; readers prefer the view and fall back to storage when needed.

Environment flags
- UNITY_TS_DISABLE_LLM_ROUTER: when set ("1"/"true"/"yes"), ActiveQueue.interject bypasses the LLM router and forwards to the current task.
- UNITY_TS_PERSIST_CHECKPOINTS: when set, execute/update checkpoints are persisted and can be reloaded across runs.
- UNITY_SIM_ACTOR_DURATION: default simulated actor step duration (seconds) for tests/sandboxes when no actor is provided.
 - UNITY_TS_LOCAL_VIEW_OFF: when set, bypasses LocalTaskView caches and always reads from storage (useful for debugging).

Tests directory mapping (tests/test_task_scheduler/)
- conftest.py: Scenario builder/fixtures for shared setup; seeds initial tasks and provides helpers to create queues.
- test_active_task.py: Verifies ActiveTask ask/interject/stop/pause/resume/result behaviours and lifecycle synchronization, including defer→reinstate flows.
- test_active_queue.py: Validates ActiveQueue orchestration: passthrough for singleton queues, numeric execute chaining, defer handling and reintegration, interjection routing (including router bypass and clarification), ask preamble content, dynamic queue edits during execution, active_task_done incremental aggregation, and composite handle adoption by execute_by_id.
- test_execute.py: Covers top‑level execute API: ask/interject/pause/resume/stop on handles, fallback ask when id missing, creation and execute flow, clarification for unknown id, activated_by set to explicit, isolation semantics (detach), and default chain semantics.
- test_task_queue.py: Core queue semantics: get/reorder/insert, primed->queued downgrade when inserted behind head, start_at migration on swaps and new front inserts, listing queues and preserving head start_at, moving tasks to new/existing queues, partitioning queues into multiple parts.
- test_reintegration.py: Detach semantics on starting head/middle, reinstate head/middle with preserved start_at and link symmetry, resilience to deleted neighbours, refusal when trigger present or while active, primed conflict downgrade, plan clearing on completion, and chain→defer restoring next head’s start_at. Uses UNITY_TS_EXEC_CHAIN only for test control.
- test_creation_deletion.py: Creation and deletion flows, including response_policy and multi‑queue creation with start times.
- test_update_tools.py: Field updates via public tools including names/descriptions/status/start_at/deadline/repetition/priority with invariant checks (e.g., scheduled head cannot be queued).
- test_cancel_tasks.py: Cancelling tasks and error on cancelling completed tasks.
- test_contact_manager_integration.py: Ensures TaskScheduler exposes ContactManager.ask as a tool and that ask/update call it in eval tests.
- test_search_tasks.py / test_taskschedule_embedding.py: Search/embedding tests (real Unify required) and derivations of column projections.
- test_taskschedule_contexts_and_clarifications.py: Parent chat context handling and clarification request paths for ask/update.
- test_taskschedule_ask.py: Semantic ask evaluation harness using LLM judgement.
- test_status_recovery_on_failure.py: Recovery semantics: fallback status downgrades on reinstate failures, orphan active guards, and validated_write rejecting active status writes.
- test_ts_tool_timing.py: Micro‑benchmarks/timing smoke tests for tools and internal operations.
- test_simulated_ts.py: Behaviour of the simulated scheduler/handle including clarification and pause/resume/stop flows.

Usage patterns (from tests)
- Construct a scheduler: ts = TaskScheduler()
- Create tasks and build queues: ts._create_task(...), ts._set_queue(queue_id=qid, order=[...], queue_start_at=...)
- Get queues: ts._list_queues(), ts._get_queue(queue_id=...), ts._get_queue_for_task(task_id=...)
- Reorder/move/partition queues: ts._reorder_queue(...), ts._move_tasks_to_queue(...), ts._partition_queue(...)
- Execute:
  • Isolated: await ts.execute(text=str(task_id), isolated=True)
  • Chain: await ts.execute(text=str(task_id)) → returns ActiveQueue
  • Numeric fast path: await ts.execute(text="<id>")
- Interact with active handles:
  • ask/interject to current task or queue; ActiveQueue adds queue context and routes interjections.
  • pause/resume/stop; defer via interjects classified by ActiveTask.
  • await handle.result() for final text, or await queue_handle.active_task_done() for per‑task JSON results.

Why each piece is needed
- Strong types (types/*): Encode invariants and clarify task/queue/trigger schemas, reducing runtime errors.
- Storage adapter (storage.py): Centralizes Unify calls, batching and normalizing IO with a local view for performance.
- Queue engine/utils/ops: Separate concerns: calculating desired queue states (queue_engine), low‑level symmetric linkage updates (_queue_utils), and higher‑level activation/attach semantics (_queue_ops). This separation makes invariants explicit and testable.
- Reintegration (reintegration.py): Guarantees deterministic restoration after isolated runs, enabling safe stop/defer flows without losing user intent or schedule semantics.
- ActiveTask/ActiveQueue: Runtime orchestration for single task vs. chained execution, exposing a consistent steerable interface to callers and LLM loops.
- Prompt builders and LLM loops: Provide tool‑rich prompts for ask/update/execute so the LLM can operate deterministically with validation/clarification.
- TaskScheduler: The cohesive orchestrator gluing storage, models, queue operations, execution handles, reintegration, and LLM tooling into one reliable surface.

Configuration summary
- Set UNITY_TS_DISABLE_LLM_ROUTER to bypass queue interjection routing and deliver to the current task only.
- Set UNITY_TS_PERSIST_CHECKPOINTS to persist execute/update checkpoints across sessions.
- Set UNITY_SIM_ACTOR_DURATION to change default simulated actor speed in tests.

Notes for contributors
- All lifecycle/queue edits should go through _validated_write or the high‑level queue tools to ensure invariants.
- Prefer batch operations (_set_queue, _batch_update_by_task_ids) to reduce backend IO.
- Maintain neighbour symmetry when changing prev/next, and ensure only heads carry start_at.
- Route tool reads/writes through LocalTaskView (e.g., write_entries/delete/get_minimal_rows_by_task_ids). Avoid calling TasksStore directly from tools.
- LocalTaskView owns the queue index and membership map; mark it stale after structural changes when precise updates are not applied.
- ReintegrationPlan must be recorded on activation detachment paths to support later restore semantics.
