## Overview

This folder contains **end-to-end steerability tests** for `HierarchicalActor` using **simulated state managers** (contacts, transcripts, knowledge, tasks).

The goal is to validate the Actor’s *in-flight control surface*:
- **Interjections**: `HierarchicalActorHandle.interject(...)` routes natural-language guidance to the correct in-flight manager handles via `SteerableToolPane`.
- **Clarifications**: bottom-up clarification events from managers are surfaced to the user and answers are routed back to the correct handle.
- **Pause/Resume**: user pause/resume applies to the plan runtime *and* in-flight manager handles.

These tests are **infrastructure-first**: they primarily assert on pane events (`steering_applied`, `clarification`) and basic completion sanity checks, rather than fragile LLM output content.

## Key patterns used throughout

- **Canned plans + deterministic gating**: plans include `TEST_GATE = asyncio.Event()` so tests can deterministically:
  - wait for N handles to be registered in the pane
  - steer (pause/interject/answer_clarification/resume)
  - then release the gate and allow completion

- **Natural language steering**: interjections are phrased like real user messages (no “broadcast this” or internal references like “main_plan”).

- **Pane-based verification**: tests validate routing/steering through pane events:
  - `type="handle_registered"`
  - `type="steering_applied"` with payload `{method, status, args...}`
  - `type="clarification"` + `pane.get_pending_clarifications()`

## Test files

| File | What it covers |
|------|----------------|
| `test_clarification_flow.py` | Clarification bubble-up + answer routing + pending tracking (deterministically forced) |
| `test_interjection_routing.py` | Targeted vs broadcast routing across mixed `ask` + `update` handles (and selective routing by manager type) |
| `test_complex_scenarios.py` | Multi-stage workflows: staged correction + sequential interjections across stages |
| `test_pause_resume.py` | Pause/resume propagation across in-flight handles + interjection while paused |
| `test_edge_cases.py` | Boundary conditions: late interjection, rapid successive interjections, interject while clarification pending, empty/vague interjections |

## Clarification forcing (test-only)

`conftest.py` defines “clarification-forcing” wrappers around simulated managers. This allows tests to deterministically trigger clarification flows **without requiring** the Actor-generated (or canned) plan to pass `_requests_clarification=True`.

## Running the tests

Run just this suite:

```bash
tests/parallel_run.sh tests/test_actor/test_state_managers/test_simulated/test_steerability/
```

Run a single file:

```bash
tests/parallel_run.sh tests/test_actor/test_state_managers/test_simulated/test_steerability/test_edge_cases.py
```

Run a single test:

```bash
tests/parallel_run.sh tests/test_actor/test_state_managers/test_simulated/test_steerability/test_edge_cases.py::test_late_interject_after_handle_completes
```

## Debugging tips

- **Pane events**:
  - `handle.pane.get_recent_events(n=500)`
  - filter for `type == "steering_applied"` and inspect `payload.method`, `payload.status`
- **Registered handles**:
  - `await handle.pane.list_handles()`
- **Pending clarifications**:
  - `await handle.pane.get_pending_clarifications()`
