# Action Simulator

Simulates `CodeActActor.act` sessions for the Console action pane at
`http://localhost:3333/assistants`. Useful for developing and testing the
action viewer UI without running a real Unity deployment.

Events faithfully replicate the real event model: `ManagerMethod`
incoming/outgoing lifecycle events, `ToolLoop` messages (user requests,
LLM thinking, tool calls, tool results, steering), nested child actions,
concurrent inner tool loops, and `_thinking_in_flight` sentinels.

## Scripts

| Script | Purpose |
|---|---|
| `simulate_action_stream.py` | Generate and deliver simulated action events |
| `clear_action_events.sh` | Wipe all action events from the local Orchestra DB |

## Flags

| Flag | Description |
|---|---|
| `--stream` | Push events via SSE to the Console with realistic delays (default if no delivery flag given) |
| `--save` | Write events to Orchestra for historical access and page-refresh persistence |
| `--scenario` | `persistent` (default) or `single_action` |
| `--speed N` | Delay multiplier for `--stream` — e.g. `2` for 2x faster, `0.5` for half speed |
| `--clear` | Wipe old events and pause for a browser refresh before starting |

`--stream` and `--save` are independent and combinable. If neither is
given, `--stream` is assumed.

## Scenarios

**persistent** — A long-running `act(persist=True)` session demonstrating:
discovery (GuidanceManager + FunctionManager search), nested
`WebSearcher.ask` with inner `_search` / `_extract` tool calls,
clarification requests, `execute_code` spawning concurrent
`KnowledgeManager.update` + `ContactManager.update` with interleaved
inner tool loops, interjections, pause/resume/stop steering.

**single_action** — A one-shot `act(persist=False)` action demonstrating:
sub-agent delegation via `execute_function(primitives.actor.act)`, the
sub-agent's own `CodeActActor` with nested `ContactManager.ask`, code
execution for email drafting, and a post-completion `StorageCheck` phase
that reviews the trajectory and decides nothing is worth storing.

## Examples

```bash
# Stream the persistent scenario (default)
.venv/bin/python scripts/dev/action_simulator/simulate_action_stream.py

# Stream single_action at half speed, clearing old data first
.venv/bin/python scripts/dev/action_simulator/simulate_action_stream.py \
    --scenario single_action --speed 0.5 --clear

# Save to Orchestra only (no streaming, instant)
.venv/bin/python scripts/dev/action_simulator/simulate_action_stream.py --save

# Stream and save simultaneously
.venv/bin/python scripts/dev/action_simulator/simulate_action_stream.py --stream --save

# Just wipe old events
scripts/dev/action_simulator/clear_action_events.sh
```

## Prerequisites

| Delivery | Requires |
|---|---|
| `--stream` | Console running at `http://localhost:3333` |
| `--save` | Local Orchestra at `http://127.0.0.1:8000` |
| `--stream --save` | Both |

Start both with `console/scripts/local.sh` (which also starts Orchestra).
