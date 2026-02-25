# ConversationManager

The `ConversationManager` is the live orchestration layer that handles real-time conversations with users across multiple mediums (phone calls, emails, SMS, web chat). It acts as the "front office" that interfaces with users while delegating complex reasoning to the `Actor` (the "brain").

## Prerequisites

### Required Dependencies

1. **Python environment** — Use the project's virtual environment
   ```bash
   # From project root
   source .venv/bin/activate
   ```

### Running Tests

```bash
tests/parallel_run.sh tests/conversation_manager/
```

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│                             External World                                │
│       (Phone/Twilio, Email/Gmail, SMS, Web Chat, LiveKit Voice)           │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                             CommsManager                                  │
│    Subscribes to GCP PubSub topics, converts external messages to events │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ In-Memory Event Broker (app:comms:*)
┌───────────────────────────────────────────────────────────────────────────┐
│                          ConversationManager                              │
│                                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────────────┐  │
│  │  ContactIndex   │  │ NotificationBar │  │      Main CM Brain        │  │
│  │  (live state)   │  │    (pending)    │  │   (async tool loop)       │  │
│  └─────────────────┘  └─────────────────┘  └───────────────────────────┘  │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                          EventHandler                               │  │
│  │    Routes events to appropriate handlers, triggers LLM runs         │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                          CallManager                                │  │
│  │    Manages voice calls (Twilio/LiveKit), realtime vs TTS modes      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (via Actions / Actor tools)
┌───────────────────────────────────────────────────────────────────────────┐
│                                 Actor                                     │
│   The "brain" — runs code-first plans and orchestrates across primitives   │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### `ConversationManager` (`conversation_manager.py`)

The central orchestrator that:
- Maintains live conversation state (`ContactIndex`, `NotificationBar`)
- Runs the "Main CM Brain" async tool loop to decide responses and actions
- Routes user input to either the Main Brain or active nested handles
- Manages voice call lifecycle via `CallManager`

Key methods:
- `wait_for_events()` — Main event loop, subscribes to event broker channels
- `run_llm()` — Triggers the Main CM Brain to process state and generate responses
- `interject_or_run(content)` — Routes user input to active `ask` handle or Main Brain

### `ConversationManagerHandle` (`handle.py`)

A steerable handle that external components (e.g., the Actor) use to interact with the live conversation:
- `ask(question, response_format)` — Ask the user a question and wait for their answer
- `interject(message)` — Inject information into the conversation (publishes a steering event)
- `get_full_transcript()` — Retrieve recent conversation history

### `EventHandler` (`domains/event_handlers.py`)

Registry-based event dispatcher that routes events to handlers:
- `@EventHandler.register(EventClass)` decorator registers handlers
- Handles all communication events (SMS, Email, Phone, etc.)
- Triggers LLM runs and updates state

### `Events` (`events.py`)

Dataclass-based event definitions:
- **Comms Events**: `SMSReceived`, `EmailReceived`, `InboundPhoneUtterance`, `OutboundPhoneUtterance`, etc.
- **Actor Events**: `ActorRequest`, `ActorResponse`, etc.
- **Control Events**: `DirectMessageEvent`, `NotificationInjectedEvent`, etc.

### `CallManager` (`domains/call_manager.py`)

Manages voice call lifecycle using the TTS (Text-to-Speech) pipeline.

---

## Voice Architecture

The voice system uses the "fast brain / slow brain" architecture where the Voice Agent (fast brain) handles conversation autonomously while the Main CM Brain (slow brain) provides guidance.

```
User speaks → STT transcribes → Fast Brain (text LLM) → TTS speaks
    ↑                                                      ↓
    └──────── Main CM Brain provides CallGuidance ─────────┘
```

Uses separate STT/TTS services with a lightweight text-based LLM (gpt-5-mini) for fast responses.

---

## The `ConversationManagerHandle.ask` Flow

When the Actor or another manager needs to ask the user a question:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                     ConversationManagerHandle.ask                         │
├───────────────────────────────────────────────────────────────────────────┤
│  1. Build context (recent transcript from ContactIndex)                   │
│  2. Start inner async tool loop (gemini-2.5-flash)                        │
│  3. Register as cm.active_ask_handle                                      │
│                                                                           │
│  Inner LLM has two paths:                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  PATH 1 (INFER): If 90%+ confident from transcript                  │  │
│  │      → Return JSON with {acknowledgment, final_answer}              │  │
│  │      → No tool calls, instant response                              │  │
│  │                                                                     │  │
│  │  PATH 2 (ASK): If genuinely ambiguous                               │  │
│  │      → Call ask_question(text) tool                                 │  │
│  │      → Publishes DirectMessageEvent → voice speaks question         │  │
│  │      → Blocks waiting for user_reply_future                         │  │
│  │      → User reply routes via cm.interject_or_run()                  │  │
│  │      → Future resolves, tool returns "User replied: ..."            │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

Key insight: While `active_ask_handle` is set, user input is routed to the nested loop, **bypassing the Main CM Brain**.

---

## Event Flow

### Event Channels

| Channel Pattern | Purpose |
|-----------------|---------|
| `app:comms:*` | Communication events (SMS, email, phone, etc.) |
| `app:actor:*` | Actor request/response events |
| `app:call:*` | Voice call control events |
| `app:logging:*` | Transcript logging events |
| `app:managers:*` | State manager operations |

### Key Events

| Event | Description |
|-------|-------------|
| `DirectMessageEvent` | Bypass Main CM Brain, send message directly to user |
| `NotificationInjectedEvent` | Inject notification into Main CM Brain's context |
| `InboundPhoneUtterance` / `InboundUnifyMeetUtterance` | User spoke during a call |
| `OutboundPhoneUtterance` / `OutboundUnifyMeetUtterance` | Assistant response during a call |
| `ActorRequest` | Request the Actor to perform an action |

---

## File Structure

```
conversation_manager/
├── __init__.py              # Public exports (start_async, stop_async, etc.)
├── base.py                  # Base class for handle abstraction
├── conversation_manager.py  # Main ConversationManager class
├── handle.py                # ConversationManagerHandle implementation
├── events.py                # Event dataclass definitions
├── event_broker.py          # In-memory event broker utilities
├── in_memory_event_broker.py # InMemoryEventBroker implementation
├── comms_manager.py         # GCP PubSub → event broker bridge
├── simulated.py             # Simulated implementation for testing
├── main.py                  # Entry point for running as a service
├── debug_logger.py          # Debug logging utilities
├── utils.py                 # Shared utilities
├── settings.py              # ConversationManager settings (SETTINGS.conversation.*)
├── prompt_builders.py       # Dynamic prompt construction (like other managers)
├── task_actions.py          # Task management utilities
├── livekit_unify_adapter.py # LiveKit LLM adapter routing through Unify
│
├── domains/                 # Sub-components
│   ├── actions.py           # Action definitions (send SMS, email, etc.)
│   ├── assistant.py         # Assistant profile dataclass
│   ├── brain.py             # BrainSpec builder for Main CM Brain
│   ├── brain_tools.py       # Read-only tools for Main CM Brain
│   ├── brain_action_tools.py # Side-effecting tools (comms, task steering)
│   ├── call_manager.py      # Voice call management
│   ├── comms_utils.py       # Communication utilities
│   ├── contact_index.py     # Live contact/conversation state
│   ├── event_handlers.py    # Event routing and handling
│   ├── llm.py               # LLM wrapper for Main CM Brain
│   ├── managers_utils.py    # Manager initialization utilities
│   ├── notifications.py     # Notification bar management
│   ├── proactive_speech.py  # Proactive silence-filling logic
│   └── renderer.py          # State → prompt rendering
│
└── medium_scripts/          # Medium-specific voice handling
    ├── call.py              # Voice calls (STT → Fast Brain → TTS)
    └── common.py            # Shared voice utilities
```

---

## Testing

Tests are located in `tests/conversation_manager/`:

| File | Description |
|------|-------------|
| `conftest.py` | Test fixtures for in-memory event broker and CM setup |
| `test_comms.py` | Integration tests for cross-medium communication (eval tests) |
| `test_managers.py` | Tests for Actor integration |
| `test_simulated.py` | Tests using simulated implementation |
| `test_utils.py` | Unit tests for utility functions |
| `test_logging.py` | Tests for logging infrastructure |
| `test_in_memory_event_broker.py` | Tests for the in-memory event broker |
| `test_voice_mode_architecture.py` | Tests for voice architecture |
| `test_event_bus_integration.py` | Tests for EventBus integration |

### Running Tests

```bash
# Run all conversation_manager tests (with per-test parallelism)
tests/parallel_run.sh tests/conversation_manager/

# Run a specific test file
tests/parallel_run.sh tests/conversation_manager/test_comms.py

# Run directly with pytest (for debugging)
.venv/bin/python -m pytest tests/conversation_manager/test_comms.py -v --timeout=300
```

### Test Categories

These tests are marked as **eval tests** (`pytest.mark.eval`) because they exercise end-to-end LLM behavior:
- Results depend on LLM responses (cached after first run)
- Some tests may be flaky due to timing-dependent voice call flows

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `Conversation manager did not subscribe` | Check CM process logs; ensure no import errors |
| Voice call tests timeout | These are timing-sensitive eval tests; retry or check LLM cache |
| `FileNotFoundError: python` | Ensure `.venv/bin/python` is used (handled by conftest) |
