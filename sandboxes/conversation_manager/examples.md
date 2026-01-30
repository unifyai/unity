# ConversationManager Sandbox — Example Scenarios

## Selecting / switching configurations

On startup you’ll be prompted to pick a mode (and it will auto-load the last-used choice from `.cm_sandbox_config`).

To switch at runtime (REPL):

```text
cm> config
⚠️  Switching configuration will:
- Restart ConversationManager
- Clear all conversation state (threads, notifications, in-flight actions)
- Auto-snapshot the project before switching (rollback is possible)

Continue? (y/N): y
Select Actor Configuration:
1. SandboxSimulatedActor (simulated managers, no computer interface)
2. CodeActActor + Simulated Managers (mock computer backend)
3. CodeActActor + Real Managers + Real Computer Interface
Enter choice (1-3) or press Enter for last used: 2
🔄 Restarting sandbox with selected configuration...
```

## Basic SMS conversation

```text
cm> sms Can you help me schedule a meeting?
[SMS → User] ...

cm> sms Next Tuesday at 2pm works for me.
[SMS → User] ...
```

## Email conversation

```text
cm> email Re: invoice | Can you confirm the outstanding amount and due date?
[Email → User] ...
```

## Phone call with steering

```text
cm> call
call> say I need help with my account

call> /i Actually, it’s a billing issue.
✅ Interjection sent

call> /ask What’s the plan?
🧾 ...

call> end_call
```

## Trace / event tree / logs (CodeAct modes)

These commands help you see what CodeAct generated and which managers were called:

```text
cm> trace 3
TRACE — Turn 1 ...

cm> tree
📊 Event Tree (Current State)
...

cm> show_logs actor
...

cm> collapse_logs all
...
```

If you want automatic traces in REPL:

```text
$ python -m sandboxes.conversation_manager.sandbox --show-trace
```

## Phone call with voice (`--voice`)

```text
$ python -m sandboxes.conversation_manager.sandbox --voice

cm> call
call> sayv
(record, transcribe)
▶️ I need help with my account
[Phone → User] ...
```

## Scenario seeding (idle-only)

```text
cm> us Generate a short SMS conversation about billing.
[generate] Building synthetic conversations – this can take a moment…
✅ Scenario generated (synthetic transcript): ...
✅ Published N inbound event(s) into ConversationManager.
```

## Real-comms mode (REPL only)

```text
$ python -m sandboxes.conversation_manager.sandbox --real-comms

cm> sms This is a test message

⚠️ REAL-COMMS MODE: Confirm action
   Medium: SMS
   Recipient: +1...
   Content:
   This is a test message
   Proceed? (Y/N) [default: N]:
```

## Mode 3 (real web mode) flags

```bash
python -m sandboxes.conversation_manager.sandbox \
  --agent-server-url http://localhost:3000 \
  --agent-mode web \
  --headless
```
