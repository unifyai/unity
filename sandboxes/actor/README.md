Actor Sandbox
=============

This folder contains an **interactive "flight simulator"** for the various `Actor` components that live in `unity/actor/`. The goal of this sandbox is to launch, monitor, and steer any of our agent implementations (`HierarchicalActor`, `CodeActActor`, etc.) as they work on a high-level goal.

## 🎥 Demo Videos
- **In Flight Modifications (#TODO: record a loom)** - How to pause, interject, and redirect running tasks
- **Generalizing A Process [Part 1](https://www.loom.com/share/d341bbe882a648baa19b759511e306f2) [Part 2](https://www.loom.com/share/9aa1d179b5d34eafa7954775a553f1e8) [Part 3](https://www.loom.com/share/e850593b76d1494caa8aa4698c0d4c61)** - How to teach a repeatable process/workflow.
- **Desktop Control (#TODO: integrate  and test desktop with the actor)** - Control a desktop terminal + a browser session to complete a task

[What are Actors?](https://github.com/unifyai/unity/blob/main/unity/actor/README.md)
----------------
**Actors** are the "brains" of the agent framework. They are responsible for taking a natural-language goal and executing a plan to achieve it. This sandbox allows you to select and run any of the following actors:

* **`HierarchicalActor`** – A sophisticated planner that generates and self-corrects Python code to solve tasks.
* **`CodeActActor`** – An agent that solves tasks primarily by writing and executing Python code in a REPL sandbox.

When an actor starts a task, it returns a `SteerableToolHandle`, which the sandbox uses to provide a rich, interactive control session.

## Prerequisites

### Magnitude Agent Service Setup
Some actors (particularly `HierarchicalActor` and `CodeActActor`) require the Magnitude BrowserAgent service to be running for web automation tasks.

The repo uses Unity's modified `magnitude-core` for the agent service (see `agent-service/package.json` dependency: `"magnitude-core": "file:../magnitude/packages/magnitude-core"`). The `magnitude/` directory contains our fork with Unity-specific enhancements.

**1. Build local magnitude-core:**
```bash
# First, clone the unity repository if you haven't already
git clone <unity-repo-url>
cd unity

# Clone Unity's magnitude fork into the magnitude/ subdirectory
git clone https://github.com/unifyai/magnitude.git magnitude
cd magnitude
git checkout unity-modifications  # Our branch with Unity enhancements

# Build magnitude-core
cd packages/magnitude-core
npm install
npm run build
```

**2. Install Agent Service deps:**
```bash
cd ../..  # Back to repo root from magnitude/packages/magnitude-core
cd agent-service
npm install
```

**3. Configure Environment:**
Create a `.env` file in the `agent-service` directory:
```bash
# agent-service/.env
ANTHROPIC_API_KEY="sk-ant-..."
UNIFY_BASE_URL="..."
UNIFY_KEY="..."
# Optional depending on configured LLM clients in magnitude-core (BAML)
GOOGLE_API_KEY="..."
OPENROUTER_API_KEY="..."
OPENAI_API_KEY="..."
```

**4. Start the Service:**
```bash
cd agent-service
npx ts-node src/index.ts
```

The service will run on `http://localhost:3000` (configurable via `--agent-url`).

> If you change code in `magnitude/packages/magnitude-core`, rebuild it and refresh the service dependency:
> ```bash
> # Rebuild local core
> cd magnitude/packages/magnitude-core
> npm run build
>
> # Reinstall in agent-service to pick up the updated local package
> cd ../..  # Back to repo root
> cd agent-service
> npm install --force
> ```

> You can also use `yalc` for a faster inner loop:
> - In `magnitude-core`: `npm run build && npx yalc publish --push`
> - In `agent-service`: `npx yalc add magnitude-core`
> - Re-run publish after changes to auto-push updates.

Running the sandbox
-------------------
The entry-point lives at `sandboxes/actor/sandbox.py` and can be executed directly or via Python's `-m` switch. You must select which actor to run.

```bash
# Run the CodeActActor in a text-only session
python -m sandboxes.actor.sandbox --actor code_act

# Run the HierarchicalActor with voice I/O enabled with persist=True mode.
python -m sandboxes.actor.sandbox --actor hierarchical --voice --persist
```

### Quick Start Example
Here's a complete setup workflow:

```bash
# Terminal 1: Build core and start the agent service
git clone <unity-repo-url> && cd unity
git clone https://github.com/unifyai/magnitude.git magnitude
cd magnitude && git checkout unity-modifications
cd packages/magnitude-core && npm i && npm run build
cd ../..  # Back to repo root
cd agent-service && npm i && npx ts-node src/index.ts

# Terminal 2: Run the actor sandbox
# (assuming you're in the unity repo root)
python -m sandboxes.actor.sandbox --actor hierarchical

# In the sandbox:
goal-for-hierarchical> find the latest news on TechCrunch
```

CLI flags
~~~~~~~~~
The sandbox uses the common helper in `sandboxes/utils.py`, so it shares a standard set of options, plus actor-specific flags:

**Actor-Specific Options:**
```
--actor / -a        [hierarchical|tooloop|code_act|browser_use] Select which actor to run (default: code_act)
--headless          Run the actor's browser in headless mode (no visible UI)
--agent-url         URL of the agent service (default: http://localhost:3000)
--persist           Enable persistent, long-running sessions that wait for interjections
```

**Project & Session Management:**
```
--project_name / -p Name of the Unify **project/context** (default: "Sandbox")
--project_version   Rollback to a specific project commit by index (-1 for latest, default)
--overwrite / -o    Delete any existing data for the chosen project before start
```

**I/O & Interaction:**
```
--voice / -v        Enable voice capture (Deepgram) + TTS playback (Cartesia)
--no_clarifications Disable interactive clarification requests (text and voice)
```

**Debugging & Logging:**
```
--debug / -d        Show full reasoning steps of every tool-loop
--traced / -t       Wrap manager calls with unify.traced for detailed logs
--log_in_terminal   Stream logs to the terminal in addition to writing file logs
--log_tcp_port      Serve main logs over TCP on localhost:PORT (-1 auto-picks; 0 off)
--http_log_tcp_port Serve HTTP request logs over TCP on localhost:PORT
```

Interactive commands inside the REPL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the sandbox starts, you will be prompted to provide a goal for the actor:

```
┌─────────────── Commands ───────────────┐
│ <your goal>         - High-level task for the actor to perform    │
│ custom              - Interactively provide a multi-line goal     │
│ save_project | sp   - Save project snapshot with current state    │
│ help | h            - Show this help message                      │
│ quit | exit         - Exit the sandbox                            │
└────────────────────────────────────────┘
```

**Command Details:**
* **`<your goal>`** – A high-level task for the actor to perform (e.g., "find the current weather in Karachi").
* **`custom`** – Enter a multi-line mode to provide a more detailed goal.
* **`save_project` / `sp`** – Save the current project state as a versioned snapshot.
* **`help` / `h`** – Show the in-session command reference.
* **`quit` / `exit`** – Exit the sandbox.

### In-flight steering (during a running task)
While an actor is working on a goal, you can steer it in-flight using commands. This functionality is powered by `sandboxes/utils.py`:

* **`/i <text>` or just plain text** – Interject with new guidance for the actor.
* **`/pause`** – Pause the running task.
* **`/resume`** – Resume a paused task.
* **`/ask <question>`** – Ask a read-only question about the current task.
* **`/stop`** – Abort the running task.
* **`/r`** (voice mode only) – Record a voice interjection.
* **`/help` / `/h`** – Show the one-line controls hint.

Notes:
- Steering commands are ignored when no task is running; you'll see a small hint if you try.
- In voice mode, during TTS playback you can press Enter to skip. Steering commands (including `/r`) are entered after playback finishes.

### Example session (text mode with CodeActActor)
```text
$ python -m sandboxes.actor.sandbox --actor code_act
Actor Sandbox
-------------
...

goal-for-code_act> find the top news headline on BBC
▶️  Starting task for goal: "find the top news headline on BBC"...
Controls: /i <text>, /pause, /resume, /ask <q>, /freeform <text>, /stop, /help

[... actor logs appear here as it works ...]

---
✅ Task Completed. Final Result:
Top headline: Major breakthrough in climate talks reported
Summary: Negotiators have reached a landmark agreement on carbon emissions after marathon talks.
---
```

### Example: clarification flow
If an actor needs more information, it will pause and ask a question:

```text
goal-for-hierarchical> book a flight for me
▶️  Starting task for goal: "book a flight for me"...
Controls: /i <text>, /pause, /resume, /ask <q>, /freeform <text>, /stop, /help, /c <answer> (clarify)
❓ Clarification requested: Sure, I can book a hotel. What city are you traveling to?
/c to Tokyo
✅ Clarification sent.
```

### Example: project management
You can save snapshots and manage project versions:

```text
goal-for-code_act> save_project
💾 Project saved at commit abc123def456
goal-for-code_act> quit

# Later, rollback to that version
$ python -m sandboxes.actor.sandbox --actor code_act --project_version 0
[version] Rolled back to commit abc123def456
```

## Project Management & Versioning

The actor sandbox supports full project lifecycle management through Unify's versioning system:

**Saving Progress:**
- Use `save_project` or `sp` command to create snapshots of your current session
- Each save creates a versioned commit with timestamp
- Snapshots include all conversation history and actor state

**Version Rollback:**
- Use `--project_version N` to rollback to a specific commit (by index)
- `--project_version -1` uses the latest version (default)
- `--project_version 0` uses the oldest saved version
- Useful for A/B testing different approaches or recovering from issues

## Troubleshooting

### Common Issues

**Voice & Audio:**
* **Deepgram / Cartesia keys** – If you use `--voice`, make sure the environment variables `DEEPGRAM_API_KEY` and `CARTESIA_API_KEY` are set.
* **Audio device access** – Ensure your microphone permissions are enabled for the terminal/Python.
* **TTS playback issues** – If voice output is garbled, try restarting the sandbox or checking your system audio settings.

**Project & Authentication:**
* **Unify backend access** – The sandbox will attempt to create contexts and logs in your configured Unify project. If your credentials (`UNIFY_KEY`, `UNIFY_BASE_URL`) are missing or invalid you may see HTTP errors.
* **Project version errors** – If `--project_version N` fails, check available commits with `unify.get_project_commits()` in a Python shell.
* **Permission issues** – Make sure your Unify account has access to create/modify projects.

**Actor-Specific:**
* **Actor selection** – Make sure to specify a valid actor name with `--actor`. If unsure, use the default `code_act`.
* **Browser automation** – For actors that use browser control, ensure Playwright is properly installed with `playwright install`.
* **Agent service connection** – If using `--agent-url`, verify the service is running and accessible at `http://localhost:3000/health`.
* **Agent service not running** – Many actors require the Magnitude service. Start it with `cd agent-service && npx ts-node src/index.ts`.
* **Agent service environment** – Ensure `.env` file in `agent-service/` has valid `ANTHROPIC_API_KEY`, `UNIFY_BASE_URL`, and `UNIFY_KEY`.
* **Magnitude branch** – Ensure you're on the `unity-modifications` branch in the `magnitude/` directory for Unity-specific enhancements.
* **Headless mode issues** – Try running without `--headless` first to see if browser automation is working.

**Performance & Logging:**
* **Slow responses** – Check your network connection to Unify services and LLM providers.
* **Log file growth** – Monitor `.logs_actor_sandbox.txt` for size; rotate or delete if needed.
* **TCP log ports** – If `--log_tcp_port` conflicts, try a different port or use `-1` for auto-selection.

### Getting Help
* Check the main Unity documentation for actor-specific configuration
* Review `.logs_actor_sandbox.txt` for detailed error traces
* Use `--debug` mode to see full reasoning steps
* Try `--traced` mode for detailed Unify API interaction logs

Happy experimenting! 🎉
