Task Scheduler Sandbox
======================

This folder contains an **interactive playground** for the `TaskScheduler` component that lives in `unity/task_scheduler/`.  The goal of the sandbox is to let you experiment with the scheduler in isolation – seed imaginary backlogs, query them in natural-language, and observe how the underlying tool-loop behaves before you integrate the manager into a larger system.

Prefer a quick demo? Watch this [video walkthrough](https://www.loom.com/share/45334c26c7c448a485dc17bd6590ce09?sid=912ca961-21b1-4698-b6ac-8fb58a1357c3)

What is the `TaskScheduler`?
----------------------------
`TaskScheduler` is an abstraction that stores **tasks** (name, description, schedule, triggers, priority, rolling summary, …) and exposes three high-level natural-language methods:

* **`ask(text)`**      – read-only questions such as *"Which tasks are due this week?"*
* **`update(text)`**    – mutations such as *"Move task 7 behind task 3 and set its priority to high."*
* **`execute_task(id)`** – begin **working** on a specific task (*"Start task 12."*).  Internally this launches a Planner that performs the concrete actions required to complete the task.

Under the hood all three methods launch a _tool-loop_ where an LLM can call a small, strongly-typed tool-kit (`_create_task`, `_update_task_status`, `_get_task_queue`, …) until it reaches a final answer.  The extensive unit-test suite in `tests/test_task_scheduler/` exercises all public and private helpers – skim through those tests if you want concrete examples of typical usage patterns, clarification flows, vector search, event logging, etc.

Running the sandbox
-------------------
The entry-point lives at `sandboxes/task_scheduler/sandbox.py` and can be executed directly or via Python’s `-m` switch:

```bash
# Basic text-only session
python -m sandboxes.task_scheduler.sandbox

# The same, but enable voice I/O via Deepgram + Cartesia
python -m sandboxes.task_scheduler.sandbox --voice
```

CLI flags
~~~~~~~~~
`sandbox.py` re-uses the common helper in `sandboxes/utils.py`, so it shares a standard set of options:

```
--voice / -v        Enable voice capture (Deepgram) + TTS playback (Cartesia)
--debug / -d        Show full reasoning steps of every tool-loop
--traced / -t       Wrap manager calls with unify.traced for detailed logs
--project_name / -p Name of the Unify **project/context** (default: "Sandbox")
--overwrite / -o    Delete any existing data for the chosen project before start
--project_version   Roll back to a specific project commit (int index)
```

Interactive commands inside the REPL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the sandbox starts you will see a prompt and a small help table.  The most important commands are:

* `us {description}`    Build **synthetic tasks** through the official public tools using `ScenarioBuilder`.
* `usv`                 Same as above but capture the description via **voice** (only when `--voice` is active).
* `r`                   Record a one-off voice query (again only with `--voice`).
* *free text*           Any other input is auto-routed to `ask`, `update` **or** `start` depending on intent.
* `save_project` / `sp` Save the current Unify project snapshot so you can roll back later.
* `help` / `h`          Show the in-session command reference.
* `quit`                Exit the sandbox.

### Steering controls (during a running request)
While an `ask`, `update` or `start` (execute) call is running, you can steer it in-flight. Type these commands (they only work while a request is active):

- **/i <text> | /interject <text> | plain text**: Interject guidance that the tool-loop should incorporate immediately. If you don’t prefix with `/`, any plain text you type during a run is treated as an interjection.
- **/pause | /p**: Pause the running call.
- **/resume**: Resume a paused call.
- **/ask <question> | /? <question>**: Ask a read-only side question about the currently running call; the answer prints inline without changing the main call’s state.
- **/freeform <text>**: Route free-form text to the best steering command (ask/interject/pause/resume/stop/status).
- **/r | /record** (voice mode only): Record a voice utterance and route it via freeform. Recording auto-cancels if the task finishes mid-capture.
- **/stop | /cancel | /s | /c**: Abort the running call.
- **/status | /st**: Print whether the call is still running or already done.
- **/help | /h**: Show the one-line controls hint.

Notes:
- Steering commands are ignored when no call is running; you’ll see a small hint if you try.
- In voice mode, during TTS playback you can press Enter to skip. Steering commands (including `/r`) are entered after playback finishes. Use `/r` to record a steering utterance and route it via freeform; it auto-cancels if the run finishes while recording.

Example session (text mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
```text
$ python -m sandboxes.task_scheduler.sandbox -d
TaskScheduler sandbox – type commands below …

us Generate a backlog of 12 realistic product-development tasks across Inbox, Next, Scheduled and Waiting queues.
[generate] Building synthetic tasks – this can take a moment…
✓ Created 12 tasks across 4 queues.

What is the next task I should work on?
[ask] → The next task is "Implement login screen" (task_id 3).

Start task 3.
[start] → Task 3 is now active.

Pause the active task.
[update] → Task 3 paused.
```

Troubleshooting
---------------
* **Deepgram / Cartesia keys** – if you use `--voice`, make sure the environment variables `DEEPGRAM_API_KEY` and `CARTESIA_API_KEY` are set.
* **Unify backend access** – the sandbox will attempt to create contexts and logs in your configured Unify project.  If your credentials (`UNIFY_KEY`, `UNIFY_BASE_URL`) are missing or invalid you may see HTTP errors.
* **Linter complaints** – the interactive session is powered by an LLM; if you hit a bug look at the `--debug` reasoning trace first.

Happy scheduling! 🎉
