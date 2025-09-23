Knowledge Manager Sandbox
=========================

This folder contains an **interactive playground** for the `KnowledgeManager` component that lives in `unity/knowledge_manager/`. The goal of the sandbox is to let you experiment with the manager in isolation – seed imaginary knowledge tables, query them in natural‑language, and observe how the underlying tool‑loop behaves (including steerable tools and clarification requests) before you integrate the manager into a larger system.

### Video walkthroughs

- (Coming soon) General overview
- (Coming soon) In‑flight steerable tools
- (Coming soon) Clarification requests flow

What is the `KnowledgeManager`?
-------------------------------
`KnowledgeManager` is an abstraction that stores structured **knowledge tables** (e.g., `Products`, `Companies`, …) and exposes three high‑level natural‑language methods:

* **`ask(text)`**     – read‑only questions such as *"Which companies shipped EVs in 2022?"*
* **`update(text)`**  – mutations such as *"Add Uptime’s 2023 APAC sales = 17,000 units."*
* **`refactor(text)`** – schema normalization and transformations such as *"Split `warranty` into `warranty_years` (int) and `warranty_coverage` (str)."*

Under the hood the methods launch a _tool‑loop_ where an LLM can call a small, strongly‑typed tool‑kit (`search`, `filter`, `search_join`, `filter_join`, `*_multi_join`, `add_rows`, `update_rows`, `rename_column`, `create_derived_column`, …) until it reaches a final answer. Skim through the tests (if present) for concrete examples of typical usage patterns, clarification flows, semantic search, event logging, etc.

Running the sandbox
-------------------
The entry‑point lives at `sandboxes/knowledge_manager/sandbox.py` and can be executed directly or via Python’s `-m` switch:

```bash
# Basic text‑only session
python -m sandboxes.knowledge_manager.sandbox

# The same, but enable voice I/O via Deepgram + Cartesia
python -m sandboxes.knowledge_manager.sandbox --voice
```

CLI flags
~~~~~~~~~
`sandbox.py` re‑uses the common helper in `sandboxes/utils.py`, so it shares a standard set of options:

```
--voice / -v        Enable voice capture (Deepgram) + TTS playback (Cartesia)
--debug / -d        Show full reasoning steps of every tool‑loop
--traced / -t       Wrap manager calls with unify.traced for detailed logs
--project_name / -p Name of the Unify **project/context** (default: "Sandbox")
--overwrite / -o    Delete any existing data for the chosen project before start
--project_version   Roll back to a specific project commit (int index)
--log_in_terminal   Stream logs to the terminal in addition to writing file logs
--no_clarifications Disable interactive clarification requests (text and voice)
--log_tcp_port      Serve main logs over TCP on localhost:PORT (-1 auto‑picks; 0 off)
--http_log_tcp_port Serve Unify Request logs over TCP on localhost:PORT (-1 auto when UNIFY_REQUESTS_DEBUG)
```

Interactive commands inside the REPL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the sandbox starts you will see a prompt and a small help table. The most important commands are:

* `us {description}`    Build **synthetic knowledge** through the official public tools using `ScenarioBuilder`.
* `usv`                 Same as above but capture the description via **voice** (only when `--voice` is active).
* `r`                   Record a one‑off voice query (again only with `--voice`).
* *free text*           Any other input is auto‑routed to `ask`, `update` or `refactor` depending on intent.
* `save_project` / `sp` Save the current Unify project snapshot so you can roll back later.
* `help` / `h`          Show the in‑session command reference.
* `quit`                Exit the sandbox.

### In‑flight steering (during a running request)
While an `ask`, `update` or `refactor` call is running, you can steer it in‑flight. Type these commands (they only work while a request is active):

- **plain text | /freeform <text> | /i <text>**: Route free-form text via the steering router. The router decides whether it's an ask, interject, pause, resume, stop, or status.
- **/pause | /p**: Pause the running call.
- **/resume**: Resume a paused call.
- **/ask <question> | /? <question>**: Ask a read‑only side question about the currently running call; the answer prints inline without changing the main call’s state.
- Plain text is now equivalent to typing `/freeform <text>`.
- **/r | /record** (voice mode only): Record a voice utterance and route it via freeform. Recording auto‑cancels if the task finishes mid‑capture.
- **/stop | /cancel | /s | /c**: Abort the running call.
- **/status | /st**: Print whether the call is still running or already done.
- **/help | /h**: Show the one‑line controls hint.

Notes:
- Steering commands are ignored when no call is running; you’ll see a small hint if you try.
- In voice mode, during TTS playback you can press Enter to skip. Steering commands (including `/r`) are entered after playback finishes. `/r` auto‑cancels if the run finishes while recording.

Example:
```text
command> Which companies shipped EVs in 2022?
Controls: /i <text>, /pause, /resume, /ask <q>, /freeform <text> (or plain text), /r, /stop, /help
/i restrict to the EU region
/ask which tables are being searched right now?
/pause
/resume
/status
```

### Example session (text mode)
```text
$ python -m sandboxes.knowledge_manager.sandbox -d
KnowledgeManager sandbox – type commands below …

us Generate 20 diverse facts about EV manufacturers including launch years, battery capacities and regional sales.
[generate] Building synthetic knowledge – this can take a moment…
✓ Created tables and inserted rows.

Which companies shipped EVs in 2022?
[ask] → Company A, Company B, …

Add Uptime’s 2023 sales = 17,000 units in APAC.
[update] → Updated 1 row in Companies.

Split `warranty` into `warranty_years` (int) and `warranty_coverage` (str).
[refactor] → Created two columns and migrated values.
```

### Clarification flow (text + voice)
```text
command> Update the warranty details for Zephyr Motors
Controls: /i <text>, /pause, /resume, /ask <q>, /freeform <text> (or plain text), /r, /stop, /help
❓ Clarification requested: Which model? We have Model S and Model X.
/c Model X
✅ Clarification sent.
```
In voice mode you could also answer with `/rc` to record a short spoken clarification.

## Scenario generation via ScenarioBuilder

`us` and `usv` build or update **synthetic knowledge** through the manager’s public tools using `ScenarioBuilder`. The LLM is only allowed to use the exposed tools and will stop when the requested scenario is fully represented. Clarification requests are supported here as well; respond the same way as during normal runs.

## Logging and debugging

- By default, logs are written to `.logs_main.txt` (overwritten each run). Pass `--log_in_terminal` to also stream logs to the terminal.
- Set `--debug` to print full reasoning steps of the tool‑loops.
- Optional TCP streams:
  - Main logs: `--log_tcp_port -1` auto‑picks an available port (or specify an explicit port). Connect with `nc 127.0.0.1 <PORT>`.
  - Unify Request logs only: `--http_log_tcp_port -1` auto‑enables when `UNIFY_REQUESTS_DEBUG` is set; connect with `nc 127.0.0.1 <PORT>`.
- A dedicated Unify Request log file is also written to `.logs_unify_requests.txt`.

### Troubleshooting
* **Deepgram / Cartesia keys** – if you use `--voice`, make sure the environment variables `DEEPGRAM_API_KEY` and `CARTESIA_API_KEY` are set.
* **Unify backend access** – the sandbox will attempt to create contexts and logs in your configured Unify project. If your credentials (`UNIFY_KEY`, `UNIFY_BASE_URL`) are missing or invalid you may see HTTP errors.
* **Linter complaints** – the interactive session is powered by an LLM; if you hit a bug look at the `--debug` reasoning trace first.

Happy exploring knowledge! 🎉
