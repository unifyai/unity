Contact Manager Sandbox
=======================

This folder contains an **interactive playground** for the `ContactManager` component that lives in `unity/contact_manager/`.  The goal of the sandbox is to let you experiment with the manager in isolation – create imaginary contacts, query them in natural-language, and observe how the underlying tool-loop behaves before you integrate the manager into a larger system.

### Video walkthroughs

- General overview (comprehensive, but older sandbox version): [Loom video](https://www.loom.com/share/31086a24fb9b4f9b8e0b69184773f942?sid=bd4588db-db8a-49a8-8f01-3d5741227006)
- In-flight steerable tools: [Loom video](https://www.loom.com/share/f3833d5e1f004c27a19ab9847dc63ab0?sid=69a5c29a-3ea8-4334-8d2d-d349b70d0429)
- Clarification requests flow: [Loom video](https://www.loom.com/share/ac773593bb8a41de8f05c6f0c3dead73?sid=3ecd861d-83e5-45d0-bcb7-7390b29f559b)

What is the `ContactManager`?
-----------------------------
`ContactManager` is an abstraction that stores contact records (first name, surname, phone, bio, rolling summary, plus any number of **custom columns**) and exposes two high-level natural-language methods:

* **`ask(text)`**   – read-only questions such as *"What is Alice's phone number?"*
* **`update(text)`** – mutations such as *"Update Bob's phone number to +123…"*

Under the hood both methods launch a _tool-loop_ where an LLM can call a small, strongly-typed tool-kit (`_search_contacts`, `_create_contact`, `update_contact`, …) until it reaches a final answer.  The extensive unit-test suite in `tests/contact/` exercises all public and private helpers – skim through those tests if you want concrete examples of typical usage patterns, clarification flows, vector search, event logging, etc.

### New since the original README

- In-flight steering (steerable tools) to interject and control runs without restarting.
- Clarification requests with text and voice responses.
- Voice mode improvements: push-to-talk recording, TTS with Enter-to-skip, and voice steering.
- Richer logging: file logs, optional terminal streaming, and TCP streams (including dedicated Unify Request logs).

Running the sandbox
-------------------
The entry-point lives at `sandboxes/contact_manager/sandbox.py` and can be executed directly or via Python’s `-m` switch:

```bash
# Basic text-only session
python -m sandboxes.contact_manager.sandbox

# The same, but enable voice I/O via Deepgram + Cartesia
python -m sandboxes.contact_manager.sandbox --voice
```

CLI flags
~~~~~~~~~
`sandbox.py` re-uses the common helper in `sandboxes/utils.py`, so it shares a standard set of options:

```
--voice / -v        Enable voice capture (Deepgram) + TTS playback (Cartesia)
--debug / -d        Show full reasoning steps of every tool-loop
--project_name / -p Name of the Unify **project/context** (default: "Sandbox")
--overwrite / -o    Delete any existing data for the chosen project before start
--project_version   Roll back to a specific project commit (int index)
--log_in_terminal   Stream logs to the terminal in addition to writing file logs
--no_clarifications Disable interactive clarification requests (text and voice)
--log_tcp_port      Serve main logs over TCP on localhost:PORT (-1 auto-picks; 0 off)
--http_log_tcp_port Serve Unify Request logs over TCP on localhost:PORT (-1 auto when UNIFY_REQUESTS_DEBUG)
```

Interactive commands inside the REPL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the sandbox starts you will see a prompt and a small help table.  The most important commands are:

* `us {description}`    Build **synthetic contacts** through the official public tools using `ScenarioBuilder`.
* `usv`                 Same as above but capture the description via **voice** (only when `--voice` is active).
* `r`                   Record a one-off voice query (again only with `--voice`).
* *free text*           Any other input is auto-routed to `ask` *or* `update` depending on intent.
* `save_project` / `sp` Save the current Unify project snapshot so you can roll back later.
* `help` / `h`          Show the in-session command reference.
* `quit`                Exit the sandbox.

### In-flight steering (during a running request)
While an `ask` or `update` call is running, you can steer it in-flight. Type these commands (they only work while a request is active):

- **plain text | /freeform <text> | /i <text>**: Route free-form text via the steering router. The router decides whether it's an ask, interject, pause, resume, stop, or status.
- **/pause | /p**: Pause the running call.
- **/resume**: Resume a paused call.
- **/ask <question> | /? <question>**: Ask a read-only side question about the currently running call; the answer prints inline without changing the main call’s state.
- Plain text is now equivalent to typing `/freeform <text>`.
- **/r | /record** (voice mode only): Record a voice utterance and route it via freeform. Recording auto-cancels if the task finishes mid-capture.
- **/stop | /cancel | /s | /c**: Abort the running call.
- **/status | /st**: Print whether the call is still running or already done.
- **/help | /h**: Show the one-line controls hint.

Notes:
- Steering commands are ignored when no call is running; you’ll see a small hint if you try.
- In voice mode, during TTS playback you can press Enter to skip. Steering commands (including `/r`) are entered after playback finishes. `/r` auto‑cancels if the run finishes while recording.

Example:
```text
command> Update Alice's contact with a new phone number +15551234
Controls: /i <text>, /pause, /resume, /ask <q>, /freeform <text> (or plain text), /r, /stop, /help
/i also add a short bio mentioning she’s based in NYC
/ask what fields have been updated so far?
/pause
/resume
/status
```

### Example session (text mode)
```text
$ python -m sandboxes.contact_manager.sandbox -d
ContactManager sandbox – type commands below …

us Create 5 contacts: Alice Smith (NYC), Bob Johnson (London)…
[generate] Building synthetic contacts – this can take a moment…
✓ Created 5 contacts …

What is Alice Smith's phone number?
[ask] → Alice Smith can be reached on 111 222 3333.

Update Bob Johnson's phone number to +15551234.
[update] → Updated contact 4 – phone_number set to +15551234.
```

### Example: clarification flow (text + voice)
```text
command> Update Alice’s contact with a short bio mentioning NYC and her role
Controls: /i <text>, /pause, /resume, /ask <q>, /freeform <text> (or plain text), /r, /stop, /help
❓ Clarification requested: Which Alice? We have Alice Smith and Alice Brown.
/c Alice Smith
✅ Clarification sent.
```

In voice mode you could also answer with `/rc` to record a short spoken clarification.

## Scenario generation via ScenarioBuilder

`us` and `usv` build or update a synthetic CRM through the manager’s public tools using `ScenarioBuilder`. The LLM is only allowed to use the exposed tools and will stop when the requested scenario is fully represented. Clarification requests are supported here as well; respond the same way as during normal `ask`/`update` runs.

## Logging and debugging

- By default, logs are written to `.logs_main.txt` (overwritten each run). Pass `--log_in_terminal` to also stream logs to the terminal.
- Set `--debug` to print full reasoning steps of the tool-loops.
- Optional TCP streams:
  - Main logs: `--log_tcp_port -1` auto-picks an available port (or specify an explicit port). Connect with `nc 127.0.0.1 <PORT>`.
  - Unify Request logs only: `--http_log_tcp_port -1` auto-enables when `UNIFY_REQUESTS_DEBUG` is set; connect with `nc 127.0.0.1 <PORT>`.
- A dedicated Unify Request log file is also written to `.logs_unify_requests.txt`.

### Troubleshooting
* **Deepgram / Cartesia keys** – if you use `--voice`, make sure the environment variables `DEEPGRAM_API_KEY` and `CARTESIA_API_KEY` are set.
* **Unify backend access** – the sandbox will attempt to create contexts and logs in your configured Unify project.  If your credentials (`UNIFY_KEY`, `UNIFY_BASE_URL`) are missing or invalid you may see HTTP errors.
* **Linter complaints** – the interactive session is powered by an LLM; if you hit a bug look at the `--debug` reasoning trace first.

Happy experimenting! 🎉
