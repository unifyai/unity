File Manager Sandbox
====================

This folder contains an interactive playground for the `FileManager` component that lives in `unity/file_manager/`. The goal of the sandbox is to let you experiment with the manager in isolation – register/import files, parse them with token‑lean returns, ask natural‑language questions about specific files, and organize your filesystem – all through its public tool‑loop APIs.

### Video walkthroughs

- Coming soon

What is the `FileManager`?
--------------------------
`FileManager` is an abstraction over filesystems and parsers. It exposes high‑level natural‑language methods:

* `ask(text)` – read‑only questions about files and their metadata
* `ask_about_file(filename, question, response_format=None)` – read‑only questions scoped to a single file; supports structured extraction via `response_format` (Pydantic model class or JSON schema)
* `organize(text)` – mutations such as rename/move/delete to keep your files tidy
* `parse(paths, config)` – ingestion/parse with configurable return modes (`compact`/`full`/`none`)

Under the hood these methods launch a tool‑loop where an LLM calls a small, strongly‑typed tool‑kit (parse, list/stat, search/join, rename/move, etc.) until it reaches a final answer. Parsing is token‑efficient by default: heavy artifacts (full text, tables, records) are persisted and referenced via `content_ref`/`tables_ref`, and only a compact typed summary is returned unless you opt‑in to `full`.

### New since the original README

- Compact parse return mode (default) with Unify references instead of verbose blobs
- Structured extraction: `response_format` for `ask_about_file` (Pydantic or JSON schema)
- Unified enums (`FileFormat`, `MimeType`) across metadata and results
- In‑flight steering and clarification flow supported during long‑running calls

Running the sandbox
-------------------
The entry point lives at `sandboxes/file_manager/managers/file_manager_sandbox.py` and can be executed directly or via Python’s `-m` switch:

```bash
# Basic text-only session
python -m sandboxes.file_manager.managers.file_manager_sandbox

# Local adapter with explicit root
python -m sandboxes.file_manager.managers.file_manager_sandbox --adapter local --root /abs/path/to/files

# Rootless local adapter (absolute paths)
python -m sandboxes.file_manager.managers.file_manager_sandbox --adapter local --rootless

# Enforce structured extraction for ask_about_file using a JSON schema
python -m sandboxes.file_manager.managers.file_manager_sandbox --schema ./report_schema.json

# Or pick a built-in response model
python -m sandboxes.file_manager.managers.file_manager_sandbox --model report_facts

# The same, but enable voice I/O via Deepgram + Cartesia
python -m sandboxes.file_manager.managers.file_manager_sandbox --voice
```

CLI flags
~~~~~~~~~
`file_manager_sandbox.py` re‑uses the common helper in `sandboxes/utils.py`, so it shares a standard set of options:

```
--voice / -v        Enable voice capture (Deepgram) + TTS playback (Cartesia)
--debug / -d        Show full reasoning steps of every tool-loop
--project_name / -p Name of the Unify project/context (default: "Sandbox")
--overwrite / -o    Delete any existing data for the chosen project before start
--project_version   Roll back to a specific project commit (int index)
--log_in_terminal   Stream logs to the terminal in addition to writing file logs
--no_clarifications Disable interactive clarification requests (text and voice)
--log_tcp_port      Serve main logs over TCP on localhost:PORT (-1 auto-picks; 0 off)
--http_log_tcp_port Serve Unify Request logs over TCP on localhost:PORT (-1 auto when UNIFY_REQUESTS_DEBUG)
```

Adapter options:

```
--adapter local (default: local)
--root <path>       Root directory for local adapter
--rootless          Use Local adapter without a root (absolute-path mode)
--return-mode       Default parse return mode for seeding: compact|full|none
--schema <path>     JSON schema for ask_about_file structured extraction
--model <name>      Built-in Pydantic model name for response_format
```

Interactive commands inside the REPL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the sandbox starts you will see a prompt and a small help table. The most important commands are:

* `seed-sample`            Import and parse sample files from `tests/file_manager/sample`
* `list`                   List files known to the adapter
* `stat <path>`            Show unified status for a path
* `askf <file> <question>` Ask about a specific file (uses `--schema/--model` when supplied)
* `parse <paths...>`       Parse specific paths using current defaults
* `r`                      Record a one‑off voice query (only when `--voice` is active)
* _free text_              Auto‑routed to `ask`, `ask_about_file`, or `organize` depending on intent
* `save_project` / `sp`    Save the current Unify project snapshot
* `help` / `h`             Show the in‑session command reference
* `quit`                   Exit the sandbox

### In‑flight steering (during a running request)
While an `ask`/`ask_about_file`/`organize` call is running, you can steer it in‑flight. Type these commands (they only work while a request is active):

- plain text | `/freeform <text>` | `/i <text>`: Route free‑form text via the steering router. The router decides whether it's an ask, interject, pause, resume, stop, or status.
- `/pause` | `/p`: Pause the running call.
- `/resume`: Resume a paused call.
- `/ask <question>` | `/? <question>`: Ask a read‑only side question about the currently running call; the answer prints inline without changing the main call’s state.
- `/r` | `/record` (voice mode only): Record a voice utterance and route it via freeform. Recording auto‑cancels if the task finishes mid‑capture.
- `/stop` | `/cancel` | `/s` | `/c`: Abort the running call.
- `/status` | `/st`: Print whether the call is still running or already done.
- `/help` | `/h`: Show the one‑line controls hint.

Notes:
- Steering commands are ignored when no call is running; you’ll see a small hint if you try.
- In voice mode, during TTS playback you can press Enter to skip. Steering commands (including `/r`) are entered after playback finishes. `/r` auto‑cancels if the run finishes while recording.

### Example session (text mode)
```text
$ python -m sandboxes.file_manager.managers.file_manager_sandbox -d
FileManager sandbox – type commands below …

seed-sample
[parse] → Parsed sample files (compact mode)

What PDFs mention "policy"?
[ask] → Found 2 PDFs that mention policy: IT_Department_Policy_Document.pdf, Security_Policy.pdf

askf IT_Department_Policy_Document.pdf Summarize the key sections
[ask_about_file] → Introduction, Scope, Access Control, Incident Response

organize Rename Security_Policy.pdf to Security-Policy-v2.pdf
[organize] → Renamed successfully
```

### Example: clarification flow (text + voice)
```text
command> Move all spreadsheets to /datasets
Controls: /i <text>, /pause, /resume, /ask <q>, /freeform <text> (or plain text), /r, /stop, /help
❓ Clarification requested: Which spreadsheets do you mean – .xlsx, .csv, or both?
/c both
✅ Clarification sent.
```

## Logging and debugging

- By default, logs are written to `.logs_main.txt` (overwritten each run). Pass `--log_in_terminal` to also stream logs to the terminal.
- Set `--debug` to print full reasoning steps of the tool‑loops.
- Optional TCP streams:
  - Main logs: `--log_tcp_port -1` auto‑picks an available port (or specify an explicit port). Connect with `nc 127.0.0.1 <PORT>`.
  - Unify Request logs only: `--http_log_tcp_port -1` auto‑enables when `UNIFY_REQUESTS_DEBUG` is set; connect with `nc 127.0.0.1 <PORT>`.
- A dedicated Unify Request log file is also written to `.logs_unify_requests.txt`.

### Troubleshooting
* **Deepgram / Cartesia keys** – if you use `--voice`, make sure the environment variables `DEEPGRAM_API_KEY` and `CARTESIA_API_KEY` are set.
* **Unify backend access** – the sandbox creates contexts and logs in your configured Unify project. If your credentials (`UNIFY_KEY`, `ORCHESTRA_URL`) are missing or invalid you may see HTTP errors.
* **File visibility** – when running rootless, ensure you pass absolute paths and have read permissions.
