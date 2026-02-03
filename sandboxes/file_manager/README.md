Global File Manager Sandbox
===========================

This folder contains an interactive playground for the `GlobalFileManager` that composes multiple `FileManager` instances (e.g., local, CodeSandbox, Interact, Google Drive). The goal is to experiment with cross‑filesystem queries and organization using only the public surface and a steerable tool‑loop.

### Video walkthroughs

- Coming soon

What is the `GlobalFileManager`?
--------------------------------
`GlobalFileManager` is an orchestrator over multiple file managers. It exposes two high‑level natural‑language methods:

* **`gask(text)`** – read‑only global questions (router decides which manager(s) to consult)
* **`gorganize(text)`** – cross‑filesystem organization requests (rename/move/delete across managers)

You can dynamically add/select local managers during the session and seed sample data across all managers. All calls support in‑flight steering and clarifications.

Running the sandbox
-------------------
The entry‑point lives at `sandboxes/file_manager/global_file_manager_sandbox.py` and can be executed via Python’s `-m` switch:

```bash
# Basic text-only session
python -m sandboxes.file_manager.global_file_manager_sandbox

# Voice I/O via Deepgram + Cartesia
python -m sandboxes.file_manager.global_file_manager_sandbox --voice
```

CLI flags
~~~~~~~~~
`global_file_manager_sandbox.py` re‑uses the common helper in `sandboxes/utils.py`, so it shares a standard set of options:

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

Interactive commands inside the REPL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the sandbox starts you will see a prompt and a small help table. The most important commands are:

* `add_local [--root=<path>|--rootless]`  Add a new local FileManager
* `list_fms`                               List available manager aliases
* `use_fm <alias>`                         Select a current manager alias
* `seed-sample`                            Import and parse sample files across all managers (compact mode)
* `gask <text>`                            Global read‑only question
* `gorganize <text>`                       Global organize request
* `save_project` / `sp`                    Save the current Unify project snapshot
* `help` / `h`                             Show the in‑session command reference
* `quit`                                   Exit the sandbox

### In‑flight steering (during a running request)
While a `gask` or `gorganize` call is running, you can steer it in‑flight. Type these commands (they only work while a request is active):

- plain text | `/freeform <text>` | `/i <text>`: Route free‑form text via the steering router. The router decides whether it's an ask, interject, pause, resume, stop, or status.
- `/pause` | `/p`: Pause the running call.
- `/resume`: Resume a paused call.
- `/ask <question>` | `/? <question>`: Ask a read‑only side question about the currently running call; the answer prints inline without changing the main call’s state.
- `/r` | `/record` (voice mode only): Record a voice utterance and route it via freeform. Recording auto‑cancels if the run finishes mid‑capture.
- `/stop` | `/cancel` | `/s` | `/c`: Abort the running call.
- `/status` | `/st`: Print whether the call is still running or already done.
- `/help` | `/h`: Show the one‑line controls hint.

Notes:
- Steering commands are ignored when no call is running; you’ll see a small hint if you try.
- In voice mode, during TTS playback you can press Enter to skip. Steering commands (including `/r`) are entered after playback finishes. `/r` auto‑cancels if the run finishes while recording.

### Example session (text mode)
```text
$ python -m sandboxes.file_manager.global_file_manager_sandbox -d
GlobalFileManager sandbox – type commands below …

add_local --rootless
add_local --root=~/Documents
list_fms
use_fm local_1
seed-sample

gask What kinds of documents do we have across all managers?
[ask] → PDFs, spreadsheets, text notes, and Word documents across 2 managers

gorganize Move any CSVs into /datasets across all managers
[organize] → Moved 3 CSV files into /datasets
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
* **Manager aliases** – remember to `list_fms` and `use_fm <alias>` when you want to target a specific manager in follow‑ups.
