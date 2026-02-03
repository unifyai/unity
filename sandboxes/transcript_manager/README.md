Transcript Manager Sandbox
==========================

This folder contains an **interactive playground** for the `TranscriptManager` component that lives in `unity/transcript_manager/`.  The goal of the sandbox is to let you experiment with the manager in isolation – seed imaginary multi-channel conversations, query them in natural-language, and observe how the underlying tool-loop behaves (including steerable tools and clarification requests) before you integrate the manager into a larger system.

### Video walkthroughs
- General overview (comprehensive, but older sandbox version): [Loom video](https://www.loom.com/share/d600aa86f59a41a3ba2f4f1cbc3089d1?sid=70e250ec-3e82-48b0-aa17-2da9024a2fff)
- In-flight steerable tools: [Loom video](https://www.loom.com/share/94af912090f2411ea95c5775c2281452?sid=246def2f-53d7-497e-98e4-4542f4c3c13c)
- Clarification requests flow: [Loom video](https://www.loom.com/share/0865f7cf33ca4cae8305000766ca45db?sid=82be14d6-64ce-4061-9a14-2c2f79c773ec)

What is the `TranscriptManager`?
--------------------------------
`TranscriptManager` is an abstraction that stores **time-stamped messages** across a variety of media (email, SMS, phone-call logs, …) and exposes one high-level natural-language method:

* **`ask(text)`** – read-only questions such as *"When did Dan last speak with Julia on the phone?"*

Under the hood the method launches a _tool-loop_ where an LLM can call a small, strongly-typed tool-kit (`_search_messages`, `_filter_messages`, `_search_contacts`, …) until it reaches a final answer.  The extensive unit-test suite in `tests/transcript_manager/` exercises all public and private helpers – skim through those tests if you want concrete examples of typical usage patterns, clarification flows, semantic search, event logging, etc.

Running the sandbox
-------------------
The entry-point lives at `sandboxes/transcript_manager/sandbox.py` and can be executed directly or via Python’s `-m` switch:

```bash
# Basic text-only session
python -m sandboxes.transcript_manager.sandbox

# The same, but enable voice I/O via Deepgram + Cartesia
python -m sandboxes.transcript_manager.sandbox --voice
```

CLI flags
---------
`sandbox.py` re-uses the common helper in `sandboxes/utils.py`, so it shares a standard set of options:

```
--voice / -v         Enable voice capture (Deepgram) + TTS playback (Cartesia)
--debug / -d         Show full reasoning steps of every tool-loop
--project_name / -p  Name of the Unify **project/context** (default: "Sandbox")
--overwrite / -o     Delete any existing data for the chosen project before start
--project_version    Roll back to a specific project commit (int index)
--log_in_terminal    Stream logs to the terminal in addition to writing file logs
--no_clarifications  Disable interactive clarification requests (text and voice)
--log_tcp_port       Serve main logs over TCP on localhost:PORT (-1 auto; 0 off)
--http_log_tcp_port  Serve Unify Request logs over TCP on localhost:PORT (-1 auto when UNIFY_REQUESTS_DEBUG)
```

Interactive commands inside the REPL
------------------------------------
Once the sandbox starts you will see a prompt and a small help table.  The most important commands are:

* `us {description}`    Build **synthetic transcripts** through the official public tools using `ScenarioBuilder` + `TranscriptGenerator`.
* `usv`                 Same as above but capture the description via **voice** (only when `--voice` is active).
* `r`                   Record a one-off voice query (again only with `--voice`).
* *free text*           Any other input is auto-routed to `ask`.
* `save_project` / `sp` Save the current Unify project snapshot so you can roll back later.
* `help` / `h`          Show the in-session command reference.
* `quit`                Exit the sandbox.

### Steering controls (during a running request)
While an `ask` call is running, you can steer it in-flight. Type these commands (they only work while a request is active):

- **plain text | /freeform <text> | /i <text>**: Route free-form text via the steering router. The router decides whether it's an ask, interject, pause, resume, stop, or status.
- **/pause | /p**: Pause the running call.
- **/resume**: Resume a paused call.
- **/ask <question> | /? <question>**: Ask a read-only side question about the currently running call; the answer prints inline without changing the main call’s state.
- Plain text is now equivalent to typing `/freeform <text>`.
- **/r | /record** (voice mode only): Record a voice utterance and route it via freeform. Recording auto-cancels if the task finishes mid-capture.
- **/stop | /cancel | /s**: Abort the running call.
- **/status | /st**: Print whether the call is still running or already done.
- **/help | /h**: Show the one-line controls hint.

Notes:
- Steering commands are ignored when no call is running; you’ll see a small hint if you try.
- In voice mode, during TTS playback you can press Enter to skip. Steering commands (including `/r`) are entered after playback finishes; `/r` auto‑cancels if the run finishes while recording.

Example:
```text
command> When did Dan last speak with Julia on the phone?
Controls: /i <text>, /pause, /resume, /ask <q>, /freeform <text> (or plain text), /r, /stop, /help
/i include messages from the last 60 days only
/ask which channels are being searched right now?
/pause
/resume
/status
```

Clarification requests
----------------------
During a run, the assistant may ask for missing details (for example, "Which Julia did you mean?"). When a clarification is pending you can:

- Type your answer directly (plain text), or
- Use `/c <answer>` to send a text clarification, or
- In voice mode, use `/rc` to record and send a spoken clarification.

Example:
```text
❓ Clarification requested: Which Julia did you mean?
/c Julia Smith
✅ Clarification sent.
```

Scenario generation via ScenarioBuilder
--------------------------------------
`us` and `usv` build or update synthetic transcripts through public tools using `ScenarioBuilder` + `TranscriptGenerator`. The LLM is constrained to the exposed tools (no back‑doors) and finishes when the requested scenario is fully represented. Clarifications are supported during generation as well.

Logging and debugging
---------------------
- The main session log is written to `.logs_main.txt` (overwritten each run). Pass `--log_in_terminal` to also stream logs to the terminal.
- Optional TCP streams:
  - Main logs: `--log_tcp_port -1` auto-picks a port (or specify one). Connect with `nc 127.0.0.1 <PORT>`.
  - Unify Request logs only: `--http_log_tcp_port -1` auto-enables when `UNIFY_REQUESTS_DEBUG` is set; connect with `nc 127.0.0.1 <PORT>`.
- A dedicated Unify Request log file is also written to `.logs_unify_requests.txt`.

Example session (text mode)
---------------------------
```text
$ python -m sandboxes.transcript_manager.sandbox -d
TranscriptManager sandbox – type commands below …

us Generate 15 realistic message exchanges across email and SMS between 5 colleagues over the last two weeks.
[generate] Building synthetic transcripts – this can take a moment…
✓ Created 60 messages across 15 exchanges.

When did Dan last speak with Julia on the phone?
[ask] → Their most recent phone-call was on 2025-04-26.

What quantity did Carlos say he wanted to buy?
[ask] → 200 units.
```

Troubleshooting
---------------
* **Deepgram / Cartesia keys** – if you use `--voice`, make sure the environment variables `DEEPGRAM_API_KEY` and `CARTESIA_API_KEY` are set.
* **Unify backend access** – the sandbox will attempt to create contexts and logs in your configured Unify project.  If your credentials (`UNIFY_KEY`, `ORCHESTRA_URL`) are missing or invalid you may see HTTP errors.
* **Linter complaints** – the interactive session is powered by an LLM; if you hit a bug look at the `--debug` reasoning trace first.

Happy exploring transcripts! 🎉
