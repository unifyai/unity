Transcript Manager Sandbox
==========================

This folder contains an **interactive playground** for the `TranscriptManager` component that lives in `unity/transcript_manager/`.  The goal of the sandbox is to let you experiment with the manager in isolation – seed imaginary multi-channel conversations, query them in natural-language, and observe how the underlying tool-loop behaves before you integrate the manager into a larger system.

Prefer a quick demo? Watch this [video walkthrough](https://www.loom.com/share/d600aa86f59a41a3ba2f4f1cbc3089d1?sid=88f1518e-de7a-4778-9c6e-d1ae8920cf1c)

What is the `TranscriptManager`?
--------------------------------
`TranscriptManager` is an abstraction that stores **time-stamped messages** across a variety of media (email, SMS, WhatsApp, phone-call logs, …) and exposes one high-level natural-language method:

* **`ask(text)`** – read-only questions such as *"When did Dan last speak with Julia on the phone?"*

Under the hood the method launches a _tool-loop_ where an LLM can call a small, strongly-typed tool-kit (`_search_messages`, `_nearest_messages`, `_search_contacts`, …) until it reaches a final answer.  The extensive unit-test suite in `tests/test_transcript_manager/` exercises all public and private helpers – skim through those tests if you want concrete examples of typical usage patterns, clarification flows, semantic search, event logging, etc.

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

* `us {description}`    Build **synthetic transcripts** through the official public tools using `ScenarioBuilder` + `TranscriptGenerator`.
* `usv`                 Same as above but capture the description via **voice** (only when `--voice` is active).
* `r`                   Record a one-off voice query (again only with `--voice`).
* *free text*           Any other input is auto-routed to `ask`.
* `save_project` / `sp` Save the current Unify project snapshot so you can roll back later.
* `help` / `h`          Show the in-session command reference.
* `quit`                Exit the sandbox.

Example session (text mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
```text
$ python -m sandboxes.transcript_manager.sandbox -d
TranscriptManager sandbox – type commands below …

us Generate 15 realistic message exchanges across email, Slack and WhatsApp between 5 colleagues over the last two weeks.
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
* **Unify backend access** – the sandbox will attempt to create contexts and logs in your configured Unify project.  If your credentials (`UNIFY_KEY`, `UNIFY_BASE_URL`) are missing or invalid you may see HTTP errors.
* **Linter complaints** – the interactive session is powered by an LLM; if you hit a bug look at the `--debug` reasoning trace first.

Happy exploring transcripts! 🎉
