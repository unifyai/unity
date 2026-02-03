WebSearcher Sandbox
===================

This folder contains an interactive playground for the `WebSearcher` component that lives in `unity/web_searcher/`. The goal of the sandbox is to let you experiment with the manager in isolation – issue natural‑language web queries, observe tool selection (search/extract/crawl/map), and iterate on prompt/policy settings before integrating into larger flows.

### Video walkthrough

- General overview, with clarification requests: [Loom video](https://www.loom.com/share/4b7257d2f13c4e85b4ce97685606346e?sid=2402a5ab-e9ec-4b23-81f2-bfb9356d5324)

What is the `WebSearcher`?
---------------------------
`WebSearcher` exposes a single high‑level natural‑language method:

* `ask(text)` – launches a tool‑loop where an LLM can call a small, strongly‑typed tool‑kit (`search`, `extract`, `crawl`, `map`) until it reaches a final answer.

The loop is guided by a concise “Decision Policy and When to Stop” section in the system prompt that encourages minimal steps and a prompt stop once sufficient evidence has been gathered.

Running the sandbox
-------------------
Entry point: `sandboxes/web_searcher/sandbox.py`.

```bash
# Basic text‑only session
python -m sandboxes.web_searcher.sandbox

# The same, but enable voice I/O (Deepgram + Cartesia via shared utils)
python -m sandboxes.web_searcher.sandbox --voice
```

CLI flags
~~~~~~~~~
This sandbox re‑uses the common helper in `sandboxes/utils.py`, so it shares the standard options:

```
--voice / -v        Enable voice capture (Deepgram) + TTS playback (Cartesia)
--debug / -d        Show full reasoning steps of every tool‑loop
--project_name / -p Name of the Unify project/context (default: "Sandbox")
--overwrite / -o    Delete any existing data for the chosen project before start
--project_version   Roll back to a specific project commit (int index)
--log_in_terminal   Stream logs to the terminal in addition to writing file logs
--no_clarifications Disable interactive clarification requests (text and voice)
--log_tcp_port      Serve main logs over TCP on localhost:PORT (-1 auto‑picks; 0 off)
--http_log_tcp_port Serve Unify Request logs over TCP on localhost:PORT (-1 auto when UNIFY_REQUESTS_DEBUG)
```

Interactive commands inside the REPL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the sandbox starts you will see a prompt and a small help table. Important commands:

* `r` (voice mode only) Record a one‑off voice query.
* free text           Any input is routed to `ask` (web research). If clarification is needed, the sandbox will prompt you and forward your answer back to the running call.
* `web:add`          Create a Website (prompts for name, host, gated, subscribed, credentials, notes).
* `web:update`       Update an existing Website (identify by id/host/name; prompts for fields).
* `web:delete`       Delete a Website by id/host/name.
* `web:filter`       Boolean filter over Websites (e.g., `gated == True`).
* `web:search`       Semantic search over Websites by `notes` (top‑k).
* `save_project` / `sp` Save the current Unify project snapshot so you can roll back later.
* `help` / `h`        Show the in‑session command reference.
* `quit`              Exit the sandbox.

In‑flight steering (during a running request)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While an `ask` call is running, you can steer it in‑flight using the same controls printed by the sandbox (interject/pause/resume/stop). Clarification questions are surfaced inline and your replies are forwarded to the manager. The handle honors the shared steerable API from `unity.common.async_tool_loop`.

Example session (text mode)
---------------------------
```text
$ python -m sandboxes.web_searcher.sandbox -d
WebSearcher sandbox – type queries below …

What changed in Q1 2025 for vector databases?
[ask] → (final answer with inline citations)

# Create a Website row
web:add
name> Medium
host> medium.com
gated (true/false)> true
subscribed (true/false)> true
credentials (secret_ids comma-separated, optional)> 101,102
notes (optional)> Tech journalism; paywalled reading list
{'outcome': 'website created', 'details': {'host': 'medium.com'}}

# Update an existing row
web:update
identify by (id/host/name)> host
host> medium.com
new name (optional)> Medium (Personal)
new host (optional)>
set gated? (true/false/skip)> true
set subscribed? (true/false/skip)> true
set credentials (comma-separated ids or blank to skip)>
new notes (optional)> Focus on subscriptions and saved lists
{'outcome': 'website updated', 'details': {...}}

# Filter rows
web:filter
filter expression (e.g., gated == True)gated == True
- id=3 name='Medium (Personal)' host='medium.com' gated=True subscribed=True

# Semantic notes search
web:search
semantic notes query> subscription sources for ML news
k (default 5)> 3
- id=3 name='Medium (Personal)' host='medium.com' gated=True subscribed=True

Login to my Medium subscription article and summarize
[ask] → (resolves Website and uses _gated_website_search)
```

Logging and debugging
---------------------

* Logs are written to `.logs_web_searcher.txt` (overwritten each run). Pass `--log_in_terminal` to also stream logs to the terminal.
* Set `--debug` to print full reasoning steps of the tool‑loop.
* Optional TCP streams:
  - Main logs: `--log_tcp_port -1` auto‑picks an available port (or specify an explicit port). Connect with `nc 127.0.0.1 <PORT>`.
  - Unify Request logs only: `--http_log_tcp_port -1` auto‑enables when `UNIFY_REQUESTS_DEBUG` is set; connect with `nc 127.0.0.1 <PORT>`.
* A dedicated Unify Request log file is also written to `.logs_unify_requests.txt`.

Troubleshooting
---------------
* Ensure the environment variable `UNITY_WEB_TAVILY_API_KEY` is set.
* If voice mode is enabled, ensure `DEEPGRAM_API_KEY` and `CARTESIA_API_KEY` are set.
* The sandbox uses your configured Unify credentials (`UNIFY_KEY`, `ORCHESTRA_URL`). Missing or invalid credentials may cause HTTP errors.
* If the loop appears to stall on tools, enable `-d` to inspect the decision policy and prompts.

Happy researching! 🎉
