Secret Manager Sandbox
======================

This folder contains an interactive playground for the `SecretManager` component in `unity/secret_manager/`. Use it to experiment with the manager in isolation – create or update secrets, resolve placeholders into real values (for trusted local use), and observe how the tool-loop behaves before integrating elsewhere.

### Video walkthroughs

- General overview, with in-flight steering: [Loom video](https://www.loom.com/share/bcdb0559d08e4f34a16e57583152bbcc?sid=38b04604-9578-4468-aa8b-4815955dd49e)

What is the `SecretManager`?
----------------------------
`SecretManager` stores fixed-schema secret records (`name`, `value`, `description`, `description_emb`) and exposes two high-level natural‑language methods:

- `ask(text)`    – read-only questions such as “list secret keys”, “show placeholder for …”
- `update(text)` – mutations such as “create ${unify_key} with this value …”, “update ${db_password} …”

Additionally it offers non‑LLM helpers:

- `from_placeholder(text)` – replace `${NAME}` placeholders with real values for trusted components (e.g., actor/browser), with value-free logging
- `to_placeholder(text)` – redact known raw values to their `${NAME}` placeholders

Running the sandbox
-------------------
The entry-point is `sandboxes/secret_manager/sandbox.py` and can be executed directly or via Python’s `-m` switch:

```bash
# Basic text-only session
python -m sandboxes.secret_manager.sandbox

# Enable voice I/O via Deepgram + Cartesia (optional)
python -m sandboxes.secret_manager.sandbox --voice
```

CLI flags (shared)
------------------
This sandbox re-uses common helpers from `sandboxes/utils.py`, so it supports the standard options (same as other manager sandboxes):

```
--voice / -v        Enable voice capture (Deepgram) + TTS playback (Cartesia)
--debug / -d        Show full reasoning steps of every tool-loop
--traced / -t       Wrap manager calls with unify.traced for detailed logs
--project_name / -p Name of the Unify project/context (default: "Sandbox")
--overwrite / -o    Delete any existing data for the chosen project before start
--project_version   Roll back to a specific project commit (int index)
--log_in_terminal   Stream logs to the terminal in addition to writing file logs
--no_clarifications Disable interactive clarification requests (text and voice)
--log_tcp_port      Serve main logs over TCP on localhost:PORT (-1 auto)
--http_log_tcp_port Serve Unify Request logs over TCP on localhost:PORT (-1 auto)
```

Interactive commands inside the REPL
------------------------------------
Once the sandbox starts you will see a prompt and a small help table. The most relevant commands are:

- `from_placeholder {text}`  Resolve `${NAME}` placeholders in text (no LLM). Prints the fully-resolved output. Values are not logged.
- `to_placeholder {text}`    Replace known raw values with their `${NAME}` placeholders.
- `r`               Record a one-off voice query (only when `--voice` is active).
- Free text         Any other input is auto-routed to `ask` or `update` depending on intent.
- `save_project`    Save the current Unify project snapshot so you can roll back later (`sp` alias).
- `help` / `h`      Show the command reference.
- `quit`            Exit the sandbox.

Notes on safety
---------------
- The tool-loop prompts and tools are designed to never reveal raw secret values to LLMs. Read paths redact `value`.
- `resolve` and `extract` bypass LLMs and publish only value-free metadata to logs.
- If you wrap the manager with `unify.traced`, traces include method calls and arguments; secret values in tool arguments should be treated carefully in higher layers. The manager itself avoids echoing values in outcomes/messages.

Example session (text mode)
---------------------------
```text
$ python -m sandboxes.secret_manager.sandbox -d
SecretManager sandbox – type commands below …

from_placeholder login with ${page_username} and ${page_password}
[from_placeholder] → login with alice@example.com and MySillyPassword123*

extract set ${db_password} = 'p@ssw0rd!'
[extract] → set ${db_password}

List secret keys and confirm that ${db_password} exists
[ask] → - Secret keys: ${db_password}, ${unify_key} …

Create a secret named ${unify_key} with a value (do not reveal it).
[update] → I can create ${unify_key}, but I need the value to store …
```

Troubleshooting
---------------
- Ensure Unify credentials are configured (project name/base URL/key) – otherwise backend calls will fail.
- Voice mode requires `DEEPGRAM_API_KEY` and `CARTESIA_API_KEY` environment variables.
- For debugging LLM tool-loops, use `--debug` to print full reasoning steps.

Happy experimenting! 🎉
