Contact Manager Sandbox
=======================

This folder contains an **interactive playground** for the `ContactManager` component that lives in `unity/contact_manager/`.  The goal of the sandbox is to let you experiment with the manager in isolation – create imaginary contacts, query them in natural-language, and observe how the underlying tool-loop behaves before you integrate the manager into a larger system.

Prefer a quick demo? Watch this [18-minute video walkthrough](
https://www.loom.com/share/31086a24fb9b4f9b8e0b69184773f942?sid=c444c524-96cf-4b95-b08d-57d35565a382) showing an end-to-end sandbox session.

What is the `ContactManager`?
-----------------------------
`ContactManager` is an abstraction that stores contact records (first name, surname, phone, WhatsApp, bio, rolling summary, plus any number of **custom columns**) and exposes two high-level natural-language methods:

* **`ask(text)`**   – read-only questions such as *"What is Alice's phone number?"*
* **`update(text)`** – mutations such as *"Add Bob's WhatsApp number +123…"*

Under the hood both methods launch a _tool-loop_ where an LLM can call a small, strongly-typed tool-kit (`_search_contacts`, `_create_contact`, `_update_contact`, …) until it reaches a final answer.  The extensive unit-test suite in `tests/test_contact/` exercises all public and private helpers – skim through those tests if you want concrete examples of typical usage patterns, clarification flows, vector search, event logging, etc.

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
--traced / -t       Wrap manager calls with unify.traced for detailed logs
--project_name / -p Name of the Unify **project/context** (default: "Sandbox")
--overwrite / -o    Delete any existing data for the chosen project before start
--project_version   Roll back to a specific project commit (int index)
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

Example session (text mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
```text
$ python -m sandboxes.contact_manager.sandbox -d
ContactManager sandbox – type commands below …

us Create 5 contacts: Alice Smith (NYC), Bob Johnson (London)…
[generate] Building synthetic contacts – this can take a moment…
✓ Created 5 contacts …

What is Alice Smith's phone number?
[ask] → Alice Smith can be reached on 111 222 3333.

Add WhatsApp number +15551234 for Bob Johnson.
[update] → Updated contact 4 – whatsapp_number set to +15551234.
```

Troubleshooting
---------------
* **Deepgram / Cartesia keys** – if you use `--voice`, make sure the environment variables `DEEPGRAM_API_KEY` and `CARTESIA_API_KEY` are set.
* **Unify backend access** – the sandbox will attempt to create contexts and logs in your configured Unify project.  If your credentials (`UNIFY_KEY`, `UNIFY_BASE_URL`) are missing or invalid you may see HTTP errors.
* **Linter complaints** – the interactive session is powered by an LLM; if you hit a bug look at the `--debug` reasoning trace first.

Happy experimenting! 🎉
