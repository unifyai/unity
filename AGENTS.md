<!--
    GENERATED FILE - DO NOT EDIT DIRECTLY.

    Regenerate with:  python3 .agents/global-rules/build_agents_md.py

    Edit the sources instead:
      .agents/repo.md              this repo's overview and always-on guidance
      .agents/rules/*.md           this repo's own rules
      .agents/shared.txt           which shared rules this repo includes
      .agents/global-rules/rules/  rules shared across all unifyai repos
                                   (submodule: unifyai/global-agent-rules)
-->

# Unify

The assistant's brain, and the cognitive core of the platform.

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first for the system design. This file
covers *how to work on the code*, not *what the code does*.

## What Unify is

Unify implements an AI assistant's brain as a **distributed back office**. A central `Actor` orchestrates specialized **state managers** (`ContactManager`, `KnowledgeManager`, `TaskScheduler`, `TranscriptManager`, `GuidanceManager`, `FunctionManager`, ...) through code-first plans. Most public manager methods run inside an **async LLM tool loop** and return a **steerable handle** that supports `ask`, `interject`, `pause`, `resume`, `stop` — all the way down the nesting tree. Typed catalogues such as Knowledge and Guidance expose direct CRUD/lifecycle methods as Actor JSON tools (`KnowledgeManager_*`, `GuidanceManager_*`) rather than NL tool loops or `primitives.*`.

Sibling repos consumed via editable installs (see `[tool.uv.sources]` in `pyproject.toml`):
- **`unisdk`** — Python SDK wrapping the Orchestra REST API
- **`unillm`** — LLM client with caching, provider normalization, observability

The open agent runtime (`unify`, `unisdk`, `unillm`) talks to the **hosted Orchestra backend** (`ORCHESTRA_URL`, default `https://api.unify.ai/v0`). `orchestra` and `console` are private/hosted and are not part of the open-source repo set.

Unify consumes data from Orchestra via `unisdk`, makes LLM calls via `unillm`, and
triggers external actions through the hosted communication stack in `unify-deploy`.
Console provides observability into its operations.

## Run the agent locally (public path)

The open-source runtime runs on your machine against the **hosted** Orchestra
backend. Provision a `UNIFY_KEY` and an assistant (`ASSISTANT_ID`) at
[console.unify.ai](https://console.unify.ai), then:

```bash
curl -fsSL https://raw.githubusercontent.com/unifyai/unify/staging/scripts/install.sh | bash
unify            # interactive local chat (alias: unify chat)
unify serve      # headless: ConversationManager + gateway
unify setup      # re-run the key/credential wizard
```

- `unify` is the CLI shim the installer drops in `~/.local/bin/`. From a
  checkout, the equivalents are `.venv/bin/python -m
  sandboxes.conversation_manager.sandbox` (chat) and `bash scripts/local.sh
  start --full` (headless).
- Configuration lives in `unify/.env`: `UNIFY_KEY`, `ASSISTANT_ID`,
  `ORCHESTRA_URL` (hosted), one LLM provider key, and optional voice/research
  keys (`scripts/prompt_byok_keys.sh`).
- No Docker, local Orchestra, or Console is involved in the public path. The
  onboarding flow, inbound channels, workspace connect, third-party app
  integrations, and screen-share are part of the hosted product.

**Internal full-local self-host stack.** The "all-repo fully local" stack
(local Orchestra + Console + Coordinator + gateway, via Docker Compose) is an
internal-only path and lives in the private **`unify-deploy`** repo under
`selfhost/` (`stack.sh`, `setup.sh`, `service.sh`, compose bundle). It drives
sibling `unify`/`console`/`orchestra` checkouts under `UNIFY_STACK_ROOT`.

## Development environment

```bash
# First-time setup (fresh clone)
pip install uv && uv sync --all-groups
```

- **Python interpreter:** always use `.venv/bin/python`. Never the system Python.
- **Bootstrap:** if `.venv/` is missing, `uv sync --all-groups` recreates it.
- **`uv.lock` and `package-lock.json` are protected** — never hand-edit them. Use the package manager.
- **Do not output `.env` or `*.key` contents to chat.**

### Running tests

Tests run in tmux sessions, each test in its own session, with logs streamed to `logs/pytest/`. The runner blocks until everything completes.

```bash
# Default — one session per test, max concurrency
tests/parallel_run.sh tests/contact_manager/

# Specific test
tests/parallel_run.sh tests/contact_manager/test_ask.py::test_name

# Serial mode (one session per file) for large suites
tests/parallel_run.sh -s tests/

# With timeout
tests/parallel_run.sh --timeout 300 tests/contact_manager/
```

Each agent (or terminal) gets an **isolated tmux server automatically**, so concurrent agents don't collide.

### When a test fails

1. **Never inspect tmux panes directly.** Read the corresponding log in `logs/pytest/<YYYY-MM-DDTHH-MM-SS_socket>/`.
2. **Use `Read` (not `cat`/`tail`)** — `logs/` is gitignored, so `Grep`/`Glob` won't find files there.
3. **Add temporary debug logs via `CURSOR_DEBUG_LOG`** — the only permitted logging mechanism for debugging. Grep for it (`rg CURSOR_DEBUG_LOG`) to find the project's util, then import and use it. Remove all calls before finalizing the fix.
4. **Clean up failed sessions** with `tests/kill_failed.sh` (or `tests/kill_server.sh` for everything).

### When CI is "stuck cancelling" / new matrices stay pending

Ordinary `gh run cancel` can no-op while jobs stay on `Run tests` for an hour+
and hold the Tests concurrency group. That is a **zombie run**, not
`if: always()` wind-down. Force-cancel immediately:

```bash
bash scripts/dev/force_cancel_stuck_tests.sh staging
```

Full decision tree: [`.agents/rules/ci-tests-cancel-zombies.md`](.agents/rules/ci-tests-cancel-zombies.md).

### Pre-commit

Install the hooks once per checkout (they run automatically on commit; CI runs the same pinned hooks):

```bash
./scripts/install-git-hooks.sh   # or: pre-commit install
```

Run them manually any time:

```bash
pre-commit run --all-files
```

## Provider integrations: no custom retries

Orchestra ``run_tool`` retries transient provider failures by default for
**every** backend (Composio, Pipedream, …): HTTP 429/5xx, timeouts, empty
bodies, GitHub GraphQL "Something went wrong while executing your query" on
**read** tools, and Pipedream action-level failures embedded in an HTTP-ok
body when attribution looks like network / upstream 429–5xx. Unify
``integrations.ops.run_tool`` additionally retries transient Orchestra
connectivity failures.

When authoring CodeAct plans, stored functions, or brain ticks that call
``primitives.integrations.*`` / Composio / Pipedream via Orchestra:

- Call the tool **once** and handle the final envelope (`ok`,
  `connect_required`, `confirmation_required`, `provider_error`, …).
- Do **not** wrap provider calls in ad-hoc `for attempt in range…` /
  `time.sleep` loops for flakiness — that duplicates policy and often misses
  GraphQL-in-payload or Pipedream embedded-action errors that Orchestra
  already classifies.
- Domain-specific **long** waits (e.g. sitting through a GitHub primary
  rate-limit reset for a multi-hour crawl) may remain at the call site; short
  transport/platform blips belong in the shared layer.

Opt out (ops/debug only): `ORCHESTRA_INTEGRATION_RETRY_MAX_ATTEMPTS=1` and/or
`UNIFY_INTEGRATION_TRANSPORT_RETRY_MAX_ATTEMPTS=1`.

## Testing philosophy

We **never** mock the LLM client. All tests use real LLM calls via `unillm.AsyncUnify`, with responses cached per unique input (`UNILLM_CACHE=true`, the default). First run is slow; subsequent runs replay from cache in milliseconds.

Tests sit on a **spectrum** between two paradigms — there's no binary classification:

- **Symbolic tests** use the LLM as a deterministic stub to exercise infrastructure (async tool loops, steering, state mutations). Failures = regression in programmatic code.
- **Eval tests** verify end-to-end *capability* ("did the assistant answer correctly?"). Failures may indicate prompt issues, tool design problems, or capability gaps.

**Never rely on sleeps** — use the trigger helpers in `tests/async_helpers.py` for deterministic ordering across cached (ms) and live (sec–min) timing.

### The cache is never the problem

"We just need to update the cache" is **never** a valid conclusion when debugging failures. The cache is a faithful replay mechanism keyed on the exact LLM input. If you change prompts or docstrings, the cache key changes automatically and you get fresh inference. If a cached response causes a failure, an LLM *actually made that decision* given that exact input — that's a prompt issue, not a stale-cache issue. Clearing the cache to "fix" a failing test is a category error.

### Tagging eval tests

```python
import pytest
pytestmark = pytest.mark.eval  # whole file

@pytest.mark.eval                # single test
async def test_natural_language_query(): ...
```

## No fast paths or heuristics

If a method needs to respond correctly to a class of user input, **always** address this by prompting the model and/or improving tool docstrings. Never apply regex-based or substring-based routing on user commands. The LLM is the router.

## State manager design

The public API of each state manager is defined by the abstract methods on `Base{SomeManager}` in `base.py`. These docstrings are the **LLM-facing contract** — they're attached to concrete implementations via `@functools.wraps`.

### Docstring rules

- **Implementation-agnostic.** Public docstrings must never reference other managers (cross-references rot) or the manager's own internal tools.
- **Tool-specific guidance lives in the tool's own docstring** — never in the prompt builder.
- **Compositional guidance (when to use tool A vs B, multi-tool patterns) lives in the prompt builder** — never in individual tool docstrings.

### Routing playbook (which manager owns what)

| Concern | Manager / primitive |
|---|---|
| People, contact records | `primitives.contacts.*` |
| Conversation history search | `primitives.transcripts.*` |
| Domain facts, typed knowledge claims | `KnowledgeManager_*` (top-level JSON tools, not primitives) |
| Durable tasks (create, execute) | `primitives.tasks.*` / `TaskScheduler` |
| Files (parse, query) | `primitives.files.*` |
| Storing new data or files, from any source | `primitives.ingestion.*` (`submit` — there is no `primitives.data.ingest`) |
| Generative UI (interactive views) | `primitives.canvas.*` |
| Web research (lightweight) | `primitives.web.*` |
| Secrets (metadata only via `ask`) | `primitives.secrets.*` |
| Procedural how-tos, SOPs | `GuidanceManager_*` (top-level JSON tools, not primitives) |
| Install/remove a packaged capability | `WorkflowManager_*` (top-level JSON tools). Not TaskScheduler: a workflow is the package you install; the tasks it plants are the work |
| Ephemeral live action | `Actor.act` (via ConversationManager) |
| Durable, tracked work | `TaskScheduler.execute` — never `update` to start work |

Full role descriptions are in [`.agents/rules/state-manager-roles.md`](.agents/rules/state-manager-roles.md).

### Cross-manager images

Images flow between managers **by filesystem path**, not by `image_id`. Receiving managers resolve to persistent storage via `ImageManager.filter_images(filter="filepath == '...'")` when needed. Managers with first-class image fields (e.g. `GuidanceManager`) accept structured `ImageRefs` types at their own API boundary.

## Additional git constraints

Beyond the shared git rules below: never use `git rebase -i` or `git add -i`
(interactive flags don't work in non-interactive shells), and never edit
`git config`.

## Repo map

```
unify/
├── unify/                   # Main package
│   ├── actor/               # CodeAct Actor, central orchestrator
│   ├── conversation_manager/ # Slow brain, live chat orchestration
│   ├── contact_manager/     # People + relationships
│   ├── knowledge_manager/   # Typed claim ledger (facts, policies, …)
│   ├── task_scheduler/      # Durable tasks, schedules, triggers
│   ├── transcript_manager/  # Conversation history
│   ├── guidance_manager/    # Procedures, SOPs
│   ├── function_manager/    # User Python functions + primitives registry
│   ├── file_manager/        # File parsing and registry
│   ├── image_manager/       # Image storage and vision queries
│   ├── web_searcher/        # Web research
│   ├── secret_manager/      # Encrypted secrets
│   ├── blacklist_manager/   # Blocked contacts
│   ├── data_manager/        # Low-level data ops
│   ├── memory_manager/      # Offline consolidation
│   ├── events/              # Typed event bus
│   ├── common/              # Async tool loop, shared infra
│   ├── deploy_runtime/      # Hosted deployment SPI (local default)
│   └── gateway/             # External comms gateway
├── agent-service/           # TypeScript service for browser-using agents
├── tests/                   # Pytest suite
├── sandboxes/               # Per-manager dev sandboxes
├── scripts/                 # Install, dev tooling
├── deploy/                  # Cloud Build, Docker, deploy configs
├── ARCHITECTURE.md          # System design (read first)
├── README.md
├── CONTRIBUTING.md
└── pyproject.toml
```

## When in doubt

- Check [`.agents/rules/`](.agents/rules/) for fuller context on any topic above.
- `ARCHITECTURE.md` is canonical for design questions.
- Code is canonical when this document and the implementation disagree — open a PR to update this doc.

---

# Repository rules

# CI Tests Cancel: Zombies vs Wind-Down

When a Tests matrix run on staging/main seems "stuck cancelling" or blocks new
`workflow_dispatch` / matrices behind concurrency, **do not** assume our
`parallel_run` trap or `if: always()` post-steps are the cause until the job
step API says so.

## Two different failure modes (do not conflate)

### A — Ordinary cancel is a no-op (the hostage / zombie)

**Symptoms**

- `gh run cancel <id>` prints success, but the run stays `in_progress` for
  **tens of minutes to an hour+**
- API: `cancel_requested_at` stays `null`
- Stuck jobs show active step **`Run tests`** (sometimes a later Post-* step)
- New Tests runs on the same branch sit `pending` / `queued` behind concurrency
  group `tests-Tests-<branch>`

**Cause**

GitHub stuck-run class: the runner/job stops responding to ordinary cancel.
This is **not** explained by `if: always()` uploads (those steps never start
while Hang/`Run tests` is still active).

**Fix (immediate)**

```bash
bash scripts/dev/force_cancel_stuck_tests.sh staging
# or a specific run:
gh api --method POST repos/unifyai/unify/actions/runs/<id>/force-cancel
```

Verified: force-cancel flips the run to `completed/cancelled` in seconds and
frees the concurrency group. Prefer the script so agents do not re-discover
the REST endpoint under pressure.

**Do not**

- Poll for 30+ minutes hoping ordinary cancel will land
- Spend the session tuning `parallel_run` traps or `always()` as the
  explanation for this symptom set
- Start another large matrix and wait — it will stay pending until the zombie
  is force-cancelled (unless `cancel-in-progress` already replaced it)

### B — Cancel worked; post-steps add ~tens of seconds to a few minutes

**Symptoms**

- Hang / `Run tests` **ends** within ~10–20s of cancel
- Job then runs `if: always()` steps (upload logs, stop orchestra, …)
- Job-level lag ≈ sum of those post-steps (empirically ~15–20s typical; can be
  longer if uploads are huge)

**Cause**

`if: always()` is true on cancel, so cleanup/upload still runs after the test
step dies. Separate from zombies.

**Fix (design)**

Skip heavy uploads on cancel (`if: success() || failure()`); keep a short
`if: cancelled()` kill (`tests/kill_server.sh --all`, orchestra stop).

## Decision checklist (agents)

1. `gh run view <id> --json status,jobs` — is any job still on **`Run tests`**?
2. `gh api repos/unifyai/unify/actions/runs/<id> --jq .cancel_requested_at`
   — still null after cancel? → **zombie → force-cancel**
3. Only if Hang already completed and post-steps are active → wind-down /
   `always()` discussion

## Related knobs

- Tests concurrency: `cancel-in-progress` is on for `pull_request` and
  `workflow_dispatch` (see `.github/workflows/tests.yml`). Push events still
  let an in-flight matrix finish.
- Cancel-latency smoke (A–E): Flow Smoke dispatch with
  `confirm_llm_spend=CANCEL_SMOKE_OK` / `scripts/dev/run_cancel_smoke.sh`

# CodeQL Comes From Default Setup, Not A Workflow

`unify` has **no `.github/workflows/codeql.yml`**, and must not gain one. Static
analysis runs from GitHub's **default setup** — the `dynamic/github-code-scanning/codeql`
workflow, which has no file in this repo. It analyses `actions`, `javascript`,
`javascript-typescript`, `python` and `typescript` on pull requests and weekly,
with the `security-extended` query suite.

## Why there is no file

There used to be one. On **2026-06-23 12:29Z** nassimberrada enabled default setup
and disabled the advanced workflow one second later, via `gh`:

```
repo.codeql_enabled         2026-06-23 12:29:52Z
workflows.disable_workflow  2026-06-23 12:29:53Z   workflow_id=296489893
```

That order is required, not incidental: GitHub does not run both configurations at
once, so enabling default setup means the advanced workflow has to go. The file was
left behind, disabled, for six weeks and read as "someone switched static analysis
off" — it had not been; default setup was analysing the whole time. It is deleted so
the next reader is not misled the same way.

## What this means in practice

- **Do not** re-enable or re-add a `codeql.yml` here. Two configurations conflict,
  and the workflow-shaped one is the one that loses.
- Change the analysis by changing default setup, not a file:
  `gh api -X PATCH repos/unifyai/unify/code-scanning/default-setup -f query_suite=extended`.
  It returns a `run_id`; the new suite applies once that run completes.
- The `CodeQL` check on a pull request is **advisory here** — the `Staging->Main`
  ruleset requires `black`, `staging-source`, `Flow smoke` and `Contract tests`, not
  CodeQL. A red CodeQL does not block a release, so do not read `BLOCKED` on a PR as
  a CodeQL problem without checking the ruleset.
- A workflow-only diff can make the check report `1 configuration not found` rather
  than a finding, because no results are produced for the languages it expects.

## Sibling repos differ — check before assuming

`orchestra` keeps a real committed `codeql.yml` (advanced setup, `security-extended`,
python) and no default setup. `unisdk`, `unillm` and `docs` are on default setup with
no file. Reading alerts for the private repos needs a token with `security_events`;
without it the API answers `403 Code Security must be enabled`, which is a scope
message and **not** evidence that scanning is off.

# Custom Source Sync: one engine, one identity contract

All git-tracked source definitions (tasks, functions, venvs, guidance,
knowledge, contacts, secrets, blacklist, data seeds,
integration registry) reconcile through the shared engine in
`unify/common/custom_sync.py`. Full contract:
[`docs/writeups/custom-source-sync.md`](../../docs/writeups/custom-source-sync.md).

## Invariants

- A managed row has `custom_key`, `custom_hash` **and** `managed_by`
  (which source reconciles it: the deployment or a workflow slug); rows
  authored by users/actors have none of them. `custom_key` is the
  identity; auto-counted ids (`task_id`, `function_id`, …) are
  environment-local handles that source code must never reference.
- Every reconcile pass is scoped to ONE `managed_by`. `live_rows` and
  `find_collision` filter with `managed_rows_filter(managed_by)`, never
  a bare `custom_hash != None` — an unscoped query hands one source its
  siblings' rows and prune then deletes them. Rows with a null
  `managed_by` belong to the deployment and are stamped on their next
  content change.
- `managed_by` is reconcile provenance only — who may overwrite/prune
  the row. Which workflows *reference* a row is membership, a separate
  multi-valued field; never encode membership into `managed_by`.
- Each manager declares its key policy in ONE place (its `custom_*.py`
  collector). Changing a key policy is an identity migration for every
  deployment — plan it, never drive-by edit it.
- Inserts write `custom_key`/`custom_hash` atomically with the row.
  No create-then-stamp second write.
- Two live managed rows with the same `custom_key` is an error the
  engine raises (`CustomSyncDuplicateKeyError`) — never silently pick a
  survivor.
- Per-entry failures are isolated and re-raised as
  `CustomSyncPartialFailure` after the pass; the aggregate hash is not
  stored on partial failure so the next reconcile retries.
- Every reconcile holds `exclusive_sync_lease` on the meta context.
- Writers either persist the collected field dict wholesale, or consume
  every field and raise on leftovers. A collector hashing a field the
  writer drops caused live rows to pin themselves "up to date" with the
  field unwritten (the task `tags` incident) — the leftover check exists
  to make that class impossible.

## Hard refuse

- A new bespoke `sync_custom_*` diff loop, or "just this one" fork of
  the engine's semantics inside a manager. Extend the engine instead.
- Adding a field to a collector's hash without routing the same field
  through the writer (and vice versa).
- Stamping identity after insert, or hand-writing rows with a
  `custom_key` outside the engine.
- Changing a manager's key derivation (or making an optional source
  `key` mandatory) without a migration plan for live rows in every
  deployment.
- Registering a surface for workflow installs while its adapter still
  queries unscoped. The `SurfaceRegistry` refuses these
  (`UnscopedSurfaceError`); routing around it reintroduces mutual prune.
- Making canvas a synced surface. A view is TypeScript that must be
  compiled, rendered and reviewed against the kit installed *now*, and its
  routing token has a lifecycle the engine does not own. Bundles ship
  canvas **source**; the install publishes through CanvasManager and the
  uninstall deletes through its own delete.

## Deviations are declared knobs, not forks

`prune=False` (secrets), `collision="yield"` (secrets),
`find_adoptable` (data seeds — deployment only, integration registry,
functions/venvs legacy rows), `find_released` (tasks: a planted task the
user has edited), `should_update` (tasks: skip while running),
`max_workers` (tasks). New deviations need a named knob on the adapter and a line in
the writeup's table.

## Handing a row to the user

A surface may end the loan on one row: clear `managed_by`, keep
`custom_key`, and set `custom_released=True`. From then on no source
reconciles it, prune never reaches it, and `find_released` stops the
next pass planting a duplicate. Releasing is a **positive flag**, never
inferred from a null `managed_by` — rows written before `managed_by`
existed also have none, and those the deployment still owns.

This Unity project is for an AI Assistant, which is implemented as a heavily distributed multi-node system. Each node in the system communicates via English language based public APIs. The assistant's "brain" is then implemented a bit like a back office, where each manager deals with different aspects of the assistant's overall emergent intelligence. For the most part (with a few exceptions, such as `CodeActActor` and `ConversationManager`) the public methods of these managers are implemented as asynchronous tool loops, whereby a central LLM handles the English language request by orchestrating lower level tools which read and mutate the manager-specific backend resources (via the unify python client, which wraps the REST API connecting to the DB). These manager methods are dynamic, and expose handles for mid-flight steering, question answering, pausing, resuming and stopping etc. These manager methods are also often **nested**, whereby the public API of one manager is exposed in the tool set of a higher level manager. The async tool loops can also steer their inner in-flight tools, enabling fully nested dynamic steering of async tool loops up to an arbitrary depth. In terms of hierarchy, the `Actor` serves as the central intelligence, orchestrating other managers through code-first plans. Importantly, we never apply "fast paths" or heuristics based on regex or substring detection from user commands. If a method needs to respond correctly to a certain type of user input, this must **always** be addressed by prompting the model and/or improving docstrings of the exposed tools in order to **nudge** the LLM in the right direction.

# Full Local Stack First

This is internal agent/developer guidance. It intentionally differs from the
public `unity` README: the README is for open-source users who run `unity`
against hosted Orchestra and do not have the private `unity-deploy` repo. For
internal cross-repo work, default to the private full local stack in
`unity-deploy/selfhost/`.

Before starting, stopping, rebuilding, or repairing any local service, assume the
full stack may already be running and inspect it first:

```bash
bash /Users/djl11/unity-deploy/selfhost/stack.sh status
```

Default commands:

- Start full source stack: `bash /Users/djl11/unity-deploy/selfhost/stack.sh up`
- Repair only Console: `bash /Users/djl11/unity-deploy/selfhost/stack.sh repair-console`
- Smoke test: `bash /Users/djl11/unity-deploy/selfhost/stack.sh smoke`

Do not run isolated commands against a live stack unless the user explicitly asks
for isolated mode: `npm run dev`, `next dev`, `next build`, `npm run ci`,
`console/scripts/local.sh start`, `orchestra/scripts/local.sh start`, or
`unity/scripts/local.sh start`.

If isolated mode is explicitly requested, use the relevant override env var and
explain which full-stack guarantees are being bypassed.

# Local Development Environment

## Python Interpreter
- **ALWAYS** use the project's virtual environment interpreter: `.venv/bin/python`.
- Do not use global python or other system interpreters.

## Environment Bootstrap (fresh clone / Cloud Agents)
- The repo virtualenv lives at **`.venv/`** and is intentionally not committed.
- If `.venv/` is missing (common in fresh clones and Cursor Cloud Agents), bootstrap it with:
  - `pip install uv && uv sync --all-groups`
- `tests/parallel_run.sh` will also auto-bootstrap `.venv/` (and install `uv` via `pip --user` if needed).
- Prefer `python3` over `python` in shell scripts; some environments don't provide a `python` shim.

## Running Tests

### Terminal Isolation (Automatic)
Each terminal session (including each A coding agent) automatically gets its own **isolated tmux server**. This means:
- Your tests don't interfere with other agents' tests
- `tmux kill-server` only affects YOUR terminal's sessions
- No configuration needed - it's automatic

### Choosing the Right Command

The script **always blocks** until all tests complete (or timeout), streaming pass/fail results inline as tests finish.

| Scenario | Command |
|----------|---------|
| **Default** | `tests/parallel_run.sh [path]` |
| **Serial mode** (one session per file) | `tests/parallel_run.sh -s [path]` |
| **With timeout** | `tests/parallel_run.sh --timeout 300 [path]` |

**Note:** Do not use `parallel_cloud_run.sh` directly. For CI, use commit message tags (see `propose-ci-tests-for-commits.md`).

- The script blocks until all tests complete, then reports success (exit 0), failure (exit 1), or timeout (exit 2).
- `--timeout N` aborts if tests don't complete within N seconds.

#### Parallelism Behavior

- By default: One tmux session per *test*. All tests run concurrently (maximum speed).
- With `-s`: One tmux session per *file*. Tests within a file run serially.

**Examples:**
```bash
# Single test file with multiple tests (default: runs all tests concurrently)
tests/parallel_run.sh tests/contact_manager/test_ask.py

# Specific test functions
tests/parallel_run.sh tests/test_foo.py::test_one tests/test_bar.py::test_two

# Small directory
tests/parallel_run.sh tests/actor/

# Large test suite with serial mode (one session per file, fewer total sessions)
tests/parallel_run.sh -s tests/

# With timeout (abort after 5 minutes)
tests/parallel_run.sh --timeout 300 tests/contact_manager/
```

### Failure Handling
- If the script exits with code 1, failures were detected.
- Do **NOT** inspect `tmux` panes directly.
- **ALWAYS** read the corresponding log file in `logs/pytest/` for the failed session.

### Log Directory Naming
Log directories use a **datetime-prefixed format** for natural time-based ordering in the filesystem:
- Format: `YYYY-MM-DDTHH-MM-SS_{socket_name}` (e.g., `2025-12-05T14-30-45_unity_dev_ttys042`)
- The datetime is when the test run started
- The socket name identifies the terminal session (for isolation)

**Finding your logs:**
- The script prints the log directory path when tests start
- Directories are sorted chronologically, so recent runs appear at the bottom of `ls` output
- Each run gets its own directory, even from the same terminal

**Example directory listing:**
```
logs/pytest/
├── 2025-12-05T09-15-22_unity_dev_ttys004/
├── 2025-12-05T10-30-45_unity_dev_ttys026/
├── 2025-12-05T14-22-18_unity_dev_ttys004/
└── 2025-12-05T15-00-00_unity_dev_ttys042/
```

**Environment variables:**
- `UNIFY_LOG_SUBDIR`: The full datetime-prefixed log directory name (set by `parallel_run.sh`)
- `UNIFY_TEST_SOCKET`: The terminal socket name for tmux isolation (e.g., `unity_dev_ttys004`)

### Cleanup (REQUIRED)
- **ALWAYS** kill failed tmux sessions after extracting failure info from `logs/pytest/`.
- Logs are persisted in `logs/pytest/`; keeping sessions open is unnecessary.
- Run: `tests/kill_failed.sh` to kill all failed sessions from YOUR terminal.
- Run: `tests/kill_server.sh` to kill the entire tmux server for YOUR terminal.
- For cross-terminal cleanup: `tests/kill_failed.sh --all` or `tests/kill_server.sh --all`

### Permissions
- Use `required_permissions: ['all']` to ensure access to `.env` and log files.

## Pre-commit Hooks
- The `pre-commit` tool is installed in the project `dev` dependencies.
- **Execution**: Run via the python module to ensure path visibility:
  - `.venv/bin/python -m pre_commit run --all-files`
- **When to run**: If you modify files and want to ensure they pass CI checks, run pre-commit *before* committing.

## Dependencies
- This project uses `uv` for dependency management.
- Config file: `pyproject.toml`

## Edit Safety
- **Protected Files**: Do not edit `uv.lock` or `package-lock.json` manually. Use the appropriate package manager commands.
- **Sensitive Files**: Do not output the contents of `.env` or `*.key` files to the chat.

# Log Directory Navigation

## Tool Behavior for Logs

The `logs/` directory is gitignored, which affects tool availability:

| Tool | Works? | Notes |
|------|--------|-------|
| **Read** | ✅ Yes | Preferred for reading log file contents |
| **Shell** | ✅ Yes | Use `ls` to explore directory structure |
| **LS** | ⚠️ Unreliable | May work with direct paths |
| **Glob** | ❌ No | Git-aware index excludes gitignored paths |
| **Grep** | ❌ No | Git-aware index excludes gitignored paths |

## Log Directories

| Directory | Purpose |
|-----------|---------|
| `logs/pytest/` | Test output logs (datetime-prefixed subdirs per run) |
| `logs/unity/` | Unity LOGGER output (async tool loop, managers) |
| `logs/unillm/` | Raw LLM request/response traces |
| `logs/unisdk/` | Unify SDK HTTP traces |
| `logs/orchestra/` | Orchestra session logs with per-request API traces |
| `logs/all/` | Cross-repo OTEL traces |

## Practical Steps

**Step 1: Explore with Shell**
```bash
# List log directories (sorted by time, newest last)
ls logs/pytest/

# List contents of a specific run
ls logs/pytest/2025-12-05T14-30-45_unity_dev_ttys042/
```

**Step 2: Read with Read tool**
```
Read: logs/pytest/2025-12-05T14-30-45_unity_dev_ttys042/contact_manager-test_ask.txt
```

## Orchestra Trace Files

Orchestra logs are organized by session with granular per-request traces:

```
logs/orchestra/
└── 2025-12-30T18-27-43/              # Session (one per orchestra start)
    └── requests/                      # Per-request API traces
        ├── 2025-12-30T18-28-03.852_DELETE_project-name_81ms_5cc61e5f.json
        ├── 2025-12-30T18-28-03.934_GET_projects_20ms_8e6fb277.json
        └── 2025-12-30T18-46-55.980_GET_projects_43ms_7be454fc.json
```

**Filename format:** `{datetime}_{METHOD}_{route}_{duration}_{trace_id_short}.json`
- `trace_id_short` = last 8 chars of the OpenTelemetry trace_id

**Trace correlation:** Each pytest run logs `TRACE_ID=<32-char-hex>` to stdout. Match the last 8 chars to Orchestra filenames:
```
# In pytest output:
[TRACE] TRACE_ID=099b207f89222185695d25977be454fc test=test_foo

# Corresponding Orchestra file:
logs/orchestra/<session>/requests/*_7be454fc.json
```

## Worktree Symlinks

In worktrees, log directories contain a `_root` symlink pointing to the main repository's logs. Use this when looking for logs from tests run in the main repo.

```bash
# List main repo's logs from a worktree
ls logs/pytest/_root/
```

## Example: Debugging a Test Failure

```bash
# 1. Find recent log directories
ls logs/pytest/

# 2. List logs in the most recent run
ls logs/pytest/2025-12-21T16-00-00_unity_dev_ttys042/
```

Then use the Read tool:
```
Read: logs/pytest/2025-12-21T16-00-00_unity_dev_ttys042/contact_manager-test_ask.txt
```

## Example: Correlating Test ↔ Orchestra Traces

When debugging why a test's API call failed:

```bash
# 1. Find the trace_id from pytest output (or grep the log file)
# Look for: [TRACE] TRACE_ID=099b207f89222185695d25977be454fc test=test_foo

# 2. Find the Orchestra session (most recent)
ls logs/orchestra/

# 3. Find the matching trace file (last 8 chars of trace_id)
ls logs/orchestra/2025-12-30T18-27-43/requests/*7be454fc*
```

Then read the trace file to see the full request/response with all spans.

# Parallelize Test Analysis

Tests run in tmux sessions and stream results inline. For long test runs, you can start them in the background and analyze early failures while remaining tests complete.

## The Mental Model

When you run `parallel_run.sh`, each test spawns in its own tmux session. The script blocks until all tests complete, streaming pass/fail results inline as tests finish. Log files are written to `logs/pytest/` as each test completes.

For long-running test suites, you can run in the background and analyze failures incrementally:
1. Start tests with `block_until_ms: 0` in the Shell tool call
2. Read log files as they appear in `logs/pytest/`
3. Analyze failures immediately
4. Check back later for final results

## Steps (Background Mode)

```bash
# 1. Start tests in background (use block_until_ms: 0 in Shell tool call)
tests/parallel_run.sh tests/some_module/

# 2. Check what logs exist (tests write here as they complete)
ls logs/pytest/<latest-run-dir>/

# 3. Read and analyze any failures that have appeared
# (Use the Read tool on specific log files)

# 4. Continue reasoning about the failure while tests run
# 5. Check back later for more results if needed
```

## The Principle: Time-to-Solution Over Turns

Prioritize **minimizing time to find a solution** rather than minimizing turns or waiting for "complete" information. Early failures often provide enough signal to begin investigation. You can always check for additional failures later.

This doesn't mean being hasty or sacrificing thoroughness. It means: **don't wait for information you don't need yet**.

## Relationship to Other Rules

- **`surgical-verification-before-tests.md`**: Covers *pre-test* optimization (quick verification scripts)
- **This rule**: Covers *mid-test* optimization (incremental analysis)
- **`log-directory-navigation.md`**: Explains how to read from `logs/pytest/` (use Shell for `ls`, Read tool for file contents)

# Product Vocabulary: One Noun Per Concept

The assistant reasons in the words we hand it. When two concepts share a
word, the model has to guess which one a prompt means, and the guess is
invisible until it routes to the wrong manager.

This rule exists because we already paid for that once. `Workflow` shipped
as a first-class type in August 2026 while "workflow" was load-bearing
prose in ~180 places across the actor and task prompts, meaning *multi-step
procedure* and *recurring task*. The design session that proposed it had
flagged the collision and concluded "give the new thing an unused name" —
then the code shipped anyway and the rename came two days later, across 39
files, invalidating LLM caches for the whole actor suite. The rename was
correct; the sequencing was the mistake.

## The canonical nouns

| Concept | Word | Owner |
|---|---|---|
| Installable package that sets the assistant up for a recurring job | **workflow** | `WorkflowManager` |
| Durable unit of scheduled or triggered work | **task** | `TaskScheduler` |
| One run of a task | **execution** | `Tasks/Executions` |
| Written-down multi-step how-to | **procedure** | `GuidanceManager` |
| Executable unit the assistant calls | **function** | `FunctionManager` |
| Durable sourced statement about the world | **claim** | `KnowledgeManager` |
| What the actor writes and runs to satisfy one request | **plan** | `Actor` |
| Ordered instructions inside a docstring | **Steps** (section header) | — |

Rule of thumb: **"workflow" is a noun you install. Everything that used to
borrow the word is a procedure, a task, or a plan.**

Do not write "workflow" to mean a multi-step anything. Do not write
"recurring workflow" — that is a recurring task. Do not title a docstring
section `### Workflow` — it now reads as a type reference; use `Steps` or
`Procedure`.

## The three stores, and how to tell them apart

`FunctionManager`, `GuidanceManager` and `KnowledgeManager` are the three
places durable know-how lives, and they are distinguished by *what kind of
thing they hold*, not by topic:

| | FunctionManager | GuidanceManager | KnowledgeManager |
|---|---|---|---|
| Role | the **what** | the **how** | the **is** |
| Holds | one callable | a multi-step procedure | one typed claim |
| Content | executable implementation | natural-language recipe | sourced statement |
| Analogy | a tool's docstring | a prompt that references tools | a fact with provenance |

Deciding where something belongs:

- **Can it run?** Function. If it is code that executed successfully and
  would be worth calling again, it is a function — not a procedure
  describing the code.
- **Does it tell someone how to act?** Procedure (guidance). Reach for it
  when composing several functions is non-obvious, or when a durable rule
  or policy governs how work is done. A procedure links the functions it
  composes via `function_ids`, which is also how a rule change finds every
  implementation that embeds it.
- **Is it true regardless of how you act on it?** Claim (knowledge).
  Facts, policies, definitions, decisions, constraints, preferences —
  carrying `source_refs` when provenance is known.

The common error is storing a procedure that merely restates one
function's docstring. If a single function's docstring already explains
its inputs, behaviour and use, store the function and stop.

Negative scope matters as much: people belong in `ContactManager`,
credentials in `SecretManager`, file bytes in `FileManager`. None of the
three stores is a dumping ground for "stuff we learned".

## Before naming a new first-class type

1. **Grep first.** `rg -ic "<candidate>" unify/ tests/`. If the word
   appears in prompts, docstrings or examples meaning something else, you
   have a collision.
2. **Prefer an unused word** over renaming existing prose. An unused word
   is free; a rename costs a sweep plus fresh LLM inference everywhere a
   prompt changed.
3. **If you keep the colliding word, do the cleanup in the same change.**
   Not "later" — the bill compounds with every prompt written against the
   ambiguous meaning, and a half-renamed codebase teaches the model both
   meanings at once.
4. **Watch for the near-synonym trap.** A new noun must not collide
   *conceptually* either. "Recipe" and "playbook" are both unused as
   types, but the three-stores table above calls guidance a
   "natural-language recipe" and a playbook *is* a procedure — either one
   would have recreated the same problem one table over.

## What gets renamed, and what does not

Rename **our** vocabulary: prompts, docstrings, tool labels, example
titles, section headers, internal identifiers, test names and fixtures.

Leave alone:

- **Simulated user speech in eval tests.** Users really do say "workflow"
  loosely about business processes. Scrubbing it removes exactly the
  ambiguity the assistant must survive, and re-runs paid inference for
  negative value.
- **Third-party product names.** HubSpot and Salesforce ship features
  called Workflows; renaming makes our integration docs factually wrong.
- **GitHub Actions terminology.** `.github/workflows/`,
  `workflow_dispatch`, `workflow_run` and `workflow_call` are API
  keywords. Renaming breaks CI.
- **The shared `global-agent-rules` submodule.** Edit it in its own repo,
  never here.

After editing anything under `.agents/`, regenerate the aggregate:
`python3 .agents/global-rules/build_agents_md.py`. Never hand-edit
`AGENTS.md`.

## Known overloads, accepted

- **`surface`** means two things: in workflows, a manager a bundle plants
  content into (`SCOPED_SURFACES`, `Surface.sync`); in canvas, a place a
  view renders (chat embed, assistants tab, standalone). Both readings are
  established and neither is user-facing, so they coexist. If this ever
  causes a real mistake, the workflow sense is the one to rename (to
  `target`), because the canvas sense is shared with Console.

## Console (`unifyai/console`) mapping

Console is the user-facing surface and does not have to mirror internal
names, but the mapping must be **deliberate**. It already is, in
`src/components/Workflows/workflowCategories.ts`, which distinguishes the
*content kind* from the *section it lives in*:

| unify surface | Console kind label | Section it lives in |
|---|---|---|
| `guidance` | Procedures | Guidance |
| `knowledge` | Knowledge claims | Knowledge |
| `functions` | Functions | Functions |
| `tasks` | Recurring tasks | Tasks |

That split is correct and matches this rule's nouns: a section is a
library ("Guidance", "Knowledge"), and what it holds is a procedure or a
claim. When adding a Console surface that renders manager content, extend
that map rather than inventing a label at the call site.

**`skill` is an umbrella, not a synonym.** unify uses "skills" for
*anything worth storing across the three stores* — the `store_skills`
tool, the `"Storing reusable skills"` review label. That is a legitimate
superset covering functions, procedures and claims together. It is wrong
only when used for one specific member: Console's `FunctionSkill` type
names a function with the umbrella word.

## Integration slugs: one id space, three connection routes

An integration is named by its **provider app slug** — lowercase, the id
space shared by Console's integrations gallery (`canonicalSlug`),
`app_slug` in the integrations primitives, and native package manifests.
`gmail`, `hubspot`, `notion`.

Two other spaces exist and are **not** interchangeable with it:

- **OAuth provider aliases** (`runtime_oauth.py`): `google` with aliases
  `gmail` / `google_workspace` / `drive`. Fine for resolving a token,
  invisible to the gallery — a workflow requirement naming
  `google_workspace` renders a chip with no logo and no connect action.
- **Integration package directory names** (unify-deploy). These *are*
  provider app slugs, but only a subset: most gallery apps have no
  package, and Workspace has none at all.

Whether an app is native or third-party is **not** a caller's concern,
because an app can offer more than one route and the route can change
without the caller changing. Anything asking "is this connected?" asks
`RequirementResolver`, which consults each authority in turn: a live
gallery connection row, then the app's own native package manifest for
the secrets that make it usable, then a caller-declared secret for BYOD
OAuth. It reports which authority answered (`via`), so a UI can say
whether the user needs the connect flow or a pasted key.

Never gate on the secret keyset alone. That was the first version of this
check and it silently held every gallery-connected app, because a
provider-backed connection is a connection row and never a secret.

**Do not couple to display labels across repos.** Console's LiveActions
view string-matches unify's `display_label` prose
(`dl === 'storing reusable skills'`) to choose icons and categories.
Those labels are human-facing copy that this rule actively encourages
rewording, and nothing on either side fails when they drift — the view
just silently falls through to generic rendering. Match on the stable
tool name instead. Until that lands, a `display_label` change in unify is
a cross-repo change.

# State Manager Interface Design

## Base Class Public APIs

The public API for all state managers (`ContactManager`, `TranscriptManager`, `TaskScheduler`, `WebSearcher` etc.) is fully contained in the docstrings of the abstract methods defined on the base class `Base{SomeManager}` in `base.py`. All high level usage instructions should be fully encapsulated in these docstrings. These docstrings are then attached to the public methods of any derived class via `@functools.wraps(Base{StateManager}.{public_method}, updated=())`. These docstrings should not make **any** reference to **other managers** (we don't want to lock in any brittle cross-references, as other managers may change) and should also not make any reference to their **internal implementation**, including the private tools used for any particular instantiation of this abstract base class, with a consistent implementation agnostic public API.

## Prompts vs Tool Docstrings

The prompts in each prompt builder file should focus on the high level usage patterns, general guidance to the LLM, and specifically how to reason about the **composition** of tools, which tool to use in which scenario with contrastive explanations etc. However, in order to have a fully modular design and maximise our separation of concerns, it's very important that we do **not** bloat these prompts with any purely tool-specific information. This belongs exclusively in the tool's unique docstring (which the LLM gets access to). If the guidance is about deciding between two tools or using these tools together for complex composite behaviour, then it belongs in the prompt for the high-level public method in `prompt_builders.py`. If it's purely tool-specific, then it belongs in the tools own docstring.

Use this to decide which manager to call, what each owns, and where its jurisdiction ends. Keep manager docstrings implementation‑agnostic; this guide is only for high‑level routing and composition.

### ConversationManager
- **Role**: Live chat orchestrator. Routes user requests to `Actor` for code-first execution and wires steering (pause/resume/interject/stop) during conversations.
- **Scope**: Conversation‑level control and message flow; returns/relays steerable handles from inner tools.
- **Connections**:
  - **Steered by**: Top-level UI/controller (outside managers).
  - **Steers**: `Actor.act` (central intelligence); relays in‑flight handles from `Actor.act` and `TaskScheduler.execute`.

### Actor
- **Role**: Central intelligence that orchestrates all state managers through code-first plans. Generates and executes Python plans that call primitives and top-level JSON tools.
- **Scope**: Code-first execution via `act()` method. Generates Python plans that orchestrate `primitives.contacts.*`, `primitives.tasks.*`, etc., plus top-level JSON tools such as `GuidanceManager_*` and `KnowledgeManager_*`. Wires in‑flight handles to `ConversationManager` for real‑time steering.
- **Connections**:
  - **Steered by**: `ConversationManager` (primary caller of `act()`).
  - **Steers**: State manager primitives (`primitives.contacts.*`, `primitives.tasks.*`, etc.), `KnowledgeManager_*` / `GuidanceManager_*` JSON tools, `TaskScheduler`, and the `ConversationManager` handle (`ask`/`interject`/`get_full_transcript`). Uses `FunctionManager` for function discovery and execution.

### Actor routing playbook
- **Read‑only questions**
  - Tasks → `primitives.tasks.ask` or `TaskScheduler.ask`
  - Contacts → `primitives.contacts.ask`
  - Transcripts → `primitives.transcripts.ask` (may call `primitives.contacts.ask` for participants)
  - Knowledge → `KnowledgeManager_search` / `KnowledgeManager_filter` / `KnowledgeManager_get_knowledge` (top-level JSON tool calls, not primitives)
  - Secrets (metadata/placeholders only) → `primitives.secrets.ask`
  - Time‑sensitive/web ("today/latest/now") → `primitives.web.ask`
  - About a specific received file (filename known) → `primitives.files.ask`
- **Mutations (create/edit/delete/merge)**
  - Tasks → `primitives.tasks.update` or `TaskScheduler.update`
  - Contacts → `primitives.contacts.update`
  - Knowledge claims → `KnowledgeManager_add_knowledge` / `KnowledgeManager_update_knowledge` / `KnowledgeManager_invalidate_knowledge` / `KnowledgeManager_supersede_knowledge` / `KnowledgeManager_delete_knowledge` (top-level JSON tool calls)
  - Guidance → `GuidanceManager_add_guidance` / `GuidanceManager_update_guidance` / `GuidanceManager_delete_guidance` (top-level JSON tool calls)
  - Secrets → `primitives.secrets.update`
- **Execution vs. interaction**
  - Ephemeral live action (UI control/one‑off interaction) → `Actor.act` (called via ConversationManager)
  - Durable, tracked work → `TaskScheduler.execute` (via `primitives.tasks.execute`)
  - Never use `TaskScheduler.update` to start work; always use `execute`.
- **Storing new data or files (any source)**
  - Rows from an API / connected app / user input, specific files, whole folders, or a reshape of a stored table → `primitives.ingestion.submit(source, target)`. One verb for every source/target pairing; returns a run handle immediately.
  - Observe and recover with `primitives.ingestion.get_status` / `get_logs` / `wait` / `retry` / `cancel` / `pause` / `resume`. `status.next_step` states the one action that makes sense. `status.files` breaks a batch down per file; `retry(files=[...])` aims at one of them, and `only="stale"` takes over an attempt whose lease lapsed.
  - Close a run with `primitives.ingestion.reconcile` before calling the data ready: it reports rows landed against rows expected **and** the columns that are blank in every row sampled. A row count alone once agreed with a run that committed 449,287 rows holding no values.
  - There is no `primitives.data.ingest` — a direct ingest blocked the plan for the length of the write and left nothing to inspect if it died part-way.
- **Generative UI (canvases)**
  - Author, revise and inspect interactive views → `primitives.canvas.*` (`create_view`, `update_view`, `refresh_props`, `preview`, `list_invocations`, …). Bind live data to stored tables (including ones an ingestion run just produced via `status.contexts`), never to a provider call.
- **File → knowledge distillation**
  - Parse with `primitives.files.parse`, then distill durable statements into typed claims via `KnowledgeManager_add_knowledge` (attach `source_refs` pointing at the file / transcript / user statement). There is no `primitives.knowledge` pipeline and no NL `ask`/`update`/`refactor` loop on KnowledgeManager.
- **Images**
  - Images are referenced **by filesystem path** across the entire stack. Screenshot directories (`Screenshots/User/`, `Screenshots/Assistant/`) and other workspace paths serve as the universal cross‑manager pointer for visual content. Managers and plans reference images via their relative filepath — no special `images` parameter or structured ref types are needed at the public API boundary.

### ImageManager
- **Role**: Persistent image store and metadata registry. Provides durable `image_id`‑keyed storage in the `Images` context, backing filesystem images with cloud persistence and queryable metadata.
- **Data model & identity**:
  - Every stored image has a unique numeric `image_id` (stable within the active assistant context).
  - Image rows store base64 bytes or a cloud object URL, plus metadata (caption, timestamp, mime/type).
  - Each image row carries an optional `filepath` field that records the local filesystem path the image was saved to. This is the bridge between the filesystem‑based reference convention and the persistent `Images` context.
- **ImageHandle wrapper**:
  - Internal code operates on an `ImageHandle` abstraction that wraps an image row and exposes:
    - `image_id`, optional `caption`/metadata, `filepath`
    - `raw()` → returns image bytes (resolves base64 vs signed URL transparently)
    - `ask(question)` → sends the image to a vision‑capable model and returns a text answer
- **Cross‑manager image convention**:
  - **Filesystem paths are the universal image reference.** When images need to flow between managers (e.g., from `Actor` plans into `GuidanceManager.update`), they are referenced by their workspace‑relative filepath (e.g., `Screenshots/User/2026-02-16T14-30-45.jpg`). The receiving manager or its internal tools can resolve filepaths to `image_id`s via `ImageManager.filter_images(filter="filepath == '...'")` when persistent storage linkage is needed.
  - Some managers accept an `images` parameter using `ImageRefs`/`AnnotatedImageRef` types for structured image attachment (e.g., `GuidanceManager.add_guidance` and `update_guidance` accept annotated image references to link visual content directly to guidance entries). This is the appropriate mechanism when a manager's data model has a first-class `images` field. The filesystem-path convention above applies at the *orchestration* boundary (Actor plans, cross-manager references); within a manager's own CRUD interface, structured image refs are the native format.
- **Connections**:
  - **Steered by**: managers that persist or query images (e.g., `GuidanceManager`, `TranscriptManager`).
  - **Steers**: — (exposes image records/handles; managers decide when/how to query or persist images).

### TaskScheduler
- **Role**: Owner of durable tasks and their execution.
- **Scope**: ask (read‑only about tasks), update (create/edit/delete tasks), execute (start tasks; returns live handle). Tasks are independent — there is no queue chaining or ordering between them.
- **Edge**: Never use `update` to "start" work; always start via `execute` (after any needed updates).
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.tasks.*`); `ConversationManager` (for direct task management).
  - **Steers**: `Actor` (to run tasks via `execute`); exposes a live `ActiveTask` handle that is wired back through `ConversationManager` for steering.

### KnowledgeManager
- **Role**: Passive typed claim ledger for durable domain knowledge (facts, policies, definitions, decisions, constraints, insights, preferences) with provenance (`source_refs`) and lifecycle status (active / superseded / invalidated).
- **Scope**: CRUD and lifecycle operations (`search`, `filter`, `get_knowledge`, `add_knowledge`, `update_knowledge`, `delete_knowledge`, `invalidate_knowledge`, `supersede_knowledge`, `reconcile_sources`, `clear`) exposed as first-class JSON tool calls on the CodeActActor (`KnowledgeManager_*`) — **not** as `primitives.knowledge.*`. No natural-language `ask` / `update` / `refactor` tool loops.
- **Negative scope**: Does **not** own people/contacts (ContactManager), procedural how-tos/SOPs (GuidanceManager), user Python functions (FunctionManager), received file bytes/parsing (FileManager), or secrets/credentials (SecretManager).
- **Writers**: KnowledgeManager is a passive store. Live writers are the Actor / ConversationManager (user-requested claim storage), StorageCheck (trajectory distillation into claims), and optionally MemoryManager (offline consolidation). Distill file/transcript content into claims with `source_refs`; do not treat KnowledgeManager as a schema-refactoring table store.
- **Connections**:
  - **Steered by**: `Actor` (via top-level `KnowledgeManager_*` JSON tool calls, not primitives); optional offline writes from `MemoryManager`.
  - **Steers**: — (exposes typed claim records; callers attach provenance and decide when to write).

### ContactManager
- **Role**: Source of truth for people/contact records.
- **Scope**: ask (read‑only), update (create/edit/delete/merge contacts).
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.contacts.*`); read‑only usage by `TranscriptManager.ask` (to resolve/compare contacts during transcript queries).
  - **Steers**: —

### TranscriptManager
- **Role**: Store and retrieval surface for message transcripts.
- **Scope**: ask (read‑only retrieval, filtering, analysis); may expose summarization in implementations.
- **Edge**: Summarize conversations here; write long‑term distilled facts to `KnowledgeManager` if needed.
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.transcripts.*`).
  - **Steers**: `ContactManager.ask` (for participant lookup/attributes in transcript answers).

### FileManager
- **Role**: Read‑only registry and parsing for received/downloaded files.
- **Scope**: exists/list, parse, ask about a specific file (read‑only tool loop), describe (storage discovery).
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.files.*`).
  - **Steers**: `DataManager` (internally delegates filter/search/reduce/join operations).

### DataManager
- **Role**: Low‑level data operations on any Unify context.
- **Scope**: Canonical implementation of filter, search, reduce, join, insert, update, delete, vectorize, plot. `ingest` exists as the low-level chunked write engine but is **not** exposed to the Actor — storing new data routes through `IngestionManager` so every write is recorded, checkpointed and recoverable.
- **Connections**:
  - **Steered by**: `FileManager` (delegates data ops), `IngestionManager` (row writes via the shared checkpointed engine), `Actor` (via `primitives.data.*`).
  - **Steers**: — (pure primitives module, no high‑level tool loops).

### IngestionManager
- **Role**: The one verb for storing data and files from anywhere — `submit(source, target)` — with a resumable, checkpointed engine behind it that in-process and worker-fleet execution share.
- **Scope**: `submit`, `get_status`, `get_logs`, `wait`, `list_runs`, `retry`, `cancel`, `pause`, `resume`, `reconcile` via `primitives.ingestion.*`. Sources: `RowsSource` (anything in hand), `FilesSource` / `FolderSource` (parse via the file pipeline), `TableSource` (reshape stored rows). Targets: `TableTarget` (one queryable context) or `CollectionTarget` (documents kept whole, inner tables extracted). Runs and their events are rows in `Ingestion/Runs` + `Ingestion/Events`.
- **Tiering**: never a caller's choice. Files parse off the assistant's process whenever a worker fleet is reachable; rows and tables run in process only under a measured row ceiling. Both tiers write the same artifacts, leases and checkpoints, so the tier affects latency and nothing else.
- **Negative scope**: does not query or reshape-in-place (DataManager), does not answer questions about file contents (FileManager `ask`), does not own provider fetches (integrations fetch, ingestion stores).
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.ingestion.*`); `FileManager` (attachment ingestion submits here).
  - **Steers**: `DataManager.ingest` (row writes), the file parse pipeline, and the hosted pipeline control plane (`/infra/pipeline/*`) for dispatched runs.

### CanvasManager
- **Role**: Assistant-authored generative React UI. The actor writes real TSX against `@unity/canvas-kit`; it is linted, typechecked, bundled, rendered headlessly and critiqued before publish; Console displays it in a genuinely isolated frame.
- **Scope**: `create_view`, `update_view`, `refresh_props`, `get_view`, `list_views`, `delete_view`, `preview`, `run_invocation`, `list_invocations` via `primitives.canvas.*`. Rows live in `Canvas/Views` / `Canvas/Actions` / `Canvas/Invocations`; a routing token is registered with the backend on publish.
- **Data plane**: query bindings (context-backed tables, executed server-side per view) and materialised props (LLM-shaped reads frozen at author/refresh time). Connected-app data must be **stored first** (via `primitives.ingestion.submit`) and bound as an ordinary table.
- **Write plane**: declared, schema-validated actions; the viewer's input is validated server-side, recorded as an invocation row, then executed by the assistant in one of three lanes (stored function, task trigger, actor request).
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.canvas.*`); `ConversationManager` (executes recorded invocations on `CanvasInvocationRequested`).
  - **Steers**: `DataManager` (binding dry-runs), `FunctionManager` (action target resolution and execution), `TaskScheduler` (task-lane triggers), the actor (assistant-lane requests).

### WebSearcher
- **Role**: Lightweight, text-based retrieval engine for quick one-off internet queries (headlines, weather, definitions, current events).
- **Scope**: ask only (search, extract, crawl, map against the public web); returns live handle. No gated-site access, no browser automation, no credentials. For authenticated or complex web procedures, use Tavily + SecretManager + ComputerPrimitives directly via code-first plans.
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.web.*`).
  - **Steers**: — (results may subsequently be persisted as typed claims via `KnowledgeManager_add_knowledge` when requested).

### SecretManager
- **Role**: Owner of secrets.
- **Scope**: ask (metadata/placeholder answers only), update (create/edit/delete secrets).
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.secrets.*`).
  - **Steers**: —

### BlacklistManager
- **Role**: Catalogue of blocked contact details per communication medium (email, SMS, phone).
- **Scope**: Primitive CRUD operations — filter/create/update/delete blacklist entries. No tool loops; purely programmatic.
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.blacklist.*`); `ConversationManager` (for direct blacklist management).
  - **Steers**: —

### FunctionManager
- **Role**: Catalogue of user‑supplied Python functions and metadata.
- **Scope**: add/list/get_precondition/delete/filter/search over functions.
- **Connections**:
  - **Steered by**: `Actor` (loads/executes functions during actions). `GuidanceManager` reads function rows via the shared "Functions" context (no direct API calls).
  - **Steers**: —


### GuidanceManager
- **Role**: Owner of procedural how-to information: step-by-step instructions, standard operating procedures, software usage walkthroughs, and strategies for composing functions together.
- **Scope**: CRUD operations (search, filter, add_guidance, update_guidance, delete_guidance) exposed as first-class JSON tool calls on the CodeActActor — **not** as primitives. Read tools are gated by the discovery-first policy; write tools are available in the doing loop (for user-requested guidance storage) and the storage review loop (for compositional strategy extraction).
- **Builtins library**: reads also federate over a global, read-only guidance catalogue (`Guidance` context in the public-read `Builtins` project) holding entries imported from the Agent Skills ecosystem with stable hash-based ids and `is_builtin=True`. Default-on for every assistant; `update_guidance`/`delete_guidance` refuse builtin ids. Seeded from the committed snapshot `unity/guidance_manager/builtins_guidance.json`, generated by `scripts.skill_migration.builtins_import` from commit-pinned upstream repos (`builtins_skills.lock.json`); upstream drift is detected (`--check`) but never auto-applied.
- **Connections**:
  - **Steered by**: `Actor` (via top-level `GuidanceManager_*` JSON tool calls, not primitives).
  - **Steers**: reads functions from the shared "Functions" context to surface linked functions.

### WorkflowManager
- **Role**: Catalogue of installable **workflows** — hand-curated, versioned packages that set an assistant up for a recurring job in one install (procedures, claims, functions and a recurring task, planted together). Owner of install state and settings, nothing else.
- **Scope**: `list_workflows`, `get_workflow`, `install_workflow`, `uninstall_workflow`, `get_installation_params` as first-class `WorkflowManager_*` JSON tool calls — **not** primitives. Installing fans out each surface's existing `sync_custom`; there is no second content store and no second reconcile loop. `reconcile_installed` and `execute_requests` are upkeep, deliberately **not** tools (boot/ops, and a single-slug update is `install_workflow` re-run). Tools enter the actor's schema only when a catalogue is configured (`UNIFY_WORKFLOWS_DIR`), so deployments without a shelf keep byte-identical tool schemas and LLM caches.
- **Negative scope**: **has no runtime.** No executions, no steering handle, no run-now, no run history — that all belongs to the tasks it plants (`TaskScheduler`). "Run the workflow now" resolves to triggering its task. It also does not own connections (declared as requirements, resolved against the integrations layer), does not author bundles (git-only, curated in unify-deploy), and never pre-seeds contacts, transcripts or blacklist entries.
- **Requirements gate arming, never planting**: an install with an unmet requirement still plants everything and returns `connect_required`; the planted tasks stay disarmed until the connection lands, and a repeat install arms them.
- **Shelf vs installation**: the catalogue listing and each artifact's published copy are platform data in the public-read `Builtins` project (`Workflows/Catalog`, `Workflows/Content`), admin-seeded and hash-guarded. Everything per-assistant — installations, params, planted rows, requested changes (`Workflows/Requests`) — lives in the assistant's own contexts.
- **Connections**:
  - **Steered by**: `Actor` (via top-level `WorkflowManager_*` JSON tool calls, not primitives); boot (`bootstrap_workflow_catalog` → `reconcile_installed` + `execute_requests`); `ConversationManager` on a `WorkflowRequestRequested` event.
  - **Steers**: each surface's `sync_custom` — `GuidanceManager`, `KnowledgeManager`, `TaskScheduler`, `FunctionManager` — plus `TaskScheduler.set_custom_tasks_enabled` to arm or hold what it planted.

### MemoryManager
- **Role**: Offline memory maintenance (periodic, non‑interactive).
- **Scope**: One‑shot methods that return strings (no live handles), e.g., updating contacts/knowledge from transcripts.
- **Connections**:
  - **Steered by**: Offline scheduler/controller (outside live chat orchestration).
  - **Steers**: `ContactManager.update`, `KnowledgeManager` typed claim APIs (`add_knowledge` / `update_knowledge` / lifecycle methods) to persist distilled facts from transcripts.

### EventBus
- **Role**: Cross‑cutting, in‑process publish/subscribe backbone and searchable event log used by all managers for telemetry and coordination.
- **Scope**: Managers publish structured events (notably `ManagerMethod` for incoming/outgoing `ask`/`update`/`execute`) via a thin logging wrapper; the bus supports `publish`, `search` (filterable queries), `join_published`/`join_callbacks` for deterministic flushing, per‑type window sizing, auto‑pinning, and callback registration.
- **Connections**:
  - **Steered by**: All public manager methods (through the logging decorator) and other components that emit operational events.
  - **Steers**: `MemoryManager` (registers callbacks to react to message and `ManagerMethod` events for maintenance passes); tests and higher‑level orchestrators query the bus to observe recent activity.

### Precedence and source of truth
- **Code is canonical**: This guide is descriptive. If the implementation ever contradicts these descriptions or relationships, the current code takes precedence.
- **Keep in sync**: As managers evolve, update this document alongside changes to cross‑manager wiring, public surfaces, or prompt composition so it remains accurate.
- **Where to update**: Prefer updating manager base docstrings (public API contracts) and prompt builders for tool composition guidance, and reflect those changes here.

# Surgical Verification Before Running Tests

When fixing infrastructure issues (especially concurrency, race conditions, or fixture problems), consider writing a quick verification script before running the full test suite.

## Why This Matters

Tests in this repo have significant overhead:
- LLM calls (cached: milliseconds, uncached: seconds to minutes)
- Backend API connections and context setup
- Fixture initialization and scenario seeding

A targeted verification script can validate a fix in **seconds** rather than waiting minutes for tests that may not even reliably reproduce the issue.

## When to Use This Pattern

✅ **Good candidates:**
- Race conditions and concurrency fixes (can simulate parallel execution directly)
- Fixture/conftest changes (test the seeding logic in isolation)
- Infrastructure plumbing (context management, caching, file locking)
- Non-deterministic failures that are hard to reproduce reliably in tests

❌ **Skip this for:**
- Simple bug fixes where one test runs quickly enough
- LLM behavior changes (need real eval tests with actual prompts)
- Cases where the verification script would be more complex than the test itself

## How to Write a Verification Script

1. **Name it clearly**: Use `tests/_verify_*.py` (underscore prefix signals temporary/debug)
2. **Keep it self-contained**: Minimal dependencies, no pytest fixtures
3. **Simulate the failure condition directly**: e.g., spawn threads to test race conditions
4. **Run fast**: Should complete in seconds, not minutes
5. **Print clear pass/fail output**: Make success/failure obvious

## Example Sequence

```bash
# 1. Implement the fix

# 2. Write quick verification script
#    tests/_verify_race_fix.py - spawns 5 threads, verifies no duplicates

# 3. Run verification (seconds)
.venv/bin/python tests/_verify_race_fix.py

# 4. If verification passes, run actual test(s)
tests/parallel_run.sh tests/path/to/relevant_test.py

# 5. Delete verification script (it served its purpose)
rm tests/_verify_race_fix.py
```

## Key Principle

This pattern **complements** running actual tests—it doesn't replace them. The goal is faster feedback during development, not skipping proper test validation.

# Test Philosophy

## Real LLMs, Never Mocked

We *never* stub the LLM client. Tests always use a real LLM via `unillm.AsyncUnify`. Responses are cached per unique input, so repeated runs are fast on CI. The LLM responses become "locked in" after the first successful run.

## Trigger-Based Synchronization, Not Sleeps

Never rely on sleeps to align events in tests. Always use the trigger helpers in `tests/async_helpers.py` to ensure each event occurs in the necessary order. This makes tests robust to significant timing differences between cached responses (milliseconds) and live LLM calls (up to a minute).

## Symbolic ↔ Eval Spectrum

Tests fall on a **spectrum** between two paradigms. Understanding where a test sits on this spectrum is essential for writing, debugging, and interpreting test results.

### Symbolic Tests (Infrastructure-Focused)

At one end, **symbolic tests** use the LLM purely as a stub. The LLM receives minimal "dummy" instructions designed to trigger specific code paths.

**Characteristics:**
- LLM behavior is deterministic and predictable
- Focus is on testing *infrastructure*: async tool loops, steering (pause/resume/interject/stop), state mutations
- The LLM's "intelligence" is irrelevant—we just need it to call the right tools in the right order
- **Failures indicate regressions in symbolic/programmatic logic**

### Eval Tests (Capability-Focused)

At the other end, **eval tests** exercise the system end-to-end. We ask a high-level question or give a directive, then verify the outcome—regardless of internal tool calls.

**Characteristics:**
- Focus is on *capability*: "Did the assistant correctly answer the question?" or "Did it complete the task?"
- Internal implementation details (tool call order, number of steps) don't matter
- Tests LLM reasoning and decision-making in realistic scenarios
- **Failures may indicate prompt issues, tool design problems, or capability gaps**

### The Spectrum (Not Binary)

Most tests sit **somewhere between** these extremes:
- Realistic prompts but only verify specific tool calls were made
- End-to-end behavior but with constrained, predictable inputs
- Combine symbolic infrastructure checks with high-level outcome assertions

Think of each test as having a "slider" between symbolic and eval—not a binary classification.

## Caching and Determinism (`UNILLM_CACHE`)

When `UNILLM_CACHE="true"` (the default), all LLM responses are cached:

1. **First run**: LLM executes normally; responses stored in cache
2. **Subsequent runs**: Cached responses replayed—no actual LLM calls

**Implications:**
- Both symbolic and eval tests become deterministic once the cache is populated
- After caching, both test types effectively verify that *symbolic logic has not regressed*
- Tests run fast on CI (milliseconds vs seconds/minutes for real LLM calls)
- To re-evaluate LLM behavior: delete `.cache.ndjson`, set `UNILLM_CACHE="false"`, or use `--no-cache`

### The Cache Is Never the Problem

"We just need to update the cache" is **never** a valid conclusion when debugging test failures. The cache is a faithful replay mechanism, not a source of bugs.

Cache keys are the exact LLM input. A cache hit means the code is sending **byte-for-byte identical** input to a previous run. If you modify prompts, tool docstrings, or system messages, the cache key changes automatically and you get fresh inference.

If a cached response causes a failure, an LLM **actually made that decision** given that exact input. This is a prompt issue, not a stale cache issue. Clearing the cache just re-runs inference on the same input—likely producing the same flawed output.

Only clear the cache to **re-evaluate** LLM behavior after prompt changes (`--no-cache`), not as a fix for failing tests.

## Tagging Tests as Eval

Mark a test file as eval with a module-level marker:

```python
import pytest

pytestmark = pytest.mark.eval  # All tests in this file are eval tests
```

For mixed files, use test-level markers:

```python
@pytest.mark.eval
@pytest.mark.asyncio
async def test_natural_language_query():
    ...
```

## Running Test Categories

```bash
# Run only eval tests
tests/parallel_run.sh --eval-only tests

# Run only symbolic tests
tests/parallel_run.sh --symbolic-only tests

# Re-evaluate LLM behavior (fresh calls, no cache)
tests/parallel_run.sh --no-cache tests

# Statistical sampling for eval reliability
tests/parallel_run.sh --no-cache --repeat 10 --eval-only tests/contact_manager
```

## When Debugging Failing Tests

1. **Determine test type**: Is it symbolic (infrastructure) or eval (capability)?
2. **For symbolic failures**: Logic bug or regression in programmatic code—debug the infrastructure
3. **For eval failures (cached)**: The cached LLM path broke—likely a code change affected tool behavior
4. **For eval failures (uncached)**: LLM reasoning changed—may need prompt tuning, tool docstring improvements, or acceptance of variance

---

# Shared rules

These apply across every unifyai repo. Edit them in the `unifyai/global-agent-rules` submodule, not here.

# Aggressive Refactoring & Zero Backward Compatibility

## Context
This project is a prototype in rapid development. We prioritize a clean, minimal, and correct codebase over stability, backward compatibility, or risk aversion.

## Critical Rules

### 1. Zero Backward Compatibility
- **Assume NO Backward Compatibility**: Unless explicitly requested, break APIs, data structures, and protocols freely.
- **Immediate Updates**: When changing an interface, update all call sites immediately. Do not create adapters, aliases, or optional parameters to "soften" the change.
- **Purge Old Patterns**: If you introduce a new design pattern (e.g., for state management), strictly remove all instances of the old pattern. Do not leave mixed patterns in the codebase.

### 2. Destructive vs. Additive Editing
- **Avoid "Stapling"**: Do not merely add new logic on top of old logic (additive editing). This creates bloat and "staples upon staples".
- **Rewrite and Simplify**: When requirements change, **rewrite** the affected code to optimally support the *new* requirements as if they were the original ones.
- **Delete Aggressively**: If code is no longer the "best" way to do something, delete it. Do not comment it out. Do not keep it "just in case".

### 3. No Defensive Coding
- **No Preemptive Exception Handling**: Do not wrap code in try/catch or try/except blocks to prevent crashes unless you are handling a specific, expected, and recoverable runtime error (e.g., network timeout). Fail loud and fast.
- **No Defensive Checks**: Do not add null checks or type checks unless strictly necessary for the logic. Trust the type system and the caller contract.
- **Clean Implementation**: Code should look like the "happy path". Avoid cluttering logic with defensive branches for edge cases that shouldn't happen in a correct system.

### 4. Review and Reflect
- **Simplify First**: Before adding a feature, ask: "Can I simplify the existing code to make this feature natural to implement?"
- **Remove Bloat**: actively look for and remove redundant code, unused variables, and overly complex abstractions.

# No Temporal or Chat-Specific Comments

## Context
Code comments and docstrings must be **timeless** and describe the code as it currently exists. They must **never** reflect the history of changes, the current chat session, or the fact that code is "new".

## Critical Rules

### 1. No "New" or "Updated" Markers
- **Forbidden**: Never use words like "New", "Updated", "Added", "Modified", "Refactored" in comments to mark changes.
- **Reasoning**: Code is only "new" for the moment it is written. Next week, it is old. These comments rot immediately and create noise.
- **Correct Approach**: Just write the comment describing what the code does. Git history tracks what is new.

### 2. No Chat Context
- **Forbidden**: Do not reference "the user request", "this chat", "per instruction", or specific reasoning from the current conversation.
- **Reasoning**: The codebase must stand alone. Context from a chat session is lost to future readers.
- **Correct Approach**: If a complex decision needs explanation, document the *technical why* (e.g., "Use X because Y is slow"), not the *conversational why* ("User asked for X").

### 3. Clean Documentation
- **Focus**: Comments should explain **why** tricky code exists or **how** it functions.
- **Avoid**: "Here is the implementation of..." or "Standardized composer utilities". The code itself shows it is an implementation.
- **Example**:
  - BAD: `NEW: Added this function to handle retries`
  - BAD: `Updated to support the new API`
  - GOOD: `Retries the request with exponential backoff to handle transient network errors.`

If a test is failing, we should **never** add test-specific information or shortcuts to production code as a hack to get the test passing. No details about specific test cases should ever make their way into production code—no special-case branches, no hardcoded values that match test inputs, no conditional logic that only exists to satisfy a test.

All fixes must be fully general and **much** broader than the specific failing test. We do **not** want to overfit production code to a specific set of tests.

# Git Safety

## Rule: Pull Before Editing a Repository

Before making any file edits in a repository, run `git pull --rebase` once to sync with the remote. This prevents the agent from working on a stale branch and silently overwriting others' commits.

- **Once per repo per session** is sufficient — no need to pull on every turn.
- **After a push rejection + rebase**: re-read any files you plan to edit next. The rebase changed them on disk but your in-memory copies are stale.
- **Exception**: only skip if the user explicitly asks you not to pull.

## Context: Explicit Path Commits
When multiple agents run in parallel in `local` mode, there is a race condition risk if they use the shared git index (staging area).
- Agent A: `git add fileA`
- Agent B: `git add fileB`
- Agent A: `git commit -m "msg"` -> Commits BOTH fileA and fileB!

## Rule: Explicit Path Commits
To eliminate this risk, **NEVER** run `git commit` without explicit file arguments.

### Incorrect
```bash
git add myfile.json
git commit -m "Update myfile"
```

### Correct
```bash
# For modified files:
git commit myfile.json -m "Update myfile"

# For new (untracked) files:
git add myfile.json
git commit myfile.json -m "Add myfile"
```

### Reasoning
Passing filenames to `git commit` bypasses the shared index for that specific commit operation, ensuring that Agent A only commits what it intends to, regardless of what Agent B has staged.

## Rule: Push Only When Explicitly Asked

Do **not** push commits unless the user explicitly asks you to. This includes all forms of pushing:

- `git push`
- `git push origin <branch>`
- `git push --force` or `git push -f` (especially dangerous)
- `git push --force-with-lease`

### Reasoning
Pushing affects shared remote state. The user must decide when and where to push. Force pushing is particularly dangerous as it rewrites remote history and can destroy other collaborators' work.

### What To Do
- Commit changes locally (following the explicit path commits rule above)
- Inform the user that changes are committed and ready for them to push
- If the user explicitly asks you to push, push to the **current branch** only (e.g., `git push origin HEAD`)
- **Never** force push unless the user explicitly requests it and understands the consequences

## Rule: Staging-First Promotion

For repositories with a `staging` branch, **never open, retarget, auto-merge, or merge a feature/fix branch directly into `main` or `master`** unless the user explicitly says to bypass staging.

Required flow:

1. Land the change into `staging` first.
2. Let the staging deployment/validation run.
3. Promote with a `staging` -> `main`/`master` PR or merge.

Before any PR creation, PR retarget, merge, or auto-merge command, verify and state the base/head:

```bash
gh pr view <number> --json baseRefName,headRefName
```

Allowed:

- `feature-or-fix-branch` -> `staging`
- `staging` -> `main`

Forbidden unless explicitly approved by the user as a staging bypass:

- `feature-or-fix-branch` -> `main`
- `feature-or-fix-branch` -> `master`
- enabling auto-merge on either forbidden pattern

If a PR is already targeting `main`/`master` from a non-`staging` branch, stop before merging, disable auto-merge if it is enabled, and retarget/recreate the PR against `staging`.

### Exception: `global-agent-rules` and `branding` have no `staging`

Both repos retired the branch — `global-agent-rules` in `e9d3e9c` ("Drop the
branch that was standing in for a check"), `branding` in August 2026. For each,
`main` is the default and only branch, and is unprotected. Commit edits
directly to `main`; staging-first does not apply, and neither does the
`magic-marty` approval step, which exists to satisfy branch protection that
these repos do not have.

They are content repos consumed as submodules, not deployed services. A staging
branch buys nothing when there is no environment to stage into: it only adds a
promotion hop between writing a rule (or a brand spec) and the consuming repos
being able to pin it.

A stale clone still shows `origin/staging`, because a plain `git fetch` does
not remove remote-tracking refs for deleted branches — and pushing that branch
**recreates it on the remote** from stale content, while the usual "am I in
sync?" check compares against the dead ref and cheerfully reports `0 0`. Run
`git fetch --prune` (or `git ls-remote --heads origin`) before trusting any
branch state there.

After changing a rule, every consuming repo needs its submodule pointer bumped
and `AGENTS.md` regenerated with
`python3 .agents/global-rules/build_agents_md.py` — a pre-commit hook enforces
freshness. The same applies to a `branding` edit: the consuming repo picks it
up only when its `branding` pointer moves.

## Rule: Agent PR Approval (`magic-marty`)

`unifyai/*` repos enforce branch protection: every PR to `main` or `staging`
requires at least one approving review from a Unify engineer **other than the
author** (SOC 2 CC8.1 separation of duties). Detect the active author first:

```bash
gh api user --jq .login
# or: gh auth status
```

When an agent authors a PR under that account, the approving review cannot come
from the same account.

The local `gh` CLI has two authenticated accounts:

| Account | Role |
|---|---|
| `<author>` (from `gh api user --jq .login`) | Author — create PR, enable auto-merge, merge after approval |
| `magic-marty` | Reviewer — approve only; satisfies the non-author review requirement |

`magic-marty` is a GitHub service account (Security Lead accountable). Use it
only for the approval step, not for authoring commits or opening PRs.

### The org's machine accounts are not interchangeable

`unifyai` has two, with deliberately different jobs. They are not a fallback
for one another, and the names do not say which is which.

| Account | Job | Access |
|---|---|---|
| `magic-marty` | **Approves PRs.** Nothing else. | Admin on the private repos |
| `unify-dev-bot` | **CI automation** — clones private dependencies, dispatches cross-repo workflows, commits dependency bumps. Owns the `CLONE_TOKEN` secret. | Read on `brain`/`branding`, write where it must push |

Never move CI credentials onto `magic-marty`. Its whole value is being a
different principal from whoever authored the change — that is the separation
of duties the branch protection exists to enforce. `CLONE_TOKEN` is shared by
`unify`, `unillm` and `unify-deploy`, so putting it on an account that holds
admin and can approve releases would mean one leaked CI secret could approve
its own merge into `main`.

Give `unify-dev-bot` the least access its job needs, and no more: an
automation credential that can edit rulesets can switch off the release gates.
It was dropped from admin to write on `unify-deploy` on 2026-08-14 for exactly
that reason, after a check found nothing requiring admin — a repository
dispatch, its only cross-repo write, needs write.

`CLONE_TOKEN` is a classic PAT and has expired at least once, silently taking
out private-dependency clones across three repos and self-host image
publishing for ten days with nothing reporting it. If private clones start
failing with `Repository not found` on a repo that plainly exists, suspect the
token before the repo: GitHub answers 404 rather than 403 when a credential
cannot see a private repository, so an expired, unauthorised, or
wrong-account token looks exactly like a missing one.

### `unify-dev-bot`'s credential already exists — find it before minting one

The bot's PAT lives in **GCP Secret Manager as `DEVBOT_GITHUB_TOKEN`**: a
classic PAT, `repo` scope, no expiry. Look there before hunting for the bot's
*password*. On 2026-08-14 an engineer spent an afternoon on that hunt and
minted a redundant second PAT while a working token sat in Secret Manager the
whole time.

**Read it from the right project.** The same secret name exists in three
projects and one of them is dead:

| Project | State |
|---|---|
| `saas-368716` | **Live — the authoritative copy** |
| `unity-assistant-vms` | Live, byte-identical to the above |
| `responsive-city-458413-a2` | **Dead — every version disabled** |

Fetching the dead copy fails, and a script that does not check the exit status
carries an empty string into `git clone` — which GitHub answers with
`Repository not found`, the same 404-instead-of-403 as above. An empty secret
and a missing repo are indistinguishable from the error alone. Confirm the
project holds an enabled version before concluding a credential is broken:

```bash
gcloud secrets versions list DEVBOT_GITHUB_TOKEN --project=saas-368716
```

**PATs are the old approach here — do not mint more.** GitHub exposes no API
for creating a personal access token, so rotating one means signing into the
web UI *as the account that owns it*. A token owned by a bot therefore needs
the bot's password and 2FA; a token owned by a person makes that person the
only one who can rotate it. Neither is acceptable, and sharing a login is not
the way out of it — that is the practice being retired, not the fix.

**A GitHub App is the direction.** An App has no login, no password and no
2FA: it signs a JWT with a private key and exchanges it for an installation
token that expires in an hour, scoped to chosen repositories and permissions.
Rotation becomes a key swap that no one has to do at a browser, and a stolen
token is worthless by the end of the hour.

The self-hosted CI runner moves first, because its credential is a *personal*
PAT expiring 2026-09-23; `CLONE_TOKEN`'s six consuming repos follow. Until
then the tokens above are simply what exists — record them, and add none.

### When this applies

- Any `staging` → `main` release PR the agent opens under the active author account
- Any feature/fix PR to `staging` or `main` the agent authored under the active author account
- After `./staging_to_main.sh` or `gh pr merge --auto` — auto-merge stays
  blocked at `REVIEW_REQUIRED` until `magic-marty` approves

### Standard release flow (`staging` → `main`)

```bash
AUTHOR=$(gh api user --jq .login)

# 1. Author — create PR and queue auto-merge
gh pr create --base main --head staging \
  --title "Release: staging → main" \
  --body "Release PR from staging to main."
gh pr merge <number> --auto --merge

# 2. Approve as magic-marty, with a token scoped to this one command
GH_TOKEN=$(gh auth token --user magic-marty) \
  gh pr review <number> --repo unifyai/<repo> --approve \
    -b "Release approval: staging CI green."

# 3. Merge as author — auth was never switched, so this is already the author
gh pr view <number> --repo unifyai/<repo> \
  --json mergeStateStatus,reviewDecision
```

**Scope the token; do not `gh auth switch`.** The switch is global machine
state, so while it is active any *other* session on the machine authors under
`magic-marty` — and several sessions routinely run in parallel across
worktrees. A `GH_TOKEN=...` prefix applies to the single command and cannot
leak into anyone else's work, and it removes the "always switch back" step
that is the failure mode when a run dies midway.

If auto-merge does not fire, merge explicitly as the author:

```bash
gh pr merge <number> --repo unifyai/<repo> --merge
```

### Invariants

- **Never self-approve.** The author account must not `gh pr review --approve` on a PR
  it authored.
- **Scope the reviewer token to the approval command** rather than switching
  auth globally (see above). Nothing then needs switching back.
- **Approvals are perishable.** `dismiss_stale_reviews_on_push` is on
  estate-wide, so any commit landing after an approval silently dismisses it.
  A PR then sits green-but-`BLOCKED`, which looks exactly like a slow check.
  On a branch several sessions push to, wait for the head to be quiet before
  approving at all, and re-read `reviewDecision` rather than trusting that an
  earlier approval still stands.
- **Verify base/head** before approving or merging (see Staging-First Promotion).
- **Approval is `magic-marty`; merge is the author.** If `magic-marty` cannot
  merge (e.g. unverified email), that is expected — only the approval must
  come from `magic-marty`.

### Batch promotions across repos

For "merge staging into main in each repo" requests, repeat per repo in order:

1. Author account — create PR + `--auto --merge`
2. `magic-marty` — `--approve` on each open PR
3. Author account — confirm `reviewDecision=APPROVED`, then let auto-merge complete

## Rule: Cross-Repo Push Semantics

When the user says "commit and push to all repos", "push across repos", or similar, interpret that as:

- For each repo that has a `staging` branch locally or on origin: commit directly to `staging` and push `staging`.
- For each repo without a `staging` branch: commit directly to `main` or `master` and push that branch.
- Do not create feature branches or PRs unless the user explicitly asks for a PR workflow.
- Do not merge a non-`staging` branch into `main`/`master` as part of a cross-repo push.

Before committing or pushing in each repo, verify its integration branch:

```bash
git branch --list staging main master
git branch -r --list origin/staging origin/main origin/master
```

If `staging` exists, use it. If it does not exist, use `main`/`master`.

# Worktree Mode: Direct Commits, No Feature Branches

## Context
When running in **worktree mode**, the mental model is fundamentally different from traditional feature development:

- **Worktrees are for small-scale parallel fixes**, not large-scale feature development
- **Multiple agents on the same branch** = multiple collaborators working in parallel, each with their own local working directory
- The overhead of `feature branch → PR → merge → cleanup` is **overkill** for this workflow

## Critical Rules

### 1. NEVER Create Feature Branches
When asked to make changes, commit, or push:
- **DO NOT** create a new branch (e.g., `git checkout -b feature/...`)
- **DO NOT** suggest creating a branch for the work
- **COMMIT DIRECTLY** to whatever branch is currently checked out

The worktree already provides isolation. Creating additional branches defeats the purpose.

### 2. NEVER Create Pull Requests
- **DO NOT** use `gh pr create` or suggest creating a PR
- **DO NOT** push to a new remote branch with the intent of opening a PR
- If the user wants changes merged, they will handle the merge strategy themselves

### 3. The Correct Workflow
```bash
# 1. Make your changes to files

# 2. Commit directly to the current branch (following git-commit-safety rules)
git commit <specific-files> -m "Description of change"

# 3. Only if the user explicitly asks you to push, push to the CURRENT branch
git push origin HEAD
```

### 4. Mental Model
Think of worktree agents as **multiple developers pair-programming on the same branch**:
- Each has their own local checkout (the worktree)
- All commit to the same branch
- No one creates personal feature branches for small fixes
- Coordination happens through communication, not branch isolation

## Why This Matters
The alternative workflow creates significant noise:
1. **Stale branches accumulate** - agents create branches, users forget to delete them
2. **PR overhead** - reviewing, merging, and closing PRs for trivial fixes wastes time
3. **Context switching** - users must mentally track multiple branches for what should be one stream of work
4. **Merge conflicts** - more branches = more opportunities for conflicts

## Exception
If the user **explicitly asks** for a feature branch or PR workflow, follow their instructions. But **never default to this behavior** in worktree mode.

# Git History for Context

## Context
This rule applies when you are trying to understand the *rationale* behind specific code blocks, the evolution of a module, or when deciding whether "weird" looking code is essential or legacy technical debt.

## Rules

### 1. Strategic Git Usage
- **Use as a Second Level of Analysis**: If the code's purpose isn't clear from the current state alone (static analysis), use `git blame` or `git log` to uncover the "why".
- **Not a Mandate**: Do not check git history for every file you touch. This creates noise. Use it selectively when you lack context.

### 2. Understanding Code Evolution
- **Identify Legacy Code**: If you suspect code is redundant or outdated, check its commit date and message. If it was added months ago for a feature that is no longer relevant, this confirms it can likely be purged.
- **Find the "Why"**: Expressive commit messages often contain the reasoning that comments lack. Use them to understand the author's original intent before refactoring or deleting complex logic.

### 3. Targeted Queries
- **Be Surgical**: When querying git, look for the history of specific lines or changes (e.g., `git blame -L n,m filename` or `git log -p filename`) rather than dumping the entire history into the context.
- **Synthesize**: Use the information to form a narrative about the code's lifecycle (e.g., "This was added in commit X to fix bug Y, but since we rewrote the bug Y subsystem, this is now dead code").

### 4. Investigating Regressions with Git Diff

When debugging test failures or regressions, git history can pinpoint exactly what changed.

**When the user proactively provides context:**
If the user says something like "the test was passing at commit `<hash>`, and the relevant changes are in `<path>`", use this optimally:
- Run `git log --oneline <hash>..HEAD -- <path>` to see which commits touched the area
- Run `git diff <hash>..HEAD -- <path>` to get the **aggregate diff** (not serial diffs commit-by-commit)
- Cross-reference the diff with commit messages to understand developer intent
- The overall diff is mathematically equivalent to composing serial diffs, but far more token-efficient and cognitively cleaner

**When debugging hits a roadblock:**
If direct code analysis and debug logging (`CURSOR_DEBUG_LOG`) aren't yielding answers, *then* ask the user:
- "Do you know when this test was last passing? If you have a commit hash and know which files/folders are likely involved, that would help narrow down what changed."
- Don't front-load this question—often the user doesn't know the answer. Try direct debugging first.

**Avoid wasteful patterns:**
- Don't ask the user to provide diffs—ask for the commit hash and run git commands yourself
- Don't read diffs commit-by-commit and mentally compose them; use the aggregate diff
- Don't dump entire file histories; scope queries to the relevant path(s)

# Staging→Main Release Gates Are Fail-Closed

Every repo's `Staging->Main` ruleset requires at least one status check whose
job makes an expensive or conditional run (full pytest matrix, paid LLM smoke
tests, E2E). Those jobs don't run on every push — they're gated on a
`[run-tests]`/`[run-flows]`-style commit-message tag, or on the PR event
itself. As of **2026-07-31**, the required context for that job is published
**unconditionally** on every push: an explicit pass or fail, never an
implicit pass from a skip.

## Why: the orchestra incident

Before 2026-07-31, the required context was only published when the gated
job actually ran; an ordinary push that skipped it published nothing, and
**GitHub counts a skipped required check as satisfied**. A staging→main
release PR shares its head SHA with whatever was last pushed to staging, so
that stale implicit pass could satisfy branch protection before the real
PR-triggered run finished. In orchestra this let four release PRs (#125,
#127, #128, #129) merge into main carrying a skipped/failing suite — #125
merged 83 seconds into an 11-minute test run that later came back failing,
leaving a broken test on main for thirteen hours.

The fix — "make the gate fail closed" — makes an aggregator job republish the
required context unconditionally, so a push that didn't run the suite now
reports that context **red**, not green-by-default.

Rolled out the same week to: orchestra (`pytest`), unillm (`pytest`), unify
(`Flow smoke`), unify-deploy (`Integration smoke`), console (`Push Gate`).
Check name and trigger tag differ per repo; the fail-closed shape is the
same. unisdk, brain, docs, and landing-page have no equivalent
expensive/conditional gate, so this doesn't currently apply there — but
treat it as the default shape for any new staging→main required check in
any repo.

## The `if:`-scoping approach does NOT work — verified 2026-08-03

An earlier version of this rule described scoping the aggregator job's own
`if:` to `pull_request`/`workflow_dispatch` (unify's `05fbdcce9`) as the
fix for the fail-closed-on-push problem below. **That is wrong, and left
orchestra and unify-deploy releases merging on a stale pass again**, plus
unify itself carrying an unnoticed stale pass on its own staging HEAD.

The reasoning behind that approach assumed an `if:`-scoped job produces *no
check run at all* when its condition is false. It does not: **GitHub Actions
still publishes a "skipped" check run for a job whose `if:` evaluates to
false**, even when that job is the required aggregator itself, and a skipped
required check is satisfied exactly like a pass. Scoping only the aggregator
job's `if:` — while the *workflow file* it lives in still triggers on
`push` — just changes what an ordinary push publishes from an explicit
failure back to a skip, which is the original 2026-07-31 bug, verbatim.

## The real fix: no `push` trigger on the gate's workflow file at all

The only way to guarantee an ordinary push publishes **nothing** under the
required context name is for the *workflow file* that defines the
aggregator to never be triggered by `push` in the first place. A workflow
with no `push` in its `on:` block simply never runs on a push event, so
none of its jobs — passing, failing, or skipped — produce a check run for
that SHA. The context sits genuinely `pending` until the real
`pull_request`-triggered run supplies an answer.

Concretely: split the gate into two workflow files.

- **The everyday/ad-hoc workflow** (existing file) keeps `push` (gated by
  the `[run-tests]`/`[run-flows]`/`[run-integration]`-style tag) and
  `workflow_dispatch`, for ordinary developer feedback. It must **not**
  define the aggregator job, and its own expensive test job must not share
  a job id/name with the required context (see the collision gotcha below).
- **A new, dedicated workflow file** (e.g. `pytest-release-gate.yml`,
  `flow-smoke-release-gate.yml`, `integration-smoke-release-gate.yml`)
  triggers **only** on `pull_request: branches: [main]` and
  `workflow_dispatch`. It contains its own copy of the expensive test job
  plus the aggregator job that publishes the required context. Because this
  file has no `push` trigger, an ordinary push can never populate that
  context under any name defined in it.

Applied 2026-08-03 to orchestra (`pytest-release-gate.yml`), unify
(`flow-smoke-release-gate.yml`), unillm (`pytest-release-gate.yml`), and
unify-deploy (`integration-smoke-release-gate.yml`). console's `Push Gate`
was checked and found **not** vulnerable to this bug: its push-triggered run
genuinely executes the full suite unconditionally (no tag-gating), so there
is no skip to exploit — no fix needed there.

## Gotcha: job-id collision reintroduces the same bug

After removing the explicit aggregator from the everyday workflow, check
that its own expensive test job doesn't accidentally publish the exact same
context name. In orchestra, the plain development `pytest:` matrix job had
no explicit `name:`, so its check run defaulted to its job id — which
happened to be the literal string `pytest`, the required context. A job
whose matrix never expands (because its own `if:` evaluated false) still
publishes **one** check run under that bare job id, not per-shard, so this
silently recreated the stale-skip bug via a different job. Fix: give the
everyday job a distinct job id and/or explicit `name:` (orchestra renamed it
to `pytest-dev`). Always verify this when doing this split — confirm the
everyday workflow's job names don't coincide with any required-check context
string, matrix or not.

## Gotcha: don't write the trigger tag in prose in a commit message

The `push` gating check on these jobs is a naive substring match against the
full commit message (subject + body) for `[run-tests]` / `[run-flows]` /
`[run-integration]`. Writing that literal bracketed tag *anywhere* in a
commit message — including inside a sentence explaining how the tag
mechanism works — re-triggers the real (sometimes paid, sometimes
live-infra) job on that push. This happened twice while rolling out the fix
above: a commit message explaining unify's ad-hoc `[run-flows]` trigger
accidentally fired a real paid Flow Smoke run (caught and cancelled before
much cost), and a commit message explaining unify-deploy's
`[run-integration]` trigger fired a real live Integration Smoke run against
staging infra that ran to completion (~18 minutes) before failing on an
unrelated live-infra teardown timeout — cleanup steps still ran, so no
resource leak, but wasted CI/infra time. When a commit touches one of these
workflow files and needs to describe its trigger tag, break up the literal
bracket sequence (e.g. quote it without brackets) so the substring match
cannot fire.

## What to do when a release PR is stuck or merges on a stale pass

**Stuck/blocked** — `reviewDecision: APPROVED`, `mergeable: MERGEABLE`,
`mergeStateStatus: BLOCKED`, and the required context shows both a
`FAILURE` and a `SUCCESS` entry for the same PR head SHA from two different
workflow runs (one push-triggered, one pull_request-triggered): this is the
pre-`if:`-scoping failure mode. Push a small commit carrying the trigger tag
to give the release PR a fresh head with a single, unambiguous run, and
apply the real fix above so it stops recurring.

**Merges anyway on a stale pass** — the required context shows `SKIPPED` on
a push-triggered run for the PR's head SHA, and the PR merges (or already
merged) before the real `pull_request`-triggered run finished: this is the
`if:`-scoping-is-insufficient failure mode described above. Apply the real
fix (dedicated no-`push` workflow file) — do not consider the job's `if:`
condition alone a fix.

In both cases: do not force-merge, disable the ruleset, or bypass the check
to route around this — the real fix satisfies the gate on its own terms.

## A gate that tests live infra races its own deploy

Where the gate exercises a deployed stack rather than the checkout, pushing a
fix does **not** mean the fix is under test. The `pull_request` run starts in
about ninety seconds; the Cloud Build deploy that ships the change to staging
takes minutes. Three unify-deploy gate runs on 2026-08-15/16 tested an image
that predated the commit under test, twice sending the investigation after
phantom regressions in a diff that was never running.

Before reading a live-infra gate result as a verdict on the change, confirm
the deploy landed first — the PR's own checks carry it (for unify-deploy,
`unity-deploy-staging (responsive-city-458413-a2)`). A red gate whose deploy
finished *after* the run started is not evidence about the change.

## `repository_dispatch` runs the DEFAULT branch's workflow

Not the ref in the payload, and not the branch that sent it. A dispatch-
triggered workflow therefore keeps executing `main`'s copy of itself no matter
what staging says, so a fix to one does nothing until it is promoted — and the
breakage is entirely invisible from staging, where the push-triggered path
passes. unify's self-host image publish failed on every dispatch for a day
this way while staging looked green.

When a dispatch-triggered workflow misbehaves, read the *default branch's*
copy of it, and treat promotion as part of the fix rather than a follow-up.

## Editing a ruleset: `PUT`, not `PATCH`

`gh api -X PATCH repos/{o}/{r}/rulesets/{id}` returns a bare `404` that reads
exactly like a permissions problem, and reproduces under a second account with
`admin:org` — which is what makes it convincing. Use `PUT` with the full
object (name, target, enforcement, bypass_actors, conditions, rules).

# Python Formatting & Pre-commit

Every first-party Python repo (`orchestra`, `unify`, `unisdk`, `unillm`,
`unify-deploy`, `docs`) enforces formatting with **black** (plus
`isort`/`autoflake` where configured), and CI rejects unformatted code. A
missing local hook or a drifting Black target/Python version is the single
most common avoidable CI failure. This rule keeps local and CI identical so
it stops blocking us — for Cursor, Claude Code, Codex, and humans alike.

## Single source of truth: the locked `lint` group

The formatters are ordinary, locked dependencies — not a version hardcoded in
the pre-commit hook or in CI YAML.

- Each repo declares its formatters in a dedicated **`lint` dependency
  group**, pinned and committed to `uv.lock`. Every first-party Python repo
  is uv-managed with a repo-local `.venv` — there are no poetry repos.
  The `dev` group includes `lint` so a normal sync gives developers everything:
  `[dependency-groups]` → `lint = ["black==X", "isort>=…", "autoflake>=…"]`,
  and `dev = [ …, {include-group = "lint"} ]`.
- **Both** the pre-commit hook and CI run that **same locked** tool via uv —
  never a separate pin:
  - pre-commit hook: `entry: uv run black`, `language: system`
    (no `additional_dependencies`).
  - CI: `uv sync --only-group lint --no-install-project --frozen` then
    `uv run --no-sync black --check .` on **Python 3.12**.
- Pin Black's language target in every Python repo so local Mac Pythons and
  CI 3.12 cannot disagree (Black 26+ defaults toward newer targets):

```toml
[tool.black]
target-version = ["py312"]
```

- Never introduce a second black version or a parallel invocation anywhere
  (CI YAML, Dockerfiles, docs, ad-hoc `pip install black`, hook
  `additional_dependencies`). The locked `lint` group is authoritative; this
  rule deliberately does not restate the number — look it up in the repo's
  `pyproject.toml` / lockfile so guidance can never drift from reality.
- Keep the version **in lockstep across all the Python repos**: a bump is one
  coordinated change per repo (the `lint` pin + lockfile) applied to every
  repo so they don't diverge.

## Committed hooks (required once per clone / worktree)

`.git/hooks/` is a per-checkout artifact. Fresh clones, Cursor/Codex/Claude
worktrees, and cloud agents start with **no** hooks, which is why unformatted
code reaches CI.

Each Python repo commits `.githooks/pre-commit`. Enable it with the shared
helper (idempotent, tool-agnostic):

```bash
python3 .agents/global-rules/ensure_git_hooks.py
```

That sets local `core.hooksPath=.githooks`. Do this before the first commit
in any new clone or worktree. Coding agents (Cursor, Claude Code, Codex)
MUST run it at session start when working in a checkout that has
`.pre-commit-config.yaml`.

`pre-commit install` alone is no longer enough — it writes into `.git/hooks/`,
which worktrees and new clones miss. Prefer `ensure_git_hooks.py`.

## Before you commit (required)

1. Ensure committed hooks are wired (above).
2. Let the hook run on `git commit`, or run it explicitly on what you changed:

```bash
pre-commit run --files <changed-files>   # or: pre-commit run --all-files
```

3. Never bypass hooks: do not use `git commit -n` / `--no-verify`.

## Formatting across multiple repos

When juggling several repos, do not invoke a globally-installed `black` —
versions drift between machines and repos and produce diffs CI rejects.
Always format through the repo's pinned tooling, which uses that repo's
locked version:

```bash
pre-commit run black --all-files          # or: uv run black .
```

## Release gates

`black` is a required status check on `staging → main` (ruleset and/or branch
protection) for the Python repos. Direct pushes to `staging` stay open for
the worktree workflow — the committed git hook is the staging-side gate.
CI still runs `black` on every push so failures are visible immediately.

## Why this matters

Without committed `.githooks` + `ensure_git_hooks.py`, the first place
formatting is checked is CI — which then burns agent turns on mundane
reformats. Pinning Black's target and CI Python to 3.12 removes the
"works on my Mac, fails in Actions" class of failures.

This rule standardizes how we add temporary debug logging during failing tests and how we clean it up afterwards. ALWAYS use this process in agent sessions WHENEVER A TEST FAILS. This is the ONLY permitted way to address failing tests.

1) Always start with hardcoded, unconditional debug logs
- Add logs immediately, without flags or guards. Do not gate behind environment variables or configuration.
- Use **only** the `CURSOR_DEBUG_LOG` function. No other logging method is permitted.
- **Finding the function**: Search for it with `rg "CURSOR_DEBUG_LOG"` to locate the utility in your project, then import and use it.
  - **Python**: `from <module> import CURSOR_DEBUG_LOG` then `CURSOR_DEBUG_LOG("message", variable)`
  - **JavaScript/TypeScript**: `import { CURSOR_DEBUG_LOG } from "<module>"` then `CURSOR_DEBUG_LOG("message", { variable })`
- Behavior: Prints an entry to stderr/stdout, making it easy to correlate with test runs.

2) Python-specific import discipline
- **Self-contained imports**: Each debug snippet must include ALL its own imports inline (e.g., `import json as _json; import os as _os;`). Never rely on the file's existing imports.
- **Prefixed names**: Use underscore-prefixed aliases (`_json`, `_os`, `_pid`) to avoid shadowing.
- **Region markers**: Wrap in `# #region agent log` / `# #endregion` for easy identification and removal.
- This prevents `NameError` crashes when debug snippets reference modules that aren't imported at that location.

3) Investigation workflow
- Step A: Add targeted debug calls around suspected code paths.
- Step B: Re-run the failing test(s) and inspect the new logs.
- Step C: If you are not 100% certain of the root cause, add more debug entries and repeat.
- Step D: Only when you are 100% confident of the cause, implement a direct fix (with or without keeping some logs briefly for confirmation).
- Step E: The user may repeatedly re-run tests and paste logs; continue iterating until the issue is definitively fixed.

4) Cleanup policy
- After the fix is confirmed, remove all temporary logging.
- Grep to find every occurrence:
  - ripgrep: `rg -n "CURSOR_DEBUG_LOG" -S`
  - grep:    `grep -Rin "CURSOR_DEBUG_LOG" .`
- Delete each call site (and any now-unused imports) before finalizing the fix.

5) Alignment with workspace rules
- No fast paths or heuristics: Logging should not add conditional shortcuts; it merely reports state unconditionally.
- No exception-handling shields: Do not add defensive exception handling (try/catch, try/except) around the logs. Keep failures visible.
- No test details in production prompts: Temporary logs must not leak test-specific information into production prompts or docstrings.
- Rapid evolution: This logging is temporary by design; remove it once the issue is resolved—do not preserve backward compatibility.

6) Intent and scope
- This logging exists solely for interactive debugging in agent sessions.
- The function name `CURSOR_DEBUG_LOG` is intentionally unique and grep-friendly to ensure quick cleanup on request.

# Full Local Stack First

This is internal agent/developer guidance. It intentionally differs from the
public `unify` README: the README is for open-source users who run `unify`
against hosted Orchestra and do not have the private `unify-deploy` repo. For
internal cross-repo work, default to the private full local stack in
`unify-deploy/selfhost/`.

Before starting, stopping, rebuilding, or repairing any local service, assume the
full stack may already be running and inspect it first:

```bash
bash ~/unify-deploy/selfhost/stack.sh status
```

Default commands:

- Start full source stack from an agent session: `bash ~/unify-deploy/selfhost/stack.sh up --durable`
- Start full source stack from a long-lived human terminal: `bash ~/unify-deploy/selfhost/stack.sh up`
- Repair only Console: `bash ~/unify-deploy/selfhost/stack.sh repair-console`
- Smoke test: `bash ~/unify-deploy/selfhost/stack.sh smoke`

When a A coding agent is asked to deploy or start the full local source stack for
the user, always use `up --durable`. Do not keep the stack alive with `nohup`, a
backgrounded shell, or a sleep loop inside a agent shell job. The durable
launcher owns the `unity-stack` tmux session, waits for readiness, and verifies
`http://localhost:3000/`.

Do not run isolated commands against a live stack unless the user explicitly asks
for isolated mode: `npm run dev`, `next dev`, `next build`, `npm run ci`,
`console/scripts/local.sh start`, `orchestra/scripts/local.sh start`, or
`unify/scripts/local.sh start`.

If isolated mode is explicitly requested, use the relevant override env var and
explain which full-stack guarantees are being bypassed.

## Isolated Orchestra and Cursor process reaping (macOS)

Prefer the durable full stack above. When isolated Orchestra is required
(`orchestra/scripts/local.sh start`, or `parallel_run.sh` bringing Orchestra up):

- **macOS has no `setsid`.** `local.sh` falls back to `bash -c … &` + `disown`.
  `disown` does **not** create a new process group — Orchestra stays in the
  agent shell’s PGID. When that shell’s process group is torn down, Orchestra
  dies and tests mid-flight see `Connection refused` on `127.0.0.1:8000`.
- **Never** treat a one-shot agent shell `local.sh start` as durable. Start
  Orchestra in a long-lived human terminal, or hold a watcher shell on the
  server pid for the whole session. `parallel_run` only keeps Orchestra alive
  for as long as *that* runner’s process group survives — and only if it
  started Orchestra itself rather than reusing an instance from a short-lived
  prior shell.
- Mid-run `Connection refused` to `:8000` is infra death until proven otherwise.
  Check `orchestra/scripts/local.sh status` / `curl` before diagnosing product
  or LLM failures.
- `local.sh stop` ends with `pkill -9 -f -- "-m orchestra"`, which kills the
  **shared** instance for every agent on the machine. Do not stop/restart
  Orchestra from one agent session while another’s tests are using it.

# Local Stack Logs: Where To Look First

When the user says something like *"we just did a local deployment and X happened,
check the logs to investigate"*, the logs almost always already exist on disk. Do
**not** re-explore the filesystem from scratch — go straight to the locations below.

## 1. Central source of truth: `$UNIFY_REPO_PATH/logs/`

Default `~/unify/logs/`. A local deployment aggregates **every** repo's logs
here — including Orchestra, which runs as a separate process.
The exact paths are set in `unify-deploy/selfhost/self_host_env.sh` (search
`*_LOG_DIR`); confirm the live values with `stack.sh status`.

| Dir | Env var | Contents |
|---|---|---|
| `logs/unillm/` | `UNILLM_LOG_DIR` | Raw LLM request/response, one `.txt` per call — system/user prompts, tool args, `reasoning_content`, model. This is **"what the model actually produced"**. |
| `logs/unisdk/` | `UNISDK_LOG_DIR` | UniSDK ↔ Orchestra HTTP traces (JSON per request). |
| `logs/orchestra/` | `ORCHESTRA_LOG_DIR` | Orchestra server-side per-request traces. |
| `logs/unify/` | `UNIFY_LOG_DIR` | Unify runtime file logs. |
| `logs/all/` | `*_OTEL_LOG_DIR` | **Combined cross-repo OTel traces** — one `{trace_id}.jsonl` per request, with unify + unisdk + unillm (+ orchestra) spans stacked together. Use this for the end-to-end story of a single request. |
| `logs/pytest/`, `logs/ci/` | — | Test runs / downloaded-CI logs. |

Deep reference (formats, env vars, examples): `<unify>/logs/README.md` (also present
in `unisdk/logs/README.md` and `unillm/logs/README.md`).

## 2. CRITICAL: these dirs are gitignored AND cursorignored

`unify/.gitignore` has `logs/*`; `unify/.cursorignore` has `logs/` and `logs/**`.
Consequence — this is the usual reason an agent "can't find the logs":

- The built-in **Read / Grep / Glob tools return nothing**, "permission denied", or
  "filtered out by .cursorignore" for anything under `logs/`.
- Plain `rg` / `grep` also **skip** these dirs (they respect `.gitignore`).

Always inspect log dirs via the **Shell tool with ignore-bypass**, and read
individual files through the shell (not the Read tool):

```bash
rg -uu -n "pattern" ~/unify/logs/unillm   # -uu = --no-ignore --hidden
rg -uu -n . ~/unify/logs/all/<trace_id>.jsonl
```

## 3. Operational logs that live OUTSIDE the `logs/` tree

- `~/.unity/service.log` — the self-host **stack supervisor**: startup, Orchestra
  boot, gateway restarts, and the CM's own log path. Location is printed by
  `stack.sh status`. (Note: may contain a one-off DB dump near a reset.)
- `/tmp/unity-local.log` — the **ConversationManager event "story"**: notifications,
  guide/speak decisions, tool calls — the human-readable narrative of a live
  conversation. Best first read for "what happened in this chat/call".
- `~/.unity/comms-bridge.log` — inbound email / SMS / WhatsApp polling.
- `~/.unity/call-tunnel.log` — cloudflared tunnel used for local phone/WhatsApp
  call webhooks. LiveKit media itself is in LiveKit Cloud for source-stack runs.

## 4. Ground truth that is NOT a file: Orchestra `Transcripts` context

What was actually **spoken / sent / received** on calls and channels lives in
Orchestra's Postgres, not in a file log. When a file log shows the *intended* text
but you need the *downstream reality* (e.g. the TTS rendering vs. the LLM text),
query the `Transcripts` context (and `Contacts`, `Tasks`, …) via the UniSDK logs API
or Console:

```bash
curl -s --get "http://127.0.0.1:8000/v0/logs" \
  --data-urlencode "project_name=Assistants" \
  --data-urlencode "context=<userId>/<assistantId>/Transcripts" \
  --data-urlencode "limit=200" \
  -H "Authorization: Bearer $KEY"
```

Local keys: the coordinator/owner API key is in `~/.unity/coordinator-runtime.json`;
`userId` is in `~/.unity/self-host-owner.json`. `unify_meet` rows are call
utterances (`sender_id` identifies the speaker).

## Quick start

1. `bash ~/unify-deploy/selfhost/stack.sh status` — running services + log paths.
2. `rg -uu` into `~/unify/logs/{all,unillm,unisdk,orchestra,unify}` for the request.
3. For one end-to-end request, open the matching `logs/all/{trace_id}.jsonl`.
4. For the conversation narrative, read `/tmp/unity-local.log`.
5. For what was truly spoken/received, query the Orchestra `Transcripts` context.

# Shared agent conversation archive

Unify keeps a private repo of **raw** agent transcripts at **`~/shared_context`**
(GitHub: `unifyai/shared_context`), keyed by **GitHub login** (e.g. `djl11`).

## Design (important)

- **Adjacent clone, not a submodule.** `shared_context` sits next to product
  checkouts (`~/unify`, `~/orchestra`, `~/brain`, …). It is **not** nested under
  any public or private product repo.
- **Why:** `unify` (and other open repos) stay public; transcript data stays
  private. Public cloners never need or see this tree. One clone serves agents
  in **every** eng repo that pulls `unifyai/global-agent-rules`.
- **Applies everywhere** this rule is loaded: `unify`, `orchestra`, `unisdk`,
  `unillm`, `unify-deploy`, `console`, `brain`, `docs`, `landing-page`, and any
  other repo that includes these global rules.

## When to load this

Use before answering questions about past investigations or decisions across the
team — e.g. "did we set up X?", "who changed Y?", "why did we do Z?" — when the
answer might live in someone else's Cursor / Claude Code / Codex session, not
only the current chat.

## How to search

Prefer ripgrep over reading whole files. Search **tracked** login trees only —
**do not** search `yours/` unless the user explicitly asks about their local /
unexported chats:

```bash
rg -n -i "keyword" ~/shared_context/derived/index.jsonl
rg -n -i "keyword" ~/shared_context -g '!yours/**' -g '!tools/**' -g '!.git/**'
```

`derived/index.jsonl` is rebuilt locally by `tools/sync.sh` / `tools/export.py`
after pull (gitignored). If it is missing, search tracked trees directly or run:

```bash
python3 ~/shared_context/tools/export.py --index-only
```

Sessions live at
`<github_login>/{cursor|codex|claude-code}/<yyyy-mm>/<id>/{meta.json,transcript.jsonl}`.

`yours/{cursor,codex,claude-code}` are local symlinks to personal stores and are
gitignored.

If `~/shared_context` is missing, say so and suggest:

```bash
git clone git@github.com:unifyai/shared_context.git ~/shared_context
```

Do **not** suggest `git submodule add` / nesting it under a product repo.

## Citing

Cite **user**, **tool**, **date**, and **path** so a human can open the same session.

## Do not

- Do not confuse this with `brain` (curated company memory).
- Do not scrub or rewrite historical transcripts.
- Do not push/sync unless the user asked you to.
- Do not grep `yours/` unless the user asked for local-only context.

# OpenAI is reached only through OpenRouter

Every OpenAI LLM call in every repo routes through **OpenRouter**, using
`OPENROUTER_API_KEY`. UniLLM does not expose a native OpenAI chat provider, and
the company's own direct OpenAI account is inactive — it answers
`429 billing_not_active` — so a native route is dead in practice as well as
unregistered.

## Canonical endpoint form

```
openai/<model-id>@openrouter      # openai/gpt-5.6-terra@openrouter
```

Never `<model-id>@openai`. `@openrouter` resolves dynamically through the
OpenRouter catalog; `@openai` is not registered and fails endpoint resolution.

The Orchestra migration `2026-08-13-00-00_openrouter_model_endpoints.py`
rewrites stored legacy assistant endpoints. Source, defaults, examples, and
tests must use the canonical OpenRouter form directly.

## Hard rules

- New LLM call sites use `openai/<id>@openrouter`. Non-OpenAI providers
  (Anthropic, Google, …) are unaffected by this rule and keep their own routing.
- Never use `OPENAI_API_KEY` or a native `openai.OpenAI()` client for LLM chat
  or text generation. Non-chat integrations such as speech-to-text, realtime
  voice, or embeddings may use that key when the call site documents its
  distinct purpose — that key is the deployment's own (self-host / BYOK), not
  the company account above.
- Env defaults and `.env.example` entries carry the `@openrouter` form, so a
  fresh checkout cannot inherit a dead route.
- Treat any surviving `@openai` model string as a bug; UniLLM rejects it.

## The one legitimate direct-OpenAI path

Masked image edits (`images.edit` with `gpt-image-2`) have no OpenRouter
equivalent — OpenRouter's unified Image API does not expose the mask parameter.
That path may use a separately-named credential (`OPENAI_DIRECT_API_KEY`), must
never fall back to reading `OPENAI_API_KEY`, and must degrade loudly when the
credential is absent. It is the only exception; adding another needs an
explicit reason, not convenience.

# Orchestra / DataManager: Server-Side Queries First

Orchestra is a Postgres-backed query engine. **DataManager** (and brain store
adapters that wrap it) is the public surface for table I/O. Both surfaces expose
the `filter` keyword, metrics, and filtered updates. Coding agents
(one-off scripts **and** production ticks) must use that surface.
Downloading a table into Python and filtering locally is almost always wrong.

In **brain**, UniSDK is not a parallel authoring API for Orchestra rows —
see `brain/.agents/rules/brain-datamanager-only.md`. Actor plans use
`primitives.data.*` (DataManager). UniSDK remains an implementation detail
under DataManager (and a few non-table exceptions listed in that rule).

## Invariant

For any lookup, count, membership check, or bulk mutation against an Orchestra
context (especially tables at 10⁴+ rows such as GTM `Prospects`):

1. **Prefer a server-side filter** — `DataManager.filter` / store
   `find_many` / `find_one` / `find_by_field` with `filter`.
2. **Prefer server-side aggregation** — `DataManager.reduce(metric="count"|…)`,
   store `count_rows`.
3. **Prefer filtered / batched writes** — `update_rows` / `update_by_ids` /
   store updates on ids from `filter(..., include_ids=True)`; bulk insert via
   `insert_rows` / `parallel_create_logs` / `ingest`. Do not invent a
   full-table client scan to decide which rows to touch.

   Note for unify Actor plans specifically: `DataManager.ingest` is the
   low-level write engine and is not exposed as a primitive there — storing
   new data goes through `primitives.ingestion.submit`, which records a
   resumable, checkpointed run over the same engine. Scripts, brain ticks and
   workers may keep calling `ingest` directly.

Equality on a typed field (e.g. `best_email == "..."`) is a sub-second
server query even on ~10⁵-row contexts. A client `iter_all` + Python `if`
is minutes of HTTP pagination and will silently throttle any tick that
calls it in a hot path.

## Required patterns

```python
# Lookup by field (server-side) — brain store adapter
rows = store.find_many(
    "Prospects",
    filter=f'best_email == "{email}"',
    limit=3,  # existence / small-cardinality checks: use a tiny limit
)

# Or DataManager directly (absolute Teams/ paths are fine)
from unify.data_manager.data_manager import DataManager
dm = DataManager()
rows = dm.filter(
    "Teams/11/Data/GTM/Prospects",
    filter=f'best_email == "{email}"',
    limit=3,
    include_ids=True,  # when a later update_by_ids is needed
)

# Counts — never download to len()
n = store.count_rows("Prospects", filter='enrich_status == "enriched"')
# or: dm.reduce(path, metric="count", columns="<any indexed col>", filter=...)
```

House examples in brain: `brain/gtm/outbound/enrollments.py`,
`brain/gtm/stargazer/enrich.py` (`_sync_email_shared_2x_flags`),
`brain.gtm.store.DataManagerGTMStore.find_by_field`. Expanded recipes:
`brain/docs/operations/orchestra-data-access.md`.

## Hard refuse (anti-patterns)

Do **not** ship or land one-off scripts that:

- Call `iter_all` / paginate with **no** (or vacuous) filter, then
  compare fields in Python (`if row["best_email"] == email`)
- Pull an entire large context to compute a count, distinct set, or "does
  any row match?"
- Nest a full-table scan inside a per-row hot loop (enrich, draft, poll)
- Reach for raw `unisdk.get_logs` / `create_logs` for Brain/Assistants
  domain tables when DataManager already covers the need

`iter_all(..., filter=...)` is acceptable only when you truly need
**every** matching row and the filter is selective enough that the result
set is bounded. Unfiltered `iter_all` on large tables requires an explicit
operator rationale in the change description.

## Before you write a scan

Ask: "Can Orchestra answer this with DataManager `filter` / `reduce` / a
small `limit`?" If yes, do that. If a capability is missing, extend
DataManager in unify — do not default to downloading the table via UniSDK.

The frozen derived-template persistence schema still uses the nested
`"filter_expr"` key. That stored key is not a public DataManager or brain store
keyword and must not be renamed.

## Related

- Auth / tenant key pitfalls: `orchestra-admin-vs-user-api-access.md`
- Brain DataManager-only invariant: `brain/.agents/rules/brain-datamanager-only.md`
- Brain env, bulk-write footguns, `limit=1000` pagination:
  `brain/.agents/rules/brain-orchestra-staging.md`
- Playbook with copy-paste recipes:
  `brain/docs/operations/orchestra-data-access.md`

# Infra command safety

Two traps here fail *silently* — wrong output rather than an error — so they
cannot be discovered by trying.

**`gcloud` is regional and the default lies.** Most resources are regional or
zonal, and several `gcloud` surfaces default to `global` and return stale or
empty output **without erroring**. Cloud Build is the worst: with no
`--region`, `gcloud builds …` hits `global`, where the main project looks
empty and the saas project hides every `orchestra`/`console` build. Reading
that as "the build never ran" or "the service doesn't exist" is a recurring
mistake. Pass the location flag explicitly — the estate is `us-central1` —
and suspect a wrong location before a wrong project.

**Some live resources keep legacy `droid-*` / `unity-*` names on purpose.**
The platform was renamed additively, so code targets those names deliberately.
A name mismatch fails at *runtime* (404/401/empty config), not at build, and
has caused silent production outages. When something infra-related 404s or
"doesn't exist", suspect a legacy name before changing the code constant.

Full topology — projects, regions, where things run, and the exhaustive
legacy-name list — is in
[`.agents/global-rules/situational/deployed-system-topology.md`](.agents/global-rules/situational/deployed-system-topology.md).
Read it before any non-trivial infra work.

# Empty is not absent

A command that returns nothing has told you one of two things, and it does not
say which: *there is nothing there*, or *I could not look*. Infra CLIs answer
the second case by exiting `0` and printing nothing, so the failure arrives
wearing the costume of a finding. Every instance below cost real time, and
several produced confident, wrong statements about production before anyone
noticed.

This rule is about `gcloud`, `kubectl`, `gh`, and third-party HTTP APIs.
Orchestra's admin-versus-user split has its own rule
(`orchestra-admin-vs-user-api-access.md`) and its own fix; do not fold the two
together.

## The ways a zero lies

**Credentials expired mid-session.** `gcloud`'s token refresh fails, and
`kubectl`'s GKE auth plugin fails with it. A `kubectl get` then returns no
rows; `gcloud builds list --format=value(...)` prints empty columns; a cost
query reads `$0.00`. Nothing errors, nothing exits non-zero. A polling loop is
the worst place for this: it will happily report "no change" every interval
for an hour while holding no credentials at all.

**The selector matches nothing.** A label that was renamed (`app=droid` when
the live label is `app=unity`), a filter with a stray space in a timestamp
(`2026-08-08 T00:00:00Z`), a `--region` left at the `global` default. The
query is valid and correctly executed against the wrong target. This is the
most dangerous of the three, because **re-running it reproduces the same
answer** — repetition feels like verification and is not.

**A response body parsed with a forgiving default.** `d.get("logs", [])`
turns `{"detail": "..."}` into `[]`, and the script prints `found: 0`. The API
said exactly what was wrong; the parser discarded it. An HTTP 200 on one
endpoint is not evidence that a credential works on a different one.

**A secret listing that only covers one of the three layers.** GitHub resolves
Actions secrets **environment > repo > org**, so `gh secret list --org <o>`
answers a narrower question than "does this secret exist anywhere". An
environment-scoped copy in `unisdk` / `unify-testing` once silently shadowed
both other layers, which made deleting the repo-level copy a no-op and made
that repo's green CI prove nothing about the org token. Auditing or retiring a
secret means enumerating all three:
`repos/{r}/actions/secrets`, `repos/{r}/environments/{e}/secrets`, and
`repos/{r}/actions/organization-secrets`.

**A search index that lags the thing you are checking.** `gh search code`
indexes **default branches only**, and lags a merge by minutes. Two minutes
after unify-deploy#157 merged, an org-wide search still listed four
`.github/workflows/` paths that `git grep origin/main` showed as zero. This is
the same lie told backwards: the stop-condition on an irreversible step —
"if any workflow still references the secret, stop" — produced a false *stop*
rather than a false clear. Resolve any search result that gates an action
against a freshly fetched ref, and remember that a PR head branch is invisible
to code search entirely.

**A search API that only pretends to match your phrase.** GitHub's
`search/commits` does no exact-phrase matching, quoted or not: `repo:X
"Co-Authored-By: Claude"` also scores commits that merely carry those tokens
separately — some unrelated `Co-authored-by:` trailer in a message that says
"claude" somewhere else, which in an agent-heavy repo is a large fraction of
them. Audited 2026-08-17, it reported 907 for openclaw/openclaw and 584 for
NousResearch/hermes-agent where full clones counted 1,164 and 652 — it
over-counted one and under-counted the other, so the error is not even a bias
you could correct for. A control query did not save it: `unifyai/brain` came
back with exactly the 128 the local checkout had, which manufactured
confidence in a method that was wrong everywhere else. The mirror image bit
too — enumerating trailers from two relevance-ordered pages of the
hermes-agent results showed *zero* against a real 652. Relevance ordering is
not a random sample, so "I checked 100 results and saw none" is not evidence
of absence.

## What to do

- **Prove the query could have answered.** Before believing a zero, run the
  same shape of command against something you know exists. A test that cannot
  fail has proved nothing.
- **Check liveness explicitly, not by inference.** `gcloud auth print-access-token
  >/dev/null 2>&1 || echo EXPIRED` costs nothing and turns a silent empty into
  a stated one. Do it before a batch and again after a long poll. This is not
  hypothetical: mid-session gcloud expiry has twice made `kubectl get pods`
  return completely empty output with a zero exit — a cluster that looks idle
  and a cluster you cannot see are the same picture.
- **Pick a probe that cannot fail for the reason you are testing.** `gh auth
  print-access-token` fails on some machines *while the credential is live*,
  in foreground and background alike, though `gh api` and
  `gh auth token --user <acct>` both work. A probe that reports the very fault
  it exists to rule out is worse than no probe. Prefer an authenticated
  round-trip such as `gh api user --jq .login`.
- **Never let a loop treat empty as a state.** A poller must distinguish "no
  result yet" from "could not ask", and stop on the second. Blank output for
  several consecutive iterations is a fault, not a plateau.
- **Never parse with a default that can absorb an error.** Inspect the status
  code and the error field first, then read the payload.
- **Count commit messages from a clone, not from a search API.** A commit-only
  clone skips trees and blobs, so it is cheap even on a huge repo — openclaw's
  2.5GB and 101,712 commits arrived in 11 seconds and 50MB:

  ```bash
  git clone --bare --filter=tree:0 https://github.com/<owner>/<repo>.git <name>.git
  git -C <name>.git log --all -i -E --grep='<pattern>' --format=%H | wc -l
  ```

  Then watch the pattern for human collateral: counting Codex trailers by
  `openai` also matched two committers whose own address is at that domain —
  people who work there, not agents — and turned 33 agent commits into 55.
- **Sanity-check the shape.** Exactly zero rows, `$0.00`, or an empty list from
  a system you know is busy is a claim about your query, not about the system.
  Suspect the tool before you report the finding.

## When reporting

If a result contradicts what the system is known to be doing, say it is
unverified rather than presenting it as a finding — and re-run it once
credentials are known good. Retracting a confident zero costs far more trust
than flagging an uncertain one.

# Deployed System Topology (shared context)

This is broad orientation so agents in **any** repo know the shape of the deployed
system without re-exploring it every time. The **authoritative, exhaustive source of
truth** (every resource name, secret, CI trigger, and rename loose-end) is the
**`unify-deploy` repo root `README.md`**. Read that before deep infra work; do not
rediscover it from scratch.

## Repos

`unify` (runtime/brain, public), `unify-deploy` (private: hosted comms app + adapters,
assistant VM/tunnel infra, self-host stack, client overlay, prod CI/CD), `orchestra`
(backend API + Postgres, hosted), `console` (Next.js UI, hosted), `unisdk` (Python SDK,
public), `unillm` (LLM layer, public). Dependency `magnitude` is consumed at branch
`main`. All repos: `main` = prod, `staging` = dev; promote `staging`→`main`.

## GCP projects (4)

| Project ID | Role |
|---|---|
| `responsive-city-458413-a2` (display "Unity LiveKit") | Main runtime: GKE cluster `unity`, Cloud Run `droid-comms-app`/`droid-adapters` (+`-staging`), Pub/Sub fleet, most buckets, tunnel servers, Artifact Registry |
| `unity-assistant-vms` | Assistant desktop VM pool: pool images/families, pool VMs, static IPs, per-assistant archives |
| `saas-368716` ("SaaS") | **Orchestra + Console + landing page** (Cloud Run) and **Cloud SQL** Postgres (`prod-ssd-usc1`/`staging-ssd-usc1`, us-central1) |
| `unify-dns-server` | Public DNS zone `unifyai` → `unify.ai` (incl. `vm.unify.ai`, `tunnel.unify.ai`) |

## gcloud regions (don't trust the defaults)

Almost every resource is **regional/zonal**, and several `gcloud` surfaces **default to the wrong location and return stale/empty output without erroring** — misreading that ("the build never ran", "service doesn't exist") is a recurring mistake. **Pass the location flag explicitly; suspect a wrong/`global` location before suspecting a wrong project.**

- **Cloud Build is the #1 trap — it's regional and the `global` default lies.** All triggers + builds for **both** `responsive-city-458413-a2` and `saas-368716` are in **`us-central1`**. With no `--region`, `gcloud builds …` hits `global`, where `responsive-city-458413-a2` is **empty/months-stale** and `saas-368716` shows **only `landing-page`** (hiding all `orchestra`/`console` builds). Always: `gcloud builds {list,describe,log,triggers list,triggers run} --region=us-central1`.
- **GKE cluster `unity`**, Cloud Run comms/adapters, tunnel/pool VMs → **`us-central1`** (VM zones `-a`/`-f`).
- **saas Cloud Run** `orchestra`/`landing-page`/Console (prod + staging) → **`us-central1`**. **Cloud SQL** `prod-ssd-usc1`/`staging-ssd-usc1` → **`us-central1`**. (Consolidated from europe-west1/west3 in July 2026; the entire estate is now `us-central1`.)
- **Secret Manager** + **Cloud DNS** → global (no region flag; `--project` only).
- Org GitHub var `GCP_LOCATION=us-central1` is the saas Cloud Run default; the whole estate now shares `us-central1`.

Full per-surface table + exact service names: `unify-deploy` README §3 "gcloud region/zone cheat-sheet".

## Where things run

- **Assistant runtime** = `unity` container as on-demand GKE Jobs (label `app=droid`) on cluster `unity`. Idle→live via Pub/Sub `droid-startup[-staging]` / `droid-{assistant_id}[-staging]`. 7-min inactivity timeout; jobs retained for logs; `job-watcher` (kopf) does crash-safe cleanup.
- **Hosted comms**: adapters (inbound webhooks) + comms app (outbound + infra control plane `/infra/*`) on Cloud Run; the comms-app image also runs the GKE `assistant-session-controller`/`-pool-controller`.
- **Assistant desktops**: pooled Ubuntu/Windows VMs in `unity-assistant-vms`; runtime syncs `~/Unity/Local` over rclone SFTP (user `unityuser`, port 2222); cross-session home persisted to `gs://droid-assistant-archives/{id}.tar.gz`; optional rathole tunnel relay (`unity-tunnel-server`) for user machines.
- **Backend/UI/DB** all in `saas-368716`. Orchestra at `https://api.unify.ai/v0`.
- **Fleet audit** (`AssistantJobs`): Orchestra `is_system` project; writes via
  comms `/infra/assistant-jobs/*` (pod `UNIFY_KEY`) or Console hosted reads with
  `ORCHESTRA_ADMIN_KEY` as `__system__`. Never a Workspace `User` API key.

## Legacy Resource Naming Reality (critical)

The platform has gone through additive renames rather than a single in-place resource cutover.
Consequence: some live GCP/GitHub resources still intentionally use legacy `droid-*` or
`unity-*` names. A name mismatch
fails at *runtime* (404/401/empty config), not at build — this has caused silent prod outages.

**Immutable / permanently `unity` (do NOT try to "fix" in code — code targets these on purpose):**
project IDs `unity-assistant-vms` & `responsive-city-458413-a2`, all service-account emails
(`pool-vm-sa@unity-assistant-vms`, `comm-sa@responsive-city-458413-a2`), and GKE cluster `unity`
(`DROID_GKE_CLUSTER_NAME` default `unity`).

**Canonical names that commonly confuse (use these, verify before assuming):**
- GKE cluster: `unity`; VM/image project: `unity-assistant-vms`.
- Tunnel: VM `unity-tunnel-server`, bucket `unity-tunnel-config` (there is **no** `droid-tunnel-config`).
- Desktop pool: Ubuntu migrated to `droid-pool-ubuntu-*` (image family `droid-pool-ubuntu-vm`); **Windows still `unity-pool-windows-*`**.
- Archive bucket: `droid-assistant-archives` (live); `unity-assistant-archives` is legacy rollback.
- Data buckets (recordings/logs/artifacts) and ~84% of Pub/Sub topics/subs are still `unity-*`.
- CI: the GitHub org secrets `UNIFY_ADAPTERS_URL`/`UNIFY_COMMS_URL` are canonical; their `UNITY_*` twins are served only until every consumer reads the new names, then go.
- Deliberate legacy-named identifiers (not typos): `UnitySystemEvent` gateway envelope (unity↔console wire contract), `UnityTests` default test project, `unity-user-filesync` SSH key comment, `WaitingForUnity` state labels.

When something infra-related "doesn't exist" or 404s/401s, suspect a legacy resource-name mismatch
and confirm the real resource name against the `unify-deploy` README (or `gcloud`/`gh`) rather
than trusting the code constant.

# Orchestra Admin vs User API Access

Orchestra (`api.unify.ai/v0` prod, `api.staging.internal.saas.unify.ai/v0` staging) has **two distinct auth paths** (`orchestra/web/api/dependencies.py`). Confusing them wastes hours.

## The two dependencies

- **`auth_api_key`** — resolves the Bearer and sets `request.state.user_id`. All **data** endpoints use it (`/logs` get/update/`atomic_field_update`, contexts, canvas tokens, etc.). A user key is **scoped to its owner**. `ORCHESTRA_ADMIN_KEY` is accepted here too, as the platform principal `__system__`.
- **`auth_admin_key`** — matches the Bearer against the server's `ORCHESTRA_ADMIN_KEY` (`secrets.compare_digest`), a Cloud Scheduler OIDC token, or an `AdminUser`'s key. Only the **`/admin/*`** routers (registered with `ADMIN_AUTH`) use it. It **gates operations; it does not grant a data scope.**

## Reading a tenant's data with the admin key

`__system__` owns the platform's own projects — `Builtins`, `AssistantJobs` —
and nothing else. Point it at an ordinary project name and it resolves
nothing, so the read returns **`{"detail":"Project X not found."}` with HTTP
200-shaped handling in most clients**, which is easily misread as "the data is
not there" rather than "you are not the principal that owns it".

Name the principal to read as:

```bash
curl -s --get "$ORCHESTRA_URL/logs" -H "Authorization: Bearer $ORCHESTRA_ADMIN_KEY" \
  --data-urlencode "project_name=Assistants" \
  --data-urlencode "context=<owner>/<agent>/Tasks" \
  --data-urlencode "as_user_id=<user_id>"          # add as_organization_id=<n> for an org context
```

- **Reads only.** Any non-`GET` carrying `as_user_id` is refused `403`. A
  deployment-wide credential that could write as someone else would leave
  mutations attributed to a principal that never made them.
- **The target is named, never inferred.** `Project.name` is not unique —
  every account has its own `Assistants` — so a bare name under `__system__`
  would return whichever row sorted first and present one tenant's data as
  another's.
- **An unknown `as_user_id` is a `404`**, not an empty result.
- Every targeted read is logged server-side.

### Consequences (do not relearn these the hard way)
- Without `as_user_id`, the admin key sees only system projects. An empty read
  is a scope answer, not a data answer — see `empty-is-not-absent.md`.
- Even an **admin user's own** `UNIFY_KEY` is data-scoped: it returns `0` for
  another tenant's contexts. Admin status does not widen `/logs` results.
- The live `ORCHESTRA_ADMIN_KEY` is the **GCP secret** in `saas-368716` (prod) — repo `.env` copies may be stale. Staging uses a different value. Fetch with `gcloud secrets versions access latest --secret=ORCHESTRA_ADMIN_KEY --project=saas-368716`.

## Writing a specific tenant's data

Writes still need the owning principal's own key. Fetch it through the admin
API, then use it on the normal data endpoints:

1. **Enumerate + get keys** (admin key): `GET /admin/assistant` returns every assistant including its `api_key`, `agent_id`, `user_id` (also `GET /admin/assistant/{id}`, `/admin/assistant/user/{user_id}`).
   ```bash
   curl -s --get "$ORCHESTRA_URL/admin/assistant" -H "Authorization: Bearer $ORCHESTRA_ADMIN_KEY"
   ```
2. **Use the assistant's `api_key`** as the Bearer on the data API — now scoped to that tenant:
   ```python
   import unify
   logs = unify.get_logs(project="Assistants", filter="'x' in content", api_key=assistant_key)
   unify.update_logs(logs=log_id, entries={"content": new}, context=ctx, api_key=assistant_key)
   ```

A cross-tenant migration = loop assistants from `/admin/assistant`, then operate per-assistant with each key (there is no superuser *write* key). `gcloud` Cloud SQL (`prod-ssd`, `saas-368716`) is the direct-DB fallback for bulk passes.

# TaskScheduler surgery (Tasks rows)

Agents frequently break recurring jobs by hand-editing `Teams/*/Tasks`
(or assistant-scoped `…/Tasks`) via DataManager / UniSDK. Follow this rule for
**any** TaskScheduler ops across brain, unify, and orchestra.

## Identity model

- **`Tasks`** is definition-only: **one row per `task_id`**, the whole series.
  `unique_keys={"task_id": "int"}`, `auto_counting={"task_id": None}`
  (`unify/task_scheduler/task_scheduler.py`).
- **`Tasks/Executions`** holds the runs: one row per wake/attempt, keyed by
  `run_key` (the idempotency key). Occurrence and attempt are the same row.
  Recurrence creates the *next* Execution when the current one **starts** — it
  does **not** clone the Tasks row.
- **`instance_id` no longer exists.** The legacy occurrence counter was purged
  (July 2026): no code writes or reads it, and there is no field to set. A
  stored `instance_id` entry on an old row is inert junk — never a lookup key,
  never identity.
- Concurrency is normal now: several Executions can be in flight against one
  definition, so a definition sitting in `active` is not a zombie by itself.

## Hard refuse

- Do **not** set or change `task_id` on an existing row (Orchestra rejects
  writes to auto-counted unique identity fields).
- Do **not** resurrect `cancelled` / `failed` / `completed` rows by flipping
  them back to `scheduled` (or rewriting their `schedule`). Same for terminal
  Executions.
- Do **not** invent a Tasks row by hand with an explicit `task_id` — go through
  TaskScheduler APIs and let Orchestra allocate it.

## Allowed ops

| Goal | How |
|---|---|
| Arm a planted custom task | Set **`enabled=True`** on the definition row (the single `task_id` row, `custom_key` set). TaskScheduler schedules the next Execution. |
| Pause | `enabled=False` on the definition row; optionally cancel open Executions. |
| One-off catch-up / run now | `POST /v0/tasks/{task_id}/trigger` (`trigger_task(task_id=…)` in `typed_tasks_client`). |
| Change cadence | Edit `tasks.jsonl` + deploy reconcile, or TaskScheduler APIs that own the schedule — not ad-hoc DM patches. |
| Stuck `active` zombie | `POST /admin/task-source/release-active` with the source task log id. |

## Pre-migration remnants

If a `task_id` resolves to more than one Tasks row, that is a pre-migration
remnant (or a bad hand-write), not a counter desync. Delete the stale
duplicate, or leave it terminal and do not re-trigger until the health check
is clean.

## Break-glass

Only with an explicit operator rationale: fail an `active` zombie via
`POST /admin/task-source/release-active`, then clean up **extra** open
Executions so one next wake remains.

Ops detail: brain `docs/operations/scheduled-jobs.md` (re-arm / disable /
catch-up). Health check: `python3 -m scripts.tasks_health_check`.
