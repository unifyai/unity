# Unity

Unity is an AI Assistant framework implemented as a heavily distributed multi-node system. Each node communicates via English-language APIs, with the assistant's intelligence emerging from specialized **state managers** that handle different aspects of cognition, memory, and action.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [State Managers](#state-managers)
- [Getting Started](#getting-started)
- [Local Development](#local-development)
- [Testing](#testing)
- [Deployment](#deployment)

---

## Architecture Overview

Unity's architecture resembles a "back office" where specialized managers handle distinct aspects of the assistant's intelligence:

```
┌───────────────────────────────────────────────────────────────────────┐
│                       ConversationManager                             │
│                    (Live chat orchestration)                          │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                            Conductor                                  │
│              (Top-level cross-domain orchestrator)                    │
│                                                                       │
│                      request (unified read/write)                     │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                         State Managers                                │
│                                                                       │
│  ContactManager    KnowledgeManager   TaskScheduler   SecretManager   │
│  TranscriptManager GuidanceManager    WebSearcher     SkillManager    │
│  FileManager       FunctionManager    ImageManager    MemoryManager   │
│                                                                       │
│                             Actor                                     │
│                   (Real-time action executor)                         │
└───────────────────────────────────────────────────────────────────────┘
```

### Key Concepts

**Asynchronous Tool Loops**: Most manager methods are implemented as async tool loops where an LLM orchestrates lower-level tools that read and mutate backend resources via the Unify Python client.

**Dynamic Steering**: Manager methods expose handles for mid-flight control—pausing, resuming, interjecting, and stopping operations. These can be nested arbitrarily deep.

**Passthrough Steering**: In-flight tools can mark themselves as "passthrough," allowing steering commands to bypass intermediate LLM reasoning and reach inner tools immediately. This enables real-time user control of the assistant's actions.

---

## State Managers

Each manager owns a specific domain. The Conductor routes requests to the appropriate manager based on the user's intent.

### Core Orchestration

| Manager | Role |
|---------|------|
| **ConversationManager** | Live chat orchestrator. Wires steering (pause/resume/interject/stop) during conversations via the Conductor. |
| **Conductor** | Top-level orchestrator unifying all managers. Single entry point: `request` (unified read/write). |

### Data & Knowledge

| Manager | Role |
|---------|------|
| **ContactManager** | Source of truth for people/contact records. |
| **KnowledgeManager** | Source of truth for domain knowledge. Supports `ask`, `update`, and `refactor` operations. |
| **TranscriptManager** | Store and retrieval for message transcripts. Read-only via `ask`. |
| **FileManager** | Read-only registry and parsing for received/downloaded files. |
| **SecretManager** | Secure storage for secrets. Returns metadata only—never raw values. |
| **GuidanceManager** | Internal guidance, policies, and instructions. |

### Execution & Action

| Manager | Role |
|---------|------|
| **Actor** | Ephemeral, real-time action executor. Can invoke functions, control browsers, read files, or use any available capability. Returns a live steerable handle. |
| **TaskScheduler** | Durable task management and execution. Use `execute` to start work, not `update`. |
| **FunctionManager** | Catalogue of reusable Python functions (created by the Actor or provided by the user). |
| **SkillManager** | Human-friendly catalogue of assistant capabilities. Read-only discovery. |

### Perception & Communication

| Manager | Role |
|---------|------|
| **ImageManager** | Low-level image store/retrieval. Managers use `ImageHandle` to work with images. |
| **ScreenShareManager** | Continuous screen-share perception. Emits annotated screenshots during screen sharing. |
| **WebSearcher** | External/web research orchestration. |

### Background Processes

| Manager | Role |
|---------|------|
| **MemoryManager** | Offline memory maintenance (non-interactive). Distills transcripts into contacts/knowledge. |
| **EventBus** | Cross-cutting pub/sub backbone for telemetry and coordination. |

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Node.js 22+ (for agent-service)
- Redis (for local development)

### Installation

```bash
# Clone the repository
git clone git@github.com:unifyai/unity.git
cd unity

# Install dependencies using uv
uv sync --all-groups

# Activate the virtual environment
source .venv/bin/activate
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required
UNIFY_KEY=<your-unify-api-key>
UNIFY_BASE_URL=https://api.unify.ai/v0

# LLM Providers
OPENAI_API_KEY=<your-openai-key>
ANTHROPIC_API_KEY=<your-anthropic-key>

# Voice/Audio (optional, for voice features)
DEEPGRAM_API_KEY=<your-deepgram-key>
CARTESIA_API_KEY=<your-cartesia-key>
LIVEKIT_URL=<your-livekit-url>
LIVEKIT_API_KEY=<your-livekit-key>
LIVEKIT_API_SECRET=<your-livekit-secret>

# Assistant Configuration
ASSISTANT_ID=<id>
ASSISTANT_NAME=<name>
```

### Cursor Worktree Mode (Optional)

If you use Cursor's worktree mode for agent windows, install [direnv](https://direnv.net/) to auto-load your `.env`:

```bash
brew install direnv

# Add hook to ~/.zshrc
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc

# Silence verbose output (direnv 2.36+)
mkdir -p ~/.config/direnv
echo '[global]
hide_env_diff = true' > ~/.config/direnv/direnv.toml

direnv allow  # run once in the repo
```

Note: Use `~/.zshrc` (not `~/.zshenv`) to ensure Homebrew's PATH is available when the hook runs.

The repo includes an `.envrc` that automatically sources the main repo's `.env` in worktrees.

### Local Unify Development (Optional)

If you're developing features in the [unify](https://github.com/unifyai/unify) package alongside Unity, you can bind your `.venv` to a local clone:

```bash
# Install local unify in editable mode (overrides the remote source)
uv pip install -e /path/to/local/unify

# To revert to the upstream version
uv sync --all-groups
```

This is useful when debugging new features that haven't been pushed upstream. Running `uv sync --all-groups` restores the locked dependency set (including pulling `unify` from its configured Git source).

---

## Local Development

### Python Interpreter

Always use the project's virtual environment:

```bash
source .venv/bin/activate
# Or use .venv/bin/python directly
```

### Running the Conversation Manager

```bash
python start.py
```

### Browser Automation (Controller Mode)

**Browser Mode** (default):

```bash
# Start the agent service
npx ts-node agent-service/src/index.ts

# The Actor will use browser mode by default (agent_mode="browser")
```

**Desktop Mode** (for full desktop automation):

```bash
# See desktop/README.md for Docker-based virtual desktop setup
# Then use agent_mode="desktop" in the Actor
```

### Pre-commit Hooks

Run before committing to ensure code quality:

```bash
.venv/bin/python -m pre_commit run --all-files
```

### Dependencies

This project uses `uv` for dependency management:

- Configuration: `pyproject.toml`
- Lock file: `uv.lock` (do not edit manually)

---

## Testing

Tests are central to Unity's development. They fall on a spectrum between **symbolic tests** (infrastructure-focused) and **eval tests** (capability-focused).

You can run tests **locally** or offload them to **GitHub Actions** for better performance and to avoid straining your machine.

### Local Testing

#### Quick Start

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_contact_manager/test_create_contact.py

# Run a specific test
pytest tests/test_contact_manager/test_create_contact.py::test_create_single_contact
```

#### Parallel Execution

For faster runs, use the parallel test runner:

```bash
# Run all tests in parallel (one tmux session per test)
tests/parallel_run.sh tests/

# Wait for completion and capture logs
tests/parallel_run.sh --wait tests/

# Run only eval tests
tests/parallel_run.sh --eval-only tests/

# Run only symbolic tests
tests/parallel_run.sh --symbolic-only tests/
```

### Cloud Test Runs (GitHub Actions)

For surgical test runs without straining your local machine, use GitHub Actions. Benefits:

- **No local CPU load** — tests run on GitHub's infrastructure
- **No rate limiting** — GitHub runners have excellent network connectivity
- **24 parallel jobs** — one per test folder, all running simultaneously
- **Full `parallel_run.sh` support** — same flags work in CI as locally

#### Triggering Tests

Tests are **off by default** to avoid unnecessary CI costs. Trigger them explicitly:

| Method | How to Trigger |
|--------|----------------|
| **Commit message** | Include `[run-tests]` anywhere in the message |
| **PR title** | Include `[run-tests]` in the pull request title |
| **Manual** | GitHub Actions → "Testing Unity with uv" → "Run workflow" |

**Recommended workflow for surgical runs:**

1. Create a branch (or use an existing feature branch)
2. Push to the branch with `[run-tests]` in the commit message, OR
3. Manually trigger via the GitHub Actions UI for full control

```bash
# Trigger full test suite on push
git commit -m "Fix contact manager bug [run-tests]"

# Regular commit (no tests)
git commit -m "Update documentation"
```

#### Manual Workflow Dispatch (Surgical Runs)

For maximum control, use the GitHub Actions UI:

1. Go to **Actions** → **"Testing Unity with uv"**
2. Click **"Run workflow"** dropdown
3. Select your branch and configure inputs:

| Input | Default | Description |
|-------|---------|-------------|
| `test_path` | `.` (all) | Path to test folder, file, or specific test |
| `parallel_run_args` | *(empty)* | Extra args passed to `parallel_run.sh` |
| `test_session_timeout` | 120 | Session timeout in minutes |
| `runner_timeout` | 130 | Overall job timeout in minutes |

#### Flexible Test Targeting

The `test_path` input supports precise targeting:

| Input Value | What Runs |
|-------------|-----------|
| *(blank or `.`)* | All 24 test folders in parallel |
| `tests/test_actor` | Only the `test_actor` folder |
| `tests/test_actor/test_code_act.py` | Only that specific file |
| `tests/test_actor/test_code_act.py::test_name` | Only that specific test |

#### Advanced Options (`parallel_run_args`)

The `parallel_run_args` input accepts any `parallel_run.sh` flags—the CI experience matches local usage exactly:

| Flag | Example | Description |
|------|---------|-------------|
| `--eval-only` | `--eval-only` | Only `@pytest.mark.eval` tests |
| `--symbolic-only` | `--symbolic-only` | Only non-eval tests |
| `--repeat N` | `--repeat 5` | Run each test N times |
| `-s` | `-s` | Serial mode (one session per file) |
| `--tags` | `--tags exp-1` | Tag runs for filtering |
| `-j N` | `-j 10` | Limit concurrent sessions |
| `--env K=V` | `--env UNIFY_CACHE=false` | Set environment variable |

**Examples** (enter in the `parallel_run_args` field):

```
--eval-only --repeat 5
--symbolic-only -s
--env UNIFY_CACHE=false
--eval-only --tags model-compare -j 15
```

#### Accessing Test Logs

After a CI run, logs are available in the GitHub Actions UI:

| Artifact | Contents |
|----------|----------|
| `all-logs-consolidated` | **One-click download** of all logs combined |
| `pytest-logs-{folder}` | Individual folder's pytest output |
| `llm-io-debug-{folder}` | Individual folder's LLM I/O traces |

**Inline Failure Summaries**: Failed jobs display collapsible failure excerpts directly in the Summary page—no download required for quick triage.

### LLM Response Caching

By default (`UNIFY_CACHE=true`), LLM responses are cached in `.cache.ndjson`:

- **First run**: Real LLM calls, responses cached
- **Subsequent runs**: Cached responses replayed (fast, deterministic)

To force fresh LLM calls:

```bash
# Locally
tests/parallel_run.sh --env UNIFY_CACHE=false tests/

# In CI (via parallel_run_args)
--env UNIFY_CACHE=false
```

### Detailed Documentation

See [tests/README.md](tests/README.md) for comprehensive testing documentation including:

- Test philosophy (symbolic vs eval spectrum)
- Parallel runner options and tmux debugging
- Grid search for model comparisons
- Test data logging and analysis

---

## Deployment

### Docker

Build and run with Docker:

```bash
docker build -t unity .
docker run -p 8000:8000 -p 6080:6080 unity
```

The container includes:
- Redis server
- Virtual desktop (X11/VNC)
- PipeWire audio
- Agent service (Node.js)
- CodeSandbox service

### Cloud Deployment

Unity uses Google Cloud Build for CI/CD:

- `cloudbuild.yaml` — Production deployment
- `cloudbuild-staging.yaml` — Staging deployment

See [INFRA.md](INFRA.md) for detailed infrastructure documentation including:

- GKE architecture and idle container system
- Pub/Sub notification routing
- Webhook system for external services
- Multi-channel communication setup

---

## Project Structure

```
unity/
├── unity/                    # Main package
│   ├── actor/               # Browser/UI automation
│   ├── conductor/           # Top-level orchestrator
│   ├── contact_manager/     # Contact records
│   ├── conversation_manager/ # Live chat orchestration
│   ├── controller/          # Browser control layer
│   ├── events/              # EventBus pub/sub
│   ├── file_manager/        # File parsing/registry
│   ├── function_manager/    # User functions
│   ├── guidance_manager/    # Policies/instructions
│   ├── image_manager/       # Image storage
│   ├── knowledge_manager/   # Domain knowledge
│   ├── memory_manager/      # Offline maintenance
│   ├── screen_share_manager/ # Screen perception
│   ├── secret_manager/      # Secrets storage
│   ├── skill_manager/       # Capability catalogue
│   ├── task_scheduler/      # Task execution
│   ├── transcript_manager/  # Message transcripts
│   ├── web_searcher/        # Web research
│   └── common/              # Shared utilities
│       └── _async_tool/     # Async tool loop infrastructure
├── tests/                   # Test suite
├── agent-service/           # Node.js browser agent
├── codesandbox-service/     # CodeSandbox integration
├── desktop/                 # Virtual desktop setup
├── scripts/                 # Utility scripts
└── sandboxes/               # Interactive development sandboxes
```

---

## Contributing

1. Create a feature branch
2. Make your changes
3. Run pre-commit hooks: `.venv/bin/python -m pre_commit run --all-files`
4. Run relevant tests: `tests/parallel_run.sh --wait tests/test_<manager>/`
5. Submit a pull request

### Code Style

- Python: Formatted with Black, imports cleaned with autoflake
- No defensive coding—fail loud and fast
- No temporal comments ("NEW:", "Updated:")
- All LLM behavior adjustments via prompts/docstrings, not heuristics
