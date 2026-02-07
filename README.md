# Unity

Unity is an AI Assistant framework implemented as a heavily distributed multi-node system. Each node communicates via English-language APIs, with the assistant's intelligence emerging from specialized **state managers** that handle different aspects of cognition, memory, and action.

## System Architecture

Unity is the central "brain" in a multi-repository system:

```
         User (Console/Phone/SMS/Email)
                      │
    ┌─────────────────┴──────────────────┐
    │           Communication            │
    │    (Webhooks, Voice, SMS, Email)   │
    └────┬───────────────────────────────┘
         │
    ┌────┴────┐    ┌─────────┐    ┌─────────┐
    │  Unity  │    │  Unify  │    │Orchestra│
    │ (Brain) │───▶│  (SDK)  │───▶│  (API)  │
    │         │    │         │    │  (DB)   │
    └────┬────┘    └────┬────┘    └────┬────┘
         │              ▲              ▲
         │              │              │
         │    ┌─────────┴─┐       ┌────┴───────┐
         └───▶│  UniLLM   │       │  Console   │
              │ (LLM API) │       │(Interfaces)│
              └───────────┘       └────────────┘
```

**This repo (Unity)** is the AI assistant's cognitive core. It depends on:
- **Unify** — Python SDK for persistence and logging
- **UniLLM** — LLM client for all inference calls

Related repositories:
- [Orchestra](https://github.com/unifyai/orchestra) — Backend API and database
- [Communication](https://github.com/unifyai/communication) — External communication gateway (voice, SMS, email)
- [Console](https://github.com/unifyai/console) — Web UI and observability dashboard

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
│                              Actor                                    │
│              (Top-level code-first orchestrator)                      │
│                                                                       │
│                           act (live execution)                        │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                         State Managers                                │
│                                                                       │
│  ContactManager    KnowledgeManager   TaskScheduler   SecretManager   │
│  TranscriptManager GuidanceManager    WebSearcher                     │
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

Each manager owns a specific domain. The Actor plans and calls the appropriate manager primitives based on the user's intent.

### Core Orchestration

| Manager | Role |
|---------|------|
| **ConversationManager** | Live chat orchestrator. Wires steering (pause/resume/interject/stop) during conversations via the Actor. |
| **Actor** | Top-level orchestrator unifying all managers. Entry point: `act` (live code-first execution). |

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
| **Actor** | Ephemeral, real-time action executor. Can invoke functions, control computer interfaces, read files, or use any available capability. Returns a live steerable handle. |
| **TaskScheduler** | Durable task management and execution. Use `execute` to start work, not `update`. |
| **FunctionManager** | Catalogue of reusable Python functions (created by the Actor or provided by the user). |

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
# Clone all three repositories as siblings
cd ~/projects  # or your preferred directory
git clone git@github.com:unifyai/unity.git
git clone git@github.com:unifyai/unify.git
git clone git@github.com:unifyai/unillm.git

# Directory structure should be:
# ~/projects/
# ├── unity/    ← this repo
# ├── unify/    ← required sibling
# └── unillm/   ← required sibling

cd unity

# Install dependencies using uv
uv sync --all-groups

# Activate the virtual environment
source .venv/bin/activate
```

> **Note:** Unity requires local clones of `unify` and `unillm` as sibling directories. This enables rapid cross-repo development—changes to unify/unillm are immediately available without pushing to GitHub.
>
> If you see `error: Distribution not found at: file:///path/to/unify`, you're missing the sibling clones.

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required
UNIFY_KEY=<your-unify-api-key>
ORCHESTRA_URL=https://api.unify.ai/v0

# LLM Providers
OPENAI_API_KEY=<your-openai-key>
ANTHROPIC_API_KEY=<your-anthropic-key>

# Voice/Audio (optional, for voice features)
DEEPGRAM_API_KEY=<your-deepgram-key>
CARTESIA_API_KEY=<your-cartesia-key>
LIVEKIT_URL=<your-livekit-url>
LIVEKIT_API_KEY=<your-livekit-key>
LIVEKIT_API_SECRET=<your-livekit-secret>

# Communication Service (auto-detected if not set)
# Staging (default for non-main branches):
UNITY_COMMS_URL=https://unity-comms-app-staging-262420637606.us-central1.run.app
# Production (main branch only):
# UNITY_COMMS_URL=https://unity-comms-app-262420637606.us-central1.run.app

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

### Parallel Development with Clones

Cursor's worktree mode (parallel agents) has a known limitation: `.cursor/rules/` files with `alwaysApply: true` are not injected into the agent's context. This means worktree agents operate without any of your project rules.

The workaround is to run multiple independent clones in local mode, where rules work correctly:

```bash
./clone_adjacent.sh fix_loop_tests    # → ../unity_fix_loop_tests
./clone_adjacent.sh refactor_actor    # → ../unity_refactor_actor
```

Each clone checks out `staging`, inits submodules, copies `.env`, and symlinks `.venv` — ready to open in a separate Cursor window with full rule support. The script uses `--reference` to borrow local git objects, so cloning is near-instant.

### Cursor Cloud Agent Secrets (Required)

Cursor Cloud Agents run in isolated VMs and need user-specific secrets. Add these in **Cursor Settings → Cloud Agents → Secrets**:

| Secret Name | Value |
|-------------|-------|
| `UNIFY_KEY` | Your personal Unify API key |
| `GIT_USER_NAME` | Your full name (e.g., `Daniel Lenton`) |
| `GIT_USER_EMAIL` | Your email (e.g., `daniel@unify.ai`) |

The git identity secrets ensure commits are attributed to you rather than `cursoragent@cursor.com`. A pre-commit hook blocks commits from `cursoragent@cursor.com` as a safety net.

### Cross-Repo Development

Unity is configured to use local sibling clones of `unify` and `unillm` by default. This means:

- **Changes are instant**: Edit code in `../unify` or `../unillm` and it's immediately available in Unity
- **No push required**: Test local commits before pushing to GitHub
- **CI uses git**: GitHub Actions automatically replaces local paths with git URLs

**Branch alignment**: Keep your local clones on `staging` for development work:

```bash
cd ../unify && git checkout staging && git pull
cd ../unillm && git checkout staging && git pull
```

**Pulling upstream changes**: When you want the latest from GitHub:

```bash
cd ../unify && git pull
cd ../unillm && git pull
cd ../unity && uv sync --all-groups  # Re-sync to pick up changes
```

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

### Web Automation (Controller Mode)

**Web Mode** (default):

```bash
# Start the agent service
npx ts-node agent-service/src/index.ts

# The Actor will use web mode by default (agent_mode="web")
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

**Syncing git dependencies:** The lock file pins git dependencies (e.g., `unifyai`, `unillm`) to specific commits. To pull the latest from upstream branches:

```bash
./scripts/sync.sh
```

This upgrades git dependencies to their latest commits before syncing. Use this instead of plain `uv sync` when you need the latest upstream changes.

---

## Testing

Run tests locally or offload them to GitHub Actions (24 parallel jobs, no local CPU load).

```bash
# Quick start (local)
tests/parallel_run.sh tests/                    # Run all tests
tests/parallel_run.sh tests/actor/         # Run one folder
tests/parallel_run.sh --timeout 300 tests/      # With 5-minute timeout

# CI trigger (via commit message)
git commit -m "Fix bug [run-tests]"                           # All tests
git commit -m "Fix bug [parallel_run.sh tests/actor]"    # Specific folder
```

See **[tests/README.md](tests/README.md)** for complete documentation:
- Local testing with `parallel_run.sh` (flags, tmux sessions, debugging)
- CI/GitHub Actions (triggers, workflow dispatch, log artifacts)
- Test philosophy (symbolic vs eval spectrum)
- Grid search, resource monitoring, and more

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
│   ├── actor/               # Top Level orchestrator
│   ├── contact_manager/     # Contact records
│   ├── conversation_manager/ # Live chat orchestration
│   ├── controller/          # Computer control layer
│   ├── events/              # EventBus pub/sub
│   ├── file_manager/        # File parsing/registry
│   ├── function_manager/    # User functions
│   ├── guidance_manager/    # Policies/instructions
│   ├── image_manager/       # Image storage
│   ├── knowledge_manager/   # Domain knowledge
│   ├── memory_manager/      # Offline maintenance
│   ├── screen_share_manager/ # Screen perception
│   ├── secret_manager/      # Secrets storage
│   ├── task_scheduler/      # Task execution
│   ├── transcript_manager/  # Message transcripts
│   ├── web_searcher/        # Web research
│   └── common/              # Shared utilities
│       └── _async_tool/     # Async tool loop infrastructure
├── tests/                   # Test suite
├── agent-service/           # Node.js web/desktop agent
├── desktop/                 # Virtual desktop setup
├── scripts/                 # Utility scripts
└── sandboxes/               # Interactive development sandboxes
```

---

## Contributing

1. Create a feature branch
2. Make your changes
3. Run pre-commit hooks: `.venv/bin/python -m pre_commit run --all-files`
4. Run relevant tests: `tests/parallel_run.sh tests/<manager>/`
5. Submit a pull request

### Code Style

- Python: Formatted with Black, imports cleaned with autoflake
- No defensive coding—fail loud and fast
- No temporal comments ("NEW:", "Updated:")
- All LLM behavior adjustments via prompts/docstrings, not heuristics
