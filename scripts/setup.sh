#!/usr/bin/env bash
# ============================================================================
# Unity setup — local backend bootstrap
# ============================================================================
# Spins up a local Orchestra instance (Postgres+pgvector in Docker + FastAPI
# server) and wires Unity's .env to use it. Idempotent: safe to re-run.
#
# Usually called automatically by scripts/install.sh; re-run directly via
# `unity setup` if you need to re-bootstrap (e.g., Docker wasn't running the
# first time, or you wiped ~/.unity).
#
# Environment (all optional):
#   UNITY_HOME              Install root (default: ~/.unity)
#   ORCHESTRA_PORT          Orchestra FastAPI port (default: 8000)
#   ORCHESTRA_DB_PORT       Postgres port (default: 5432)
#   UNITY_SKIP_ORCHESTRA    If "1", skip the Orchestra spin-up (env only)
# ============================================================================

set -e

# --- Config ---------------------------------------------------------------
UNITY_HOME="${UNITY_HOME:-$HOME/.unity}"
UNITY_REPO="${UNITY_HOME}/unity"
ORCHESTRA_REPO="${UNITY_HOME}/orchestra"
ORCHESTRA_PORT="${ORCHESTRA_PORT:-8000}"
ORCHESTRA_DB_PORT="${ORCHESTRA_DB_PORT:-5432}"

# Ensure user-local tool dirs are on PATH. `uv` and tools `uv` installs
# (e.g. poetry) land here, and in a fresh shell they may not be picked up.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# --- Colors / logging -----------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log_info()    { echo -e "${CYAN}→${NC} $1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_warn()    { echo -e "${YELLOW}⚠${NC} $1"; }
log_error()   { echo -e "${RED}✗${NC} $1"; }

# --- Docker ---------------------------------------------------------------
detect_os() {
    case "$(uname -s)" in
        Linux*) echo "linux" ;;
        Darwin*) echo "macos" ;;
        *) echo "unknown" ;;
    esac
}

install_docker_interactive() {
    local os; os="$(detect_os)"
    log_warn "Docker is not installed (required for local Orchestra)."
    case "$os" in
        macos)
            echo "  On macOS, install Docker Desktop:"
            echo "    • via Homebrew:  brew install --cask docker"
            echo "    • or download:   https://www.docker.com/products/docker-desktop"
            if [ -t 0 ] && command -v brew >/dev/null 2>&1; then
                read -r -p "  Install via Homebrew now? [y/N] " ans
                if [[ "$ans" =~ ^[Yy]$ ]]; then
                    brew install --cask docker
                    log_info "After Docker Desktop opens and finishes initial setup, re-run: unity setup"
                fi
            fi
            ;;
        linux)
            echo "  On Linux, install Docker Engine:"
            echo "    • Debian/Ubuntu:  curl -fsSL https://get.docker.com | sh"
            echo "    • Then:           sudo usermod -aG docker \$USER  (and log out/in)"
            if [ -t 0 ]; then
                read -r -p "  Run the official Docker install script now? [y/N] " ans
                if [[ "$ans" =~ ^[Yy]$ ]]; then
                    curl -fsSL https://get.docker.com | sh
                    log_info "After Docker is configured (sudo usermod -aG docker \$USER), re-run: unity setup"
                fi
            fi
            ;;
        *)
            echo "  Your OS isn't auto-detected. See https://docs.docker.com/get-docker/"
            ;;
    esac
    return 1
}

ensure_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        install_docker_interactive
        return 1
    fi
    if ! docker info >/dev/null 2>&1; then
        log_warn "Docker is installed but the daemon isn't running."
        if [ "$(detect_os)" = "macos" ] && [ -d /Applications/Docker.app ]; then
            log_info "Starting Docker Desktop..."
            open -a Docker
            local i
            for i in $(seq 1 60); do
                if docker info >/dev/null 2>&1; then break; fi
                sleep 1
            done
            if ! docker info >/dev/null 2>&1; then
                log_error "Docker didn't become ready within 60s. Please start it and re-run: unity setup"
                return 1
            fi
        else
            log_error "Start your Docker daemon and re-run: unity setup"
            return 1
        fi
    fi
    log_success "Docker: $(docker --version 2>/dev/null | head -1)"
}

# --- Poetry (for orchestra) -----------------------------------------------
ensure_poetry() {
    if command -v poetry >/dev/null 2>&1; then
        log_success "poetry: $(poetry --version 2>/dev/null)"
        return 0
    fi
    if ! command -v uv >/dev/null 2>&1; then
        log_error "uv not on PATH. scripts/install.sh should have installed it; re-run install.sh?"
        return 1
    fi
    log_info "Installing poetry via uv tool..."
    uv tool install poetry >/dev/null 2>&1 || {
        log_error "Failed to install poetry via uv tool"
        return 1
    }
    # uv tool installs into ~/.local/bin
    if ! command -v poetry >/dev/null 2>&1 && [ -x "$HOME/.local/bin/poetry" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    log_success "poetry installed: $(poetry --version 2>/dev/null)"
}

# --- Orchestra clone + install --------------------------------------------
ensure_orchestra_repo() {
    if [ -d "$ORCHESTRA_REPO/.git" ]; then
        log_info "Updating orchestra at $ORCHESTRA_REPO..."
        git -C "$ORCHESTRA_REPO" fetch --quiet origin main
        git -C "$ORCHESTRA_REPO" checkout --quiet main 2>/dev/null || log_warn "Couldn't checkout main (uncommitted changes?)"
        git -C "$ORCHESTRA_REPO" pull --quiet --ff-only origin main 2>/dev/null || log_warn "Non-ff pull skipped in orchestra; leaving as-is."
    else
        log_info "Cloning unifyai/orchestra into $ORCHESTRA_REPO..."
        mkdir -p "$UNITY_HOME"
        git clone --quiet --branch main https://github.com/unifyai/orchestra.git "$ORCHESTRA_REPO"
    fi
    log_success "orchestra: $(git -C "$ORCHESTRA_REPO" rev-parse --short HEAD)"
}

install_orchestra_deps() {
    log_info "Installing orchestra dependencies via poetry (first-time: a few minutes)..."
    (cd "$ORCHESTRA_REPO" && poetry install --no-interaction --quiet) || {
        log_error "poetry install failed in $ORCHESTRA_REPO"
        log_info "Try manually: cd $ORCHESTRA_REPO && poetry install"
        return 1
    }
    log_success "Orchestra dependencies installed"
}

# --- Orchestra spin-up ----------------------------------------------------
start_local_orchestra() {
    log_info "Starting local Orchestra (Docker Postgres+pgvector + FastAPI)..."

    # Disable auto-shutdown: local installs should stay up until `unity stop`
    export ORCHESTRA_INACTIVITY_TIMEOUT_SECONDS=0
    export ORCHESTRA_PORT
    export ORCHESTRA_DB_PORT
    export ORCHESTRA_REPO_PATH="$ORCHESTRA_REPO"

    # Run orchestra's local.sh; tee to terminal AND a log file so we can parse
    # the final `export UNIFY_BASE_URL=... / UNIFY_KEY=...` lines out of it.
    local tmp_log
    tmp_log="$(mktemp)"
    bash "$ORCHESTRA_REPO/scripts/local.sh" start 2>&1 | tee "$tmp_log"
    local start_exit=${PIPESTATUS[0]}

    if (( start_exit != 0 )); then
        rm -f "$tmp_log"
        log_error "Orchestra failed to start (local.sh exit=$start_exit). See output above."
        log_info "Common causes: port $ORCHESTRA_PORT / $ORCHESTRA_DB_PORT in use, Docker daemon not running."
        log_info "Re-run with:  unity setup"
        return 1
    fi

    # Extract the export lines from the log
    local env_block
    env_block=$(grep -E '^export (UNIFY_BASE_URL|UNIFY_KEY)=' "$tmp_log" | tail -2)
    rm -f "$tmp_log"

    if [ -z "$env_block" ]; then
        log_error "Orchestra started but didn't emit UNIFY_BASE_URL / UNIFY_KEY lines."
        return 1
    fi

    # Source the block (safe: local.sh emits simple 'export KEY=value' lines)
    eval "$env_block"

    # Sanity: reject the staging-URL fallback that local.sh emits when checks fail
    case "${UNIFY_BASE_URL:-}" in
        http://127.0.0.1:*|http://localhost:*) ;;
        *)
            log_error "Orchestra emitted a non-local URL (${UNIFY_BASE_URL:-empty}). Refusing to wire."
            log_info "This usually means local.sh's internal checks failed and fell back to staging."
            return 1
            ;;
    esac

    log_success "Orchestra URL:   ${UNIFY_BASE_URL}"
    log_success "Local UNIFY_KEY: ${UNIFY_KEY}"
}

# --- Wire into Unity's .env -----------------------------------------------
wire_unity_env() {
    local env_file="$UNITY_REPO/.env"
    if [ ! -f "$env_file" ]; then
        if [ -f "$UNITY_REPO/.env.example" ]; then
            cp "$UNITY_REPO/.env.example" "$env_file"
            log_info "Created $env_file from .env.example"
        else
            touch "$env_file"
        fi
    fi

    # Idempotent upsert of KEY=VALUE
    upsert() {
        local key="$1" val="$2"
        if grep -qE "^${key}=" "$env_file"; then
            # Replace existing line (portable sed for macOS + linux)
            python3 - "$env_file" "$key" "$val" <<'PYEOF'
import sys, re
path, key, val = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    lines = f.readlines()
pat = re.compile(rf'^{re.escape(key)}=')
for i, line in enumerate(lines):
    if pat.match(line):
        lines[i] = f'{key}={val}\n'
        break
with open(path, 'w') as f:
    f.writelines(lines)
PYEOF
        else
            printf '%s=%s\n' "$key" "$val" >> "$env_file"
        fi
    }

    upsert "ORCHESTRA_URL" "$UNIFY_BASE_URL"
    upsert "UNIFY_KEY" "$UNIFY_KEY"

    log_success "Wrote ORCHESTRA_URL and UNIFY_KEY to $env_file"
    log_info "Edit this file to add OPENAI_API_KEY or ANTHROPIC_API_KEY for the LLM provider."
}

# --- Main -----------------------------------------------------------------
main() {
    echo ""
    echo -e "${BOLD}Unity setup${NC} — bootstrapping local Orchestra"
    echo ""

    if [ ! -d "$UNITY_REPO" ]; then
        log_error "Unity is not installed at $UNITY_REPO. Run scripts/install.sh first."
        exit 1
    fi

    if [ "${UNITY_SKIP_ORCHESTRA:-0}" = "1" ]; then
        log_warn "UNITY_SKIP_ORCHESTRA=1 — skipping Orchestra spin-up."
        log_info "Set ORCHESTRA_URL + UNIFY_KEY manually in $UNITY_REPO/.env to point at a remote backend."
        exit 0
    fi

    ensure_docker || exit 1
    ensure_poetry || exit 1
    ensure_orchestra_repo
    install_orchestra_deps || exit 1
    start_local_orchestra || exit 1
    wire_unity_env

    echo ""
    echo -e "${GREEN}${BOLD}Setup complete.${NC}"
    echo ""
    echo "  Orchestra is running at $UNIFY_BASE_URL"
    echo "  Stop it any time with:  unity stop"
    echo ""
    echo "  Next: add an LLM key to $UNITY_REPO/.env, then run  ${CYAN}unity${NC}"
    echo ""
}

main "$@"
