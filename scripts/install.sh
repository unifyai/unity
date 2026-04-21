#!/usr/bin/env bash
# ============================================================================
# Unity Installer
# ============================================================================
# Installs Unity (https://github.com/unifyai/unity) locally on macOS / Linux /
# WSL2. Clones unity, unify, and unillm as siblings under $UNITY_HOME and
# editable-installs them with uv.
#
# Quick install:
#   curl -fsSL https://raw.githubusercontent.com/unifyai/unity/main/scripts/install.sh | bash
#
# Options:
#   --dir PATH        Installation directory (default: ~/.unity)
#   --branch NAME     Git branch to install (default: main)
#   --no-cli          Skip creating the `unity` command shim
#   --skip-deps       Skip system-dependency checks (PortAudio, etc.)
#   --skip-setup      Skip the local Orchestra spin-up at the end
#                     (just install the code; run `unity setup` later)
#   -h, --help        Show this help
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Configuration
UNITY_HOME="${UNITY_HOME:-$HOME/.unity}"
UNITY_REPO="${UNITY_HOME}/unity"
UNIFY_REPO="${UNITY_HOME}/unify"
UNILLM_REPO="${UNITY_HOME}/unillm"
BRANCH="main"
PYTHON_VERSION="3.12"
CREATE_CLI=true
CHECK_DEPS=true
RUN_SETUP=true
CLI_DIR="${CLI_DIR:-$HOME/.local/bin}"
REPO_BASE="https://github.com/unifyai"

# ----------------------------------------------------------------------------
# Parse options
# ----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --dir) UNITY_HOME="$2"; UNITY_REPO="$UNITY_HOME/unity"; UNIFY_REPO="$UNITY_HOME/unify"; UNILLM_REPO="$UNITY_HOME/unillm"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --no-cli) CREATE_CLI=false; shift ;;
        --skip-deps) CHECK_DEPS=false; shift ;;
        --skip-setup) RUN_SETUP=false; shift ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's|^# ||;s|^#$||'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
log_info()    { echo -e "${CYAN}→${NC} $1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_warn()    { echo -e "${YELLOW}⚠${NC} $1"; }
log_error()   { echo -e "${RED}✗${NC} $1"; }

print_banner() {
    echo ""
    echo -e "${MAGENTA}${BOLD}"
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│                  Unity Installer                         │"
    echo "├─────────────────────────────────────────────────────────┤"
    echo "│  Steerable AI agent orchestration, open-sourced by      │"
    echo "│  Unify. https://github.com/unifyai/unity                │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
}

# ----------------------------------------------------------------------------
# OS detection
# ----------------------------------------------------------------------------
detect_os() {
    case "$(uname -s)" in
        Linux*)
            OS="linux"
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                DISTRO="$ID"
            else
                DISTRO="unknown"
            fi
            ;;
        Darwin*)
            OS="macos"
            DISTRO="macos"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            log_error "Native Windows is not supported."
            log_info "Please install Unity inside WSL2:"
            log_info "  https://learn.microsoft.com/en-us/windows/wsl/install"
            exit 1
            ;;
        *)
            OS="unknown"
            DISTRO="unknown"
            log_warn "Unknown operating system: $(uname -s). Proceeding at your own risk."
            ;;
    esac
    log_success "Detected: $OS ($DISTRO)"
}

# ----------------------------------------------------------------------------
# uv
# ----------------------------------------------------------------------------
ensure_uv() {
    log_info "Checking for uv..."
    if command -v uv &> /dev/null; then
        UV_CMD="uv"
    elif [ -x "$HOME/.local/bin/uv" ]; then
        UV_CMD="$HOME/.local/bin/uv"
    elif [ -x "$HOME/.cargo/bin/uv" ]; then
        UV_CMD="$HOME/.cargo/bin/uv"
    else
        log_info "Installing uv (fast Python package manager)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || {
            log_error "Failed to install uv. See https://docs.astral.sh/uv/getting-started/installation/"
            exit 1
        }
        if [ -x "$HOME/.local/bin/uv" ]; then
            UV_CMD="$HOME/.local/bin/uv"
        elif [ -x "$HOME/.cargo/bin/uv" ]; then
            UV_CMD="$HOME/.cargo/bin/uv"
        elif command -v uv &> /dev/null; then
            UV_CMD="uv"
        else
            log_error "uv installed but not found on PATH. Add ~/.local/bin to your PATH and re-run."
            exit 1
        fi
    fi
    log_success "uv: $($UV_CMD --version 2>/dev/null)"
}

# ----------------------------------------------------------------------------
# Python
# ----------------------------------------------------------------------------
ensure_python() {
    log_info "Ensuring Python $PYTHON_VERSION is available..."
    if ! $UV_CMD python find "$PYTHON_VERSION" &> /dev/null; then
        log_info "Python $PYTHON_VERSION not found locally — uv will download it during sync."
    else
        log_success "Python $PYTHON_VERSION: $($UV_CMD python find "$PYTHON_VERSION")"
    fi
}

# ----------------------------------------------------------------------------
# System deps (informational — we don't attempt install)
# ----------------------------------------------------------------------------
check_system_deps() {
    [ "$CHECK_DEPS" = "false" ] && return 0
    log_info "Checking system dependencies..."

    local hard_missing=() warn_missing=()

    case "$OS" in
        macos)
            # Xcode CLI tools — required for any package building from sdist (pyaudio etc.)
            xcode-select -p >/dev/null 2>&1 || hard_missing+=("Xcode Command Line Tools  (install: xcode-select --install)")
            pkg-config --exists portaudio-2.0 2>/dev/null || warn_missing+=("portaudio  (install: brew install portaudio)")
            ;;
        linux)
            # pyaudio has no manylinux wheel for Py3.12; it builds from sdist and needs gcc + Python headers.
            command -v gcc >/dev/null 2>&1 || hard_missing+=("gcc / build-essential  (install: sudo apt-get install -y build-essential python3-dev)")
            pkg-config --exists portaudio-2.0 2>/dev/null || hard_missing+=("portaudio development headers  (install: sudo apt-get install -y portaudio19-dev)")
            ;;
    esac

    if [ ${#hard_missing[@]} -gt 0 ]; then
        log_error "Required system packages are missing:"
        for m in "${hard_missing[@]}"; do echo "    - $m"; done
        echo ""
        case "$OS" in
            linux)
                echo "    Quick fix:"
                echo "      sudo apt-get update && sudo apt-get install -y build-essential python3-dev portaudio19-dev"
                echo "    Then re-run: curl -fsSL https://raw.githubusercontent.com/unifyai/unity/main/scripts/install.sh | bash"
                ;;
            macos)
                echo "    Quick fix:"
                echo "      xcode-select --install   # if not already installed"
                echo "      brew install portaudio"
                echo "    Then re-run install.sh."
                ;;
        esac
        echo ""
        echo "    (Bypass with --skip-deps at your own risk; uv sync will fail on native extensions.)"
        exit 1
    fi

    if [ ${#warn_missing[@]} -gt 0 ]; then
        log_warn "Optional system packages missing (voice features will be limited):"
        for m in "${warn_missing[@]}"; do echo "    - $m"; done
    else
        log_success "System dependencies OK"
    fi
}

# ----------------------------------------------------------------------------
# Clone a repo (idempotent — pulls if already present, clones otherwise)
# ----------------------------------------------------------------------------
clone_or_update() {
    local name="$1"
    local dest="$2"
    local url="$REPO_BASE/$name.git"

    if [ -d "$dest/.git" ]; then
        log_info "Updating $name at $dest..."
        git -C "$dest" fetch --quiet origin "$BRANCH"
        git -C "$dest" checkout --quiet "$BRANCH" || {
            log_warn "Couldn't checkout $BRANCH in $dest (uncommitted changes?). Skipping update."
            return 0
        }
        git -C "$dest" pull --quiet --ff-only origin "$BRANCH" || log_warn "Fast-forward pull failed in $dest; leaving as-is."
    else
        log_info "Cloning $url into $dest..."
        git clone --quiet --branch "$BRANCH" "$url" "$dest"
    fi
    log_success "$name: $(git -C "$dest" rev-parse --short HEAD)"
}

# ----------------------------------------------------------------------------
# Core install
# ----------------------------------------------------------------------------
do_install() {
    mkdir -p "$UNITY_HOME"
    clone_or_update "unify" "$UNIFY_REPO"
    clone_or_update "unillm" "$UNILLM_REPO"
    clone_or_update "unity" "$UNITY_REPO"

    log_info "Syncing dependencies via uv (this pulls Python 3.12 if missing, may take a few minutes)..."
    (cd "$UNITY_REPO" && $UV_CMD sync)
    log_success "Dependencies installed in $UNITY_REPO/.venv"

    # .env scaffolding
    if [ ! -f "$UNITY_REPO/.env" ] && [ -f "$UNITY_REPO/.env.example" ]; then
        cp "$UNITY_REPO/.env.example" "$UNITY_REPO/.env"
        log_success "Created $UNITY_REPO/.env from .env.example — edit with your API keys"
    fi
}

# ----------------------------------------------------------------------------
# CLI shim at ~/.local/bin/unity
# ----------------------------------------------------------------------------
create_cli() {
    [ "$CREATE_CLI" = "false" ] && return 0
    mkdir -p "$CLI_DIR"
    local shim="$CLI_DIR/unity"

    cat > "$shim" <<EOF
#!/usr/bin/env bash
# Unity CLI shim — dispatches subcommands, falls through to the sandbox.
# Generated by install.sh; safe to edit.
set -e
UNITY_HOME="${UNITY_HOME}"
UNITY_REPO="${UNITY_REPO}"
ORCHESTRA_REPO="\$UNITY_HOME/orchestra"
export UNITY_HOME

if [ ! -d "\$UNITY_REPO" ]; then
    echo "Unity is not installed at \$UNITY_REPO. Re-run install.sh or set UNITY_HOME." >&2
    exit 1
fi

case "\${1:-}" in
    setup)
        shift
        exec bash "\$UNITY_REPO/scripts/setup.sh" "\$@"
        ;;
    stop)
        if [ -x "\$ORCHESTRA_REPO/scripts/local.sh" ]; then
            exec bash "\$ORCHESTRA_REPO/scripts/local.sh" stop
        else
            echo "Orchestra not installed at \$ORCHESTRA_REPO — nothing to stop." >&2
            exit 1
        fi
        ;;
    status)
        if [ -x "\$ORCHESTRA_REPO/scripts/local.sh" ]; then
            exec bash "\$ORCHESTRA_REPO/scripts/local.sh" status
        else
            echo "Orchestra not installed at \$ORCHESTRA_REPO." >&2
            exit 1
        fi
        ;;
    restart)
        if [ -x "\$ORCHESTRA_REPO/scripts/local.sh" ]; then
            exec bash "\$ORCHESTRA_REPO/scripts/local.sh" restart
        else
            echo "Orchestra not installed at \$ORCHESTRA_REPO — run \\\`unity setup\\\` first." >&2
            exit 1
        fi
        ;;
    help|--help|-h)
        cat <<'HELP'
Unity CLI

Usage:
  unity [--project_name NAME ...]   Launch the ConversationManager sandbox
  unity setup                        Bootstrap / re-bootstrap local Orchestra
  unity stop                         Stop local Orchestra
  unity status                       Show local Orchestra status
  unity restart                      Restart local Orchestra (wipes DB)
  unity help                         Show this message

Any unrecognized first argument is passed through to the sandbox, so
existing flags like --project_name, --overwrite, --real-comms still work.
HELP
        ;;
    *)
        cd "\$UNITY_REPO"
        # shellcheck disable=SC1091
        source .venv/bin/activate
        exec python -m sandboxes.conversation_manager.sandbox "\$@"
        ;;
esac
EOF
    chmod +x "$shim"
    log_success "Installed \`unity\` command at $shim"

    # Check PATH
    case ":$PATH:" in
        *":$CLI_DIR:"*) ;;
        *)
            log_warn "$CLI_DIR is not on your PATH."
            log_info "Add this to your shell profile:"
            echo "    export PATH=\"$CLI_DIR:\$PATH\""
            ;;
    esac
}

# ----------------------------------------------------------------------------
# Run setup.sh at the end (clones orchestra, spins up local backend, writes .env)
# ----------------------------------------------------------------------------
run_setup() {
    [ "$RUN_SETUP" = "false" ] && {
        log_info "Skipping local Orchestra spin-up (--skip-setup). Run \`unity setup\` later."
        SETUP_OK=skipped
        return 0
    }

    if [ ! -x "$UNITY_REPO/scripts/setup.sh" ]; then
        log_warn "setup.sh not found at $UNITY_REPO/scripts/setup.sh — skipping Orchestra spin-up."
        SETUP_OK=failed
        return 0
    fi

    # Export UNITY_HOME so setup.sh uses the same layout. Export PATH
    # additions so setup.sh finds uv (which install.sh just installed
    # to ~/.local/bin but didn't add to PATH globally yet).
    UNITY_HOME="$UNITY_HOME" \
        PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" \
        bash "$UNITY_REPO/scripts/setup.sh"
    local setup_exit=$?

    if (( setup_exit == 0 )); then
        SETUP_OK=ok
    else
        SETUP_OK=failed
        log_warn "Local Orchestra setup didn't complete (exit=$setup_exit)."
        log_info "Re-run after fixing the issue:  unity setup"
    fi
}

# ----------------------------------------------------------------------------
# Post-install hints
# ----------------------------------------------------------------------------
print_next_steps() {
    echo ""
    if [ "${SETUP_OK:-}" = "failed" ]; then
        echo -e "${YELLOW}${BOLD}Installation partially complete.${NC}"
        echo "  Code is installed; local Orchestra didn't start. See warnings above."
    else
        echo -e "${GREEN}${BOLD}Installation complete.${NC}"
    fi
    echo ""
    echo -e "${BOLD}Next steps:${NC}"
    echo ""
    if [ "${SETUP_OK:-}" = "ok" ]; then
        echo "  1. Add an LLM provider key to $UNITY_REPO/.env"
        echo "     OPENAI_API_KEY or ANTHROPIC_API_KEY"
        echo "     (ORCHESTRA_URL and UNIFY_KEY are already wired to local Orchestra.)"
    else
        echo "  1. Bootstrap local Orchestra:"
        echo -e "     ${CYAN}\$ unity setup${NC}"
        echo ""
        echo "  2. Add an LLM provider key to $UNITY_REPO/.env"
        echo "     OPENAI_API_KEY or ANTHROPIC_API_KEY"
    fi
    echo ""
    if [ "$CREATE_CLI" = "true" ]; then
        echo "  2. Start the conversation-manager sandbox:"
        echo -e "     ${CYAN}\$ unity --project_name Sandbox --overwrite${NC}"
        echo ""
        echo "  Also available:"
        echo -e "     ${CYAN}\$ unity status${NC}    Local Orchestra status"
        echo -e "     ${CYAN}\$ unity stop${NC}      Stop local Orchestra"
        echo -e "     ${CYAN}\$ unity help${NC}      Subcommand reference"
    else
        echo "  2. Activate the venv and start the sandbox:"
        echo "     \$ cd $UNITY_REPO"
        echo -e "     ${CYAN}\$ source .venv/bin/activate${NC}"
        echo -e "     ${CYAN}\$ python -m sandboxes.conversation_manager.sandbox --project_name Sandbox --overwrite${NC}"
    fi
    echo ""
    echo "  Documentation: $UNITY_REPO/README.md"
    echo "  Architecture:  $UNITY_REPO/ARCHITECTURE.md"
    echo ""
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
main() {
    print_banner
    detect_os
    ensure_uv
    ensure_python
    check_system_deps
    do_install
    create_cli
    run_setup
    print_next_steps
}

main "$@"
