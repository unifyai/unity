#!/usr/bin/env bash
# Common shell utilities for repairs query scripts.
#
# This file is sourced by parallel_queries.sh and other helper scripts.
#
# Usage (in other scripts):
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
#   source "$SCRIPT_DIR/_repairs_common.sh"

# ---- UTF-8 Locale for Unicode emoji support ----
# Session names use emojis (⏳ ✅ ❌) to indicate status.
# Try to set UTF-8 locale, but don't fail if not available.
if locale -a 2>/dev/null | grep -qi 'en_US.utf'; then
  export LC_ALL=en_US.UTF-8
  export LANG=en_US.UTF-8
elif locale -a 2>/dev/null | grep -qi 'C.utf'; then
  export LC_ALL=C.UTF-8
  export LANG=C.UTF-8
else
  # Fallback: just set LANG, skip LC_ALL to avoid warnings
  export LANG="${LANG:-C.UTF-8}"
fi

# ---- Terminal-based tmux socket isolation ----
# Each terminal session gets its own isolated tmux server via a unique socket.
_derive_socket_name() {
  local tty_id
  tty_id=$(tty 2>/dev/null)
  if [[ "$tty_id" == "not a tty" || -z "$tty_id" || ! "$tty_id" =~ ^/ ]]; then
    tty_id="pid$$"
  else
    tty_id=$(echo "$tty_id" | sed 's|/|_|g')
  fi
  echo "repairs${tty_id}"
}

# Default socket name (can be overridden via REPAIRS_TMUX_SOCKET env var)
REPAIRS_TMUX_SOCKET="${REPAIRS_TMUX_SOCKET:-$(_derive_socket_name)}"

# ---- Log directory naming ----
# Format: YYYY-MM-DDTHH-MM-SS_{socket_name}
_derive_log_subdir() {
  local socket_name="$1"
  local datetime
  datetime=$(date +"%Y-%m-%dT%H-%M-%S")
  echo "${datetime}_${socket_name}"
}

# ---- Timeout command wrapper ----
_setup_timeout_cmd() {
  if command -v timeout >/dev/null 2>&1; then
    REPAIRS_TIMEOUT_CMD="timeout 1"
  elif command -v gtimeout >/dev/null 2>&1; then
    REPAIRS_TIMEOUT_CMD="gtimeout 1"
  else
    REPAIRS_TIMEOUT_CMD=""
  fi
}
_setup_timeout_cmd

# ---- Tmux helpers ----
_tmux_ls() {
  local sock="$1"
  if [[ -n "$REPAIRS_TIMEOUT_CMD" ]]; then
    $REPAIRS_TIMEOUT_CMD tmux -L "$sock" ls 2>/dev/null || true
  else
    tmux -L "$sock" ls 2>/dev/null || true
  fi
}

# Get all repairs* tmux sockets for the current user
_get_repairs_sockets() {
  local socket_dir="/tmp/tmux-$(id -u)"
  if [[ -d "$socket_dir" ]]; then
    for sock in "$socket_dir"/repairs*; do
      [[ -e "$sock" ]] && basename "$sock"
    done
  fi
}

# ---- Color output helpers ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

info() {
  echo -e "${BLUE}ℹ${NC} $*"
}

success() {
  echo -e "${GREEN}✓${NC} $*"
}

warn() {
  echo -e "${YELLOW}⚠${NC} $*" >&2
}

error() {
  echo -e "${RED}✗${NC} $*" >&2
}

# Status indicators (matching tests/parallel_run.sh style)
print_pass() {
  echo -e "  ${GREEN}p ✅${NC} $*"
}

print_fail() {
  echo -e "  ${RED}f ❌${NC} $*"
}

print_pending() {
  echo -e "  ${YELLOW}r ⏳${NC} $*"
}
