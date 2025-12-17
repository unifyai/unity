#!/usr/bin/env bash
# Common shell utilities for test helper scripts.
#
# This file is sourced by parallel_run.sh, attach.sh, kill_failed.sh,
# kill_server.sh, list_runs.sh, and watch_tests.sh to eliminate code duplication.
#
# Usage (in other scripts):
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
#   source "$SCRIPT_DIR/_shell_common.sh"

# ---- UTF-8 Locale for Unicode emoji support ----
# Session names use emojis (⏳ ✅ ❌) to indicate status. Without proper locale,
# these get corrupted to underscores, breaking prefix detection and failure detection.
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# ---- Terminal-based tmux socket isolation ----
# Each terminal session gets its own isolated tmux server via a unique socket.
# This prevents agents from interfering with each other.
#
# The socket is derived from the terminal's TTY device, which is unique
# and stable for each terminal session.
_derive_socket_name() {
  local tty_id
  tty_id=$(tty 2>/dev/null)
  if [[ "$tty_id" == "not a tty" || -z "$tty_id" || ! "$tty_id" =~ ^/ ]]; then
    tty_id="pid$$"
  else
    tty_id=$(echo "$tty_id" | sed 's|/|_|g')
  fi
  echo "unity${tty_id}"
}

# Default socket name (can be overridden via UNITY_TEST_SOCKET env var)
UNITY_TMUX_SOCKET="${UNITY_TEST_SOCKET:-$(_derive_socket_name)}"

# ---- Timeout command wrapper ----
# Determines the best available timeout command for avoiding hangs on dead sockets.
# Sets UNITY_TIMEOUT_CMD to either "timeout 1", "gtimeout 1", or empty string.
_setup_timeout_cmd() {
  if command -v timeout >/dev/null 2>&1; then
    UNITY_TIMEOUT_CMD="timeout 1"
  elif command -v gtimeout >/dev/null 2>&1; then
    UNITY_TIMEOUT_CMD="gtimeout 1"
  else
    UNITY_TIMEOUT_CMD=""
  fi
}
_setup_timeout_cmd

# ---- Tmux helpers ----
# List sessions from a socket with timeout protection
_tmux_ls() {
  local sock="$1"
  if [[ -n "$UNITY_TIMEOUT_CMD" ]]; then
    $UNITY_TIMEOUT_CMD tmux -L "$sock" ls 2>/dev/null || true
  else
    tmux -L "$sock" ls 2>/dev/null || true
  fi
}

# Get all unity* tmux sockets for the current user
_get_unity_sockets() {
  local socket_dir="/tmp/tmux-$(id -u)"
  if [[ -d "$socket_dir" ]]; then
    for sock in "$socket_dir"/unity*; do
      [[ -e "$sock" ]] && basename "$sock"
    done
  fi
}
