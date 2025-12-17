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

# ---- Worktree log symlink management ----
# When running tests from a git worktree, create symlinks in the main repo's
# log directories pointing to this worktree's logs. This lets you browse all
# worktree logs from a single location (the main repo).
#
# Creates symlinks like:
#   /main/repo/.pytest_logs/worktree-oty -> /path/to/worktree/oty/.pytest_logs
#   /main/repo/.llm_io_debug/worktree-oty -> /path/to/worktree/oty/.llm_io_debug

# Check if we're in a git worktree (not the main repo)
_is_git_worktree() {
  local git_dir
  git_dir="$(git rev-parse --git-dir 2>/dev/null)" || return 1
  # In a worktree, .git is a file pointing to the main repo's .git/worktrees/<name>
  # In the main repo, .git is a directory
  [[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/.git" ]]
}

# Get the main (non-worktree) repo path
_get_main_repo_path() {
  local git_common_dir
  git_common_dir="$(git rev-parse --git-common-dir 2>/dev/null)" || return 1
  # git-common-dir returns the main .git directory (e.g., /main/repo/.git)
  # We want the parent (the repo root)
  dirname "$git_common_dir"
}

# Get a descriptive name for the current worktree (used in symlink names)
_get_worktree_name() {
  local worktree_root
  worktree_root="$(git rev-parse --show-toplevel 2>/dev/null)" || return 1
  # Use the last component of the path as the name
  # e.g., /Users/djl11/.cursor/worktrees/unity/oty -> oty
  basename "$worktree_root"
}

# Ensure log directory symlinks exist in the main repo
# Call this from parallel_run.sh before running tests
_ensure_worktree_log_symlinks() {
  # Only proceed if we're in a worktree
  _is_git_worktree || return 0

  local main_repo worktree_name worktree_root
  main_repo="$(_get_main_repo_path)" || return 0
  worktree_name="$(_get_worktree_name)" || return 0
  worktree_root="$(git rev-parse --show-toplevel 2>/dev/null)" || return 0

  # Create symlinks for each log directory
  for log_dir in .pytest_logs .llm_io_debug; do
    local main_log_dir="$main_repo/$log_dir"
    local worktree_log_dir="$worktree_root/$log_dir"
    local symlink_path="$main_log_dir/worktree-$worktree_name"

    # Ensure the main repo's log directory exists
    mkdir -p "$main_log_dir" 2>/dev/null || continue

    # Ensure the worktree's log directory exists
    mkdir -p "$worktree_log_dir" 2>/dev/null || continue

    # Create or update the symlink (remove stale symlink first)
    if [[ -L "$symlink_path" ]]; then
      # Symlink exists - check if it points to the right place
      local current_target
      current_target="$(readlink "$symlink_path" 2>/dev/null)"
      if [[ "$current_target" != "$worktree_log_dir" ]]; then
        rm -f "$symlink_path"
        ln -s "$worktree_log_dir" "$symlink_path"
      fi
    elif [[ ! -e "$symlink_path" ]]; then
      # No file/symlink exists - create it
      ln -s "$worktree_log_dir" "$symlink_path"
    fi
    # If something else exists at that path, leave it alone
  done
}
