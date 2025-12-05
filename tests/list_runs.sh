#!/usr/bin/env bash
set -euo pipefail

# List all active test run sockets and their sessions
#
# Usage:
#   list_runs.sh           # List sockets with active sessions
#   list_runs.sh --all     # Include empty sockets too
#   list_runs.sh --quiet   # Just list socket names (for scripting)

# ---- Terminal-based isolation ----
# Uses the same socket detection as parallel_run.sh
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

CURRENT_SOCKET="$(_derive_socket_name)"

QUIET=0
SHOW_ALL=0

while (( "$#" )); do
  case "$1" in
    -q|--quiet)
      QUIET=1
      shift
      ;;
    -a|--all)
      SHOW_ALL=1
      shift
      ;;
    -h|--help)
      echo "Usage: list_runs.sh [-q|--quiet] [-a|--all]"
      echo ""
      echo "List all active test run sockets and their sessions."
      echo ""
      echo "By default, only shows sockets with active sessions."
      echo ""
      echo "Options:"
      echo "  -a, --all    Include empty sockets (no active sessions)"
      echo "  -q, --quiet  Just list socket names (for scripting)"
      echo "  -h, --help   Show this help"
      echo ""
      echo "Use the socket name with other commands:"
      echo "  watch_tests --socket <socket>         Watch a specific socket"
      echo "  attach --socket <socket> <session>    Attach to a session"
      echo "  kill_failed --socket <socket>         Kill failed sessions"
      echo "  kill_server --socket <socket>         Kill the entire server"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# Find all unity sockets
SOCKETS=()
for sock in /tmp/tmux-"$(id -u)"/unity*; do
  [ -e "$sock" ] || continue
  SOCKETS+=( "$(basename "$sock")" )
done

if (( ${#SOCKETS[@]} == 0 )); then
  if (( QUIET )); then
    exit 0
  else
    echo "No test sockets found."
    exit 0
  fi
fi

# Count sockets with sessions (first pass)
sockets_with_sessions=0
empty_sockets=0
for socket in "${SOCKETS[@]}"; do
  sessions_output=$(LC_ALL=en_US.UTF-8 tmux -L "$socket" ls 2>/dev/null || true)
  if [[ -n "$sessions_output" ]]; then
    ((sockets_with_sessions++)) || true
  else
    ((empty_sockets++)) || true
  fi
done

if (( sockets_with_sessions == 0 )); then
  if (( QUIET )); then
    exit 0
  else
    if (( SHOW_ALL )); then
      echo "Found ${#SOCKETS[@]} socket(s), but none have active sessions."
    else
      echo "No active test runs found."
      echo ""
      echo "Tip: Use --all to see ${#SOCKETS[@]} empty socket(s)"
    fi
    exit 0
  fi
fi

# Detailed output (or quiet mode)
for socket in "${SOCKETS[@]}"; do
  sessions_output=$(LC_ALL=en_US.UTF-8 tmux -L "$socket" ls 2>/dev/null || true)
  has_sessions=0
  [[ -n "$sessions_output" ]] && has_sessions=1

  # Skip empty sockets unless --all
  if (( ! SHOW_ALL && ! has_sessions )); then
    continue
  fi

  if (( QUIET )); then
    echo "$socket"
    continue
  fi

  # Check if this is the current terminal's socket
  if [[ "$socket" == "$CURRENT_SOCKET" ]]; then
    marker="(current terminal)"
  else
    marker="(orphaned)"
  fi

  echo "=== $socket $marker ==="

  if (( ! has_sessions )); then
    echo "  (no sessions)"
    echo ""
    continue
  fi

  # Count sessions by status
  running=0
  passed=0
  failed=0

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    session_name="${line%%:*}"
    case "$session_name" in
      "r "*)  ((running++)) || true ;;
      "p "*)  ((passed++)) || true ;;
      "f "*)  ((failed++)) || true ;;
    esac
    echo "  $session_name"
  done <<< "$sessions_output"

  # Summary line
  echo ""
  echo "  Summary: $running running, $passed passed, $failed failed"
  echo ""
done

if (( QUIET )); then
  exit 0
fi

# Show count of hidden sockets if not --all
if (( ! SHOW_ALL && empty_sockets > 0 )); then
  echo "($empty_sockets empty socket(s) hidden - use --all to show)"
  echo ""
fi

echo "Tip: Use --socket <name> with watch_tests, attach, kill_failed, or kill_server"
