#!/usr/bin/env bash
set -euo pipefail

# Attach to a tmux test session
#
# Usage:
#   attach.sh <session-name>                    # Attach in THIS terminal's socket
#   attach.sh --socket <name> <session-name>    # Attach to a session in a specific socket
#   attach.sh --all                             # List sessions from ALL terminals

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

TMUX_SOCKET="${UNITY_TEST_SOCKET:-$(_derive_socket_name)}"

show_help() {
  echo "Usage: attach.sh [--socket <name>] <session-name>"
  echo ""
  echo "Attach to a tmux test session."
  echo ""
  echo "Arguments:"
  echo "  <session-name>  Name of the session to attach to"
  echo ""
  echo "Options:"
  echo "  -s, --socket NAME  Use a specific socket (use 'list_runs.sh' to find names)"
  echo "  --all              List sessions from ALL terminals"
  echo "  -h, --help         Show this help"
  echo ""
  echo "Examples:"
  echo "  attach.sh 'p ✅ test_contact_manager-test_ask'"
  echo "  attach.sh --socket unity_dev_ttys042 'f ❌ test_actor-test_code_act'"
}

LIST_ALL=0
EXPLICIT_SOCKET=""
SESSION_NAME=""

while (( "$#" )); do
  case "$1" in
    --all)
      LIST_ALL=1
      shift
      ;;
    -s|--socket)
      if [[ -n "${2-}" ]]; then
        EXPLICIT_SOCKET="$2"
        shift 2
      else
        echo "Error: --socket requires a socket name argument." >&2
        echo "Use 'list_runs.sh' to see available sockets." >&2
        exit 2
      fi
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      SESSION_NAME="$1"
      shift
      ;;
  esac
done

# Use explicit socket if provided
if [[ -n "$EXPLICIT_SOCKET" ]]; then
  TMUX_SOCKET="$EXPLICIT_SOCKET"
fi

if (( LIST_ALL )); then
  echo "Sessions across all terminals:"
  echo ""
  for sock in /tmp/tmux-"$(id -u)"/unity*; do
    [ -e "$sock" ] || continue
    name=$(basename "$sock")
    echo "=== $name ==="
    tmux -L "$name" ls 2>/dev/null || echo "(no sessions)"
    echo
  done
  exit 0
fi

if [[ -z "$SESSION_NAME" ]]; then
  echo "Error: No session name provided." >&2
  echo "" >&2
  show_help >&2
  exit 2
fi

exec tmux -L "$TMUX_SOCKET" attach -t "$SESSION_NAME"
