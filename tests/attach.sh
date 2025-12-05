#!/usr/bin/env bash
set -euo pipefail

# Attach to a tmux test session
#
# Usage:
#   attach.sh <session-name>   # Attach to a session in THIS terminal's socket
#   attach.sh --all            # List sessions from ALL terminals (to help find the right one)

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
  echo "Usage: attach.sh <session-name>"
  echo ""
  echo "Attach to a tmux test session."
  echo ""
  echo "Arguments:"
  echo "  <session-name>  Name of the session to attach to (use watch_tests.sh to see available sessions)"
  echo ""
  echo "Options:"
  echo "  --all           List sessions from ALL terminals"
  echo "  -h, --help      Show this help"
  echo ""
  echo "Examples:"
  echo "  attach.sh 'd ✅ test_contact_manager-test_ask'"
  echo "  attach.sh 'f ❌ test_actor-test_code_act'"
}

if [[ $# -eq 0 ]]; then
  echo "Error: No session name provided." >&2
  echo "" >&2
  show_help >&2
  exit 2
fi

case "$1" in
  --all)
    echo "Sessions across all terminals:"
    echo ""
    for sock in /tmp/tmux-$(id -u)/unity*; do
      [ -e "$sock" ] || continue
      name=$(basename "$sock")
      echo "=== $name ==="
      tmux -L "$name" ls 2>/dev/null || echo "(no sessions)"
      echo
    done
    exit 0
    ;;
  -h|--help)
    show_help
    exit 0
    ;;
  *)
    exec tmux -L "$TMUX_SOCKET" attach -t "$1"
    ;;
esac
