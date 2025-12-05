#!/usr/bin/env bash
set -euo pipefail

# Watch test progress for the current terminal session
#
# Usage:
#   ./.watch_tests.sh       # Watch tests from THIS terminal
#   ./.watch_tests.sh --all # Watch tests from ALL terminals

# ---- Terminal-based isolation ----
# Uses the same socket detection as .parallel_run.sh
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

WATCH_ALL=0

while (( "$#" )); do
  case "$1" in
    --all)
      WATCH_ALL=1
      shift
      ;;
    -h|--help)
      echo "Usage: ./.watch_tests.sh [--all]"
      echo ""
      echo "Watch test progress in real-time."
      echo ""
      echo "By default, shows only tests from THIS terminal (isolated socket)."
      echo ""
      echo "Options:"
      echo "  --all      Show tests from ALL terminals"
      echo "  -h, --help Show this help"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if (( WATCH_ALL )); then
  # Watch all unity sockets
  exec watch -n 0.5 '
    for sock in /tmp/tmux-$(id -u)/unity*; do
      [ -e "$sock" ] || continue
      name=$(basename "$sock")
      echo "=== $name ==="
      tmux -L "$name" ls 2>/dev/null || echo "(no sessions)"
      echo
    done
  '
else
  # Watch just this terminal's socket
  exec watch -n 0.5 "tmux -L $TMUX_SOCKET ls 2>/dev/null || echo '(no sessions)'"
fi
