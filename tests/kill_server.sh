#!/usr/bin/env bash
set -euo pipefail

# Kill the tmux server for the current terminal session
#
# Usage:
#   ./.kill_server.sh       # Kill THIS terminal's tmux server
#   ./.kill_server.sh --all # Kill ALL unity test tmux servers

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

KILL_ALL=0

while (( "$#" )); do
  case "$1" in
    --all)
      KILL_ALL=1
      shift
      ;;
    -h|--help)
      echo "Usage: ./.kill_server.sh [--all]"
      echo ""
      echo "Kill the tmux server for test sessions."
      echo ""
      echo "By default, kills only THIS terminal's tmux server (isolated socket)."
      echo ""
      echo "Options:"
      echo "  --all      Kill ALL unity test tmux servers across all terminals"
      echo "  -h, --help Show this help"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if (( KILL_ALL )); then
  # Kill all unity* servers
  count=0
  for sock in /tmp/tmux-"$(id -u)"/unity*; do
    [ -e "$sock" ] || continue
    name=$(basename "$sock")
    if tmux -L "$name" kill-server 2>/dev/null; then
      echo "Killed server: $name"
      ((count++)) || true
    fi
  done
  if (( count == 0 )); then
    echo "No unity test servers found."
  else
    echo "Killed $count server(s)."
  fi
else
  # Kill just this terminal's server
  if tmux -L "$TMUX_SOCKET" kill-server 2>/dev/null; then
    echo "Killed server: $TMUX_SOCKET"
  else
    echo "No server running for socket: $TMUX_SOCKET"
  fi
fi
